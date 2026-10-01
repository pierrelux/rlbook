"""Deterministic PDE plume recordings, independent of surrogate/MPPI artifacts."""
from dataclasses import replace
from functools import partial
from pathlib import Path
import hashlib
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from thermal_reactor import (ReactorConfig, inlet_profile, velocity_profile,
                             rhs_with_inlet, stable_step_size, warm_state,
                             field_metrics, valid_fields)

PLUME_CONFIG = ReactorConfig(velocity_shear=.2)
FRAME_DT = .25
DURATION = 80.
SCENARIOS = (("constant", True, False), ("boost", True, True),
             ("constant-reference", False, False), ("boost-reference", False, True))


def pulse_envelope(t):
    phase = jnp.clip((jnp.asarray(t) - 6.) / 8., 0., 1.)
    return jnp.where((t > 6.) & (t < 14.), jnp.sin(jnp.pi * phase)**2, 0.)


def plume_inlet(t, pulse=True, config=PLUME_CONFIG):
    y = (jnp.arange(config.ny) + .5) * config.width / config.ny
    patch = jnp.exp(-.5 * ((y - .55) / .22)**2)
    return inlet_profile(0., config) - 100 * jnp.asarray(pulse)[..., None] * pulse_envelope(t) * patch


def plume_controls(t, boost=False):
    on = jnp.asarray(boost) * ((t >= 6.) & (t < 18.))
    return jnp.broadcast_to(jnp.float32(.48), on.shape + (4,)).at[..., 0].add(.12 * on)


@partial(jax.jit, static_argnames=("config",))
def plume_step(fields, start, elapsed, pulse=True, boost=False, config=PLUME_CONFIG,
               held_controls=None):
    """Nonautonomous SSPRK2, exact switches and stage-time inlet forcing.

    Prescribed heating is constant within a substep. Sampling at its midpoint
    chooses the correct one-sided value at a heater switching boundary.
    ``held_controls`` optionally supplies arbitrary heater inputs for the whole
    caller interval, preserving the original prescribed schedule by default.
    """
    fields = jnp.asarray(fields, jnp.float32)
    def advance(carry):
        local_t, x = carry
        t = start + local_t
        switches = jnp.array([6., 14., 18.])
        next_switch = jnp.min(jnp.where(switches > t + 1e-5, switches - t, jnp.inf))
        h = jnp.minimum(stable_step_size(x, config), jnp.minimum(elapsed - local_t, next_switch))
        u = plume_controls(t + h / 2, boost) if held_controls is None else held_controls
        first = x + h * rhs_with_inlet(x, u, plume_inlet(t, pulse, config), config)
        second = .5 * x + .5 * (first + h * rhs_with_inlet(first, u, plume_inlet(t + h, pulse, config), config))
        return local_t + h, second
    return jax.lax.while_loop(lambda z: z[0] < elapsed - 1e-6, advance, (jnp.float32(0), fields))[1]


@partial(jax.jit, static_argnames=("config", "duration", "frame_dt"))
def plume_rollout(initial, pulse=True, boost=False, config=PLUME_CONFIG,
                  duration=DURATION, frame_dt=FRAME_DT):
    times = jnp.arange(int(round(duration / frame_dt)), dtype=jnp.float32) * frame_dt
    def advance(x, t):
        y = plume_step(x, t, frame_dt, pulse, boost, config)
        return y, y
    following = jax.lax.scan(advance, initial, times)[1]
    return jnp.concatenate([initial[None], following], axis=0)


def motion_metrics(fields, reference, config=PLUME_CONFIG):
    rows = []
    x = (np.arange(config.nx) + .5) * config.length / config.nx
    for t in (20, 30, 40):
        k = int(t / FRAME_DT)
        cold = np.maximum(reference[k, 0] - fields[k, 0], 0)
        rows.append({"time_s": t, "maximum_cold_contrast_K": float(cold.max()),
                     "cold_centroid_x_m": float(np.sum(cold * x) / cold.sum()),
                     "maximum_reactant_excess": float((fields[k, 1] - reference[k, 1]).max())})
    return rows


def refinement(initial, coarse, config=PLUME_CONFIG):
    """Same initial cell averages and forcings, compared at all saved times."""
    fine_t = np.asarray(plume_rollout(initial, True, False, replace(config, max_dt=.05)))
    fine_cfg = replace(config, nx=2 * config.nx, ny=2 * config.ny, max_dt=.025)
    fine_initial = np.repeat(np.repeat(initial, 2, -1), 2, -2)
    fine_x = np.asarray(plume_rollout(fine_initial, True, False, fine_cfg))
    restricted = fine_x.reshape(len(fine_x), 2, config.ny, 2, config.nx, 2).mean(axis=(3, 5))
    def errors(other, other_conversion):
        return {"temperature_rmse_K": float(np.sqrt(np.mean((coarse[:, 0] - other[:, 0])**2))),
                "peak_temperature_difference_K": float(np.max(np.abs(coarse[:, 0].max(axis=(1, 2)) - other[:, 0].max(axis=(1, 2))))),
                "outlet_conversion_mae": float(np.mean(np.abs(field_metrics(coarse, config)['outlet_conversion'] - other_conversion)))}
    return {"time_refinement": errors(fine_t, field_metrics(fine_t, config)['outlet_conversion']),
            "space_refinement": errors(restricted, field_metrics(fine_x, fine_cfg)['outlet_conversion']),
            "fine_mesh": [fine_cfg.ny, fine_cfg.nx], "duration_s": DURATION,
            "initial_transfer": "piecewise constant prolongation preserves each coarse cell average",
            "note": "First-order upwind spreading includes numerical diffusion of about v*dx/2 (roughly 0.013 m²/s), exceeding the physical 0.0005 m²/s; refinement measures this discretization sensitivity."}


def validate_assets(assets):
    assets = Path(assets)
    m = json.loads((assets / 'manifest.json').read_text())
    times = np.asarray(m['times_s'])
    np.testing.assert_allclose(np.diff(times), FRAME_DT)
    assert times[0] == 0 and times[-1] == DURATION
    arrays = {}
    for row in m['runs']:
        desc = row['fields']; raw = (assets / desc['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == desc['sha256']
        a = np.frombuffer(raw, dtype='<f4').reshape(desc['shape'])
        assert valid_fields(a)
        metrics = field_metrics(a, ReactorConfig(**m['config']))
        for key in metrics:
            np.testing.assert_allclose(metrics[key], row[key], atol=1e-6)
        assert len(a) == len(times) == len(row['controls'])
        arrays[row['id']] = a
    for method in ('constant', 'boost'):
        a, b = arrays[method], arrays[method + '-reference']
        np.testing.assert_array_equal(a[:25], b[:25])  # identical through pulse onset at6s
        runs = {r['id']: r for r in m['runs']}
        assert runs[method]['controls'] == runs[method + '-reference']['controls']
    assert np.array_equal(arrays['constant'][0], arrays['boost'][0])
    motion = motion_metrics(arrays['constant'], arrays['constant-reference'])
    assert all(r['maximum_cold_contrast_K'] > 10 for r in motion)
    assert all(r['maximum_reactant_excess'] > .05 for r in motion)
    assert all(b['cold_centroid_x_m'] - a['cold_centroid_x_m'] > 1 for a, b in zip(motion, motion[1:]))
    return {"arrays_checked": len(arrays), "frames_per_run": len(times),
            "physical_states_valid": True, "matched_references_verified": True,
            "motion": motion}


def storyboard(arrays, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(3, 4, figsize=(12, 4.8), layout='constrained')
    for col, t in enumerate((10, 20, 30, 40)):
        k = int(t / FRAME_DT)
        values = (arrays[k, 0, 0], 1 - arrays[k, 0, 1], arrays[k, 0, 0] - arrays[k, 2, 0])
        for row, (v, limits, cmap) in enumerate(zip(values, ((680, 1120), (0, 1), (-100, 100)), ('magma', 'viridis', 'RdBu_r'))):
            im = ax[row, col].imshow(v, origin='lower', extent=(0, 8, 0, 2), vmin=limits[0], vmax=limits[1], cmap=cmap)
            ax[row, col].set(xticks=[0, 4, 8], yticks=[0, 2], xlabel='Downstream position (m)')
            ax[row, col].axvline(4, color='white', ls=':', lw=.6)
            ax[row, col].axhline(1, color='white', ls=':', lw=.6)
            if col == 0:
                ax[row, col].set_ylabel('Width (m)')
            if row == 0:
                ax[row, col].set_title(f'{t} s')
            if col == 3:
                fig.colorbar(im, ax=ax[row, :], shrink=.8, fraction=.018, pad=.015,
                             label=('Temperature (K)', 'Conversion', 'Pulse effect (K)')[row])
    dest = Path(root) / '_static' / 'thermal_reactor'; dest.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(dest / f'plume-storyboard.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def generate_plume(root):
    root = Path(root); assets = root / 'interactive' / 'thermal-plume-data'
    reports = root / 'artifacts' / 'thermal_plume'
    assets.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    initial = warm_state(PLUME_CONFIG)
    print('Warm shear-flow state generated', flush=True)
    pulse = jnp.array([s[1] for s in SCENARIOS]); boost = jnp.array([s[2] for s in SCENARIOS])
    init = jnp.broadcast_to(initial, (4,) + initial.shape)
    arrays = np.asarray(plume_rollout(init, pulse, boost))
    assert valid_fields(arrays)
    times = np.arange(len(arrays)) * FRAME_DT
    import matplotlib
    manifest = {"version": 1, "scenario": "prescribed-shear-cold-pulse", "config": PLUME_CONFIG.to_dict(),
                "times_s": times.tolist(), "frame_dt_s": FRAME_DT, "duration_s": DURATION,
                "velocity_m_s": np.asarray(velocity_profile(PLUME_CONFIG)).tolist(),
                "inlet_y_m": ((np.arange(PLUME_CONFIG.ny) + .5) * PLUME_CONFIG.width / PLUME_CONFIG.ny).tolist(),
                "pulse": {"start_s": 6, "end_s": 14, "amplitude_K": -100, "center_y_m": .55, "sigma_y_m": .22},
                "warmup": {"duration_s": 400, "heater_fraction": .48},
                "limits": {"temperature_K": [680, 1120], "conversion": [0, 1],
                           "temperature_error_K": [-100, 100], "conversion_error": [-.2, .2]},
                "colormaps": {name: (matplotlib.colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).astype(int).tolist()
                              for name in ('magma', 'viridis', 'RdBu_r')},
                "runs": [], "note": "Numerical PDE responses under prescribed heating; not learned or optimized controls. Outlet conversion is weighted by outgoing reactant flux."}
    for i, (name, has_pulse, has_boost) in enumerate(SCENARIOS):
        a = np.asarray(arrays[:, i], dtype='<f4'); path = assets / f'{name}.bin'; a.tofile(path)
        metrics = field_metrics(a, PLUME_CONFIG)
        controls = np.asarray(jax.vmap(lambda t: plume_controls(t, has_boost))(jnp.asarray(times)))
        inlet = np.asarray(jax.vmap(lambda t: plume_inlet(t, has_pulse))(jnp.asarray(times)))
        energy = np.r_[0., np.cumsum(controls[:-1].astype(np.float64).mean(axis=-1)) * FRAME_DT]
        row = {"id": name, "pulse": has_pulse, "boost": has_boost,
               "reference_id": name + '-reference' if has_pulse else None,
               "fields": {"file": path.name, "shape": list(a.shape), "dtype": "float32-le",
                          "channels": ['temperature_K', 'unreacted_fraction'], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
               "controls": controls.tolist(), "inlet_temperature_K": inlet.tolist(),
               "energy_s": energy.tolist(), **{k: v.tolist() for k, v in metrics.items()},
               "summary": {"maximum_temperature_K": float(metrics['peak_temperature_K'].max()),
                           "minimum_outlet_conversion": float(metrics['outlet_conversion'].min()),
                           "temperature_violation_duration_s": float(np.sum(metrics['peak_temperature_K'][:-1] > 1070) * FRAME_DT),
                           "conversion_violation_duration_s": float(np.sum(metrics['outlet_conversion'][:-1] < .95) * FRAME_DT),
                           "energy_s": float(energy[-1])}}
        manifest['runs'].append(row)
    print('Four plume/reference recordings generated; checking refinement', flush=True)
    manifest['numerical_validation'] = refinement(initial, arrays[:, 0])
    (assets / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    checks = validate_assets(assets)
    checks['numerical_validation'] = manifest['numerical_validation']
    checks['elapsed_s'] = time.perf_counter() - started
    (reports / 'validation.json').write_text(json.dumps(checks, indent=2) + '\n')
    (reports / 'summary.json').write_text(json.dumps({r['id']: r['summary'] for r in manifest['runs']}, indent=2) + '\n')
    storyboard(arrays, root)
    print(json.dumps(checks, indent=2), flush=True)
    return checks
