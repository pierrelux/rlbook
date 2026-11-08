"""Direct-PDE MPPI restoration of a disturbed reactor's operating field.

The plant and future inlet are deterministic. Randomness belongs to the
optimizer; no surrogate checkpoint is loaded by this experiment.
"""
from dataclasses import dataclass, asdict, replace
from functools import partial
from pathlib import Path
import hashlib
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit, logit

from mppi_control import gaussian_mppi_update
from thermal_reactor import (physics_step, velocity_profile, warm_state,
                             field_metrics, valid_fields)
from thermal_plume import PLUME_CONFIG, plume_step


@dataclass(frozen=True)
class RecoveryConfig:
    horizon: int = 20
    knots: int = 6
    candidates: int = 256
    iterations: int = 4
    control_dt: float = 2.
    frame_dt: float = .25
    baseline: float = .46
    latent_scale: float = .12
    variance: float = .49
    temperature: float = .01
    temperature_scale: float = 20.
    conversion_scale: float = .1
    control_scale: float = .05
    deviation_weight: float = .01
    slew_weight: float = .005
    terminal_weight: float = 1.
    validation_tail_steps: int = 20
    start_s: float = 14.
    end_s: float = 80.
    seed: int = 19744


P = RecoveryConfig()


def target_and_initial(config=PLUME_CONFIG, planner=P):
    target = warm_state(config, planner.baseline)
    initial = np.asarray(plume_step(jnp.asarray(target), 0., planner.start_s, True,
                                   False, config, jnp.full(4, planner.baseline)))
    return target, initial


def seed_knots(planner=P):
    knots = np.full((planner.knots, 4), planner.baseline)
    knots[:2, 0] = .50
    return knots


def interpolate_knots(knots, planner=P):
    knots = np.asarray(knots)
    at = np.linspace(0, planner.horizon - 1, planner.knots)
    basis = np.stack([np.interp(np.arange(planner.horizon), at, row)
                      for row in np.eye(planner.knots)], axis=-1)
    return np.einsum('hk,...kj->...hj', basis, knots).astype(np.float32)


def decode(latent, reference_knots, planner=P):
    return interpolate_knots(expit(logit(np.clip(reference_knots, 1e-6, 1 - 1e-6))
                                  + planner.latent_scale * np.asarray(latent)), planner)


def proposal_center(tape, planner=P):
    """An approximation for sampling only; never replaces the exact tape."""
    at = np.linspace(0, planner.horizon - 1, planner.knots)
    return np.stack([np.interp(at, np.arange(planner.horizon), tape[:, j]) for j in range(4)], axis=-1)


def tracking_components(fields, target, planner=P):
    fields, target = jnp.asarray(fields), jnp.asarray(target)
    temperature = jnp.mean(((fields[..., 0, :, :] - target[0]) / planner.temperature_scale)**2, axis=(-2, -1))
    # Squared conversion error equals squared remaining-fraction error.
    conversion = jnp.mean(((fields[..., 1, :, :] - target[1]) / planner.conversion_scale)**2, axis=(-2, -1))
    return temperature, conversion


def tracking_error(fields, target, planner=P):
    a, b = tracking_components(fields, target, planner)
    return a + b


def control_cost(controls, previous, planner=P):
    deviation = planner.deviation_weight * jnp.mean(((controls - planner.baseline) / planner.control_scale)**2, axis=-1)
    slew = planner.slew_weight * jnp.mean(((controls - previous) / planner.control_scale)**2, axis=-1)
    return deviation + slew


@partial(jax.jit, static_argnames=('config', 'planner'))
def score_batch(initial, tapes, target, previous, config=PLUME_CONFIG, planner=P):
    """Costs at 2-second states; invalid/infeasible candidates have zero weight."""
    n = len(tapes)
    x = jnp.broadcast_to(initial, (n,) + initial.shape)
    v = velocity_profile(config)
    def advance(carry, u):
        x, old, total, feasible = carry
        y = physics_step(x, u, jnp.zeros(n), planner.control_dt, config)
        e = tracking_error(y, target, planner)
        conversion = 1 - jnp.sum(y[:, 1, :, -1] * v, axis=-1) / v.sum()
        physical = (jnp.isfinite(y).all(axis=(1, 2, 3)) & (y[:, 0].min(axis=(1, 2)) > 0)
                    & (y[:, 1].min(axis=(1, 2)) >= -2e-5) & (y[:, 1].max(axis=(1, 2)) <= 1 + 2e-5))
        ok = physical & (y[:, 0].max(axis=(1, 2)) <= config.temperature_limit) & (conversion >= config.conversion_target)
        return (y, u, total + e + control_cost(u, old, planner), feasible & ok), None
    final, _, total, feasible = jax.lax.scan(advance, (x, jnp.broadcast_to(previous, (n, 4)),
                                                     jnp.zeros(n), jnp.ones(n, dtype=bool)), tapes.swapaxes(0, 1))[0]
    raw = total / planner.horizon + planner.terminal_weight * tracking_error(final, target, planner)
    feasible = feasible & jnp.isfinite(raw)
    return jnp.where(feasible, raw, jnp.inf), raw, feasible


@partial(jax.jit, static_argnames=('config', 'planner'))
def recorded_rollout(initial, tape, config=PLUME_CONFIG, planner=P):
    """Actual PDE frames at the output cadence, including the initial state."""
    repeats = int(round(planner.control_dt / planner.frame_dt))
    controls = jnp.repeat(tape, repeats, axis=0)
    def advance(x, u):
        y = physics_step(x, u, jnp.zeros(x.shape[:-3]), planner.frame_dt, config)
        return y, y
    fields = jax.lax.scan(advance, initial, controls)[1]
    return jnp.concatenate([initial[None], fields], axis=0)


def validate_tape(initial, tape, config=PLUME_CONFIG, planner=P):
    extended = np.concatenate([tape, np.full((planner.validation_tail_steps, 4), planner.baseline)])
    fields = np.asarray(recorded_rollout(jnp.asarray(initial), jnp.asarray(extended), config, planner))
    m = field_metrics(fields, config)
    details = {'maximum_temperature_K': float(m['peak_temperature_K'].max()),
               'minimum_outlet_conversion': float(m['outlet_conversion'].min()),
               'duration_s': len(extended) * planner.control_dt,
               'frame_dt_s': planner.frame_dt}
    feasible = (valid_fields(fields) and details['maximum_temperature_K'] <= config.temperature_limit
                and details['minimum_outlet_conversion'] >= config.conversion_target)
    return feasible, details


class RecoveryController:
    def __init__(self, target, config=PLUME_CONFIG, planner=P, seed=None):
        self.target, self.config, self.planner = target, config, planner
        self.rng = np.random.default_rng(planner.seed if seed is None else seed)
        self.incumbent = interpolate_knots(seed_knots(planner), planner)
        self.previous = np.full(4, planner.baseline)
        self.last_plan = self.incumbent.copy()

    def score(self, x, tapes):
        parts = []
        # Reuse one compiled batch size at 128, 256, and512 candidates.
        for offset in range(0, len(tapes), 32):
            block = tapes[offset:offset + 32]
            n = len(block)
            if n < 32:
                block = np.concatenate([block, np.repeat(block[-1:], 32 - n, axis=0)])
            parts.append(tuple(np.asarray(a)[:n] for a in score_batch(jnp.asarray(x), jnp.asarray(block),
                         jnp.asarray(self.target), jnp.asarray(self.previous), self.config, self.planner)))
        return tuple(np.concatenate([part[i] for part in parts]) for i in range(3))

    def validate(self, x, tape):
        return validate_tape(x, tape, self.config, self.planner)

    def plan(self, x, capture=False):
        p = self.planner; started = time.perf_counter()
        incumbent = self.incumbent.copy()
        feasible, initial_check = self.validate(x, incumbent)
        record = {'iterations': [], 'planning_failure': not feasible,
                  'initial_validation': initial_check, 'sampling_seed': p.seed}
        if not feasible:
            record['runtime_s'] = time.perf_counter() - started
            return None, record, None
        incumbent_cost = float(self.score(x, incumbent[None])[1][0])
        record['initial_objective'] = incumbent_cost
        reference = proposal_center(incumbent, p)
        # This reference/decoder stays FIXED as the sampling mean changes.
        mean = np.zeros((p.knots, 4))
        snapshot = None
        for iteration in range(p.iterations):
            samples = mean + np.sqrt(p.variance) * self.rng.normal(size=(p.candidates, p.knots, 4))
            tapes = decode(samples, reference, p)
            masked, costs, candidate_feasible = self.score(x, tapes)
            before_cost = incumbent_cost
            entry = {'iteration': iteration + 1, 'feasible_candidates': int(candidate_feasible.sum()),
                     'accepted': False, 'alpha': 0., 'incumbent_before_objective': before_cost,
                     'ess': 0., 'weighted_proposal_objective': None, 'weighted_proposal_feasible': False}
            proposal = None; weights = np.zeros(p.candidates); proposed_tape = incumbent.copy()
            try:
                proposal, weights, ess = gaussian_mppi_update(samples, masked, mean, p.variance, p.temperature)
                entry['ess'] = float(ess)
                proposed_tape = decode(proposal, reference, p)
                _, raw, ok = self.score(x, proposed_tape[None])
                entry['weighted_proposal_objective'] = float(raw[0]) if np.isfinite(raw[0]) else None
                entry['weighted_proposal_feasible'] = bool(ok[0])
                for alpha in (1., .5, .25, .125):
                    trial_mean = mean + alpha * (proposal - mean)
                    trial = decode(trial_mean, reference, p)
                    _, raw, ok = self.score(x, trial[None])
                    if not ok[0] or not np.isfinite(raw[0]) or raw[0] >= incumbent_cost - 1e-7:
                        continue
                    valid, check = self.validate(x, trial)
                    if not valid:
                        continue
                    mean, incumbent, incumbent_cost = trial_mean, trial, float(raw[0])
                    entry.update({'accepted': True, 'alpha': alpha, 'accepted_validation': check})
                    break
            except ValueError as error:
                entry['failure_reason'] = str(error)
            entry['incumbent_after_objective'] = incumbent_cost
            record['iterations'].append(entry)
            if capture and iteration == p.iterations - 1:
                order = np.argsort(-weights)
                selected = order[[0, 1, 2, p.candidates // 4, p.candidates // 2, p.candidates - 1]]
                snapshot = {'initial': np.asarray(x), 'candidate_controls': tapes[selected],
                            'candidate_costs': costs[selected], 'candidate_weights': weights[selected],
                            'candidate_feasible': candidate_feasible[selected], 'candidate_indices': selected,
                            'weighted_controls': proposed_tape, 'used_controls': incumbent.copy(),
                            'weighted_proposal_exists': proposal is not None,
                            'reference_knots': reference, 'iteration_record': entry.copy()}
        self.last_plan = incumbent.copy()
        action = incumbent[0].copy()
        # Retain the exact tape. Resampling knots happens only for a proposal.
        self.incumbent = np.concatenate([incumbent[1:], np.full((1, 4), p.baseline)]).astype(np.float32)
        self.previous = action.copy()
        record.update({'final_objective': incumbent_cost,
                       'runtime_s': time.perf_counter() - started,
                       'accepted': sum(r['accepted'] for r in record['iterations']),
                       'rejected': sum(not r['accepted'] for r in record['iterations'])})
        return action, record, snapshot


def write_binary(path, values):
    a = np.asarray(values, dtype='<f4')
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite field recording')
    a.tofile(path)
    return {'file': path.name, 'shape': list(a.shape), 'dtype': 'float32-le',
            'channels': ['temperature_K', 'unreacted_fraction'],
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def summarize(fields, tape, target, config=PLUME_CONFIG, planner=P):
    metrics = field_metrics(fields, config)
    et, ec = (np.asarray(a) for a in tracking_components(fields, target, planner))
    integrated = float(np.trapezoid(et + ec, dx=planner.frame_dt))
    duration = (len(fields) - 1) * planner.frame_dt
    return {**{k: v.tolist() for k, v in metrics.items()},
            'temperature_tracking_error': et.tolist(), 'conversion_tracking_error': ec.tolist(),
            'summary': {'duration_s': duration, 'integrated_field_error_s': integrated,
                        'mean_field_error': integrated / duration,
                        'temperature_rmse_K': float(np.sqrt(np.trapezoid(et, dx=planner.frame_dt) / duration) * planner.temperature_scale),
                        'conversion_rmse': float(np.sqrt(np.trapezoid(ec, dx=planner.frame_dt) / duration) * planner.conversion_scale),
                        'terminal_field_error': float(et[-1] + ec[-1]),
                        'energy_s': float(np.sum(np.mean(tape.astype(np.float64), axis=-1)) * planner.control_dt),
                        'maximum_temperature_K': float(metrics['peak_temperature_K'].max()),
                        'minimum_outlet_conversion': float(metrics['outlet_conversion'].min()),
                        'temperature_violation_duration_s': float(np.sum(metrics['peak_temperature_K'][:-1] > config.temperature_limit) * planner.frame_dt),
                        'quality_violation_duration_s': float(np.sum(metrics['outlet_conversion'][:-1] < config.conversion_target) * planner.frame_dt)}}


def generate_recovery(root):
    root = Path(root); config, planner = PLUME_CONFIG, P
    assets = root / 'interactive' / 'thermal-recovery-data'; assets.mkdir(parents=True, exist_ok=True)
    reports = root / 'artifacts' / 'thermal_recovery'; reports.mkdir(parents=True, exist_ok=True)
    target, initial = target_and_initial(config, planner)
    if not valid_fields(initial):
        raise RuntimeError('Invalid disturbed initial state')
    stationary = np.asarray(physics_step(target, np.full(4, planner.baseline), 0., 2., config))
    drift = float(np.max(np.abs(stationary[0] - target[0])))
    if drift > .01:
        raise RuntimeError('Warm target has not settled')
    steps = int(round((planner.end_s - planner.start_s) / planner.control_dt))
    baseline = np.full((steps, 4), planner.baseline, np.float32)
    seed_tape = baseline.copy(); seed_tape[:planner.horizon] = interpolate_knots(seed_knots(planner), planner)
    tapes = {'unchanged': baseline, 'seed': seed_tape}
    runs = {key: np.asarray(recorded_rollout(jnp.asarray(initial), jnp.asarray(tape), config, planner)) for key, tape in tapes.items()}
    ctl = RecoveryController(target, config, planner)
    fields, actions, decisions, snapshots = [initial], [], [], []
    for step in range(steps):
        t = planner.start_s + step * planner.control_dt
        action, record, snapshot = ctl.plan(fields[-1], capture=t in (14., 22., 34.))
        record['time_s'] = t
        decisions.append(record)
        if action is None:
            print(f'Planning failure at {t}s: {record}', flush=True)
            break
        if snapshot:
            snapshots.append((t, snapshot))
        future = np.asarray(recorded_rollout(jnp.asarray(fields[-1]), jnp.asarray(action[None]), config, planner))
        fields.extend(future[1:]); actions.append(action)
        print(json.dumps({'time_s': t, 'accepted': record['accepted'], 'rejected': record['rejected'],
                          'objective': record['final_objective'], 'runtime_s': record['runtime_s'],
                          'action': action.tolist()}), flush=True)
    runs['mppi'] = np.asarray(fields); tapes['mppi'] = np.asarray(actions)
    manifest = {'version': 1, 'config': config.to_dict(), 'planner': asdict(planner),
                'start_s': planner.start_s, 'end_s': planner.end_s,
                'target': write_binary(assets / 'target.bin', target[None]),
                'target_two_second_drift_K': drift, 'runs': [], 'snapshots': [],
                'limits': {'temperature_K': [680, 1120], 'conversion': [0, 1],
                           'temperature_error_K': [-100, 100], 'conversion_error': [-.2, .2]},
                'method_note': 'Direct numerical PDE planning and execution. The inlet is known and normal after14s. Only the optimizer samples randomly. Supplied feasible seed and Gaussian regularization do not imply global optimality.'}
    import matplotlib
    manifest['colormaps'] = {name: (matplotlib.colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).astype(int).tolist()
                             for name in ('magma', 'viridis', 'RdBu_r')}
    for name, values in runs.items():
        metrics = summarize(values, tapes[name], target, config, planner)
        manifest['runs'].append({'id': name, 'fields': write_binary(assets / f'{name}.bin', values),
            'times_s': (planner.start_s + np.arange(len(values)) * planner.frame_dt).tolist(),
            'controls': tapes[name].tolist(), 'completed': len(tapes[name]) == steps,
            'decisions': decisions if name == 'mppi' else [], **metrics})
    # Snapshot field generation is outside the reported optimizer runtime.
    for t, snapshot in snapshots:
        stem = f'decision-{int(t)}'
        init = snapshot.pop('initial')
        candidate_tapes = snapshot['candidate_controls']
        candidate_fields = np.asarray(recorded_rollout(jnp.broadcast_to(init, (6,) + init.shape),
                                                       jnp.asarray(candidate_tapes.swapaxes(0, 1)), config, planner))
        descriptor = {'time_s': t, 'times_ahead_s': (np.arange(len(candidate_fields)) * planner.frame_dt).tolist(),
                      'reference_knots': snapshot['reference_knots'].tolist(), 'iteration': snapshot['iteration_record'],
                      'weighted_proposal_exists': snapshot['weighted_proposal_exists'], 'candidates': []}
        for i in range(6):
            descriptor['candidates'].append({'index': int(snapshot['candidate_indices'][i]),
                  'objective': float(snapshot['candidate_costs'][i]) if np.isfinite(snapshot['candidate_costs'][i]) else None,
                  'weight': float(snapshot['candidate_weights'][i]), 'coarse_feasible': bool(snapshot['candidate_feasible'][i]),
                  'controls': candidate_tapes[i].tolist(),
                  'fields': write_binary(assets / f'{stem}-candidate-{i}.bin', candidate_fields[:, i])})
        for name in ('weighted', 'used'):
            tape = snapshot[name + '_controls']
            values = np.asarray(recorded_rollout(jnp.asarray(init), jnp.asarray(tape), config, planner))
            descriptor[name] = {'controls': tape.tolist(), 'fields': write_binary(assets / f'{stem}-{name}.bin', values)}
        manifest['snapshots'].append(descriptor)
    (reports / 'decisions.json').write_text(json.dumps(decisions, indent=2) + '\n')
    (reports / 'summary.json').write_text(json.dumps({r['id']: r['summary'] for r in manifest['runs']}, indent=2) + '\n')
    print('Recovery recording complete; checking sampling sensitivity', flush=True)
    sensitivity = []
    for count in (128, 256, 512):
        for seed in (19744, 19745, 19746):
            p = replace(planner, candidates=count, seed=seed)
            trial = RecoveryController(target, config, p)
            _, record, _ = trial.plan(initial)
            ok, validation = trial.validate(initial, trial.last_plan)
            sensitivity.append({'candidates': count, 'seed': seed, 'feasible': ok,
                                'initial_objective': record['initial_objective'], 'final_objective': record['final_objective'],
                                'runtime_s': record['runtime_s'], 'ess': [a['ess'] for a in record['iterations']],
                                'accepted': record['accepted'], **validation})
            print(json.dumps(sensitivity[-1]), flush=True)
    manifest['sensitivity'] = sensitivity
    (reports / 'sensitivity.json').write_text(json.dumps(sensitivity, indent=2) + '\n')
    manifest['numerical_validation'] = recovery_refinement(initial, runs['mppi'], tapes['mppi'], config, planner)
    manifest['runtime_note'] = 'Offline planning; 2s is simulated control time. Per-decision runtimes include scoring and fine validation, and first-use compilation where needed, but exclude rendering snapshot movies.'
    (assets / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    checks = validate_recovery(assets)
    (reports / 'validation.json').write_text(json.dumps(checks, indent=2) + '\n')
    recovery_figures(root, manifest, runs, target)
    print(json.dumps(checks, indent=2), flush=True)
    return manifest


def recovery_refinement(initial, coarse, tape, config=PLUME_CONFIG, planner=P):
    fine_t = np.asarray(recorded_rollout(jnp.asarray(initial), jnp.asarray(tape), replace(config, max_dt=.05), planner))
    cfg = replace(config, nx=config.nx * 2, ny=config.ny * 2, max_dt=.025)
    fine_initial = np.repeat(np.repeat(initial, 2, axis=-1), 2, axis=-2)
    fine_x = np.asarray(recorded_rollout(jnp.asarray(fine_initial), jnp.asarray(tape), cfg, planner))
    restricted = fine_x.reshape(len(fine_x), 2, config.ny, 2, config.nx, 2).mean(axis=(3, 5))
    def error(other, full, other_config):
        m = field_metrics(full, other_config)
        return {'temperature_rmse_K': float(np.sqrt(np.mean((other[:, 0] - coarse[:, 0])**2))),
                'outlet_conversion_mae': float(np.mean(np.abs(m['outlet_conversion'] - field_metrics(coarse, config)['outlet_conversion']))),
                'maximum_temperature_K': float(m['peak_temperature_K'].max()),
                'minimum_outlet_conversion': float(m['outlet_conversion'].min()),
                'temperature_violation_duration_s': float(np.sum(m['peak_temperature_K'][:-1] > config.temperature_limit) * planner.frame_dt),
                'quality_violation_duration_s': float(np.sum(m['outlet_conversion'][:-1] < config.conversion_target) * planner.frame_dt)}
    return {'time_refinement': error(fine_t, fine_t, config), 'space_refinement': error(restricted, fine_x, cfg),
            'fine_mesh': [cfg.ny, cfg.nx], 'initial_transfer': 'piecewise constant conservative prolongation',
            'note': 'Same recorded controls and initial cell averages. First-order upwind spreading includes numerical diffusion, exceeding physical diffusion on the base mesh. Checks are on discrete output grids, not continuous-time guarantees.'}


def validate_recovery(assets):
    assets = Path(assets); m = json.loads((assets / 'manifest.json').read_text())
    def read(desc):
        raw = (assets / desc['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == desc['sha256']
        values = np.frombuffer(raw, dtype='<f4').reshape(desc['shape'])
        assert valid_fields(values)
        return values
    target = read(m['target'])[0]; runs = {}
    for r in m['runs']:
        values = read(r['fields']); runs[r['id']] = values
        assert len(values) == len(r['times_s']) == len(r['controls']) * 8 + 1
        np.testing.assert_allclose(np.diff(r['times_s']), .25)
        metrics = summarize(values, np.asarray(r['controls']), target)
        for key in ('peak_temperature_K', 'outlet_conversion', 'temperature_tracking_error', 'conversion_tracking_error'):
            np.testing.assert_allclose(metrics[key], r[key], atol=1e-6)
        if r['id'] in ('seed', 'mppi'):
            assert metrics['summary']['temperature_violation_duration_s'] == 0
            assert metrics['summary']['quality_violation_duration_s'] == 0
    for a in runs.values():
        np.testing.assert_array_equal(a[0], runs['unchanged'][0])
    for snapshot in m['snapshots']:
        k = int(round((snapshot['time_s'] - m['start_s']) / P.frame_dt))
        for candidate in snapshot['candidates'] + [snapshot['weighted'], snapshot['used']]:
            values = read(candidate['fields'])
            assert len(values) == len(snapshot['times_ahead_s'])
            np.testing.assert_array_equal(values[0], runs['mppi'][k])
        assert sum(c['weight'] for c in snapshot['candidates']) <= 1 + 1e-6
        for c in snapshot['candidates']:
            if not c['coarse_feasible']:
                assert c['weight'] == 0
    mppi = next(r for r in m['runs'] if r['id'] == 'mppi')
    assert mppi['completed'], 'Recovery did not complete; inspect decisions.json'
    for decision in mppi['decisions']:
        for iteration in decision['iterations']:
            assert iteration['incumbent_after_objective'] <= iteration['incumbent_before_objective'] + 1e-9
    return {'completed': True, 'run_count': len(runs), 'snapshot_count': len(m['snapshots']),
            'field_arrays_checked': 1 + len(runs) + 8 * len(m['snapshots']),
            'sample_weights_and_timestamps_verified': True,
            'mppi_summary': mppi['summary'], 'numerical_validation': m['numerical_validation']}


def recovery_figures(root, manifest, runs, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    dest = Path(root) / '_static' / 'thermal_reactor'; dest.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(dest / f'{name}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(10, 6.5), layout='constrained')
    for col, t in enumerate((22, 34, 46)):
        k = int((t - P.start_s) / P.frame_dt)
        for row, (name, channel) in enumerate((('unchanged', 0), ('mppi', 0), ('unchanged', 1), ('mppi', 1))):
            error = (runs[name][k, channel] - target[channel]) * (1 if channel == 0 else -1)
            limits = (-100, 100) if channel == 0 else (-.2, .2)
            im = axes[row, col].imshow(error, origin='lower', extent=(0, 8, 0, 2), vmin=limits[0], vmax=limits[1], cmap='RdBu_r')
            axes[row, col].set(xticks=[0, 4, 8], yticks=[0, 2], xlabel='Downstream position (m)')
            axes[row, col].axvline(4, color='gray', ls=':', lw=.6); axes[row, col].axhline(1, color='gray', ls=':', lw=.6)
            if col == 0:
                axes[row, col].set_ylabel(('Unchanged' if name == 'unchanged' else 'MPPI') + '\nWidth (m)')
            if row == 0:
                axes[row, col].set_title(f'{t} s')
            if col == 2:
                fig.colorbar(im, ax=axes[row, :], fraction=.018, pad=.015, shrink=.8,
                             label='Temperature error (K)' if channel == 0 else 'Conversion error')
    save(fig, 'recovery-storyboard')
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    colors = {'unchanged': '#6b7280', 'seed': '#b95525', 'mppi': '#176787'}
    labels = {'unchanged': 'Unchanged heating', 'seed': 'Supplied feasible schedule', 'mppi': 'MPPI'}
    for r in manifest['runs']:
        ts = r['times_s']; col = colors[r['id']]
        error = np.asarray(r['temperature_tracking_error']) + r['conversion_tracking_error']
        axes[0, 0].plot(ts, error, label=labels[r['id']], color=col)
        axes[0, 1].plot(ts, r['outlet_conversion'], color=col)
        axes[1, 0].plot(ts, r['peak_temperature_K'], color=col)
        if r['id'] == 'mppi':
            u = np.asarray(r['controls']); control_t = P.start_s + np.arange(len(u) + 1) * 2
            for j in range(4):
                axes[1, 1].step(control_t, np.r_[u[:, j], u[-1, j]] - P.baseline, where='post', label=f'H{j+1}')
    axes[0, 0].legend(fontsize=8); axes[1, 1].legend(fontsize=8)
    axes[0, 1].axhline(.95, color='black', ls='--', lw=.8); axes[1, 0].axhline(1070, color='black', ls='--', lw=.8)
    axes[1, 1].axhline(0, color='gray', lw=.6)
    for ax, label in zip(axes.flat, ('Normalized field error', 'Outlet conversion', 'Maximum temperature (K)', 'Heater fraction above nominal')):
        ax.set(xlabel='Time (s)', ylabel=label)
    save(fig, 'recovery-control')
