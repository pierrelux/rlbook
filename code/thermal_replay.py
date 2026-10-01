"""Export numerical thermal-reactor fields and static scientific comparisons."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from thermal_reactor import ReactorConfig, physics_step, field_metrics, warm_state


def write_fields(path, fields):
    """Explicit little-endian float32; metadata records exact shape and units."""
    values = np.asarray(fields, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite replay fields: {path}")
    values.tofile(path)
    return {"file": Path(path).name, "shape": list(values.shape), "dtype": "float32-le",
            "channels": ["temperature_K", "unreacted_fraction"]}


def numerical_validation(config=ReactorConfig()):
    from scipy.ndimage import zoom
    initial = warm_state(config)
    u = np.array([.45, .50, .48, .46], np.float32)
    base = np.asarray(physics_step(initial, u, -10., 40., config))
    fine_t = np.asarray(physics_step(initial, u, -10., 40., replace(config, max_dt=.05)))
    fine_cfg = replace(config, nx=2 * config.nx, ny=2 * config.ny, max_dt=.025)
    # Interpolate the SAME initial field at cell centres (not independent warm starts).
    refined_initial = zoom(initial, (1, 2, 2), order=1, grid_mode=True, mode="nearest")
    fine_x = np.asarray(physics_step(refined_initial, u, -10., 40., fine_cfg))
    restricted = fine_x.reshape(2, config.ny, 2, config.nx, 2).mean(axis=(2, 4))
    def errors(other):
        return {"temperature_rmse_K": float(np.sqrt(np.mean((base[0] - other[0])**2))),
                "outlet_conversion_difference": float(abs(base[1, :, -1].mean() - other[1, :, -1].mean())),
                "peak_temperature_difference_K": float(abs(base[0].max() - other[0].max()))}
    return {"duration_s": 40, "base_shape": [config.ny, config.nx],
            "time_refinement": errors(fine_t), "space_refinement": errors(restricted),
            "fine_shape": [fine_cfg.ny, fine_cfg.nx],
            "note": "Same initial field and controls; spatial refinement is a sensitivity check, not an exact solution."}


def export(output_dir, root):
    output_dir, root = Path(output_dir), Path(root)
    summary = json.loads((output_dir / "evaluation.json").read_text())
    config = ReactorConfig(**summary["config"])
    assets = root / "interactive" / "thermal-reactor-data"
    assets.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "config": summary["config"], "planner": summary["planner"],
                "status": summary["surrogate_status"], "runs": [], "snapshots": [],
                "time_interval_s": config.control_dt,
                "limits": {"temperature_K": [750, 1120], "conversion": [0, 1],
                           "temperature_error_K": [-20, 20], "conversion_error": [-.05, .05]},
                "summary": summary}
    for row in summary["rows"]:
        stem = f"trial-{row['trial']}-{row['controller']}"
        with np.load(output_dir / f"{stem}.npz", allow_pickle=False) as data:
            fields = data["fields"]
            metrics = field_metrics(fields)
            desc = write_fields(assets / f"{stem}.bin", fields)
            manifest["runs"].append({"id": stem, "trial": row["trial"], "controller": row["controller"],
                                     "fields": desc, "controls": data["controls"].tolist(),
                                     "disturbances": data["disturbances"].tolist(),
                                     "peak_temperature_K": metrics["peak_temperature_K"].tolist(),
                                     "outlet_conversion": metrics["outlet_conversion"].tolist(),
                                     "decisions": json.loads(str(data["decisions"])), "metrics": row})
    for path in sorted(output_dir.glob("candidates-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            stem = path.stem
            mode, step = stem.removeprefix("candidates-").rsplit("-", 1)
            initial = data["initial"]
            candidates = []
            for j in range(6):
                frames = np.concatenate([initial[None], data["candidate_fields"][:, j]])
                candidates.append({"index": int(data["candidate_indices"][j]),
                                   "cost": float(data["candidate_costs"][j]) if np.isfinite(data["candidate_costs"][j]) else None,
                                   "weight": float(data["candidate_weights"][j]),
                                   "controls": data["candidate_controls"][j].tolist(),
                                   "fields": write_fields(assets / f"{stem}-candidate-{j}.bin", frames)})
            pred = np.concatenate([initial[None], data["proposed_surrogate"]])
            pde = np.concatenate([initial[None], data["proposed_pde"]])
            manifest["snapshots"].append({"id": stem, "controller": mode, "step": int(step),
                 "time_s": int(step) * config.control_dt, "candidates": candidates,
                 "accepted": bool(data["proposed_accepted"]), "mean_feasible": bool(data["proposed_mean_feasible"]),
                 "ess": float(data["ess"]), "forecast": data["forecast"].tolist(),
                 "ensemble_size": int(data["full_forecasts"].shape[1]),
                 "proposed_cost": float(data["proposed_cost"]) if np.isfinite(data["proposed_cost"]) else None,
                 "incumbent_cost": float(data["incumbent_cost"]) if np.isfinite(data["incumbent_cost"]) else None,
                 "proposed_controls": data["proposed_controls"].tolist(),
                 "surrogate": write_fields(assets / f"{stem}-surrogate.bin", pred),
                 "pde": write_fields(assets / f"{stem}-pde.bin", pde)})
    manifest["surrogate_validation"] = json.loads((output_dir / "surrogate_validation.json").read_text())["summary"]
    manifest["sensitivity"] = json.loads((output_dir / "sensitivity.json").read_text())
    numerical = numerical_validation(config)
    (output_dir / "numerical_validation.json").write_text(json.dumps(numerical, indent=2))
    manifest["numerical_validation"] = numerical
    import matplotlib
    manifest["colormaps"] = {name: (matplotlib.colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).astype(int).tolist()
                             for name in ("magma", "viridis", "RdBu_r")}
    (assets / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    static_figures(output_dir, root, manifest)
    checks = validate_replay(assets)
    (output_dir / "artifact_validation.json").write_text(json.dumps(checks, indent=2))
    print(f"Exported {len(manifest['runs'])} runs and {len(manifest['snapshots'])} planning snapshots", flush=True)


def static_figures(output_dir, root, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9,
                         "axes.titlesize": 10, "legend.fontsize": 8, "mathtext.fontset": "cm",
                         "axes.spines.top": False, "axes.spines.right": False})
    dest = root / "_static" / "thermal_reactor"; dest.mkdir(parents=True, exist_ok=True)
    cfg = manifest["config"]
    names = {"constant": "Constant heating", "mean": "Mean-inlet MPPI", "stochastic": "Stochastic MPPI"}
    colors = {"constant": "#6b7280", "mean": "#0072B2", "stochastic": "#D55E00"}
    def save(fig, name):
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(dest / f"{name}.{suffix}", bbox_inches="tight", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10, 3.2), layout="constrained")
    mappables = [None, None]
    for col, method in enumerate(names):
        with np.load(output_dir / f"trial-0-{method}.npz") as a:
            frame = a["fields"][-1]
            for row, (field, limits, cmap) in enumerate(((frame[0], (750, 1120), "magma"), (1 - frame[1], (0, 1), "viridis"))):
                im = axes[row, col].imshow(field, origin="lower", extent=[0, 8, 0, 2], cmap=cmap, vmin=limits[0], vmax=limits[1], aspect="equal")
                axes[row, col].set(xlabel="Streamwise position (m)", ylabel="Width (m)")
                axes[row, col].axvline(4, color="white", ls=":", lw=.7)
                axes[row, col].axhline(1, color="white", ls=":", lw=.7)
                mappables[row] = im
            axes[0, col].set_title(f"{names[method]}\nRecorded endpoint: {(len(a['fields']) - 1) * 2} s")
    for row, label in enumerate(("Temperature (K)", "Conversion")):
        fig.colorbar(mappables[row], ax=axes[row, :], label=label, shrink=.65, aspect=12, fraction=.02, pad=.025)
    save(fig, "fields")
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), layout="constrained")
    for method in names:
        run = next(r for r in manifest["runs"] if r["trial"] == 0 and r["controller"] == method)
        times = np.arange(len(run["peak_temperature_K"])) * 2
        axes[0].plot(times, run["peak_temperature_K"], color=colors[method], label=names[method])
        axes[1].plot(times, run["outlet_conversion"], color=colors[method])
        energy = np.r_[0, np.cumsum(np.mean(run["controls"], axis=-1)) * 2]
        axes[2].plot(times, energy, color=colors[method])
    axes[0].axhline(1070, color="black", ls="--", lw=.8)
    axes[1].axhline(.95, color="black", ls="--", lw=.8)
    for ax, ylabel in zip(axes, ("Maximum temperature (K)", "Outlet conversion", "Normalized heating energy (s)")):
        ax.set(xlabel="Time (s)", ylabel=ylabel)
    axes[0].legend()
    save(fig, "control")
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), layout="constrained")
    rows = manifest["summary"]["rows"]
    metrics = ("normalized_heating_energy_s", "minimum_outlet_conversion", "maximum_temperature_K")
    for ax, metric, label in zip(axes, metrics, ("Heating energy (s)", "Minimum outlet conversion", "Maximum temperature (K)")):
        for index, method in enumerate(names):
            records = [r for r in rows if r["controller"] == method]
            for j, r in enumerate(records):
                ax.scatter(index + (j - (len(records) - 1) / 2) * .035, r[metric], color=colors[method], marker="o" if r["completed"] else "x", s=22)
        ax.set(xticks=range(3), xticklabels=["Constant", "Mean", "Stochastic"], ylabel=label)
    axes[1].axhline(.95, color="black", ls="--", lw=.8)
    axes[2].axhline(1070, color="black", ls="--", lw=.8)
    save(fig, "paired-trials")


def validate_replay(assets):
    """Check binary layouts, frame alignment, and saved scalar diagnostics."""
    assets = Path(assets)
    manifest = json.loads((assets / 'manifest.json').read_text())
    cfg = manifest['config']; count = 0
    def read(desc):
        nonlocal count
        shape = tuple(desc['shape'])
        if shape[1:] != (2, cfg['ny'], cfg['nx']):
            raise AssertionError('Unexpected replay field shape')
        a = np.fromfile(assets / desc['file'], dtype='<f4')
        if a.size != np.prod(shape) or not np.isfinite(a).all():
            raise AssertionError('Invalid binary field array')
        count += 1
        return a.reshape(shape)
    for run in manifest['runs']:
        x = read(run['fields']); u = np.asarray(run['controls'])
        if len(x) != len(u) + 1 or len(u) != run['metrics']['steps_completed']:
            raise AssertionError('Executed fields and control timestamps disagree')
        if np.any((u < 0) | (u > 1)):
            raise AssertionError('Executed control outside physical bounds')
        m = field_metrics(x)
        np.testing.assert_allclose(m['peak_temperature_K'], run['peak_temperature_K'], atol=1e-4)
        np.testing.assert_allclose(m['outlet_conversion'], run['outlet_conversion'], atol=1e-7)
    for snapshot in manifest['snapshots']:
        predicted, physical = read(snapshot['surrogate']), read(snapshot['pde'])
        np.testing.assert_array_equal(predicted[0], physical[0])
        if len(predicted) != manifest['planner']['horizon'] + 1:
            raise AssertionError('Prediction timestamps disagree with planning horizon')
        if len(snapshot['candidates']) != 6:
            raise AssertionError('Expected six displayed candidate futures')
        weight = 0.
        for candidate in snapshot['candidates']:
            x = read(candidate['fields'])
            np.testing.assert_array_equal(x[0], physical[0])
            if x.shape != predicted.shape or len(candidate['controls']) != len(x) - 1:
                raise AssertionError('Candidate and updated trajectories are not synchronized')
            w = candidate['weight']
            if not np.isfinite(w) or w < 0 or w > 1:
                raise AssertionError('Invalid candidate probability')
            weight += w
        if weight > 1 + 1e-6:
            raise AssertionError('Displayed subset weights exceed the whole batch probability')
    return {'binary_arrays_verified': count, 'runs': len(manifest['runs']),
            'planning_snapshots': len(manifest['snapshots']),
            'timestamps_and_scalar_diagnostics_verified': True,
            'field_channels': ['temperature_K', 'remaining_reactant_fraction'],
            'binary_encoding': 'little-endian float32'}
