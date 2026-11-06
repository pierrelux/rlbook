#!/usr/bin/env python3
"""Rebuild deterministic thermoacoustic pull-down results and teaching figures.

Run from the repository root: uv run python scripts/build_thermoacoustic_pulldown_artifacts.py
Solver times exclude JAX compilation and are diagnostic, not benchmarks.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from thermoacoustic_pulldown import (  # noqa: E402
    FridgeParameters, SCENARIOS, audit, initial_controls, initial_state,
    make_problem, powers,
)
from trajectory_optimization import SolverOptions, solve_trajectory  # noqa: E402

STATIC = ROOT / "_static" / "thermoacoustic_pulldown"
ARTIFACTS = ROOT / "artifacts" / "thermoacoustic_pulldown"
BLUE, ORANGE = "#0072B2", "#D55E00"
GREEN, PURPLE, GRAY = "#009E73", "#CC79A7", "#666666"
METHOD_LABEL = {"ilqr": "iLQR", "ddp": "DDP"}
EXPECTED_COST = {
    "baseline": 51.224,
    "no_load": 20.652,
    "heavy_hot": 45.716,
    "no_minor_loss": 33.353,
    "unreachable": 119.582,
}


def energy_j(states, controls, parameters):
    """Integrate the model's acoustic work over piecewise-constant controls."""
    work = jax.vmap(lambda x, u: powers(x, u, parameters)[1])(
        jnp.asarray(states[:-1]), jnp.asarray(controls))
    return float(parameters.dt * np.sum(np.asarray(work)))


def save(fig, name):
    for suffix in ("svg", "pdf", "png"):
        options = {"metadata": {"Date": None}} if suffix == "svg" else {}
        if suffix == "pdf":
            options = {"metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(STATIC / f"{name}.{suffix}", bbox_inches="tight",
                    pad_inches=.04, **options)
    plt.close(fig)


def figure_style():
    # Match the boat figures so the two experiments share a visual vocabulary.
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .65,
        "lines.linewidth": 1.4, "figure.dpi": 160,
        "legend.frameon": False, "savefig.dpi": 180,
        "svg.fonttype": "none", "svg.hashsalt": "thermoacoustic-pulldown",
        "pdf.fonttype": 42,
    })


def figures(results, full_amplitude, parameters):
    figure_style()
    baseline = results["baseline"]["ilqr"]
    time = np.arange(parameters.steps + 1) * parameters.dt
    control_time = time[:-1]

    fig, (temperatures, amplitude) = plt.subplots(
        2, 1, figsize=(7.2, 4.05), sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1]}, layout="constrained")
    temperatures.plot(time, baseline.states[:, 0], color=BLUE,
                      label="Cold side, optimized")
    temperatures.plot(time, baseline.states[:, 1], color=ORANGE,
                      label="Hot side, optimized")
    temperatures.plot(time, full_amplitude["states"][:, 0], color=GRAY,
                      ls=":", label="Cold side, full amplitude")
    temperatures.plot(time, full_amplitude["states"][:, 1], color=GRAY,
                      ls="-.", label="Hot side, full amplitude")
    temperatures.axhline(parameters.target, color=".4", ls="--", lw=.9,
                         label="Cold target")
    temperatures.set(ylabel="Temperature (°C)", xlim=(0, time[-1]))
    temperatures.legend(loc="center right", ncol=2, fontsize=7.4,
                        columnspacing=1.1, handlelength=2.5)
    amplitude.step(control_time, baseline.controls[:, 0], where="post",
                   color=BLUE, label="Optimized amplitude")
    amplitude.step(control_time, full_amplitude["controls"][:, 0],
                   where="post", color=GRAY, ls=":", label="Full amplitude")
    amplitude.set(xlabel="Time (s)", ylabel="Driver amplitude", ylim=(-.03, 1.08))
    amplitude.legend(loc="lower right")
    save(fig, "pulldown")

    fig, ax = plt.subplots(figsize=(7.2, 2.6), layout="constrained")
    variants = [
        ("baseline", "ilqr", BLUE, "-"),
        ("no_load", "ilqr", ORANGE, "--"),
        ("heavy_hot", "ilqr", GREEN, "-."),
        ("no_minor_loss", "ddp", PURPLE, ":"),
    ]
    for scenario, method, color, style in variants:
        ax.step(control_time, results[scenario][method].controls[:, 0],
                where="post", color=color, ls=style,
                label=SCENARIOS[scenario]["name"])
    ax.set(xlabel="Time (s)", ylabel="Driver amplitude", xlim=(0, time[-1]),
           ylim=(-.03, 1.08))
    ax.legend(loc="lower right", ncol=2, fontsize=7.5,
              columnspacing=1.0, handlelength=2.5)
    save(fig, "variants")

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.65),
                            sharey=True, layout="constrained")
    for ax, scenario in zip(axs, ("baseline", "no_minor_loss")):
        for method, color, style in (("ilqr", BLUE, "-"),
                                     ("ddp", ORANGE, "--")):
            costs = np.array([item["cost"] for item in
                              results[scenario][method].history])
            updates = np.arange(len(costs))
            ax.semilogy(updates, costs, color=color, ls=style,
                        label=METHOD_LABEL[method])
            ax.plot(updates[-1], costs[-1], marker="o" if method == "ilqr" else "s",
                    ms=3.5, color=color)
        ax.set(xlabel="Accepted update (0 = initial)",
               title=SCENARIOS[scenario]["name"])
        ax.legend(loc="upper right")
    axs[0].set_ylabel("Nonlinear cost (log scale)")
    save(fig, "convergence")


def run_report(scenario, method, result, parameters, reachable):
    report = audit(result.states, result.controls, parameters,
                   require_target=reachable)
    history_costs = [float(item["cost"]) for item in result.history]
    report.update({
        "scenario": scenario,
        "method": method,
        "status": result.status,
        "initial_cost": history_costs[0],
        "cost": history_costs[-1],
        "accepted_iterations": len(history_costs) - 1,
        "accepted_costs": history_costs,
        "energy_J": energy_j(result.states, result.controls, parameters),
        "terminal_cold_grid_C": float(result.states[-1, 0]),
        "terminal_hot_grid_C": float(result.states[-1, 1]),
        "controls_sampled_25s": result.controls[::25, 0].tolist(),
        "minimum_control": float(np.min(result.controls)),
        "maximum_control": float(np.max(result.controls)),
        "projected_gradient": float(result.projected_gradient),
        "forward_evaluations": result.forward_evaluations,
        "backward_attempts": result.backward_attempts,
        "solver_seconds_excluding_compilation": result.solver_seconds,
        "states": result.states.tolist(),
        "controls": result.controls.tolist(),
    })
    return report


def comparison_report(problem, parameters):
    controls = np.ones((parameters.steps, 1))
    states = problem.rollout(initial_state(parameters), controls)
    return {
        "description": "Baseline constant full-amplitude rollout",
        "cost": problem.cost(states, controls),
        "energy_J": energy_j(states, controls, parameters),
        "terminal_cold_C": float(states[-1, 0]),
        "terminal_hot_C": float(states[-1, 1]),
        "audit": audit(states, controls, parameters, require_target=False),
        "states": states,
        "controls": controls,
    }


def source_provenance(options):
    source_files = [
        "code/trajectory_optimization.py",
        "code/thermoacoustic_pulldown.py",
        "scripts/build_thermoacoustic_pulldown_artifacts.py",
    ]
    return {
        "sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                   for path in source_files},
        "solver_options": asdict(options),
    }


def write_results(reports, full_amplitude):
    lines = [
        "| Experiment | Method | Status | Accepted updates | Cost | Energy (J) | Cold at 300 s (°C) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        lines.append(
            f"| {SCENARIOS[report['scenario']]['name']} | "
            f"{METHOD_LABEL[report['method']]} | "
            f"{report['status']} | {report['accepted_iterations']} | "
            f"{report['cost']:.3f} | {report['energy_J']:.0f} | "
            f"{report['terminal_cold_C']:.2f} |")
    lines.extend([
        "",
        "Costs are evaluated on the final accepted nonlinear trajectory. "
        "The iteration-limit row is a retained local plan, not a converged solve.",
        "",
        f"For comparison, constant full amplitude in the baseline case ends at "
        f"{full_amplitude['terminal_cold_C']:.2f} °C, uses "
        f"{full_amplitude['energy_J']:.0f} J, and has cost "
        f"{full_amplitude['cost']:.2f}.",
    ])
    (ARTIFACTS / "results.md").write_text("\n".join(lines) + "\n")


def write_readme():
    (ARTIFACTS / "README.md").write_text(
        "# Thermoacoustic pull-down with iLQR and DDP\n\n"
        "These deterministic local trajectory-optimization experiments support "
        "`iterative-trajectory-optimization.md`. The refrigerator is a synthetic "
        "teaching model, not a measured device.\n\n"
        "Regenerate from the repository root with:\n\n"
        "```bash\n"
        "uv run python scripts/build_thermoacoustic_pulldown_artifacts.py\n"
        "```\n\n"
        "The builder solves five variants with both iLQR and DDP from the "
        "same constant-amplitude seed. It also evaluates a constant full-amplitude "
        "plan and two other baseline seeds. It writes:\n\n"
        "- `metrics.json`: parameters, source hashes, solver settings, final plans, "
        "accepted costs, energy, status, and replay checks;\n"
        "- `results.md`: the chapter's numerical comparison;\n"
        "- `../../_static/thermoacoustic_pulldown/`: matching SVG, PDF, and PNG "
        "teaching figures.\n\n"
        "The state columns are cold and hot temperatures in degrees Celsius. "
        "Controls are driver-amplitude fractions in [0, 1]. One RK4 step lasts "
        "1 s, and the 300 controls cover a fixed 300 s horizon. Energy is the "
        "sum of acoustic work over those intervals. The running cost weights "
        "that energy, and the terminal cost penalizes the final cold-temperature "
        "error.\n\n"
        "Each final plan is checked for finite states, control bounds, and "
        "agreement with a four-times-finer RK4 replay. Reachable-target runs "
        "must end within 0.6 K of their targets. The 300 s out-of-reach variant "
        "still undergoes bounds and replay checks. All accepted costs must "
        "decrease. The no-minor-loss iLQR run intentionally retains its last "
        "accepted plan at the iteration limit; its DDP counterpart converges. "
        "An unexpected status or failed check causes the builder to exit "
        "with an error after writing its diagnostic reports.\n\n"
        "Single-run solver times exclude JAX compilation and vary by machine. "
        "They are diagnostic only. Ordinary book builds read these committed "
        "artifacts and do not run the optimizer.\n")


def main():
    for directory in (STATIC, ARTIFACTS):
        directory.mkdir(parents=True, exist_ok=True)
    options = SolverOptions(max_iterations=200)
    results, reports, issues = {}, [], []
    for scenario, cfg in SCENARIOS.items():
        parameters = cfg["parameters"]
        problem = make_problem(parameters)
        results[scenario] = {}
        for method in ("ilqr", "ddp"):
            result = solve_trajectory(
                problem, initial_state(parameters), initial_controls(parameters),
                method=method, options=options)
            results[scenario][method] = result
            report = run_report(scenario, method, result, parameters,
                                cfg["target_reachable"])
            reports.append(report)
            expected_status = ("iteration_limit" if scenario == "no_minor_loss"
                               and method == "ilqr" else "converged")
            if report["status"] != expected_status:
                issues.append(f"{scenario}/{method}: status {report['status']}, "
                              f"expected {expected_status}")
            if not report["pulldown_pass"]:
                issues.append(
                    f"{scenario}/{method}: audit failed: "
                    f"cold error {report['terminal_cold_error_K']:.6g} K, "
                    f"replay difference "
                    f"{report['fine_replay_max_temperature_difference_K']:.6g} K, "
                    f"control violation {report['control_violation']:.6g}")
            if not np.all(np.diff(report["accepted_costs"]) < 0):
                issues.append(f"{scenario}/{method}: accepted costs did not decrease")
            if method == "ilqr" and abs(report["cost"] - EXPECTED_COST[scenario]) > .005:
                issues.append(f"{scenario}/{method}: cost {report['cost']:.6f} "
                              f"differs from prototype {EXPECTED_COST[scenario]:.3f}")
            print(f"{scenario}/{method}: {report['status']}; "
                  f"{report['accepted_iterations']} accepted; "
                  f"cost {report['cost']:.6f}; energy {report['energy_J']:.3f} J",
                  flush=True)

    baseline_parameters = SCENARIOS["baseline"]["parameters"]
    baseline_problem = make_problem(baseline_parameters)
    full_amplitude = comparison_report(baseline_problem, baseline_parameters)
    if not full_amplitude["audit"]["pulldown_pass"]:
        issues.append("baseline full-amplitude replay or bounds check failed")

    seed_sensitivity = []
    for seed in (.1, .9):
        controls = np.full((baseline_parameters.steps, 1), seed)
        result = solve_trajectory(
            baseline_problem, initial_state(baseline_parameters), controls,
            method="ilqr", options=options)
        seed_sensitivity.append({
            "initial_amplitude": seed,
            "status": result.status,
            "cost": float(result.history[-1]["cost"]),
            "terminal_cold_C": float(result.states[-1, 0]),
            "energy_J": energy_j(result.states, result.controls, baseline_parameters),
            "accepted_iterations": len(result.history) - 1,
        })
        print(f"baseline seed {seed:.1f}: {result.status}; "
              f"cost {result.history[-1]['cost']:.6f}", flush=True)

    figures(results, full_amplitude, baseline_parameters)
    provenance = source_provenance(options)
    data = {
        "schema_version": 1,
        "parameters": asdict(FridgeParameters()),
        "scenario_parameters": {name: asdict(cfg["parameters"])
                                for name, cfg in SCENARIOS.items()},
        "state_columns": ["cold_temperature_C", "hot_temperature_C"],
        "control_columns": ["driver_amplitude_fraction"],
        "provenance": provenance,
        "runs": reports,
        "full_amplitude_comparison": {
            key: value for key, value in full_amplitude.items()
            if key not in ("states", "controls")},
        "seed_sensitivity": seed_sensitivity,
        "validation_issues": issues,
    }
    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n")
    write_results(reports, full_amplitude)
    write_readme()
    if issues:
        raise RuntimeError("thermoacoustic artifact validation failed: "
                           + "; ".join(issues))


if __name__ == "__main__":
    main()
