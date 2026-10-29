#!/usr/bin/env python3
"""Rebuild the four docking solves, browser data, and vector teaching figures.

Run from the repository root: uv run python scripts/build_boat_docking_artifacts.py
Numerical results are deterministic. Reported solver times are single local
measurements after compilation, not performance benchmarks.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from boat_docking import BoatParameters, SCENARIOS, audit, initial_controls, make_problem
from trajectory_optimization import SolverOptions, solve_trajectory

STATIC = ROOT / "_static" / "boat_docking"
ARTIFACTS = ROOT / "artifacts" / "boat_docking"
BLUE, ORANGE = "#0072B2", "#D55E00"


def serializable(value):
    if isinstance(value, np.ndarray):
        return np.round(value, 9).tolist()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def hull(x, p, scale=1.):
    body = scale * np.array([[-p.length/2, -p.width/2], [p.length*.3, -p.width/2],
                            [p.length/2, 0.], [p.length*.3, p.width/2], [-p.length/2, p.width/2]])
    cs, sn = np.cos(x[2]), np.sin(x[2])
    return body @ np.array([[cs, sn], [-sn, cs]]) + x[:2]


def boat(ax, x, p, color=BLUE, alpha=1., fill=False):
    ax.add_patch(Polygon(hull(x, p), facecolor=color if fill else "none",
                        edgecolor=color, linewidth=.8, alpha=alpha, zorder=4))


def harbor(ax, p, limits=(-9.5, 2., -.8, 6.)):
    ax.axhspan(limits[2], 0, color=".92", zorder=-4)
    ax.axhline(0, color=".45", lw=.8)
    target = np.array([p.berth_x, p.berth_y, 0., 0., 0., 0.])
    ax.add_patch(Polygon(hull(target, p), fill=False, edgecolor=".4", linestyle=":", lw=.8))
    ax.set(xlim=limits[:2], ylim=limits[2:], aspect="equal", xlabel="$p_x$ (m)", ylabel="$p_y$ (m)")


def save(fig, name):
    for suffix in ("svg", "pdf", "png"):
        options = {"metadata": {"Date": None}} if suffix == "svg" else {}
        if suffix == "pdf":
            options = {"metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(STATIC / f"{name}.{suffix}", bbox_inches="tight", pad_inches=.04, **options)
    plt.close(fig)


def figures(results, p):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "lines.linewidth": 1.4, "figure.dpi": 160,
        "legend.frameon": False,
        "savefig.dpi": 180, "svg.fonttype": "none", "svg.hashsalt": "boat-docking",
        "pdf.fonttype": 42})
    history = results["angled"]["ilqr"].history
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.65), layout="constrained", sharex=True, sharey=True)
    for ax, index in zip(axs, [0, 1, len(history)-1]):
        record = history[index]
        harbor(ax, p)
        xs = record["states"]
        ax.plot(xs[:, 0], xs[:, 1], color=BLUE)
        for t in [0, 40, 80, 120, 160, 200]:
            boat(ax, xs[t], p, alpha=.35 if t not in (0, 200) else 1.)
        ax.set_title(f"Iteration {index}\nCost {record['cost']:.2f}")
    axs[1].set_ylabel(""); axs[2].set_ylabel("")
    save(fig, "iterations")

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained", sharex=True, sharey=True)
    for ax, (scenario, runs) in zip(axs, results.items()):
        # Same displayed extent in the pair, including the wider DDP maneuver.
        harbor(ax, p, (-9.5, 3., -.8, 6.))
        seed = runs["ilqr"].history[0]["states"]
        ax.plot(seed[:, 0], seed[:, 1], color=".45", ls=":", label="Initial coast")
        for method, color, style in [("ilqr", BLUE, "-"), ("ddp", ORANGE, "--")]:
            xs = runs[method].states
            ax.plot(xs[:, 0], xs[:, 1], color=color, ls=style, label="iLQR" if method == "ilqr" else "DDP")
            for t in [0, 40, 80, 120, 160, 200]:
                boat(ax, xs[t], p, color=color, alpha=.45 if t != 200 else .9)
        ax.set_title(SCENARIOS[scenario]["name"])
        ax.legend(loc="upper right")
    axs[1].set_ylabel("")
    save(fig, "paths")

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7), layout="constrained", sharey=True)
    for ax, (scenario, runs) in zip(axs, results.items()):
        for method, color, style in [("ilqr", BLUE, "-"), ("ddp", ORANGE, "--")]:
            costs = [record["cost"] for record in runs[method].history]
            ax.semilogy(np.arange(len(costs)), costs, color=color, ls=style,
                        marker="o" if method == "ilqr" else "s", markevery=5, ms=2.5,
                        label="iLQR" if method == "ilqr" else "DDP")
        ax.set(xlabel="Accepted iteration (0 = initial)", title=SCENARIOS[scenario]["name"])
        ax.legend()
    axs[0].set_ylabel("Nonlinear trajectory cost (log scale)")
    save(fig, "convergence")

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.1), layout="constrained")
    xs = results["angled"]["ilqr"].states
    us = results["angled"]["ilqr"].controls
    index = 20
    harbor(axs[0], p)
    axs[0].plot(xs[:index+1, 0], xs[:index+1, 1], color=".5", label="Traveled")
    axs[0].plot(xs[index:, 0], xs[index:, 1], color=BLUE, ls="--", label="Planned future")
    for t in range(index+20, p.steps+1, 20):
        boat(axs[0], xs[t], p, alpha=.22)
    boat(axs[0], xs[index], p, fill=True)
    axs[0].set_title("At 2 s: future poses every 2 s")
    axs[0].legend(loc="upper right")
    time = np.arange(p.steps+1)*p.dt
    axs[1].plot(time, np.linalg.norm(xs[:, 3:5], axis=1), color=BLUE, label="Speed (m/s)")
    axs[1].set(xlabel="Simulation time (s)", ylabel="Speed (m/s)", ylim=(0, None))
    twin = axs[1].twinx()
    twin.step(time, np.r_[us[:, 0], us[-1, 0]]*p.thrust_limit, where="post", color=ORANGE,
              ls="--", alpha=.8, label="Left thrust")
    twin.step(time, np.r_[us[:, 1], us[-1, 1]]*p.thrust_limit, where="post", color=".35",
              ls=":", label="Right thrust")
    twin.set_ylabel("Thrust (N)")
    twin.set_ylim(-30, 30)
    axs[1].axvline(2., color=".65", lw=.8)
    handles1, labels1 = axs[1].get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axs[1].legend(handles1+handles2, labels1+labels2, loc="upper right", fontsize=7.5)
    axs[1].set_title("Angled approach, final iLQR plan")
    save(fig, "future-and-motion")


def main():
    for directory in (STATIC, ARTIFACTS):
        directory.mkdir(parents=True, exist_ok=True)
    p, options = BoatParameters(), SolverOptions()
    problem = make_problem(p)
    results, scenarios, metrics = {}, {}, []
    for scenario, cfg in SCENARIOS.items():
        results[scenario] = {}
        scenarios[scenario] = {**cfg, "runs": {}}
        for method in ("ilqr", "ddp"):
            result = solve_trajectory(problem, cfg["initial"], initial_controls(p), method, options)
            report = audit(result.states, result.controls, p)
            if not result.converged or not report["docking_pass"]:
                raise RuntimeError(f"{scenario}/{method}: {result.status}, {report}")
            report.update(scenario=scenario, method=method, status=result.status,
                          initial_cost=result.history[0]["cost"], cost=result.history[-1]["cost"],
                          accepted_iterations=len(result.history)-1,
                          projected_gradient=result.projected_gradient,
                          forward_evaluations=result.forward_evaluations,
                          backward_attempts=result.backward_attempts,
                          solver_seconds_excluding_compilation=result.solver_seconds)
            results[scenario][method] = result
            scenarios[scenario]["runs"][method] = {"history": result.history, "metrics": report}
            metrics.append(report)
            print(f"{scenario}/{method}: {report['accepted_iterations']} accepted updates; "
                  f"cost {report['cost']:.6f}; docking passed", flush=True)
    figures(results, p)
    source_files = ["code/trajectory_optimization.py", "code/boat_docking.py",
                    "scripts/build_boat_docking_artifacts.py"]
    provenance = {"sha256": {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                              for path in source_files}, "solver_options": asdict(options)}
    data = {"schema_version": 1, "parameters": asdict(p), "provenance": provenance,
            "state_columns": ["p_x_m", "p_y_m", "heading_rad", "v_x_mps", "v_y_mps", "omega_radps"],
            "control_columns": ["left_thrust_fraction", "right_thrust_fraction"],
            "scenarios": scenarios}
    (ROOT/"interactive"/"boat-docking-data.json").write_text(
        json.dumps(serializable(data), separators=(",", ":"), allow_nan=False)+"\n")
    (ARTIFACTS/"metrics.json").write_text(json.dumps({"parameters": asdict(p),
        "provenance": provenance, "runs": metrics}, indent=2, allow_nan=False)+"\n")
    rows = ["| Approach | Method | Cost | Accepted updates | Position error (m) | Heading error (°) | Speed (m/s) | Clearance (m) |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for report in metrics:
        rows.append(f"| {SCENARIOS[report['scenario']]['name']} | "
                    f"{'iLQR' if report['method']=='ilqr' else 'DDP'} | {report['cost']:.4f} | "
                    f"{report['accepted_iterations']} | {report['position_error_m']:.3g} | "
                    f"{report['heading_error_deg']:.3g} | {report['terminal_speed_mps']:.3g} | "
                    f"{report['minimum_clearance_m']:.3f} |")
    (ARTIFACTS/"results.md").write_text("\n".join(rows)+"\n")


if __name__ == "__main__":
    main()
