#!/usr/bin/env python3
"""Rebuild the two single-shooting truck solves, browser data, and figures.

Run from the repository root: uv run python scripts/build_truck_parking_artifacts.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from truck_parking import (  # noqa: E402
    SCENARIOS, TruckParameters, audit, configuration, initial_controls,
    linearized_amplification, pull_out_plan, rollout, solve_single_shooting,
    steering_amplification, trailer_axle,
)

STATIC = ROOT / "_static" / "truck_parking"
ARTIFACTS = ROOT / "artifacts" / "truck_parking"
REPLAY_DATA = ROOT / "interactive" / "truck-parking-data.json"
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#666666"
SOLVER_OPTIONS = dict(max_iterations=1500, gradient_tolerance=1e-5, relative_tolerance=1e-10)


def serializable(value, decimals=9):
    if isinstance(value, np.ndarray):
        rounded = np.round(value.astype(float), decimals)
        return [None if np.isnan(v) else v for v in rounded.tolist()] if rounded.ndim == 1 \
            else rounded.tolist()
    if isinstance(value, dict):
        return {key: serializable(item, decimals) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item, decimals) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    return value


def axle_path(states, p):
    return np.column_stack((states[:, 0] - p.trailer_length * np.cos(states[:, 3]),
                            states[:, 1] - p.trailer_length * np.sin(states[:, 3])))


def rectangle(origin, heading, back, front, width):
    body = np.array([[-back, -width / 2], [front, -width / 2], [front, width / 2], [-back, width / 2]])
    cs, sn = np.cos(heading), np.sin(heading)
    return body @ np.array([[cs, sn], [-sn, cs]]) + origin


def rig(ax, x, p, color=BLUE, alpha=1., fill=False, linestyle="-"):
    trailer = rectangle(x[:2], x[3], p.trailer_rear, p.trailer_front, p.trailer_width)
    tractor = rectangle(x[:2], x[2], p.tractor_rear, p.tractor_front, p.tractor_width)
    for body in (trailer, tractor):
        ax.add_patch(Polygon(body, facecolor=color if fill else "none", edgecolor=color,
                             linewidth=.7, alpha=alpha, linestyle=linestyle, zorder=4))


def yard(ax, goal, p, limits=(-13., 36., -4., 30.), goal_label="Bay"):
    ax.axhspan(limits[2], 0, color=".92", zorder=-4)
    ax.axhline(0, color=".45", lw=.8)
    ax.text(limits[0] + 1.5, -2.6, "DOCK", color=".45", fontsize=7)
    rig(ax, goal, p, color=".4", linestyle=":")
    if goal_label:
        anchor = goal[:2] + 1.6 * np.array([np.cos(goal[2]), np.sin(goal[2])])
        ax.text(anchor[0] + 2.2, anchor[1], goal_label, color=".4", fontsize=7, va="center")
    ax.set(xlim=limits[:2], ylim=limits[2:], aspect="equal", xlabel="$x$ (m)", ylabel="$y$ (m)")


def save(fig, name):
    for suffix in ("svg", "pdf", "png"):
        options = {"metadata": {"Date": None}} if suffix == "svg" else {}
        if suffix == "pdf":
            options = {"metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(STATIC / f"{name}.{suffix}", bbox_inches="tight", pad_inches=.04, **options)
    svg_path = STATIC / f"{name}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in
                                  svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def figure_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "lines.linewidth": 1.4, "figure.dpi": 160,
        "legend.frameon": False, "savefig.dpi": 180, "svg.fonttype": "none",
        "svg.hashsalt": "truck-parking", "pdf.fonttype": 42})


def figures(results, p):
    figure_style()
    dock, pull = results["dock"], results["pull_out"]
    time = np.arange(p.steps + 1) * p.dt
    control_time = time[:-1]
    stride = int(round(5 / p.dt))

    # Checkpoints of the backing solve: the seed and three later plans.
    history = dock["result"].history
    wanted = [0, 3, 10, len(history) - 1]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.35), layout="constrained", sharex=True, sharey=True)
    for ax, index in zip(axs, wanted):
        record = history[index]
        yard(ax, dock["goal"], p, goal_label=None)
        xs = record["states"]
        ax.plot(*axle_path(xs, p).T, color=BLUE)
        for t in range(0, p.steps + 1, 2 * stride):
            rig(ax, xs[t], p, alpha=.35 if t not in (0, p.steps) else 1.)
        ax.set_title(f"Iteration {record['iteration']}\nCost {record['cost']:.1f}")
        if ax is not axs[0]:
            ax.set_ylabel("")
    save(fig, "iterations")

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.), layout="constrained", sharex=True, sharey=True)
    for ax, (key, run) in zip(axs, results.items()):
        yard(ax, run["goal"], p, goal_label="Bay" if key == "dock" else "Lane target")
        seed = run["result"].history[0]["states"]
        ax.plot(*axle_path(seed, p).T, color=".45", ls=":", label="Straight seed")
        xs = run["result"].states
        ax.plot(*axle_path(xs, p).T, color=BLUE, label="Trailer axle")
        ax.plot(xs[:, 0], xs[:, 1], color=ORANGE, ls="--", lw=1., label="Tractor axle")
        for t in range(0, p.steps + 1, stride):
            rig(ax, xs[t], p, alpha=.3 if t not in (0, p.steps) else .9)
        ax.set_title(SCENARIOS[key]["name"])
        ax.legend(loc="lower right")
    axs[1].set_ylabel("")
    save(fig, "paths")

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")
    us, xs = dock["result"].controls, dock["result"].states
    axs[0].step(control_time, us[:, 0], where="post", color=BLUE, label="Speed (m/s)")
    axs[0].set(xlabel="Time (s)", ylabel="Speed (m/s)", ylim=(-1.7, 1.7), xlim=(0, time[-1]))
    twin = axs[0].twinx()
    twin.step(control_time, np.rad2deg(us[:, 1]), where="post", color=ORANGE, ls="--",
              label="Steering (°)")
    twin.set(ylabel="Steering (°)", ylim=(-34, 34))
    twin.spines["right"].set_visible(True)
    handles = [Line2D([], [], color=BLUE), Line2D([], [], color=ORANGE, ls="--")]
    axs[0].legend(handles, ["Speed (m/s)", "Steering (°)"], loc="lower right")
    axs[0].set_title("Backing plan: the 320 decision variables")
    for key, run, color, style in [("dock", dock, BLUE, "-"), ("pull_out", pull, ORANGE, "--")]:
        states = run["result"].states
        axs[1].plot(time, np.rad2deg(states[:, 2] - states[:, 3]), color=color, ls=style,
                    label=SCENARIOS[key]["name"])
    limit = np.rad2deg(p.hitch_limit)
    axs[1].axhline(limit, color=".6", lw=.7, ls=":")
    axs[1].axhline(-limit, color=".6", lw=.7, ls=":")
    axs[1].set(xlabel="Time (s)", ylabel="Hitch angle (°)", xlim=(0, time[-1]), ylim=(-70, 70))
    axs[1].set_title("Hitch angle along both plans")
    axs[1].legend(loc="upper right")
    save(fig, "plan")

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    ax = axs[0]
    amp = dock["amplification"]
    ax.semilogy(control_time, amp["linearized"], color=BLUE, lw=1.,
                label="Backing plan, per-interval product")
    ax.semilogy(control_time, amp["autodiff"], ".", color=BLUE, ms=2.6,
                label="Backing plan, reverse mode")
    ax.semilogy(control_time, amp["reversed_linearized"], color=ORANGE, lw=1., ls="--",
                label="Same path forward, per-interval product")
    ax.semilogy(control_time, amp["reversed_autodiff"], ".", color=ORANGE, ms=2.6,
                label="Same path forward, reverse mode")
    ax.axhline(1., color=".5", lw=.7)
    ax.set(xlabel="Control interval start (s)", ylabel="Amplification of a steering error",
           xlim=(0, time[-1]))
    ax.legend(loc="upper right", fontsize=6.8)
    ax.set_title("Terminal hitch angle vs. steering at $t$")
    ax = axs[1]
    for key, run, color, style in [("dock", dock, BLUE, "-"), ("pull_out", pull, ORANGE, "--")]:
        trace = run["result"].cost_trace
        ax.semilogy(np.arange(len(trace)), trace, color=color, ls=style, label=SCENARIOS[key]["name"])
    ax.set(xlabel="L-BFGS-B iteration", ylabel="Objective $J(\\mathbf{u})$ (log scale)")
    ax.set_title("Convergence from the straight seeds")
    ax.legend()
    save(fig, "conditioning")


def main():
    for directory in (STATIC, ARTIFACTS):
        directory.mkdir(parents=True, exist_ok=True)
    p = TruckParameters()
    results, metrics = {}, []
    for key, cfg in SCENARIOS.items():
        initial, goal = configuration(cfg["initial"], p), configuration(cfg["goal"], p)
        result = solve_single_shooting(initial, goal, initial_controls(p, cfg["seed_speed"]), p,
                                       **SOLVER_OPTIONS)
        report = audit(initial, result.controls, goal, p)
        if result.status not in ("converged", "small_decrease") or not report["docking_pass"]:
            raise RuntimeError(f"{key}: {result.status}, {report}")
        forward = steering_amplification(initial, result.controls, p)
        linear = linearized_amplification(result.states, result.controls, p)
        back_initial, back_controls = pull_out_plan(result.states, result.controls)
        back_states = np.asarray(rollout(back_initial, back_controls, p))
        reversed_forward = steering_amplification(back_initial, back_controls, p)
        reversed_linear = linearized_amplification(back_states, back_controls, p)
        moving = ~np.isnan(forward)
        amplification = dict(autodiff=forward, linearized=linear,
                             reversed_autodiff=reversed_forward, reversed_linearized=reversed_linear)
        report.update(
            scenario=key, status=result.status, iterations=result.iterations,
            objective_evaluations=result.objective_evaluations,
            initial_cost=float(result.cost_trace[0]), cost=float(result.cost_trace[-1]),
            projected_gradient=result.projected_gradient,
            solver_seconds_excluding_compilation=result.solver_seconds,
            checkpoints=len(result.history),
            amplification_first_interval=float(forward[0]),
            amplification_first_interval_linearized=float(linear[0]),
            amplification_max_moving=float(np.nanmax(forward)),
            amplification_min_moving=float(np.nanmin(forward)),
            amplification_max_relative_mismatch=float(np.max(np.abs(forward[moving] / linear[moving] - 1))),
            reversed_plan_first_moving_amplification=float(reversed_forward[moving][0]) if key == "dock"
            else float(reversed_forward[~np.isnan(reversed_forward)][0]),
            reversed_plan_return_error_m=float(np.linalg.norm(back_states[-1][:2] - initial[:2])))
        results[key] = dict(result=result, initial=initial, goal=goal, report=report,
                            amplification=amplification)
        metrics.append(report)
        print(f"{key}: {result.status} after {result.iterations} iterations; cost "
              f"{report['cost']:.4f}; amplification at start {forward[0]:.2f}; docking passed",
              flush=True)
    figures(results, p)

    source_files = ["code/truck_parking.py", "scripts/build_truck_parking_artifacts.py"]
    provenance = {"sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                             for path in source_files}, "solver_options": SOLVER_OPTIONS}
    scenarios = {}
    for key, run in results.items():
        history = [dict(iteration=int(h["iteration"]), cost=h["cost"],
                        projected_gradient=h["projected_gradient"],
                        states=np.round(h["states"], 6), controls=np.round(h["controls"], 6))
                   for h in run["result"].history]
        scenarios[key] = {"name": SCENARIOS[key]["name"], "initial": run["initial"],
                          "goal": run["goal"], "seed_speed": SCENARIOS[key]["seed_speed"],
                          "history": history, "cost_trace": run["result"].cost_trace,
                          "metrics": run["report"], "amplification": run["amplification"]}
    data = {"schema_version": 1, "parameters": asdict(p), "provenance": provenance,
            "state_columns": ["p_x_m", "p_y_m", "tractor_heading_rad", "trailer_heading_rad"],
            "control_columns": ["speed_mps", "steering_rad"], "scenarios": scenarios}
    REPLAY_DATA.write_text(json.dumps(serializable(data), separators=(",", ":"),
                                      allow_nan=False) + "\n")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(serializable({
        "parameters": asdict(p), "provenance": provenance, "runs": metrics,
        "final_plans": {key: {"states": run["result"].states, "controls": run["result"].controls}
                        for key, run in results.items()}}), indent=2, allow_nan=False) + "\n")

    rows = ["| Scenario | Stop | Iterations | Objective evaluations | Initial cost | Final cost |",
            "| :--- | :--- | ---: | ---: | ---: | ---: |"]
    stop = {"converged": "gradient", "small_decrease": "small decrease"}
    for r in metrics:
        rows.append(f"| {SCENARIOS[r['scenario']]['name']} | {stop[r['status']]} | "
                    f"{r['iterations']} | {r['objective_evaluations']} | {r['initial_cost']:.1f} | "
                    f"{r['cost']:.4f} |")
    rows.extend(["", "The arrival checks below use the finer replay.", "",
                 "| Scenario | Trailer-axle error (m) | Trailer heading error (°) | Hitch angle (°) | "
                 "Clearance (m) | Largest hitch angle (°) | Distance (m) |",
                 "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for r in metrics:
        rows.append(f"| {SCENARIOS[r['scenario']]['name']} | {r['axle_error_m']:.3g} | "
                    f"{r['trailer_heading_error_deg']:.3g} | {r['hitch_error_deg']:.3g} | "
                    f"{r['minimum_clearance_m']:.3f} | {r['maximum_hitch_deg']:.1f} | "
                    f"{r['distance_driven_m']:.1f} |")
    rows.extend(["", "Amplification of a steering perturbation into the terminal hitch angle, "
                 "at the first control interval of each plan.", "",
                 "| Plan | Reverse mode | Per-interval product |",
                 "| :--- | ---: | ---: |"])
    for r in metrics:
        rows.append(f"| {SCENARIOS[r['scenario']]['name']} | {r['amplification_first_interval']:.3g} | "
                    f"{r['amplification_first_interval_linearized']:.3g} |")
    dock = results["dock"]
    moving = ~np.isnan(dock["amplification"]["reversed_autodiff"])
    rows.append(f"| Backing plan driven forward | "
                f"{dock['amplification']['reversed_autodiff'][moving][0]:.3g} | "
                f"{dock['amplification']['reversed_linearized'][moving][0]:.3g} |")
    (ARTIFACTS / "results.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
