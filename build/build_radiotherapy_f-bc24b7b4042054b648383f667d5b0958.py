#!/usr/bin/env python3
"""Rebuild the radiotherapy fractionation solves, teaching figures, and replay data.

Run from the repository root: uv run python scripts/build_radiotherapy_fractionation_artifacts.py
Solver times exclude JAX compilation and are diagnostic, not benchmarks.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from radiotherapy_fractionation import (  # noqa: E402
    BEAMS, STRUCTURES, WEEKDAYS, CourseParameters, Phantom, beam_doses,
    dose_matrix, gaps, grid, initial_controls, initial_state, make_problem,
    masks, uniform_curative_plan,
)
from trajectory_optimization import SolverOptions, solve_trajectory  # noqa: E402

STATIC = ROOT / "_static" / "radiotherapy_fractionation"
ARTIFACTS = ROOT / "artifacts" / "radiotherapy_fractionation"
REPLAY_DATA = ROOT / "interactive" / "radiotherapy-fractionation-data.json"
# Okabe-Ito colors: one per beam, and the organ each beam crosses shares it.
BEAM_COLORS = ("#E69F00", "#CC79A7", "#009E73")
STRUCTURE_COLORS = ("#0072B2", "#E69F00", "#CC79A7", "#009E73")
GRAY = "#7b858a"
METHOD_LABEL = {"ilqr": "iLQR", "ddp": "DDP"}
DOSE_MAP = LinearSegmentedColormap.from_list(
    "dose", ["#ffffff00", "#fde9c8", "#f6b26b", "#d55e00", "#8c2d04"])


def save(fig, name):
    for suffix in ("svg", "pdf", "png"):
        options = {"metadata": {"Date": None}} if suffix == "svg" else {}
        if suffix == "pdf":
            options = {"metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(STATIC / f"{name}.{suffix}", bbox_inches="tight",
                    pad_inches=.04, **options)
    svg_path = STATIC / f"{name}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in
                                  svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def figure_style():
    # Match the boat and refrigerator figures in the same chapter.
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .65,
        "lines.linewidth": 1.4, "figure.dpi": 160,
        "legend.frameon": False, "savefig.dpi": 180,
        "svg.fonttype": "none", "svg.hashsalt": "radiotherapy-fractionation",
        "pdf.fonttype": 42,
    })


def draw_phantom(ax, phantom, total_dose):
    X, Y = grid(phantom)
    body, _ = masks(phantom)
    outline = Ellipse((0, 0), 2 * phantom.body_axes[0], 2 * phantom.body_axes[1],
                      fc="#f4f1ec", ec="#8a979e", lw=.8, zorder=0)
    ax.add_patch(outline)
    # The ellipse clips the smooth dose map; masking by grid cells would leave steps.
    image = ax.imshow(total_dose, extent=(X.min(), X.max(), Y.min(), Y.max()),
                      origin="lower", cmap=DOSE_MAP, vmin=0, vmax=total_dose[body].max(),
                      interpolation="bilinear", zorder=1)
    image.set_clip_path(outline)
    cx, cy, r = phantom.cord
    shapes = [
        Circle((0, 0), phantom.tumor_radius),
        Circle((cx, cy), r),
        Ellipse(phantom.heart[:2], 2 * phantom.heart[2], 2 * phantom.heart[3]),
        Ellipse(phantom.lung[:2], 2 * phantom.lung[2], 2 * phantom.lung[3]),
    ]
    for shape, color, name in zip(shapes, STRUCTURE_COLORS, STRUCTURES):
        shape.set(fc="none", ec=color, lw=1.4, zorder=3)
        ax.add_patch(shape)
    offsets = {"tumor": (0, -3.1), "spinal cord": (3.6, -9.2),
               "heart": (-6.06, 7.0), "lung": (6.5, 7.25)}
    for name, color in zip(STRUCTURES, STRUCTURE_COLORS):
        ax.text(*offsets[name], name, ha="center", va="center", fontsize=7.5,
                color=color, zorder=4)
    for source, color, name in zip(phantom.beam_sources, BEAM_COLORS, BEAMS):
        angle = np.deg2rad(source)
        start = 15.6 * np.array([np.cos(angle), np.sin(angle)])
        ax.annotate("", xy=.62 * start, xytext=start, zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))
    ax.set(xlim=(-16, 16), ylim=(-13, 13), aspect="equal")
    ax.axis("off")
    return image


def figures(results, uniform, A, phantom, p):
    figure_style()
    plan = results["ddp"]
    maps = beam_doses(phantom)
    total = np.tensordot(plan.controls.sum(axis=0), maps, axes=1)
    sessions = np.arange(1, p.sessions + 1)

    fig = plt.figure(figsize=(7.4, 3.25), layout="constrained")
    grid_spec = fig.add_gridspec(2, 2, width_ratios=[1, 1.35])
    slice_ax = fig.add_subplot(grid_spec[:, 0])
    beam_ax = fig.add_subplot(grid_spec[0, 1])
    health_ax = fig.add_subplot(grid_spec[1, 1], sharex=beam_ax)
    image = draw_phantom(slice_ax, phantom, total)
    bar = fig.colorbar(image, ax=slice_ax, location="bottom", shrink=.7, pad=.01, aspect=30)
    bar.set_label("Course dose of the optimized plan (Gy)", fontsize=7.5)
    bar.ax.tick_params(labelsize=7)
    bottom = np.zeros(p.sessions)
    for j, (name, color) in enumerate(zip(BEAMS, BEAM_COLORS)):
        beam_ax.bar(sessions, plan.controls[:, j], bottom=bottom, color=color,
                    width=.78, label=f"beam through {name}")
        bottom += plan.controls[:, j]
    beam_ax.step(np.r_[sessions - .5, sessions[-1] + .5],
                 np.r_[uniform[1].sum(axis=1), uniform[1][-1].sum()],
                 where="post", color=GRAY, ls="--", lw=1, label="uniform plan")
    beam_ax.set(ylabel="Tumor dose (Gy)", ylim=(0, 7))
    beam_ax.legend(loc="upper right", fontsize=7, ncol=2, columnspacing=.9,
                   handlelength=1.4)
    plt.setp(beam_ax.get_xticklabels(), visible=False)
    days = np.arange(p.sessions + 1) + .5
    for k, (name, color) in enumerate(zip(STRUCTURES, STRUCTURE_COLORS)):
        health_ax.plot(days, plan.states[:, k], color=color, label=name)
        health_ax.plot(days, uniform[0][:, k], color=color, ls=":", lw=1.1)
    health_ax.axhline(-1, color=".35", ls="--", lw=.8)
    health_ax.text(20.6, -1.07, "tolerance", ha="right", va="top", fontsize=7, color=".35")
    health_ax.axhline(0, color=".75", lw=.6)
    health_ax.set(xlabel="Session (weekend after 5, 10, 15)", ylabel="Health",
                  xlim=(.4, 20.6), ylim=(-2.35, 1.08), xticks=[1, 5, 10, 15, 20])
    health_ax.legend(loc="lower left", fontsize=7, ncol=4, columnspacing=.9,
                     handlelength=1.6)
    save(fig, "plan")

    fig, (cost_ax, dose_ax) = plt.subplots(1, 2, figsize=(7.4, 2.6), layout="constrained")
    for method, color, style in (("ilqr", "#0072B2", "-"), ("ddp", "#D55E00", "--")):
        costs = np.array([item["cost"] for item in results[method].history])
        cost_ax.semilogy(np.arange(len(costs)), costs, color=color, ls=style,
                         label=METHOD_LABEL[method])
        cost_ax.plot(len(costs) - 1, costs[-1], marker="o" if method == "ilqr" else "s",
                     ms=3.5, color=color)
    cost_ax.set(xlabel="Accepted update (0 = uniform 2.1 Gy plan)",
                ylabel="Nonlinear cost (log scale)")
    cost_ax.legend(loc="upper right")
    history = results["ddp"].history
    shown = [0, 1, 2, 4, len(history) - 1]
    shades = plt.cm.Blues(np.linspace(.35, 1, len(shown)))
    for index, shade in zip(shown, shades):
        dose = history[index]["controls"] @ A[0]
        dose_ax.step(np.r_[sessions - .5, 20.5], np.r_[dose, dose[-1]], where="post",
                     color=shade, lw=1.1 if index < shown[-1] else 1.6,
                     label=f"update {index}")
    dose_ax.set(xlabel="Session", ylabel="Tumor dose, DDP iterates (Gy)", xlim=(.5, 20.5),
                ylim=(0, 8.6), xticks=[1, 5, 10, 15, 20])
    dose_ax.legend(loc="upper center", fontsize=7, ncol=3, columnspacing=.9,
                   handlelength=1.6)
    save(fig, "iterations")


def course_summary(states, controls, A, problem):
    dose = controls @ A.T
    return {
        "cost": problem.cost(states, controls),
        "final_health": dict(zip(STRUCTURES, states[-1].tolist())),
        "lowest_health": dict(zip(STRUCTURES[1:], states[:, 1:].min(axis=0).tolist())),
        "course_dose_Gy": dict(zip(STRUCTURES, dose.sum(axis=0).tolist())),
        "beam_totals": dict(zip(BEAMS, controls.sum(axis=0).tolist())),
    }


def rounded(array, digits=5):
    return np.round(np.asarray(array, dtype=float), digits).tolist()


def write_replay_data(results, uniform, A, phantom, p, provenance):
    """Write the data read by interactive/radiotherapy-fractionation.html.

    The page replays recorded accepted iterates and the uniform plan; it does
    not rerun the optimizer. Dose maps are subsampled to every second grid
    point for the slice drawing.
    """
    maps = beam_doses(phantom)[:, ::2, ::2]
    X, _ = grid(phantom)
    runs = {}
    for method, result in results.items():
        runs[method] = {
            "status": result.status,
            "iterates": [{"cost": float(item["cost"]), "states": rounded(item["states"]),
                          "controls": rounded(item["controls"])}
                         for item in result.history],
        }
    data = {
        "schema_version": 1,
        "structures": list(STRUCTURES),
        "beams": list(BEAMS),
        "weekdays": list(WEEKDAYS),
        "gaps_days": gaps(p).tolist(),
        "dose_matrix": rounded(A),
        "parameters": asdict(p),
        "phantom": asdict(phantom),
        "dose_grid": {"extent": float(X.max()), "size": int(maps.shape[1]),
                      "maps": [rounded(m.ravel(), 3) for m in maps]},
        "runs": runs,
        "uniform": {"cost": float(uniform[2]), "states": rounded(uniform[0]),
                    "controls": rounded(uniform[1])},
        "provenance": provenance,
    }
    REPLAY_DATA.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")


def write_results(reports, uniform_summary):
    lines = [
        "| Plan | Accepted updates | Cost | Tumor dose (Gy) | Final tumor health | Lowest cord, heart, lung health |",
        "| :--- | ---: | ---: | ---: | ---: | :--- |",
    ]

    def row(name, updates, summary):
        low = ", ".join(f"{v:.2f}" for v in summary["lowest_health"].values())
        return (f"| {name} | {updates} | {summary['cost']:.3f} | "
                f"{summary['course_dose_Gy']['tumor']:.1f} | "
                f"{summary['final_health']['tumor']:.3f} | {low} |")

    lines.append(row("Uniform, 3 Gy per session", "–", uniform_summary))
    for report in reports:
        lines.append(row(f"{METHOD_LABEL[report['method']]} ({report['status']})",
                         report["accepted_iterations"], report["summary"]))
    lines += ["", "Health $-1$ is the tolerance of each organ. Both solvers start "
              "from equal beams delivering 2.1 Gy per session, a plan that "
              "leaves the tumor uncontrolled."]
    (ARTIFACTS / "results.md").write_text("\n".join(lines) + "\n")


def write_readme():
    (ARTIFACTS / "README.md").write_text(
        "# Radiotherapy fractionation with iLQR and DDP\n\n"
        "These deterministic experiments support the fractionation section of "
        "`iterative-trajectory-optimization.md`. The phantom and the dose-response "
        "coefficients are teaching values, not clinical data. The problem is "
        "adapted from https://github.com/pierrelux/ddp-fractionation.\n\n"
        "Regenerate from the repository root with:\n\n"
        "```bash\n"
        "uv run python scripts/build_radiotherapy_fractionation_artifacts.py\n"
        "```\n\n"
        "The builder solves the 20-session course with iLQR and DDP from the "
        "same uniform 2.1 Gy plan and evaluates the uniform equal-beam plan "
        "that just controls the tumor. It writes `metrics.json`, `results.md`, "
        "the figures in `../../_static/radiotherapy_fractionation/`, and the "
        "browser replay data `../../interactive/radiotherapy-fractionation-data.json`.\n\n"
        "The state is the health of the tumor, spinal cord, heart, and lung. "
        "Controls are the three beam intensities, in gray at the tumor per "
        "session. The builder exits with an error if a solve does not "
        "converge, if accepted costs fail to decrease, or if the optimized "
        "plan leaves the tumor uncontrolled or an organ beyond tolerance.\n")


def main():
    for directory in (STATIC, ARTIFACTS):
        directory.mkdir(parents=True, exist_ok=True)
    phantom, p = Phantom(), CourseParameters()
    options = SolverOptions(max_iterations=200)
    A = dose_matrix(phantom)
    problem = make_problem(A, p)
    results, reports, issues = {}, [], []
    for method in ("ilqr", "ddp"):
        result = solve_trajectory(problem, initial_state(), initial_controls(p),
                                  method=method, options=options)
        results[method] = result
        costs = [float(item["cost"]) for item in result.history]
        summary = course_summary(result.states, result.controls, A, problem)
        reports.append({
            "method": method, "status": result.status,
            "accepted_iterations": len(costs) - 1, "accepted_costs": costs,
            "projected_gradient": float(result.projected_gradient),
            "forward_evaluations": result.forward_evaluations,
            "backward_attempts": result.backward_attempts,
            "solver_seconds_excluding_compilation": result.solver_seconds,
            "summary": summary,
            "states": result.states.tolist(), "controls": result.controls.tolist(),
        })
        if result.status != "converged":
            issues.append(f"{method}: status {result.status}")
        if not np.all(np.diff(costs) < 0):
            issues.append(f"{method}: accepted costs did not decrease")
        if summary["final_health"]["tumor"] > 0.01:
            issues.append(f"{method}: tumor not controlled")
        if min(summary["lowest_health"].values()) < -1:
            issues.append(f"{method}: an organ exceeds tolerance")
        print(f"{method}: {result.status}; {len(costs) - 1} accepted; "
              f"cost {costs[-1]:.6f}", flush=True)
    if abs(reports[0]["summary"]["cost"] - reports[1]["summary"]["cost"]) > 1e-5:
        issues.append("iLQR and DDP reached different plans")
    states, controls = uniform_curative_plan(problem, p)
    uniform_summary = course_summary(states, controls, A, problem)
    uniform = (states, controls, uniform_summary["cost"])

    figures(results, uniform, A, phantom, p)
    provenance = {
        "sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                   for path in ("code/trajectory_optimization.py",
                                "code/radiotherapy_fractionation.py",
                                "scripts/build_radiotherapy_fractionation_artifacts.py")},
        "solver_options": asdict(options),
    }
    write_replay_data(results, uniform, A, phantom, p, provenance)
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "schema_version": 1, "parameters": asdict(p), "phantom": asdict(phantom),
        "dose_matrix": A.tolist(), "provenance": provenance, "runs": reports,
        "uniform_plan": {"beam_per_session": float(controls[0, 0]), **uniform_summary},
        "validation_issues": issues,
    }, indent=2, allow_nan=False) + "\n")
    write_results(reports, uniform_summary)
    write_readme()
    if issues:
        raise RuntimeError("radiotherapy artifact validation failed: " + "; ".join(issues))


if __name__ == "__main__":
    main()
