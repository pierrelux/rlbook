#!/usr/bin/env python3
"""Draw the tube of nearby trajectories around the final iLQR docking plan.

Run from the repository root: uv run python scripts/build_boat_tube_figure.py
The boat follows the angled-approach plan, receives a small push at 2 s, and
then either replays the planned thrusts or applies the local feedback law
u = u_bar + K (x - x_bar), clipped to the thrust bounds. The shaded tube is
the envelope of the feedback runs after the small pushes. The larger pushes
appear only in the animation. The script writes
the static figure and the data for interactive/boat-tube.html.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
from boat_docking import BoatParameters, SCENARIOS, initial_controls, make_problem, step
from build_boat_docking_artifacts import BLUE, ORANGE, harbor, save
from trajectory_optimization import SolverOptions, solve_trajectory

PUSH_STEP, PUSHES, SEED, DOCKED = 20, 12, 3, .1
PUSH_SIZES = {"small": .3, "large": .6}
TUBE, TUBE_EDGE = "#cfe3f0", "#7fb2d4"


def simulate(x_push, plan_states, plan_controls, gains, feedback):
    xs = [x_push]
    for t in range(PUSH_STEP, len(plan_controls)):
        u = plan_controls[t]
        if feedback:
            u = np.clip(u + gains[t] @ (xs[-1] - plan_states[t]), -1., 1.)
        xs.append(np.asarray(step(xs[-1], u)))
    return np.array(xs)


def tube_radius(plan, paths, floor=.05, window=4):
    """Largest distance from the planned path, attributed to the nearest planned point.

    Each run point is assigned to its nearest point on the plan after the push,
    so the tube measures how far runs stray sideways, not how far they lag in
    time. A running maximum over a few steps smooths the outline.
    """
    centre = plan[PUSH_STEP:, :2]
    radius = np.full(len(centre), floor)
    for path in paths:
        distance = np.linalg.norm(path[:, None, :2] - centre[None], axis=2)
        nearest = distance.argmin(axis=1)
        np.maximum.at(radius, nearest, distance[np.arange(len(path)), nearest])
    smooth = np.array([radius[max(0, j-window):j+window+1].max() for j in range(len(radius))])
    return np.r_[np.zeros(PUSH_STEP), smooth]


def tube_polygon(plan, radius):
    """Union of the disks of the given radii along the plan after the push."""
    from shapely.geometry import MultiPoint, Point
    from shapely.ops import unary_union
    disks = [Point(*plan[t, :2]).buffer(radius[t], 32) for t in range(PUSH_STEP, len(plan))]
    hulls = [MultiPoint(list(a.exterior.coords) + list(b.exterior.coords)).convex_hull
             for a, b in zip(disks, disks[1:])]
    outline = np.asarray(unary_union(hulls).simplify(.002).exterior.coords)
    return outline[:, 0], outline[:, 1]


def pushes(rng, size):
    """Random velocity kicks up to `size` m/s per axis, with proportional heading and spin changes."""
    kicks = np.zeros((PUSHES, 6))
    kicks[:, 2] = rng.uniform(-.2, .2, PUSHES) * size
    kicks[:, 3:5] = rng.uniform(-1., 1., (PUSHES, 2)) * size
    kicks[:, 5] = rng.uniform(-.125, .125, PUSHES) * size
    return kicks


def main():
    p = BoatParameters()
    result = solve_trajectory(make_problem(p), SCENARIOS["angled"]["initial"],
                              initial_controls(p), "ilqr", SolverOptions())
    if not result.converged:
        raise RuntimeError(result.status)
    xs, us, K = result.states, result.controls, result.gains
    berth_error = lambda path: float(np.linalg.norm(path[-1, :2] - xs[-1, :2]))
    sets = {}
    for name, size in PUSH_SIZES.items():
        # The same random directions at each size, so only the magnitude changes.
        kicks = pushes(np.random.default_rng(SEED), size)
        runs = {feedback: [simulate(xs[PUSH_STEP] + kick, xs, us, K, feedback) for kick in kicks]
                for feedback in (False, True)}
        sets[name] = dict(size=size, runs=runs,
                          docked={feedback: [berth_error(path) <= DOCKED for path in runs[feedback]]
                                  for feedback in runs})
    small = sets["small"]["runs"]
    if not all(sets["small"]["docked"][True]):
        raise RuntimeError("a small push was not absorbed by the feedback")
    radius = tube_radius(xs, small[True])
    tube = tube_polygon(xs, radius)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "lines.linewidth": 1.4, "figure.dpi": 160,
        "legend.frameon": False, "savefig.dpi": 180, "svg.fonttype": "none",
        "svg.hashsalt": "boat-tube", "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.15), layout="constrained", sharex=True, sharey=True)
    titles = ["Replay the planned thrusts", r"Local feedback $\bar u_t+K_t(x_t-\bar x_t)$"]
    for ax, feedback, color, title in zip(axs, (False, True), (ORANGE, BLUE), titles):
        harbor(ax, p, (-6.8, 2.2, -.6, 4.4))
        ax.fill(*tube, facecolor=TUBE, edgecolor=TUBE_EDGE, lw=.6, zorder=1)
        ax.plot(xs[:PUSH_STEP+1, 0], xs[:PUSH_STEP+1, 1], color=".35", lw=1.1, zorder=2)
        for path in small[feedback]:
            ax.plot(path[:, 0], path[:, 1], color=color, lw=.75, alpha=.6, zorder=3)
            ax.plot(*path[-1, :2], "o", ms=2.6, mfc="white", mec=color, mew=.9, zorder=5)
        ax.plot(xs[PUSH_STEP:, 0], xs[PUSH_STEP:, 1], color="k", lw=1.3, zorder=4)
        ax.plot(*xs[PUSH_STEP, :2], "o", color="k", ms=4, zorder=6)
        ax.annotate("push at 2 s", xs[PUSH_STEP, :2], xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=".2")
        docked = sum(sets["small"]["docked"][feedback])
        ax.text(.98, .97, f"{docked} of {PUSHES} within {100*DOCKED:.0f} cm of the berth",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=color)
        ax.set_title(title)
    axs[1].set_ylabel("")
    handles = [Line2D([], [], color="k", lw=1.3, label="Plan"),
               Patch(facecolor=TUBE, edgecolor=TUBE_EDGE, lw=.6, label="Tube"),
               Line2D([], [], color=".4", lw=.75, label="Pushed runs (orange: replay, blue: feedback)"),
               Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mec=".4", label="Position at 20 s")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, columnspacing=1.2,
               handlelength=2.)
    save(fig, "tube")

    rounded = lambda a: np.round(np.asarray(a)[:, :3], 4).tolist()
    data = {"dt": p.dt, "push_step": PUSH_STEP, "length": p.length, "width": p.width,
            "berth": [p.berth_x, p.berth_y], "docked_radius": DOCKED, "plan": rounded(xs),
            "radius": np.round(radius, 4).tolist(),
            "tube": np.round(np.column_stack(tube), 4).tolist(),
            "pushes": {name: {"size": entry["size"],
                              "open_loop": [rounded(path) for path in entry["runs"][False]],
                              "feedback": [rounded(path) for path in entry["runs"][True]]}
                       for name, entry in sets.items()}}
    (ROOT / "interactive" / "boat-tube-data.json").write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    for name, entry in sets.items():
        for feedback, label in ((False, "replay"), (True, "feedback")):
            errors = np.array([berth_error(path) for path in entry["runs"][feedback]])
            print(f"{name} pushes ({entry['size']} m/s), {label}: {sum(entry['docked'][feedback])} "
                  f"of {PUSHES} docked; berth error mean {errors.mean():.3f} m, max {errors.max():.3f} m")


if __name__ == "__main__":
    main()
