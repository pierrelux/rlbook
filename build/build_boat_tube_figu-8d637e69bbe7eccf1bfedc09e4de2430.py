#!/usr/bin/env python3
"""Draw the tube of nearby trajectories around the final iLQR docking plan.

Run from the repository root: uv run python scripts/build_boat_tube_figure.py
The boat follows the angled-approach plan, receives a small push at 2 s, and
then either replays the planned thrusts or applies the local feedback law
u = u_bar + K (x - x_bar), clipped to the thrust bounds. The shaded tube is
the envelope of the feedback runs that return to the plan. The script writes
the static figure and the data for interactive/boat-tube.html.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
from boat_docking import BoatParameters, SCENARIOS, initial_controls, make_problem, step
from build_boat_docking_artifacts import BLUE, ORANGE, boat, harbor, save
from trajectory_optimization import SolverOptions, solve_trajectory

PUSH_STEP, PUSHES, SEED = 20, 12, 3


def simulate(x_push, plan_states, plan_controls, gains, feedback):
    xs = [x_push]
    for t in range(PUSH_STEP, len(plan_controls)):
        u = plan_controls[t]
        if feedback:
            u = np.clip(u + gains[t] @ (xs[-1] - plan_states[t]), -1., 1.)
        xs.append(np.asarray(step(xs[-1], u)))
    return np.array(xs)


def tube_radius(plan, paths, floor=.04):
    """Largest distance from the plan at each time step over the given runs."""
    radius = np.full(len(plan), floor)
    for path in paths:
        distance = np.linalg.norm(path[:, :2] - plan[PUSH_STEP:, :2], axis=1)
        radius[PUSH_STEP:] = np.maximum(radius[PUSH_STEP:], distance)
    return radius


def tube_polygon(plan, radius):
    tangent = np.gradient(plan[:, :2], axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    upper = plan[:, :2] + radius[:, None] * normal
    lower = plan[:, :2] - radius[:, None] * normal
    outline = np.r_[upper, lower[::-1]]
    return outline[:, 0], outline[:, 1]


def main():
    p = BoatParameters()
    result = solve_trajectory(make_problem(p), SCENARIOS["angled"]["initial"],
                              initial_controls(p), "ilqr", SolverOptions())
    if not result.converged:
        raise RuntimeError(result.status)
    xs, us, K = result.states, result.controls, result.gains
    rng = np.random.default_rng(SEED)
    # Velocity kicks of up to about 0.4 m/s and small heading and spin changes.
    kicks = np.zeros((PUSHES, 6))
    kicks[:, 2] = rng.uniform(-.08, .08, PUSHES)
    kicks[:, 3:5] = rng.uniform(-.4, .4, (PUSHES, 2))
    kicks[:, 5] = rng.uniform(-.05, .05, PUSHES)
    # A push "escapes" when even the feedback run ends more than 0.5 m from the berth.
    escaped = [i for i, kick in enumerate(kicks) if np.linalg.norm(
        simulate(xs[PUSH_STEP] + kick, xs, us, K, True)[-1, :2] - xs[-1, :2]) > .5]

    runs = {feedback: [simulate(xs[PUSH_STEP] + kick, xs, us, K, feedback) for kick in kicks]
            for feedback in (False, True)}
    radius = tube_radius(xs, [runs[True][i] for i in range(PUSHES) if i not in escaped])
    tube = tube_polygon(xs, radius)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "lines.linewidth": 1.4, "figure.dpi": 160,
        "legend.frameon": False, "savefig.dpi": 180, "svg.fonttype": "none",
        "svg.hashsalt": "boat-tube", "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained", sharex=True, sharey=True)
    titles = ["Replay the planned thrusts", r"Local feedback $\bar u_t+K_t(x_t-\bar x_t)$"]
    errors = {}
    for ax, feedback, color, title in zip(axs, (False, True), (ORANGE, BLUE), titles):
        harbor(ax, p, (-9.5, 3., -.8, 6.))
        finals = []
        for i, kick in enumerate(kicks):
            path = runs[feedback][i]
            if i in escaped:
                ax.plot(path[:, 0], path[:, 1], color=color, lw=1., ls="--",
                        label="Push outside the tube" if i == escaped[0] else None)
            else:
                ax.plot(path[:, 0], path[:, 1], color=color, lw=.7, alpha=.55)
            finals.append(path[-1])
            boat(ax, path[-1], p, color=color, alpha=.35)
        ax.fill(*tube, color=BLUE, alpha=.18, lw=0, label="Tube")
        ax.plot(xs[:, 0], xs[:, 1], color="k", lw=1.6, label="Plan")
        ax.plot(*xs[PUSH_STEP, :2], "o", color="k", ms=3.5, label="Push at 2 s")
        finals = np.array(finals)
        errors[feedback] = np.linalg.norm(finals[:, :2] - xs[-1, :2], axis=1)
        ax.set_title(title)
    for ax in axs:
        ax.legend(loc="upper right")
    axs[1].set_ylabel("")
    save(fig, "tube")
    rounded = lambda a: np.round(np.asarray(a)[:, :3], 4).tolist()
    data = {"dt": p.dt, "push_step": PUSH_STEP, "length": p.length, "width": p.width,
            "berth": [p.berth_x, p.berth_y], "plan": rounded(xs),
            "radius": np.round(radius, 4).tolist(), "escaped": escaped,
            "open_loop": [rounded(path) for path in runs[False]],
            "feedback": [rounded(path) for path in runs[True]]}
    (ROOT / "interactive" / "boat-tube-data.json").write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    for feedback, name in ((False, "open loop"), (True, "feedback")):
        rest = np.delete(errors[feedback], escaped)
        print(f"{name}: final position error over the {len(rest)} contained pushes "
              f"mean {rest.mean():.3f} m, max {rest.max():.3f} m; escaped pushes "
              f"{np.round(errors[feedback][escaped], 3).tolist()} m")


if __name__ == "__main__":
    main()
