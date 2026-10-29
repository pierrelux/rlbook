#!/usr/bin/env python3
"""Rebuild the balancing-stick figures and metrics for Receding-Horizon Control.

Run from the repository root:
uv run python scripts/build_receding_horizon_artifacts.py
Every number quoted in the chapter is written to
artifacts/receding_horizon/metrics.json.
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
from receding_horizon import (  # noqa: E402
    StickParameters,
    discretize,
    lyapunov_decrease_margin,
    make_mpc,
    riccati_limit,
    saturated_lqr,
    simulate,
    solve_mpc,
    tap,
)

STATIC = ROOT / "_static" / "receding_horizon"
ARTIFACTS = ROOT / "artifacts" / "receding_horizon"
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "0.55"
X0 = np.array([0.2, 0.6])


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .65, "lines.linewidth": 1.4,
        "legend.frameon": False, "svg.fonttype": "path", "svg.hashsalt": "receding-horizon",
    })


def save(fig, name: str) -> None:
    fig.savefig(STATIC / f"{name}.svg", bbox_inches="tight", pad_inches=.04,
                metadata={"Date": None})
    plt.close(fig)


def limit_band(ax, value: float) -> None:
    for sign in (-1, 1):
        ax.axhline(sign * value, color=GREY, lw=.7, ls=":")


def replanning_figure(params: StickParameters, metrics: dict) -> None:
    """Open-loop execution of one plan against replanning, on a mismatched plant."""
    mpc = make_mpc(params, 40, "cost+set")
    plant = discretize(params, length=0.85)
    push = tap(0.5, 20)
    steps = 60
    nominal = simulate(mpc, X0, steps, replan=False)
    frozen = simulate(mpc, X0, steps, plant=plant, disturbance=push, replan=False)
    replanned = simulate(mpc, X0, steps, plant=plant, disturbance=push)
    time = params.period * np.arange(steps + 1)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.2, 3.6), sharex=True)
    ax0.plot(time, nominal.x[:, 0], color=GREY, lw=1.0, ls="--", label="plan, nominal model")
    ax0.plot(time, frozen.x[:, 0], color=ORANGE, label="plan executed open loop")
    ax0.plot(time, replanned.x[:, 0], color=BLUE, label="replanned every step")
    limit_band(ax0, params.theta_max)
    ax0.axvline(20 * params.period, color=GREY, lw=.6)
    ax0.text(20 * params.period + .02, .42, "tap", fontsize=8, color="0.35")
    ax0.set(ylim=(-.5, .5), ylabel=r"angle $\theta$ (rad)")
    ax0.legend(loc="lower right", ncol=1)
    ax1.step(time[:-1], frozen.u[:, 0], where="post", color=ORANGE)
    ax1.step(time[:-1], replanned.u[:, 0], where="post", color=BLUE)
    limit_band(ax1, params.u_max)
    ax1.set(xlabel="time (s)", ylabel=r"command $u$ (rad/s$^2$)", ylim=(-6, 6))
    fig.align_ylabels()
    save(fig, "replanning")
    leave = np.argmax(np.abs(frozen.x[:, 0]) > params.theta_max)
    metrics["replanning"] = {
        "plant_length": 0.85, "tap_rate": 0.5, "tap_time": 20 * params.period,
        "open_loop_leaves_limit_at": float(leave * params.period),
        "replanned_max_angle": float(np.max(np.abs(replanned.x[:, 0]))),
        "replanned_all_feasible": bool(replanned.feasible.all()),
        "replanned_max_command": float(np.max(np.abs(replanned.u))),
    }


def horizon_figure(params: StickParameters, metrics: dict) -> None:
    """Closed loops without terminal ingredients for several horizons."""
    steps = 80
    time = params.period * np.arange(steps + 1)
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    shades = {1: "#F0B27A", 3: "#E67E22", 5: ORANGE, 10: "#7F3C10"}
    runs = {}
    for N, color in shades.items():
        run = simulate(make_mpc(params, N, "none"), X0, steps)
        runs[N] = run
        lost = np.flatnonzero(~run.feasible)
        end = lost[0] + 1 if lost.size else steps
        ax.plot(time[:end + 1], run.x[:end + 1, 0], color=color, label=rf"no terminal, $N={N}$")
        if lost.size:
            ax.plot(time[end], run.x[end, 0], "x", color=color, ms=5)
    ref = simulate(make_mpc(params, 10, "cost+set"), X0, steps)
    ax.plot(time, ref.x[:, 0], color=BLUE, label=r"terminal cost and set, $N=10$")
    limit_band(ax, params.theta_max)
    ax.set(xlabel="time (s)", ylabel=r"angle $\theta$ (rad)", ylim=(-.12, .42))
    ax.legend(loc="center right", ncol=1)
    save(fig, "horizon")
    metrics["horizon"] = {
        str(N): {
            "first_infeasible_step": int(np.flatnonzero(~r.feasible)[0]) if (~r.feasible).any() else None,
            "final_angle": float(r.x[-1, 0]),
            "cost": float(r.cost) if r.feasible.all() else None,
        }
        for N, r in runs.items()
    }
    metrics["horizon"]["reference_cost"] = float(ref.cost)


def value_figure(params: StickParameters, metrics: dict) -> None:
    """Finite-horizon values on both sides of the infinite-horizon value."""
    horizons = np.arange(1, 31)
    v_inf = solve_mpc(make_mpc(params, 300, "cost"), X0).value
    values = {term: [solve_mpc(make_mpc(params, int(N), term), X0).value for N in horizons]
              for term in ("none", "cost", "cost+set")}
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.axhline(v_inf, color="k", lw=.8, label=r"$V_\infty(\mathbf{x}_0)$")
    ax.plot(horizons, values["none"], "o-", ms=3, color=ORANGE, label="no terminal ingredients")
    ax.plot(horizons, values["cost"], "s-", ms=3, color=GREEN, label=r"terminal cost $\mathbf{x}^\top P\mathbf{x}$ only")
    finite = np.isfinite(values["cost+set"])
    ax.plot(horizons[finite], np.array(values["cost+set"])[finite], "D", ms=3.5, color=BLUE,
            label="terminal cost and set")
    first = int(horizons[finite][0])
    ax.axvspan(.5, first - .5, color="0.93", zorder=-3)
    ax.text(1.2, 0.35 * v_inf, "terminal set\nunreachable", fontsize=8, color="0.35")
    ax.set(xlabel=r"horizon $N$", ylabel=r"$V_N(\mathbf{x}_0)$", xlim=(.5, 30.5))
    ax.legend(loc="lower right")
    save(fig, "values")
    metrics["values"] = {
        "v_inf": float(v_inf),
        "first_feasible_horizon_with_set": first,
        "none": {str(N): float(v) for N, v in zip(horizons, values["none"])},
        "cost": {str(N): float(v) for N, v in zip(horizons, values["cost"])},
        "cost+set": {str(N): (float(v) if np.isfinite(v) else None)
                     for N, v in zip(horizons, values["cost+set"])},
    }


def feasible_set_figure(params: StickParameters, metrics: dict) -> None:
    """Feasible sets X_N, the terminal ellipse, and one slice of the MPC law."""
    thetas = np.linspace(-params.theta_max, params.theta_max, 57)
    rates = np.linspace(-3.0, 3.0, 61)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw={"width_ratios": [1.15, 1]})
    colors = {20: "#DCE9F5", 10: "#9CC3E6", 5: "#4A90C8"}
    gap = 0.0
    areas = {}
    for N, color in colors.items():
        mpc = make_mpc(params, N, "cost+set")
        grid = np.zeros((len(rates), len(thetas)))
        for i, w in enumerate(rates):
            for j, th in enumerate(thetas):
                sol = solve_mpc(mpc, np.array([th, w]))
                grid[i, j] = sol.feasible
                if sol.feasible:
                    gap = max(gap, abs(sol.u[0, 0] - saturated_lqr(mpc, np.array([th, w]))[0]))
        areas[str(N)] = float(grid.mean())
        ax0.contourf(thetas, rates, grid, levels=[.5, 1.5], colors=[color])
        ax0.contour(thetas, rates, grid, levels=[.5], colors=["0.3"], linewidths=.5)
    mpc = make_mpc(params, 20, "cost+set")
    s = np.linspace(0, 2 * np.pi, 200)
    L = np.linalg.cholesky(np.linalg.inv(mpc.P))
    ellipse = np.sqrt(mpc.alpha) * (L @ np.vstack([np.cos(s), np.sin(s)]))
    ax0.plot(*ellipse, color="k", lw=1.0)
    ax0.plot(*X0, "k*", ms=6)
    ax0.text(.01, -.35, r"$\mathcal{X}_f$", fontsize=9)
    ax0.set(xlabel=r"$\theta$ (rad)", ylabel=r"$\dot\theta$ (rad/s)", title=r"feasible sets $\mathcal{X}_5\subset\mathcal{X}_{10}\subset\mathcal{X}_{20}$")
    rate = 0.8
    fine = np.linspace(-params.theta_max, params.theta_max, 141)
    kappa = []
    for th in fine:
        sol = solve_mpc(mpc, np.array([th, rate]))
        kappa.append(sol.u[0, 0] if sol.feasible else np.nan)
    sat = [saturated_lqr(mpc, np.array([th, rate]))[0] for th in fine]
    ax1.plot(fine, sat, color=GREY, lw=2.6, alpha=.5, label=r"$\mathrm{sat}(K\mathbf{x})$")
    ax1.plot(fine, kappa, color=BLUE, lw=1.2, label=r"$\kappa_{20}(\mathbf{x})$")
    limit_band(ax1, params.u_max)
    ax1.set(xlabel=r"$\theta$ (rad)", ylabel=r"$u$ (rad/s$^2$)",
            title=rf"first action along $\dot\theta={rate}$")
    ax1.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "feasible-sets")
    feasible = np.isfinite(kappa)
    metrics["feasible_sets"] = {
        "grid_fraction_feasible": areas,
        "max_gap_to_saturated_lqr": float(gap),
        "slice_rate": rate,
        "slice_infeasible_above_theta": float(fine[feasible][-1]),
    }


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    style()
    params = StickParameters()
    mpc = make_mpc(params, 10)
    metrics = {
        "parameters": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(params).items()
                       if np.isfinite(np.asarray(v, dtype=float)).all()},
        "A": mpc.A.tolist(), "B": mpc.B.ravel().tolist(), "P": mpc.P.tolist(),
        "K": mpc.K.ravel().tolist(), "alpha": mpc.alpha,
        "riccati_iteration_gap": float(np.max(np.abs(riccati_limit(mpc.A, mpc.B, mpc.Q, mpc.R) - mpc.P))),
        "lyapunov_decrease_worst": lyapunov_decrease_margin(mpc),
        "x0": X0.tolist(), "x0_P_x0": float(X0 @ mpc.P @ X0),
    }
    replanning_figure(params, metrics)
    horizon_figure(params, metrics)
    value_figure(params, metrics)
    feasible_set_figure(params, metrics)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ("alpha", "x0_P_x0", "replanning", "feasible_sets")}, indent=2))


if __name__ == "__main__":
    main()
