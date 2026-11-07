#!/usr/bin/env python3
"""Rebuild the fishery-near-a-tipping-point solves, figures and table.

Run from the repository root: uv run python scripts/build_fishery_tipping_point_artifacts.py
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
from book_figure_theme import INK, MUTED, PALETTE, TEXT_WIDTH, apply_style, label, save  # noqa: E402
from fishery_tipping_point import (  # noqa: E402
    BASES, C, GAMMA, K, N_FINE, N_SHOCK, PRICE, R, SIGMA, X_MAX, Bellman,
    assess, collocate, growth, reed_escapement, solve_exact,
)

STATIC = ROOT / "_static" / "fishery_tipping_point"
ARTIFACTS = ROOT / "artifacts" / "fishery_tipping_point"
FEATURED_N = 12                      # basis size shown in the comparison figure
SWEEP = list(range(4, 25)) + [28, 32, 40, 48, 64]
TABLE_N = [6, 10, 12, 20, 40]
COLORS = {"chebyshev": PALETTE["vermilion"], "linear": PALETTE["blue"], "exact": INK}


def sweep(bellman, exact):
    rows = []
    for n in SWEEP:
        for key, Basis in BASES.items():
            row = assess(collocate(Basis(n), bellman), exact, bellman)
            row["key"] = key
            rows.append(row)
    return rows


def figure_problem(bellman, exact):
    x, v, s = exact.x, exact.v, exact.escapement
    x_hat, s_star = exact.fish_out_level, exact.constant_escapement
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.4))

    ax = axes[0]
    grid = np.linspace(0, 1.15, 400)
    lo, hi = np.exp(-1.2816 * SIGMA - SIGMA ** 2 / 2), np.exp(1.2816 * SIGMA - SIGMA ** 2 / 2)
    ax.fill_between(grid, lo * growth(grid), hi * growth(grid), color=PALETTE["blue"], alpha=.15, lw=0)
    ax.plot(grid, grid, color=MUTED, lw=.8, ls="--")
    ax.plot(grid, growth(grid), color=PALETTE["blue"])
    ax.axvline(C, color=PALETTE["vermilion"], lw=.8, ls=":")
    label(ax, C + .03, 1.1, "tipping\npoint $C$", PALETTE["vermilion"], va="top")
    label(ax, .62, .84, "$f(s)$", PALETTE["blue"], ha="right")
    label(ax, 1.13, 1.05, "$x' = s$", MUTED, ha="right", va="bottom")
    ax.set_xlim(0, 1.15), ax.set_ylim(0, 1.15)
    ax.set_xlabel("escapement $s$")
    ax.set_ylabel(r"stock next year $x'$")
    ax.set_title("(a) Growth with an Allee effect")

    for ax in axes[1:]:
        ax.axvspan(0, x_hat, color=PALETTE["vermilion"], alpha=.09, lw=0)
        ax.axvspan(x_hat, s_star, color=PALETTE["green"], alpha=.09, lw=0)
        ax.set_xlim(0, X_MAX)
        ax.set_xlabel("stock $x$ before the season")

    ax = axes[1]
    ax.plot(x, v, color=INK)
    ax.set_ylim(0, 4.8)
    ax.set_ylabel(r"$v^\star(x)$")
    ax.set_title("(b) Optimal value")
    ax.annotate("cliff at the\ntipping point", xy=(.335, 1.9), xytext=(.62, .7), fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=.6, shrinkA=1, shrinkB=2))

    ax = axes[2]
    ax.plot([0, X_MAX], [0, X_MAX], color=MUTED, lw=.8, ls="--")
    ax.plot(x, s, color=INK)
    label(ax, x_hat / 2, 1.47, "fish\nout", PALETTE["vermilion"], ha="center", va="top")
    label(ax, (x_hat + s_star) / 2, 1.47, "close", PALETTE["green"], ha="center", va="top")
    label(ax, 1.17, 1.47, "harvest\ndown to $S^\star$", INK, ha="center", va="top")
    label(ax, X_MAX - .02, s_star + .06, f"$S^\\star = {s_star:.2f}$", INK, ha="right", va="bottom")
    ax.set_ylim(0, 1.5)
    ax.set_ylabel(r"escapement $s^\star(x)$")
    ax.set_title("(c) Optimal policy")
    fig.tight_layout(w_pad=1.4)
    save(fig, STATIC, "problem")


def figure_collocation(bellman, exact, results):
    x = bellman.x
    exact_style = dict(color=PALETTE["gray"], lw=3.4, alpha=.4, solid_capstyle="round")
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.4))

    ax = axes[0]
    ax.plot(x, exact.v, **exact_style)
    for key, res in results.items():
        f = res.v
        ax.plot(x, f(x), color=COLORS[key], lw=1.2)
        ax.plot(res.basis.nodes, f(res.basis.nodes), "o", ms=3, color=COLORS[key], mec="none")
    label(ax, .05, 2.25, "Chebyshev", COLORS["chebyshev"], va="bottom")
    label(ax, .5, .9, "piecewise\nlinear", COLORS["linear"], va="center")
    label(ax, 1.05, 3.55, "exact", PALETTE["gray"], va="top")
    ax.set_xlim(0, X_MAX), ax.set_ylim(0, 4.8)
    ax.set_xlabel("stock $x$")
    ax.set_ylabel(r"$\hat v(x)$")
    ax.set_title(f"(a) Approximations, $n = {FEATURED_N}$")

    ax = axes[1]
    ax.axhline(0, color=MUTED, lw=.6)
    for key, res in results.items():
        f = res.v
        ax.plot(x, bellman.apply(f) - f(x), color=COLORS[key], lw=1.0)
        ax.plot(res.basis.nodes, np.zeros(res.basis.n), "o", ms=3, color=COLORS[key], mec="none", zorder=3)
    ax.set_xlim(0, X_MAX), ax.set_ylim(-.24, .24)
    ax.set_xlabel("stock $x$")
    ax.set_ylabel(r"$R(x;\theta)$")
    ax.set_title("(b) Bellman residual")

    ax = axes[2]
    ax.axvline(C, color=PALETTE["vermilion"], lw=.8, ls=":")
    ax.plot(x, exact.escapement, **exact_style)
    for key, res in results.items():
        ax.plot(x, bellman.greedy(res.v), color=COLORS[key], lw=1.2)
    label(ax, C + .02, .95, "tipping\npoint", PALETTE["vermilion"], va="top")
    label(ax, .52, .62, "optimal", PALETTE["gray"], ha="right")
    ax.set_xlim(0, 1.0), ax.set_ylim(0, 1.0)
    ax.set_xlabel("stock $x$")
    ax.set_ylabel("greedy escapement")
    ax.set_title("(c) Greedy policies")
    fig.tight_layout(w_pad=1.4)
    save(fig, STATIC, "collocation")


def figure_sweep(rows):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.45))
    panels = ((axes[0], "value_error", "(a) Value error", r"$\max_x |\hat v(x) - v^\star(x)|$"),
              (axes[1], "policy_loss", "(b) Policy loss", r"$\max_x\, (v^\star - v^{\hat\pi})(x)$"))
    for ax, metric, title, ylabel in panels:
        for key in BASES:
            ok = [r for r in rows if r["key"] == key and not r["diverged"]]
            ax.plot([r["n"] for r in ok], [r[metric] for r in ok], "o-", ms=3, lw=1.1, mec="none",
                    color=COLORS[key])
            bad = [r["n"] for r in rows if r["key"] == key and r["diverged"]]
            if bad:
                ax.plot(bad, [30.0] * len(bad), "x", ms=5, mew=1.3, color=COLORS[key], clip_on=False)
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32, 64], labels=["4", "8", "16", "32", "64"])
        ax.set_xlim(3.6, 72)
        ax.set_xlabel("coefficients $n$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        label(ax, 13, 30.0, "diverged", COLORS["chebyshev"], fontsize=8)
    axes[0].set_ylim(5e-3, 40)
    axes[1].set_ylim(1e-5, 40)
    label(axes[0], 66, 1.4e-2, "Chebyshev", COLORS["chebyshev"], ha="right", va="top")
    label(axes[0], 66, 6e-2, "piecewise\nlinear", COLORS["linear"], ha="right", va="bottom")
    label(axes[1], 60, 1.2e-5, "Chebyshev", COLORS["chebyshev"], ha="right", va="bottom")
    label(axes[1], 66, 4e-3, "piecewise\nlinear", COLORS["linear"], ha="right", va="bottom")
    fig.tight_layout(w_pad=2.2)
    save(fig, STATIC, "sweep")


def write_results(rows, exact):
    by = {(r["key"], r["n"]): r for r in rows}
    lines = ["| $n$ | Chebyshev: value error | fishes out below | policy loss "
             "| Piecewise linear: value error | fishes out below | policy loss |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in TABLE_N:
        cells = [str(n)]
        for key in ("chebyshev", "linear"):
            r = by[(key, n)]
            if r["diverged"]:
                cells += ["diverged", "", ""]
            else:
                cells += [f"{r['value_error']:.2f}", f"{r['fish_out_level']:.2f}", f"{r['policy_loss']:.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"Collocation by function iteration with $n$ coefficients. *Value error* is "
        r"$\max_x |\hat v(x) - v^\star(x)|$; the optimal value ranges from 0 to "
        f"{exact.v.max():.1f}. *Fishes out below* is the largest stock that the greedy policy of "
        r"$\hat v$ harvests to zero; the optimal policy fishes out stocks below "
        f"{exact.fish_out_level:.2f}, and the tipping point is at {C}. *Policy loss* is "
        r"$\max_x (v^\star - v^{\hat\pi})(x)$, with the greedy policy $\hat\pi$ evaluated exactly. "
        "*Diverged*: the coefficients grew without bound.")
    (ARTIFACTS / "results.md").write_text("\n".join(lines) + "\n")


def main():
    STATIC.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    bellman = Bellman()
    exact = solve_exact(bellman)
    issues = []
    reed = reed_escapement()
    if not (reed - 1e-3 <= exact.constant_escapement <= reed + 0.01):
        issues.append(f"constant escapement {exact.constant_escapement:.4f} far from Reed's {reed:.4f}")
    if np.abs(bellman.evaluate(exact.escapement) - exact.v).max() > 1e-6:
        issues.append("policy evaluation disagrees with value iteration")
    if not (0.0 < exact.fish_out_level < C < exact.constant_escapement < K):
        issues.append("optimal policy does not have the fish-out / close / escapement structure")
    coarse = solve_exact(Bellman(n_fine=(N_FINE - 1) // 2 + 1))
    if np.abs(coarse(exact.x) - exact.v).max() > 1e-3:
        issues.append("reference solution not converged in the grid size")

    rows = sweep(bellman, exact)
    if any(r["diverged"] for r in rows if r["key"] == "linear"):
        issues.append("piecewise-linear iteration diverged (it is an averager and must not)")
    if any(r["policy_loss"] < -1e-8 for r in rows if not r["diverged"]):
        issues.append("a greedy policy beats the optimal one")
    if issues:
        raise SystemExit("\n".join(issues))

    featured = {key: collocate(Basis(FEATURED_N), bellman) for key, Basis in BASES.items()}
    apply_style("fishery-tipping-point")
    figure_problem(bellman, exact)
    figure_collocation(bellman, exact, featured)
    figure_sweep(rows)
    write_results(rows, exact)
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "schema_version": 1,
        "model": "single stock, Allee growth x' = z s exp(r (1 - s/K)(s - C)/K), log-normal z, escapement control",
        "parameters": {"r": R, "K": K, "C": C, "sigma": SIGMA, "gamma": GAMMA, "price": PRICE,
                       "x_max": X_MAX, "fine_grid": N_FINE, "gauss_hermite_nodes": N_SHOCK},
        "exact": {"value_iteration_sweeps": exact.iterations,
                  "fish_out_level": exact.fish_out_level,
                  "constant_escapement": exact.constant_escapement,
                  "reed_escapement": reed,
                  "max_value": float(exact.v.max())},
        "featured_n": FEATURED_N,
        "sweep": [{k: v for k, v in r.items() if k != "basis"} | {"basis": r["basis"]} for r in rows],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
