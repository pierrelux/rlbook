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
from book_figure_theme import INK, MUTED, PALETTE, RULE, SOFT, apply_style, save  # noqa: E402
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
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.45))

    ax = axes[0]
    grid = np.linspace(0, 1.15, 400)
    lo, hi = np.exp(-1.2816 * SIGMA - SIGMA ** 2 / 2), np.exp(1.2816 * SIGMA - SIGMA ** 2 / 2)
    ax.fill_between(grid, lo * growth(grid), hi * growth(grid), color=PALETTE["blue"], alpha=.16, lw=0,
                    label="80% of years")
    ax.plot(grid, growth(grid), color=PALETTE["blue"], lw=1.6, label=r"mean $f(s)$")
    ax.plot(grid, grid, color=MUTED, lw=.8, ls="--", label="no change")
    ax.axvline(C, color=PALETTE["vermilion"], lw=.8, ls=":")
    ax.text(C + .02, 1.08, f"tipping point\n$C = {C}$", color=PALETTE["vermilion"], fontsize=7, va="top")
    ax.set_xlim(0, 1.15), ax.set_ylim(0, 1.15)
    ax.set_xlabel("escapement $s$ (stock left to spawn)")
    ax.set_ylabel(r"stock next year $x'$")
    ax.set_title("(a) Growth with an Allee effect")
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.4)

    ax = axes[1]
    ax.axvspan(0, x_hat, color=PALETTE["vermilion"], alpha=.10, lw=0)
    ax.axvspan(x_hat, s_star, color=PALETTE["green"], alpha=.10, lw=0)
    ax.plot(x, v, color=INK, lw=1.6)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel("stock $x$ before the season")
    ax.set_ylabel(r"$v^\star(x)$")
    ax.set_title("(b) Optimal value")
    ax.annotate("cliff at the\ntipping point", xy=(.33, 1.6), xytext=(.55, .55), fontsize=7, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=.6))

    ax = axes[2]
    ax.axvspan(0, x_hat, color=PALETTE["vermilion"], alpha=.10, lw=0)
    ax.axvspan(x_hat, s_star, color=PALETTE["green"], alpha=.10, lw=0)
    ax.plot(x, s, color=INK, lw=1.6)
    ax.plot([0, X_MAX], [0, X_MAX], color=MUTED, lw=.8, ls="--")
    ax.text(x_hat / 2, 1.42, "fish\nout", ha="center", va="top", fontsize=7, color=PALETTE["vermilion"])
    ax.text((x_hat + s_star) / 2, 1.42, "close the\nfishery", ha="center", va="top", fontsize=7,
            color=PALETTE["green"])
    ax.text(1.18, 1.42, "harvest down\nto $S^\\star$", ha="center", va="top", fontsize=7, color=INK)
    ax.text(X_MAX - .02, s_star + .04, f"$S^\\star = {s_star:.3f}$", ha="right", fontsize=7, color=INK)
    ax.set_xlim(0, X_MAX), ax.set_ylim(0, 1.5)
    ax.set_xlabel("stock $x$ before the season")
    ax.set_ylabel(r"escapement $s^\star(x)$")
    ax.set_title("(c) Optimal policy")
    fig.tight_layout(w_pad=1.6)
    save(fig, STATIC, "problem")


def figure_collocation(bellman, exact, results):
    x = bellman.x
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.45))
    ax = axes[0]
    ax.plot(x, exact.v, color=INK, lw=2.2, alpha=.35, label="exact")
    for key, res in results.items():
        f = res.v
        ax.plot(x, f(x), color=COLORS[key], lw=1.1, label=res.basis.name)
        ax.plot(res.basis.nodes, f(res.basis.nodes), "o", ms=2.6, color=COLORS[key])
    ax.set_xlim(0, X_MAX)
    ax.set_xlabel("stock $x$")
    ax.set_ylabel("value")
    ax.set_title(f"(a) Approximations, $n = {FEATURED_N}$")
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.4)

    ax = axes[1]
    for key, res in results.items():
        f = res.v
        ax.plot(x, bellman.apply(f) - f(x), color=COLORS[key], lw=1.0)
        ax.plot(res.basis.nodes, np.zeros(res.basis.n), "o", ms=2.6, color=COLORS[key])
    ax.axhline(0, color=MUTED, lw=.6)
    ax.set_xlim(0, X_MAX)
    ax.set_xlabel("stock $x$")
    ax.set_ylabel(r"$(\mathrm{L}\hat v)(x) - \hat v(x)$")
    ax.set_title("(b) Bellman residual")

    ax = axes[2]
    ax.plot(x, exact.escapement, color=INK, lw=2.2, alpha=.35, label="optimal")
    for key, res in results.items():
        ax.plot(x, bellman.greedy(res.v), color=COLORS[key], lw=1.1)
    ax.axvline(C, color=PALETTE["vermilion"], lw=.8, ls=":")
    ax.set_xlim(0, 1.0), ax.set_ylim(0, 1.0)
    ax.set_xlabel("stock $x$")
    ax.set_ylabel("escapement")
    ax.set_title("(c) Greedy policies")
    ax.text(C + .015, .93, "tipping\npoint", fontsize=7, color=PALETTE["vermilion"], va="top")
    fig.tight_layout(w_pad=1.6)
    save(fig, STATIC, "collocation")


def figure_sweep(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.45))
    for ax, metric, title in ((axes[0], "value_error", r"(a) Value error $\max_x |\hat v - v^\star|$"),
                              (axes[1], "policy_loss", r"(b) Policy loss $\max_x (v^\star - v^{\hat\pi})$")):
        for key in BASES:
            ok = [r for r in rows if r["key"] == key and not r["diverged"]]
            ax.semilogy([r["n"] for r in ok], [r[metric] for r in ok], "o-", ms=2.8, lw=1.0,
                        color=COLORS[key], label=ok[0]["basis"])
            bad = [r["n"] for r in rows if r["key"] == key and r["diverged"]]
            if bad:
                ax.plot(bad, [12.0] * len(bad), "x", ms=4.5, mew=1.2, color=COLORS[key],
                        label="iteration diverged" if key == "chebyshev" else None)
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32, 64], labels=["4", "8", "16", "32", "64"])
        ax.set_xlabel("coefficients $n$")
        ax.set_title(title)
    axes[0].legend(loc="lower left", fontsize=6.5, handlelength=1.4)
    fig.tight_layout(w_pad=2)
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
