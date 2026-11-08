#!/usr/bin/env python3
"""Rebuild the mallard-harvest solves, policy maps and error table.

Run from the repository root: uv run python scripts/build_mallard_harvest_artifacts.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from book_figure_theme import INK, MUTED, PALETTE, apply_style, save  # noqa: E402
from mallard_harvest import (  # noqa: E402
    CLOSED_BELOW, CODES, GAMMA, GOAL, MODEL_WEIGHTS, MODELS, N_MAX, N_MIN, P_MAX, PACKAGES,
    Grid, GridFit, NetworkFit, TreeFit, evaluate_policy, fitted_value_iteration, greedy,
    published_2012, sample_states, value_iteration,
)

STATIC = ROOT / "_static" / "mallard_harvest"
ARTIFACTS = ROOT / "artifacts" / "mallard_harvest"
REFERENCE_STEP = 0.1        # million birds and million ponds
ITERATIONS = 400            # fitted value iterations; gamma**400 < 3e-4
SAMPLES = 2000              # sampled states for the regressors
COLORS = [PALETTE["gray"], PALETTE["orange"], PALETTE["sky"], PALETTE["blue"]]
MAP_N = np.linspace(4.0, 9.0, 201)
MAP_P = np.linspace(1.0, 6.5, 221)


def fits():
    states = sample_states(SAMPLES, seed=0)
    return [GridFit(2.0), GridFit(1.0), GridFit(0.5), TreeFit(states), NetworkFit(states)]


def run_fits(reference):
    rows, policies = [], {}
    test = reference.grid.points
    for fit in fits():
        start = time.time()
        v, errors = fitted_value_iteration(fit, ITERATIONS, test)
        eps = max(errors)
        actions = greedy(test, v)
        loss = reference.values - evaluate_policy(reference.grid, actions)
        rows.append({
            "fit": fit.name, "parameters": int(fit.states.shape[0]),
            "fit_error": eps,
            "value_bound": eps / (1 - GAMMA),
            "value_error": float(np.abs(v(test) - reference.values).max()),
            "policy_bound": 2 * GAMMA * eps / (1 - GAMMA) ** 2,
            "policy_loss": float(loss.max()),
            "policy_loss_mean": float(loss.mean()),
            "agreement": float((actions == reference.policy).mean()),
            "seconds": round(time.time() - start, 1),
        })
        policies[fit.name] = v
        print(f"{fit.name}: {rows[-1]}", flush=True)
    return rows, policies


def policy_map(v):
    NN, PP = np.meshgrid(MAP_N, MAP_P, indexing="ij")
    return greedy(np.column_stack([NN.ravel(), PP.ravel()]), v).reshape(NN.shape)


def draw_map(ax, actions, title):
    cmap = ListedColormap(COLORS)
    ax.imshow(actions, origin="lower", aspect="auto", cmap=cmap, vmin=-.5, vmax=3.5, interpolation="nearest",
              extent=(MAP_P[0], MAP_P[-1], MAP_N[0], MAP_N[-1]))
    ax.set_title(title)
    ax.set_xlabel("May ponds $P$ (millions)")
    ax.grid(False)


def figure_policies(reference, network):
    population, ponds, table = published_2012()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)
    # Published table: each cell drawn over the pond and population ranges it covers.
    pub = np.full((MAP_N.size, MAP_P.size), -1)
    for i, n in enumerate(population):
        lo = n - 0.125 if i > 0 else MAP_N[0]
        hi = n + 0.125 if i < len(population) - 1 else MAP_N[-1]
        if i == 0:                              # the closed row covers everything below the constraint
            hi = CLOSED_BELOW
        if i == 1:                              # the 4.75-6.0 row of the table starts at the constraint
            lo = CLOSED_BELOW
        for j, p in enumerate(ponds):
            plo = p - 0.25 if j > 0 else MAP_P[0]
            phi = p + 0.25 if j < len(ponds) - 1 else MAP_P[-1]
            pub[np.ix_((MAP_N >= lo) & (MAP_N <= hi), (MAP_P >= plo) & (MAP_P <= phi))] = table[i, j]
    draw_map(axes[0], np.ma.masked_less(pub, 0), "(a) Published, USFWS 2012")
    draw_map(axes[1], policy_map(reference.v), "(b) Grid solution, step 0.1")
    draw_map(axes[2], policy_map(network), "(c) Network, 2,000 states")
    axes[0].set_ylabel("breeding population $N$ (millions)")
    for ax in axes:
        ax.axhline(CLOSED_BELOW, color=INK, lw=.6, ls=":")
    handles = [Patch(color=c, label=p) for c, p in zip(COLORS, PACKAGES)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, handlelength=1.2,
               bbox_to_anchor=(.5, -.03))
    fig.tight_layout(rect=(0, .07, 1, 1), w_pad=1.2)
    save(fig, STATIC, "policies")


def write_results(rows):
    lines = ["| Fit | Targets per iteration | Fit error $\\varepsilon$ | Bound $\\varepsilon/(1-\\gamma)$ "
             "| Value error | Bound $2\\gamma\\varepsilon/(1-\\gamma)^2$ | Policy loss | Same package |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['fit'].capitalize()} | {r['parameters']:,} | {r['fit_error']:.2f} | {r['value_bound']:.0f} "
                     f"| {r['value_error']:.2f} | {r['policy_bound']:,.0f} | {r['policy_loss']:.3f} "
                     f"| {100 * r['agreement']:.1f}% |")
    lines.append("")
    lines.append(
        f"Fitted value iteration for {ITERATIONS} iterations with γ = {GAMMA}. The fit error is the largest "
        r"$|v_{k+1} - \mathrm{L} v_k|$ over the 12,231 states of the reference grid in the last 20 iterations. "
        "Value error and policy loss are measured against the reference solution, in millions of birds of "
        "discounted harvest; the optimal value ranges from about 30 to 51. *Same package* is the share of "
        "reference states where the greedy policy picks the reference package.")
    (ARTIFACTS / "results.md").write_text("\n".join(lines) + "\n")


def main():
    STATIC.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    issues = []
    start = time.time()
    reference = value_iteration(Grid(REFERENCE_STEP, REFERENCE_STEP))
    reference_seconds = time.time() - start
    check = value_iteration(Grid(2 * REFERENCE_STEP, 2 * REFERENCE_STEP))
    grid_change = float(np.abs(check.v(reference.grid.points) - reference.values).max())
    if grid_change > 0.1:
        issues.append(f"reference grid not converged: halving the step changes v by {grid_change:.3f}")
    if np.abs(evaluate_policy(reference.grid, reference.policy) - reference.values).max() > 1e-4:
        issues.append("policy evaluation disagrees with value iteration")

    population, ponds, table = published_2012()
    cells = np.array([[n, p] for n in population for p in ponds])
    ours = greedy(cells, reference.v).reshape(table.shape)
    agree, within_one = float((ours == table).mean()), float((np.abs(ours - table) <= 1).mean())
    if agree < 0.8 or within_one < 1.0:
        issues.append(f"policy far from the published table: {agree:.2f} exact, {within_one:.2f} within one")

    rows, values = run_fits(reference)
    for r in rows:
        if r["value_error"] > r["value_bound"] + 1e-6 or r["policy_loss"] > r["policy_bound"] + 1e-6:
            issues.append(f"{r['fit']}: a bound is violated")
        if r["policy_loss"] < -1e-6:
            issues.append(f"{r['fit']}: greedy policy beats the reference")
    if issues:
        raise SystemExit("\n".join(issues))

    apply_style("mallard-harvest")
    figure_policies(reference, values["neural network"])
    write_results(rows)
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "schema_version": 1,
        "model": "USFWS midcontinent mallard AHM, 2012 models and weights, four regulation packages",
        "parameters": {"gamma": GAMMA, "goal": GOAL, "closed_below": CLOSED_BELOW,
                       "models": [list(m) for m in MODELS], "model_weights": MODEL_WEIGHTS.tolist(),
                       "N_range": [N_MIN, N_MAX], "P_range": [0.0, P_MAX]},
        "reference": {"step": REFERENCE_STEP, "states": int(reference.grid.size),
                      "iterations": reference.iterations, "seconds": round(reference_seconds, 1),
                      "change_when_step_doubled": grid_change,
                      "value_range": [float(reference.values.min()), float(reference.values.max())]},
        "published_2012": {"cells": int(table.size), "same_package": agree, "within_one_package": within_one,
                           "ours": ["".join(CODES[a] for a in row) for row in ours],
                           "published": ["".join(CODES[a] for a in row) for row in table],
                           "population_rows": population.tolist(), "pond_columns": ponds.tolist()},
        "fits": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
