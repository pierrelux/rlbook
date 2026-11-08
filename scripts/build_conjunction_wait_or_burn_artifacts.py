#!/usr/bin/env python3
"""Rebuild the wait-or-burn collision-avoidance experiment, figure and table.

Run from the repository root: uv run python scripts/build_conjunction_wait_or_burn_artifacts.py
Run it alone: with another builder running at the same time, the tree and network fits
came out differently (Nov 2026), and the published numbers were not reproduced.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from book_figure_theme import INK, MUTED, PALETTE, TEXT_WIDTH, apply_style, save  # noqa: E402
from conjunction_wait_or_burn import (  # noqa: E402
    COLLISION_COST, DATA, DEADLINE, RISK_FLOOR, burn_cost, cross_validated_fqi, fitted_q_iteration,
    load_epochs, realized_costs, rule_as_soon_as_above, rule_at_deadline, rule_hindsight,
    rule_never, summarize,
)

STATIC = ROOT / "_static" / "conjunction_wait_or_burn"
ARTIFACTS = ROOT / "artifacts" / "conjunction_wait_or_burn"
FEATURES = ["time_to_tca", "risk", "age"]
FIT_FLOOR = -7.0          # epochs with a smaller log10 probability are never fitted (never worth a maneuver)
FQI = dict(features=FEATURES, fit_floor=FIT_FLOOR, iterations=30, n_estimators=60, min_samples_leaf=20)
SPLITS = (0, 1, 2)        # seeds of the event-level 5-fold splits
BREAK_EVEN = np.log10(1.0 / COLLISION_COST)
POLICIES = {
    "never": "Never maneuver",
    "soon": "Maneuver at the first alarm",
    "deadline": "Decide at the deadline",
    "fqi_relative": "Fitted Q-iteration, relative target",
    "fqi_raw": "Fitted Q-iteration, raw target",
    "hindsight": "Hindsight (knows the final estimate)",
}


def run(epochs):
    results = {"never": summarize(realized_costs(epochs, rule_never(epochs))),
               "soon": summarize(realized_costs(epochs, rule_as_soon_as_above(epochs))),
               "deadline": summarize(realized_costs(epochs, rule_at_deadline(epochs))),
               "hindsight": summarize(realized_costs(epochs, rule_hindsight(epochs)))}
    for key, relative in (("fqi_relative", True), ("fqi_raw", False)):
        runs = []
        for seed in SPLITS:
            W = cross_validated_fqi(epochs, split_seed=seed, relative=relative, seed=100 * seed, **FQI)
            runs.append(summarize(realized_costs(epochs, W > epochs.burn)))
            print(key, seed, runs[-1], flush=True)
        mean = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        mean["splits"] = runs
        results[key] = mean
    return results


def dominant_event(epochs):
    """In-sample fits with and without the costliest event, for both targets."""
    big = epochs.event[np.argmax(epochs.outcome)]
    ratio = epochs.outcome / (COLLISION_COST * 10.0 ** epochs.risk)
    out = {"event": int(big), "final_probability": float(epochs.outcome.max() / COLLISION_COST),
           "cost_if_never": float(epochs.outcome.max()),
           "relative_target_range": [float(ratio[epochs.event == big].min()),
                                     float(ratio[epochs.event == big].max())]}
    for key, relative in (("relative", True), ("raw", False)):
        for name, train in (("all", np.ones(epochs.t.size, bool)), ("without", epochs.event != big)):
            W = fitted_q_iteration(epochs, train, relative=relative, seed=0, **FQI)
            out[f"{key}_{name}"] = summarize(realized_costs(epochs, W > epochs.burn))
            print("dominant event", key, name, out[f"{key}_{name}"], flush=True)
    return out


def decision_map(full_fit):
    """Maneuver (True) or wait at a message that has just arrived, on a grid of time and log-probability."""
    t = np.linspace(DEADLINE, 7, 120)
    r = np.linspace(-7, -2, 160)
    TT, RR = np.meshgrid(t, r)
    X = np.column_stack([TT.ravel(), RR.ravel(), np.zeros(TT.size)])
    return TT, RR, full_fit(X).reshape(TT.shape) > burn_cost(TT)


def figure(epochs, results, burn_map):
    raw = pd.read_csv(DATA).sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    alarmed = np.unique(epochs.event[rule_as_soon_as_above(epochs)])     # the events of the first-alarm rule
    final = raw.groupby("event_id").risk.last()
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.55), gridspec_kw={"width_ratios": [1.15, 1, 1]})

    ax = axes[0]
    ax.axvspan(0, DEADLINE, color=PALETTE["gray"], alpha=.12, lw=0)
    for needed in (False, True):
        for e in alarmed:
            if (final[e] >= BREAK_EVEN) != needed:
                continue
            g = raw[raw.event_id == e]
            # Below 1e-8 a message means "no meaningful risk" (41% of messages report 1e-30): the line
            # leaves the panel there instead of plunging to the bottom edge.
            risk = g.risk.where(g.risk > -8)
            ax.plot(g.time_to_tca, risk, lw=.7 if needed else .4,
                    color=PALETTE["vermilion"] if needed else PALETTE["gray"], alpha=.75 if needed else .08)
    ax.axhline(BREAK_EVEN, color=INK, lw=.8, ls="--")
    ax.set_xlim(7, 0), ax.set_ylim(-8, -1)
    ax.set_xticks([6, 4, 2, 0])
    ax.set_xlabel("days to closest approach")
    ax.set_ylabel(r"$\log_{10}$ collision probability")
    ax.set_title("(a) Alarms before the deadline")
    ax.text(DEADLINE / 2, -1.3, "too late\nto plan", ha="center", va="top", fontsize=8, color=MUTED)

    ax = axes[1]
    order = ["soon", "fqi_raw", "deadline", "fqi_relative", "hindsight"]
    labels = ["first alarm", "FQI, raw", "deadline", "FQI, relative", "hindsight"]
    y = np.arange(len(order))
    man = np.array([results[k]["maneuver_cost"] for k in order])
    col = np.array([results[k]["collision_cost"] for k in order])
    ax.barh(y, man, color=PALETTE["blue"], height=.6)
    ax.barh(y, col, left=man, color=PALETTE["vermilion"], height=.6)
    for yi, k in zip(y, order):
        ax.text(results[k]["total_cost"] + 10, yi, f"{results[k]['total_cost']:.0f}", va="center", fontsize=8.5)
    ax.text(man[0] / 2, -.62, "maneuvers", color=PALETTE["blue"], fontsize=8, ha="center", va="bottom")
    ax.text(man[0] + col[0] / 2, -.62, "collisions", color=PALETTE["vermilion"], fontsize=8,
            ha="left", va="bottom")
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(order) - .5, -1.1)
    ax.set_xlim(0, 1.25 * max(results[k]["total_cost"] for k in order))
    ax.set_xlabel("total cost over all events")
    ax.set_title(f"(b) Cost (never: {results['never']['total_cost']:,.0f})")

    ax = axes[2]
    TT, RR, burn = burn_map
    t = TT[0]
    ax.contourf(TT, RR, burn.astype(float), levels=[.5, 1.5], colors=[PALETTE["blue"]], alpha=.25)
    ax.contour(TT, RR, burn.astype(float), levels=[.5], colors=[PALETTE["blue"]], linewidths=1.2)
    ax.plot(t, np.log10(burn_cost(t) / COLLISION_COST), color=INK, lw=.8, ls="--")
    ax.set_xlim(7, DEADLINE), ax.set_ylim(-7, -2)
    ax.set_xticks([7, 6, 5, 4, 3, 2])
    ax.set_xlabel("days to closest approach")
    ax.set_ylabel(r"$\log_{10}$ collision probability")
    ax.set_title("(c) Learned rule at a new message")
    ax.text(6.8, -2.3, "maneuver", color=PALETTE["blue"], fontsize=8.5, va="top")
    ax.text(6.8, -6.75, "wait", color=INK, fontsize=8.5)
    ax.text(2.1, np.log10(burn_cost(2.1) / COLLISION_COST) - .2, "break-even\nof the message", fontsize=8,
            color=INK, ha="right", va="top")
    fig.tight_layout(w_pad=1.0)
    save(fig, STATIC, "wait_or_burn")


def write_results(results, epochs):
    lines = ["| Policy | Maneuvers | Maneuver cost | Expected collision cost | Total | Missed |",
             "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for key, name in POLICIES.items():
        r = results[key]
        cells = [f"{r['maneuvers']:.0f}", f"{r['maneuver_cost']:.0f}", f"{r['collision_cost']:.0f}",
                 f"{r['total_cost']:.0f}", f"{r['missed']:.0f} of {r['needed']:.0f}"]
        if key == "fqi_relative":
            cells = [f"**{c}**" for c in cells]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    n_events = np.unique(epochs.event).size
    lines.append("")
    lines.append(
        f"Totals over the {n_events:,} conjunctions with a message at least {DEADLINE:g} days before closest "
        "approach, in units of a maneuver two days out. The expected collision cost of an event without a "
        f"maneuver is {COLLISION_COST:.0e} times the collision probability of its last message. *Missed* "
        "counts the events whose last message exceeds the break-even probability of $10^{-5}$ and that "
        "received no maneuver. The fitted Q-iteration rows are out of sample: each event is decided by a "
        "policy fitted without it (5-fold cross-validation by event), averaged over three random splits.")
    (ARTIFACTS / "results.md").write_text("\n".join(lines) + "\n")


def main():
    STATIC.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    epochs = load_epochs()
    results = run(epochs)
    issues = []
    if results["hindsight"]["total_cost"] > min(r["total_cost"] for r in results.values()) + 1e-9:
        issues.append("a policy beats hindsight")
    if not results["fqi_relative"]["total_cost"] < results["soon"]["total_cost"]:
        issues.append("fitted Q-iteration does not beat maneuvering at the first alarm")
    if issues:
        raise SystemExit("\n".join(issues))

    sensitivity = dominant_event(epochs)
    _, full_fit = fitted_q_iteration(epochs, np.ones(epochs.t.size, bool), relative=True, seed=0,
                                     return_model=True, **FQI)

    apply_style("conjunction-wait-or-burn")
    figure(epochs, results, decision_map(full_fit))
    write_results(results, epochs)
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "schema_version": 1,
        "data": "ESA Kelvins Collision Avoidance Challenge training set (Uriot et al. 2022), CC BY 4.0",
        "parameters": {"deadline_days": DEADLINE, "collision_cost": COLLISION_COST,
                       "burn_cost": "(1 + 2 / t) / 2", "risk_floor": RISK_FLOOR, "fqi": FQI,
                       "splits": list(SPLITS)},
        "events": int(np.unique(epochs.event).size), "decision_epochs": int(epochs.t.size),
        "results": results,
        "dominant_event": sensitivity,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
