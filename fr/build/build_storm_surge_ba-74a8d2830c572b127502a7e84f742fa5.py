#!/usr/bin/env python3
"""Rebuild the storm-surge-barrier solves, teaching figure, table, and replay data.

Run from the repository root: uv run python scripts/build_storm_surge_barrier_artifacts.py
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
from storm_surge_barrier import (  # noqa: E402
    ATTEMPT_HOURS, CLOSED, F_MIN, FLOOD_COST, GRID, HORIZON, N_F, OPEN,
    PRIOR_MEAN, PRIOR_SD, Q_FAIL, R_REPAIR, REOPEN_HOURS, SHIP_COST,
    SIGMA_STEP, STATUS, THRESHOLD, WARN_HOURS, backward_induction,
    build_model, evaluate, exceed_probability, forecast_kernel, outcome_rates,
    p_not_closed, prior, sample_forecast_path, simulate, threshold_rule,
)

STATIC = ROOT / "_static" / "storm_surge_barrier"
ARTIFACTS = ROOT / "artifacts" / "storm_surge_barrier"
REPLAY_DATA = ROOT / "interactive" / "storm-surge-barrier-data.json"
FLOOD_COSTS = (200, 500, 1000, 2000, 5000)
REPLAY_COSTS = (200, 1000, 5000)
MARGINS = range(-10, 41)          # cm below the criterion at which a threshold rule orders the closure
LATEST_LEADS = range(6, HORIZON + 1)
N_STORMS_SAMPLED = 4000
SEED = 20231221                   # date of the first automatic closure of the Maeslant barrier
BLUE, ORANGE, GREEN, VERMILION, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#7b858a"
NOMINAL = WARN_HOURS + ATTEMPT_HOURS


def thresholds(close):
    """Lowest forecast (cm) at which the policy orders the closure, per lead; None if never."""
    out = []
    for h in range(HORIZON + 1):
        idx = np.nonzero(close[h])[0]
        if idx.size and idx[-1] - idx[0] + 1 != idx.size:
            raise SystemExit(f"closing region at lead {h} is not an upper set of forecasts")
        if idx.size and idx[-1] != N_F - 1:
            raise SystemExit(f"closing region at lead {h} does not reach the top of the grid")
        out.append(int(GRID[idx[0]]) if idx.size else None)
    return out


def best_threshold_rule(m, start):
    best = None
    for margin in MARGINS:
        for lead in LATEST_LEADS:
            cost = float(start @ evaluate(m, threshold_rule(margin, lead))[HORIZON][:, OPEN])
            if best is None or cost < best[0] - 1e-12:
                best = (cost, margin, lead)
    _, margin, lead = best
    return outcome_rates(m, threshold_rule(margin, lead), start), margin, lead


def solve_all():
    start = prior()
    results = {}
    for C in FLOOD_COSTS:
        m = build_model(C)
        sol = backward_induction(m)
        dp = outcome_rates(m, sol.close, start)
        if abs(dp["cost"] - float(start @ sol.values[HORIZON][:, OPEN])) > 1e-8:
            raise SystemExit("policy evaluation disagrees with backward induction")
        tuned, margin, lead = best_threshold_rule(m, start)
        literal = outcome_rates(m, threshold_rule(0, HORIZON), start)
        if dp["cost"] > min(tuned["cost"], literal["cost"]) + 1e-9:
            raise SystemExit(f"C={C}: backward induction worse than a threshold rule")
        results[C] = {"model": m, "solution": sol, "dp": dp, "tuned": tuned, "literal": literal,
                      "rule": {"margin": margin, "latest_lead": lead},
                      "dp_threshold": thresholds(sol.close)}
    return results


def policies(results, C):
    r = results[C]
    return {"dp": r["solution"].close,
            "tuned": threshold_rule(r["rule"]["margin"], r["rule"]["latest_lead"]),
            "literal": threshold_rule(0, HORIZON)}


def sample_storms(results):
    """Sample storms and pick a few that show the typical cases, all with the same random numbers."""
    rng = np.random.default_rng(SEED)
    K, start = forecast_kernel(), prior()
    pols = policies(results, FLOOD_COST)
    storms = []
    for _ in range(N_STORMS_SAMPLED):
        path = sample_forecast_path(rng, start, K)
        u = rng.random(HORIZON + 1)
        runs = {k: simulate(p, path, u) for k, p in pols.items()}
        storms.append((path, u, runs))
    # Monte Carlo check of the exact evaluation.
    for k in pols:
        mc = np.mean([s[2][k]["cost"] for s in storms])
        exact = results[FLOOD_COST][k]["cost"]
        se = np.std([s[2][k]["cost"] for s in storms]) / np.sqrt(len(storms))
        if abs(mc - exact) > 4 * se + 0.05:
            raise SystemExit(f"Monte Carlo cost of {k} ({mc:.2f}) far from exact ({exact:.2f})")

    def peak(s):
        return int(GRID[s[0][0]])

    used = set()

    def pick(name, cond):
        for i, s in enumerate(storms):
            if i not in used and cond(s):
                used.add(i)
                return name, s
        raise SystemExit(f"no sampled storm of type {name}")
    picks = [
        pick("Severe surge", lambda s: peak(s) > 330 and s[2]["dp"]["status"][-1] == CLOSED),
        pick("Late rise", lambda s: s[2]["literal"]["flooded"] and not s[2]["dp"]["flooded"]
             and GRID[s[0][HORIZON]] < 285),
        pick("Early scare", lambda s: s[2]["literal"]["order_lead"] is not None and s[2]["literal"]["order_lead"] > 16
             and s[2]["dp"]["order_lead"] is None and peak(s) < 290),
        pick("Near miss, closed anyway", lambda s: s[2]["dp"]["order_lead"] is not None and 285 <= peak(s) <= 298
             and s[2]["tuned"]["order_lead"] is not None),
        pick("Closing fault", lambda s: s[2]["dp"]["order_lead"] is not None and peak(s) > THRESHOLD
             and STATUS.index("delayed") in s[2]["dp"]["status"]),
        pick("Quiet storm", lambda s: max(GRID[s[0]]) < 280),
        pick("Narrow exceedance", lambda s: THRESHOLD < peak(s) <= 306 and s[2]["dp"]["status"][-1] == CLOSED),
        pick("Threshold rule waits too long", lambda s: s[2]["tuned"]["flooded"] and not s[2]["dp"]["flooded"]),
    ]
    return picks


def replay_data(results, picks):
    K = forecast_kernel()
    p_exceed = np.array([exceed_probability(h, K) for h in range(HORIZON + 1)])
    data = {
        "params": {"threshold": THRESHOLD, "f_min": F_MIN, "horizon": HORIZON, "sigma_step": SIGMA_STEP,
                   "warn_hours": WARN_HOURS, "closing_hours": ATTEMPT_HOURS, "q_fault": Q_FAIL,
                   "r_repair": R_REPAIR, "reopen_hours": REOPEN_HOURS, "ship_cost": SHIP_COST,
                   "prior_mean": PRIOR_MEAN, "prior_sd": PRIOR_SD},
        "status_names": list(STATUS),
        "default_cost": FLOOD_COST,
        "costs": {},
        "storms": [],
    }
    for C in REPLAY_COSTS:
        r = results[C]
        data["costs"][str(C)] = {
            "dp_threshold": r["dp_threshold"],
            "rule": r["rule"],
            "metrics": {k: {kk: round(vv, 6) for kk, vv in r[k].items()} for k in ("dp", "tuned", "literal")},
        }
    for name, (path, u, _runs) in picks:
        storm = {"name": name, "forecast": [int(GRID[path[h]]) for h in range(HORIZON, -1, -1)],
                 "p_exceed": [round(float(p_exceed[h, path[h]]), 5) for h in range(HORIZON, -1, -1)],
                 "runs": {}}
        for C in REPLAY_COSTS:
            storm["runs"][str(C)] = {}
            for k, pol in policies(results, C).items():
                run = simulate(pol, path, u, flood_cost=C)
                storm["runs"][str(C)][k] = {"status": run["status"], "order_lead": run["order_lead"],
                                            "flooded": run["flooded"], "cost": round(run["cost"], 6)}
        data["storms"].append(storm)
    REPLAY_DATA.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    return data


def write_results(results):
    lines = [
        "| Flood damage (hours of halted shipping) | Policy | Expected cost | False closures | Missed floods | Hours halted |",
        "| ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for C in FLOOD_COSTS:
        r = results[C]
        rule = r["rule"]
        rows = [("Backward induction", r["dp"]),
                (f"Best threshold rule ({THRESHOLD - rule['margin']} cm, {NOMINAL} to {rule['latest_lead']} h ahead)", r["tuned"]),
                (f"Criterion as soon as reached ({THRESHOLD} cm, {NOMINAL} to {HORIZON} h ahead)", r["literal"])]
        for i, (name, x) in enumerate(rows):
            cost = f"{x['cost']:.1f}"
            lines.append(f"| {C if i == 0 else ''} | {name} | {cost} | {100 * x['p_false']:.1f}% | "
                         f"{100 * x['p_miss']:.2f}% | {x['hours_halted']:.1f} |")
    lines.append("")
    lines.append(f"Exact values over storms whose peak forecast {HORIZON} h ahead is drawn from a normal "
                 f"distribution with mean {PRIOR_MEAN:.0f} cm and standard deviation {PRIOR_SD:.0f} cm above NAP; "
                 f"{100 * results[FLOOD_COST]['dp']['p_exceed']:.1f}% of these storms exceed {THRESHOLD} cm. "
                 "Costs are in hours of halted shipping. A false closure is a closure ordered for a peak that stays "
                 f"at or below {THRESHOLD} cm; a missed flood is a peak above {THRESHOLD} cm with the barrier "
                 "not closed. All model parameters are teaching values except the closing criterion.")
    (ARTIFACTS / "results.md").write_text("\n".join(lines) + "\n")


def figure_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9,
        "legend.fontsize": 7.5, "figure.dpi": 160, "savefig.dpi": 180,
        "svg.fonttype": "none", "svg.hashsalt": "storm-surge-barrier",
        "pdf.fonttype": 42,
    })


def save(fig, name):
    for suffix in ("svg", "pdf", "png"):
        options = {"metadata": {"Date": None}} if suffix == "svg" else {}
        if suffix == "pdf":
            options = {"metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(STATIC / f"{name}.{suffix}", bbox_inches="tight", pad_inches=.04, **options)
    svg_path = STATIC / f"{name}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def figure(results, picks):
    r = results[FLOOD_COST]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
    leads = np.arange(HORIZON + 1)
    lo, hi = 245, 345
    # Optimal closing region, drawn as hourly cells.
    for h, f in enumerate(r["dp_threshold"]):
        if f is not None:
            ax.add_patch(plt.Rectangle((h - .5, f - .5), 1, hi - f + .5, color=BLUE, alpha=.22, lw=0))
    ax.axvspan(-.5, NOMINAL - .5, color=GRAY, alpha=.12, lw=0)
    ax.text(NOMINAL / 2 - .5, lo + 4, "too late\nto close", ha="center", va="bottom", fontsize=7, color="#4a5a63")
    rule = r["rule"]
    f_rule = THRESHOLD - rule["margin"]
    ax.plot([NOMINAL - .5, NOMINAL - .5, rule["latest_lead"] + .5, rule["latest_lead"] + .5], [hi, f_rule, f_rule, hi],
            color=ORANGE, lw=1.4,
            ls="--", label=f"best threshold rule ({f_rule} cm, {NOMINAL} to {rule['latest_lead']} h ahead)")
    ax.axhline(THRESHOLD, color=VERMILION, lw=.9, ls=":", label=f"closing criterion ({THRESHOLD} cm)")
    shown = {"Late rise", "Early scare", "Severe surge"}
    for name, (path, u, runs) in picks:
        if name not in shown:
            continue
        f = GRID[path]
        ax.plot(leads, f, color="#33434c", lw=.8, alpha=.85)
        o = runs["dp"]["order_lead"]
        if o is not None:
            ax.plot([o], [f[o]], "o", ms=4.5, color=BLUE, zorder=4)
        ax.plot([0], [f[0]], "s", ms=3.5, color=VERMILION if f[0] > THRESHOLD else "#33434c", zorder=4)
        ax.text(-.9, f[0], name.lower(), fontsize=6.8, va="center", ha="left", color="#33434c", clip_on=False)
    ax.plot([], [], color=BLUE, alpha=.4, lw=6, label="backward induction: close")
    ax.set_xlim(HORIZON + .5, -.5)
    ax.set_ylim(lo, hi)
    ax.set_xticks([24, 20, 16, 12, 10, 8, 6, 4, 2, 0])
    ax.set_xlabel("hours before the forecast peak")
    ax.set_ylabel("forecast of the peak (cm above NAP)")
    ax.set_title("(a) When to order the closure", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, frameon=False, fontsize=6.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for key, color, label in (("dp", BLUE, "backward induction"), ("tuned", ORANGE, "best threshold rule")):
        xs = [100 * results[C][key]["p_false"] for C in FLOOD_COSTS]
        ys = [100 * results[C][key]["p_miss"] for C in FLOOD_COSTS]
        bx.plot(xs, ys, "-o", color=color, ms=3.5, lw=1.1, label=label)
    for C in (FLOOD_COSTS[0], FLOOD_COSTS[-1]):
        x, y = 100 * results[C]["dp"]["p_false"], 100 * results[C]["dp"]["p_miss"]
        bx.annotate(f"$C={C}$", (x, y), xytext=(-5, -3), textcoords="offset points", fontsize=6.8, color=BLUE,
                    ha="right", va="top")
    lit = results[FLOOD_COST]["literal"]
    bx.plot([100 * lit["p_false"]], [100 * lit["p_miss"]], "D", color=VERMILION, ms=4,
            label=f"close once forecast ≥ {THRESHOLD} cm")
    bx.set_yscale("log")
    bx.set_ylim(.015, 6)
    bx.set_xlim(10, 50)
    bx.set_xlabel("false closures (% of storms)")
    bx.set_ylabel("missed floods (% of storms)")
    bx.set_title("(b) False closures against missed floods", loc="left")
    bx.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=1, frameon=False, fontsize=6.8)
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "stopping_region")


def main():
    STATIC.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    results = solve_all()
    # Sanity checks on the structure of the optimal policy.
    for C, r in results.items():
        th = r["dp_threshold"]
        if any(th[h] is not None for h in range(NOMINAL)):
            raise SystemExit(f"C={C}: closing ordered when it can no longer finish")
    th = results[FLOOD_COST]["dp_threshold"]
    first = min(h for h in range(HORIZON + 1) if th[h] is not None)
    if th[first] >= THRESHOLD:
        raise SystemExit("expected the optimal policy to close below the criterion near the deadline")
    picks = sample_storms(results)
    replay_data(results, picks)
    write_results(results)
    figure_style()
    figure(results, picks)
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "schema_version": 1,
        "model": "finite-horizon optimal stopping; Gaussian martingale forecast on a 1 cm grid; teaching values",
        "params": {"threshold_cm": THRESHOLD, "horizon_h": HORIZON, "sigma_step_cm_per_sqrt_h": SIGMA_STEP,
                   "warn_hours": WARN_HOURS, "closing_hours": ATTEMPT_HOURS, "q_fault": Q_FAIL,
                   "r_repair": R_REPAIR, "reopen_hours": REOPEN_HOURS, "prior_mean_cm": PRIOR_MEAN,
                   "prior_sd_cm": PRIOR_SD,
                   "p_not_closed_by_order_lead": {h: p_not_closed(h) for h in range(NOMINAL, NOMINAL + 5)}},
        "results": {str(C): {"dp": r["dp"], "best_threshold_rule": {**r["tuned"], **r["rule"]},
                             "criterion_as_soon_as_reached": r["literal"], "dp_threshold_cm_by_lead": r["dp_threshold"]}
                    for C, r in results.items()},
        "replay_storms": [name for name, _ in picks],
    }, indent=1) + "\n")


if __name__ == "__main__":
    main()
