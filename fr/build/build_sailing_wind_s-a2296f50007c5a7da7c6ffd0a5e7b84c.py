#!/usr/bin/env python3
"""Rebuild the sailing-in-wind-shifts solves, teaching figure, table, and replay data.

Run from the repository root: uv run python scripts/build_sailing_wind_shifts_artifacts.py
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
from sailing_wind_shifts import (  # noqa: E402
    DIRS, DX, DY, LEG_TIME, POINT_OF_SAIL, SCENARIO_LABELS, SCENARIOS, STARBOARD, PORT,
    TACK_DELAY, TACKS, Lake, bellman, build, certainty_equivalent_policy, evaluate,
    naive_policy, policy_iteration, simulate, start_value, value_iteration, wind_matrix,
)

STATIC = ROOT / "_static" / "sailing_wind_shifts"
ARTIFACTS = ROOT / "artifacts" / "sailing_wind_shifts"
REPLAY_DATA = ROOT / "interactive" / "sailing-wind-shifts-data.json"
FEATURED = "oscillating"
LAKE = Lake()
N_RACES = 8                 # races recorded for the replay, per scenario
N_MC = 4000                 # Monte Carlo races per scenario for the histogram and the check
SEED = 20061
VI_TOL = 1e-9
VI_WINDS = (7, 0, 1)        # NW, N, NE: value-iteration frames recorded for these winds
NORTH = 0
BLUE, ORANGE, GREEN, GRAY, RED = "#0072B2", "#E69F00", "#009E73", "#7b858a", "#b2182b"


def solve(kind):
    model = build(LAKE, wind_matrix(kind))
    vi = value_iteration(model, tol=VI_TOL, record=set(range(0, 200)))
    naive = naive_policy(model)
    v_naive = evaluate(model, naive)
    ce = certainty_equivalent_policy(LAKE)
    v_ce = evaluate(model, ce)
    v_opt = evaluate(model, vi.policy)
    pi_pol, pi_v, pi_values, _ = policy_iteration(model, naive)
    return dict(model=model, vi=vi, naive=naive, v_naive=v_naive, ce=ce, v_ce=v_ce,
                v_opt=v_opt, pi_policy=pi_pol, pi_v=pi_v, pi_values=pi_values)


def races(sol, rng):
    """Paired races: the optimal boat and the layline sailor see the same wind draws, leg by leg."""
    model = sol["model"]
    _, t_opt = start_value(sol["vi"].v, LAKE, NORTH)
    _, t_naive = start_value(sol["v_naive"], LAKE, NORTH)
    recorded, times = [], np.zeros((N_MC, 2))
    for r in range(N_MC):
        u = rng.random(400)
        a = simulate(model, sol["vi"].policy, u, NORTH, t_opt)
        b = simulate(model, sol["naive"], u, NORTH, t_naive)
        times[r] = a[-1][4], b[-1][4]
        if r < N_RACES:
            recorded.append({"optimal": [[x, y, w, t, round(c, 3)] for x, y, w, t, c in a],
                             "naive": [[x, y, w, t, round(c, 3)] for x, y, w, t, c in b]})
    return recorded, times


def tack_on_header_share(sol):
    """Share of open-water states with a 45-degree shift where the optimal boat ends on the lifted tack.

    With the wind from NE, starboard tack points north (lifted) and port tack points east (headed);
    with the wind from NW it is the reverse. Open water: below the mark and at least three cells
    inside both mean-wind laylines.
    """
    pi, model = sol["vi"].policy, sol["model"]
    mx, my = LAKE.mark
    total = lifted = 0
    for w, lifted_tack in ((1, STARBOARD), (7, PORT)):
        for t in range(2):
            for y in range(LAKE.n):
                for x in range(LAKE.n):
                    if my - y < 4 or abs(x - mx) > my - y - 3:
                        continue
                    total += 1
                    lifted += model.next_tack[pi[t, w, y, x], t, w] == lifted_tack
    return lifted / total, total


def tack_boundary(policy, model, wind, tack):
    """For each row below the mark, the x at which a boat on `tack` first chooses to tack (or None)."""
    out = []
    for y in range(LAKE.mark[1]):
        xs = [x for x in range(LAKE.n) if model.next_tack[policy[tack, wind, y, x], tack, wind] != tack]
        out.append(xs)
    return out


def max_legs(sol):
    """Longest number of legs to the mark under the optimal policy over every possible wind sequence.

    Raises if the policy's transition graph has a cycle (then no such bound exists).
    """
    model, pi = sol["model"], sol["vi"].policy
    n, mark = LAKE.n, LAKE.mark
    memo, active = {}, set()

    def legs(s):
        t, w, y, x = s
        if (x, y) == mark:
            return 0
        if s in memo:
            return memo[s]
        if s in active:
            raise SystemExit("optimal policy graph has a cycle")
        active.add(s)
        h = pi[s]
        nt = model.next_tack[h, t, w]
        out = 1 + max(legs((nt, u, y + DY[h], x + DX[h])) for u in np.nonzero(model.P[w])[0])
        active.discard(s)
        memo[s] = out
        return out

    sys.setrecursionlimit(20000)
    return max(legs((t, w, y, x)) for t in range(2) for w in range(8) for y in range(n) for x in range(n))


def check(sols, mc):
    issues = []
    steady = sols["steady"]
    rows = LAKE.mark[1] - LAKE.start[1]
    exact = rows * LEG_TIME[1] * np.sqrt(2) + TACK_DELAY
    if abs(start_value(steady["vi"].v, LAKE)[0] - exact) > 1e-9:
        issues.append("steady wind: optimal time differs from the one-tack beat")
    if abs(start_value(steady["v_naive"], LAKE)[0] - exact) > 1e-9:
        issues.append("steady wind: the layline sailor is not optimal")
    for kind, s in sols.items():
        v = s["vi"].v
        if np.max(np.abs(bellman(s["model"], v)[0] - v)) > 1e-9:
            issues.append(f"{kind}: v is not a fixed point of the Bellman operator")
        if np.max(np.abs(s["v_opt"] - v)) > 1e-6:
            issues.append(f"{kind}: evaluating the greedy policy does not give v*")
        if np.max(np.abs(s["pi_v"] - v)) > 1e-6:
            issues.append(f"{kind}: policy iteration and value iteration disagree")
        for name in ("v_naive", "v_ce"):
            if np.any(s[name] < v - 1e-9):
                issues.append(f"{kind}: {name} beats the optimal value somewhere")
        times = mc[kind]
        for j, key in enumerate(("vi", "naive")):
            exact_v = start_value(s["vi"].v if key == "vi" else s["v_naive"], LAKE, NORTH)[0]
            se = times[:, j].std(ddof=1) / np.sqrt(len(times))
            if abs(times[:, j].mean() - exact_v) > 4 * se + 1e-9:
                issues.append(f"{kind}: Monte Carlo mean for {key} is {times[:, j].mean():.2f}, exact {exact_v:.2f}")
    if issues:
        raise SystemExit("\n".join(issues))


def summary(sols, mc):
    out = {}
    for kind, s in sols.items():
        times = mc[kind]
        d = times[:, 1] - times[:, 0]
        out[kind] = {
            "label": SCENARIO_LABELS[kind],
            "wind_matrix": wind_matrix(kind).tolist(),
            "optimal_minutes": round(start_value(s["vi"].v, LAKE, NORTH)[0], 4),
            "optimal_start_tack": TACKS[start_value(s["vi"].v, LAKE, NORTH)[1]],
            "certainty_equivalent_minutes": round(start_value(s["v_ce"], LAKE, NORTH)[0], 4),
            "layline_minutes": round(start_value(s["v_naive"], LAKE, NORTH)[0], 4),
            "value_iteration_sweeps": len(s["vi"].residuals),
            "value_iteration_final_residual": s["vi"].residuals[-1],
            "policy_iteration_evaluations": len(s["pi_values"]),
            "policy_iteration_improvements": len(s["pi_values"]) - 1,
            "policy_iteration_errors": [float(np.max(np.abs(v - s["vi"].v))) for v in s["pi_values"]],
            "monte_carlo_races": len(times),
            "monte_carlo_mean_optimal": round(float(times[:, 0].mean()), 3),
            "monte_carlo_mean_layline": round(float(times[:, 1].mean()), 3),
            "share_races_optimal_faster": round(float(np.mean(d > 1e-9)), 4),
            "share_races_tied": round(float(np.mean(np.abs(d) <= 1e-9)), 4),
            "max_legs_optimal_any_wind": max_legs(s),
            "first_sweep_with_error_below_1_minute": next(
                k for k, it in enumerate(s["vi"].iterates) if np.max(np.abs(it - s["vi"].v)) < 1),
        }
        if kind != "steady":
            share, count = tack_on_header_share(s)
            out[kind]["lifted_tack_share_after_shift"] = round(float(share), 4)
            out[kind]["lifted_tack_states"] = count
    return out


def write_results(stats):
    lines = [
        "| Wind | Optimal (value iteration) | Certainty-equivalent | Layline sailor | Value-iteration sweeps | Policy-iteration improvements |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind in SCENARIOS:
        s = stats[kind]
        lines.append(f"| {s['label']} | {s['optimal_minutes']:.1f} | {s['certainty_equivalent_minutes']:.1f} | "
                     f"{s['layline_minutes']:.1f} | {s['value_iteration_sweeps']} | {s['policy_iteration_improvements']} |")
    lines.append("")
    lines.append("Expected time in minutes from the start to the windward mark, 16 rows upwind on a 21 by 21 lake, "
                 "with the wind from the north at the start. *Certainty-equivalent* plans each leg as if the wind "
                 "seen now will never change; the *layline sailor* tacks only on the laylines of the mean wind. "
                 "Value iteration starts from zero and stops when the sup-norm residual is below "
                 f"{VI_TOL:g} minutes; policy iteration starts from the layline sailor.")
    (ARTIFACTS / "results.md").write_text("\n".join(lines) + "\n")


def rnd(a, k=1):
    return [round(float(x), k) for x in np.asarray(a).ravel()]


def replay_data(sols, recorded, stats, mc):
    data = {"n": LAKE.n, "start": list(LAKE.start), "mark": list(LAKE.mark), "dirs": list(DIRS),
            "tacks": list(TACKS), "leg_time": [None if not np.isfinite(x) else x for x in LEG_TIME],
            "tack_delay": TACK_DELAY, "featured": FEATURED, "scenarios": {}}
    for kind, s in sols.items():
        times = mc[kind]
        lo, hi = float(np.floor(times.min() / 5) * 5), float(np.ceil(times.max() / 5) * 5)
        bins = np.arange(lo, hi + 5, 2.5)
        entry = {
            "label": SCENARIO_LABELS[kind],
            "P": wind_matrix(kind).tolist(),
            # Arrays flattened in [tack][wind][y][x] order.
            "v": rnd(s["vi"].v),
            "policy": [int(h) for h in s["vi"].policy.ravel()],
            "naive": [int(h) for h in s["naive"].ravel()],
            "stats": stats[kind],
            "races": recorded[kind],
            "hist": {"edges": [float(b) for b in bins],
                     "optimal": np.histogram(times[:, 0], bins)[0].tolist(),
                     "naive": np.histogram(times[:, 1], bins)[0].tolist()},
            "residuals": [float(r) for r in s["vi"].residuals],
            "vi_errors": [float(np.max(np.abs(it - s["vi"].v))) for it in s["vi"].iterates],
            "pi_errors": stats[kind]["policy_iteration_errors"],
        }
        if kind == FEATURED:
            frames = np.stack(s["vi"].iterates)[:, :, list(VI_WINDS)]
            entry["vi_winds"] = list(VI_WINDS)
            entry["vi_frames"] = [[int(round(x)) for x in f.ravel()] for f in frames]
        data["scenarios"][kind] = entry
    REPLAY_DATA.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")


def figure_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9,
        "legend.fontsize": 8, "figure.dpi": 160, "savefig.dpi": 180,
        "svg.fonttype": "none", "svg.hashsalt": "sailing-wind-shifts",
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


def lake_panel(ax, sol, wind, tack, title):
    n, (mx, my), (sx, sy) = LAKE.n, LAKE.mark, LAKE.start
    v, pi, model = sol["vi"].v, sol["vi"].policy, sol["model"]
    ax.set_facecolor("#f4f8fa")
    levels = np.arange(0, 140, 10)
    cs = ax.contour(np.arange(n), np.arange(n), v[tack, wind], levels=levels, colors="#9fb3bf", linewidths=.6)
    ax.clabel(cs, levels[1::2], fontsize=6, fmt="%d", inline_spacing=1)
    for y in range(n):
        for x in range(n):
            if (x, y) == (mx, my) or (x + y) % 2:
                continue
            h = pi[tack, wind, y, x]
            tacks = model.next_tack[h, tack, wind] != tack
            color = RED if tacks else BLUE
            dx, dy = DX[h], DY[h]
            norm = np.hypot(dx, dy)
            ax.annotate("", xy=(x + .42 * dx / norm, y + .42 * dy / norm), xytext=(x - .42 * dx / norm, y - .42 * dy / norm),
                        arrowprops=dict(arrowstyle="-|>,head_length=.25,head_width=.12", color=color, lw=.7,
                                        shrinkA=0, shrinkB=0))
    # Mean-wind laylines through the mark.
    ax.plot([mx, mx - my], [my, 0], color=GRAY, lw=.8, ls="--")
    ax.plot([mx, mx + my], [my, 0], color=GRAY, lw=.8, ls="--")
    ax.scatter([mx], [my], s=50, marker="^", color="#20333f", zorder=5)
    ax.scatter([sx], [sy], s=22, marker="s", color="#20333f", zorder=5)
    # Wind arrow (blowing from `wind`).
    ang = np.deg2rad(45 * wind)
    ux, uy = np.sin(ang), np.cos(ang)
    cx, cy = .955, 1.045
    ax.annotate("", xy=(cx - .035 * ux, cy - .035 * uy), xytext=(cx + .035 * ux, cy + .035 * uy),
                xycoords="axes fraction", textcoords="axes fraction", annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>,head_length=.4,head_width=.25", color="#20333f", lw=1.4))
    ax.set_xlim(-.7, n - .3)
    ax.set_ylim(-.7, n - .3)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c8d2d7")
    ax.set_title(title, loc="left")


def figure(sols, recorded, stats):
    sol = sols[FEATURED]
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(2, 2, hspace=.24, wspace=.16)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0]); d = fig.add_subplot(gs[1, 1])
    lake_panel(a, sol, NORTH, STARBOARD, "(a) Wind from N, starboard tack")
    lake_panel(b, sol, 7, STARBOARD, "(b) Wind from NW, starboard tack")

    # (c) one recorded race.
    n, (mx, my) = LAKE.n, LAKE.mark
    race = recorded[FEATURED][0]
    c.set_facecolor("#f4f8fa")
    for key, color, name in (("naive", ORANGE, "layline sailor"), ("optimal", BLUE, "optimal policy")):
        p = np.array(race[key])
        c.plot(p[:, 0], p[:, 1], color=color, lw=1.6, marker="o", ms=2.2,
               label=f"{name}: {p[-1, 4]:.1f} min")
        tacks = np.nonzero(np.diff(p[:, 3]))[0] + 1
        c.scatter(p[tacks, 0], p[tacks, 1], s=26, facecolors="none", edgecolors=color, lw=1, zorder=4)
    c.plot([mx, mx - my], [my, 0], color=GRAY, lw=.8, ls="--")
    c.plot([mx, mx + my], [my, 0], color=GRAY, lw=.8, ls="--")
    c.scatter([mx], [my], s=50, marker="^", color="#20333f", zorder=5)
    c.scatter([LAKE.start[0]], [LAKE.start[1]], s=22, marker="s", color="#20333f", zorder=5)
    winds = [w for _, _, w, _, _ in race["optimal"]][:-1]
    c.text(.03, .97, "wind each leg: " + " ".join(DIRS[w] for w in winds[:12]) + (" ..." if len(winds) > 12 else ""),
           transform=c.transAxes, fontsize=6.2, va="top", family="monospace")
    c.set_xlim(-.7, n - .3); c.set_ylim(-.7, n - .3); c.set_aspect("equal")
    c.set_xticks([]); c.set_yticks([])
    for sp in c.spines.values():
        sp.set_color("#c8d2d7")
    c.legend(loc="upper center", bbox_to_anchor=(.5, -.01), ncol=2, frameon=False, fontsize=7)
    c.set_title("(c) One race, shared wind draws", loc="left")

    # (d) convergence.
    vi_err = [np.max(np.abs(it - sol["vi"].v)) for it in sol["vi"].iterates]
    res = sol["vi"].residuals
    d.semilogy(range(len(vi_err)), np.maximum(vi_err, 1e-9), color=BLUE, lw=1.4, label=r"VI error $\|v_n-v^*\|_\infty$")
    d.semilogy(range(1, len(res) + 1), np.maximum(res, 1e-9), color=BLUE, lw=1, ls=":", label=r"VI residual $\|v_{n}-v_{n-1}\|_\infty$")
    pe = stats[FEATURED]["policy_iteration_errors"]
    d.semilogy(range(len(pe)), np.maximum(pe, 1e-9), color=GREEN, lw=1.4, marker="o", ms=4,
               label=r"PI error $\|v^{\pi_k}-v^*\|_\infty$")
    d.set_xlabel("sweep $n$ (VI) or improvement step $k$ (PI)")
    d.set_ylabel("minutes")
    d.set_ylim(5e-10, 400)
    d.text(len(res) - 2.5, 3e-9, "exactly 0", ha="right", va="bottom", fontsize=7, color=BLUE)
    d.grid(True, which="major", color="#e3e9ec", lw=.6)
    d.legend(loc="lower left", fontsize=7, frameon=True, framealpha=.9)
    d.set_title("(d) Value and policy iteration", loc="left")
    d.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .505, "Blue arrow: keep the tack.   Red arrow: change tack on this leg.   "
             "Dashed: laylines of the mean wind.   Contours: expected minutes to the mark.",
             ha="center", fontsize=7.2, color="#33434c")
    save(fig, "sailing")


def main():
    STATIC.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    sols = {kind: solve(kind) for kind in SCENARIOS}
    recorded, mc = {}, {}
    for i, kind in enumerate(SCENARIOS):
        recorded[kind], mc[kind] = races(sols[kind], np.random.default_rng(SEED + i))
    check(sols, mc)
    stats = summary(sols, mc)
    for kind in SCENARIOS:
        # The recorded race must end at the time the simulator reports and use only allowed legs.
        for race in recorded[kind]:
            for key in ("optimal", "naive"):
                assert tuple(race[key][-1][:2]) == LAKE.mark
    replay_data(sols, recorded, stats, mc)
    write_results(stats)
    figure_style()
    figure(sols, recorded, stats)
    sol = sols[FEATURED]
    boundary = {f"{TACKS[t]}_{DIRS[w]}": tack_boundary(sol["vi"].policy, sol["model"], w, t)
                for w in (7, 0, 1) for t in range(2)}
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "schema_version": 1,
        "model": "Vanderbei sailing problem as an undiscounted stochastic shortest-path MDP",
        "lake": {"n": LAKE.n, "start": LAKE.start, "mark": LAKE.mark},
        "leg_time_minutes": {POINT_OF_SAIL[k]: (None if not np.isfinite(LEG_TIME[k]) else float(LEG_TIME[k]))
                             for k in range(5)},
        "diagonal_factor": "sqrt(2)",
        "tack_delay_minutes": TACK_DELAY,
        "start_wind": DIRS[NORTH],
        "scenarios": stats,
        "featured_tack_states_by_row": boundary,
    }, indent=1) + "\n")


if __name__ == "__main__":
    main()
