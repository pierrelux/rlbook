#!/usr/bin/env python3
"""Rebuild the tamarisk river-network solves, figures, table, and replay data.

Run from the repository root: uv run python scripts/build_tamarisk_river_artifacts.py
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
from tamarisk_river import (  # noqa: E402
    ACTION_COST, ADVANTAGE, BETA, CHI, DEATH, DOWN, EXO_NATIVE, EXO_TAMARISK,
    GAMMA, HEURISTICS, INVADED_COST, NATIVE, NOTHING, RESTORE, SEEDS,
    SLOT_COST, TAMARISK, UP, River, decode, depth, encode, evaluate, parent,
    policy_iteration, rule_policy, simulate_step, stage_cost, value_iteration,
)

STATIC = ROOT / "_static" / "tamarisk_river"
ARTIFACTS = ROOT / "artifacts" / "tamarisk_river"
REPLAY_DATA = ROOT / "interactive" / "tamarisk-river-data.json"
N = 7                       # reaches in the featured river
N_MAX_BUDGET = 8            # largest river solved with the one-treatment budget
N_MAX_FREE = 7              # largest river solved with 3**n joint actions
N_MAX_PI = 7
YEARS = 40
SEEDS_REPLAY = 12
POLICIES = {
    "optimal": "Optimal (value iteration)",
    "upstream": "Restore the most upstream invaded reach",
    "downstream": "Restore the most downstream invaded reach",
    "all": "Restore every invaded reach (no budget)",
    "nothing": "Do nothing",
}
STARTS = {
    "full": ("Every reach invaded", [TAMARISK] * N),
    "upstream": ("Outbreak in two headwater reaches", [NATIVE, NATIVE, NATIVE, TAMARISK, TAMARISK, NATIVE, NATIVE]),
    "outlet": ("Outbreak at the outlet", [TAMARISK, NATIVE, NATIVE, NATIVE, NATIVE, NATIVE, NATIVE]),
}
COLORS = {"empty": "#d9dee2", "native": "#009E73", "tamarisk": "#D55E00"}
CODE_COLOR = [COLORS["empty"], COLORS["native"], COLORS["tamarisk"]]
INK = "#20333f"


def fastest(solve, repeats):
    """Run a solver several times and keep the fastest run (timings are noisy on shared machines)."""
    runs = [solve() for _ in range(repeats)]
    return min(runs, key=lambda r: r.seconds)


def scaling():
    rows = []
    for n in range(1, N_MAX_BUDGET + 1):
        reps = 3 if n <= 6 else 1
        river = River(n)
        vi = fastest(lambda: value_iteration(river), reps)
        row = {"n": n, "states": 3 ** n, "joint_actions": 3 ** n, "budget_actions": len(river.actions),
               "dense_entries_budget": 9 ** n * len(river.actions), "dense_entries_free": 27 ** n,
               "build_s": river.build_seconds, "vi_iterations": vi.iterations, "vi_s": vi.seconds}
        if n <= N_MAX_PI:
            pi = fastest(lambda: policy_iteration(river), reps)
            assert np.abs(pi.v - vi.v).max() < 1e-4, n
            row.update(pi_iterations=pi.iterations, pi_s=pi.seconds)
        if n <= N_MAX_FREE:
            free = River(n, budget=None)
            fvi = fastest(lambda: value_iteration(free), reps)
            assert np.all(fvi.v <= vi.v + 1e-6), n         # more actions can only lower the cost
            row.update(free_vi_s=fvi.seconds + free.build_seconds, free_vi_iterations=fvi.iterations)
        rows.append(row)
        print(f"n={n}: {row}")
    return rows


def solve_featured():
    river, free = River(N), River(N, budget=None)
    vi, pi, fvi = value_iteration(river), policy_iteration(river), value_iteration(free)
    policies = {"optimal": vi.policy}
    values = {"optimal": evaluate(river, vi.policy)}
    for key in ("upstream", "downstream", "nothing", "all"):
        policies[key] = rule_policy(N, HEURISTICS[key])
        values[key] = evaluate(free if key == "all" else river, policies[key])
    values["free_optimal"] = fvi.v
    return river, vi, pi, fvi, policies, values


def rollouts(river, policies, values):
    rng = np.random.default_rng(20120613)
    data = {}
    for start_key, (_, x0) in STARTS.items():
        episodes = []
        for _ in range(SEEDS_REPLAY):
            u = rng.random((YEARS, N, 3))
            ep = {}
            for key, pol in policies.items():
                x, states, acts, costs = np.array(x0, dtype=np.int8), [list(map(int, x0))], [], []
                for t in range(YEARS):
                    a = pol[encode(x)]
                    costs.append(round(stage_cost(x, a), 4))
                    acts.append(list(a))
                    x = simulate_step(x, a, river.K, u[t])
                    states.append(list(map(int, x)))
                ep[key] = {"states": states, "actions": acts, "costs": costs}
            episodes.append(ep)
        s0 = encode(x0)
        data[start_key] = {"episodes": episodes,
                           "expected": {k: round(float(v[s0]), 4) for k, v in values.items() if k in policies}}
    return data


def check_rollouts(roll, values):
    issues = []
    for start_key, (_, x0) in STARTS.items():
        s0 = encode(x0)
        for key in POLICIES:
            disc = [sum(GAMMA ** t * c for t, c in enumerate(ep[key]["costs"])) for ep in roll[start_key]["episodes"]]
            # Twelve episodes are noisy; only flag gross disagreement with the exact value.
            if abs(np.mean(disc) - values[key][s0]) > 0.35 * values[key][s0] + 1.0:
                issues.append(f"{start_key}/{key}: sampled {np.mean(disc):.2f} vs exact {values[key][s0]:.2f}")
    return issues


def figure_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9,
        "legend.fontsize": 8, "figure.dpi": 160, "savefig.dpi": 180,
        "svg.fonttype": "none", "svg.hashsalt": "tamarisk-river",
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


def junctions(n):
    """Upstream junction of each reach; the outlet sits below reach 0."""
    max_d = max(depth(j) for j in range(n))
    pos = {}
    for j in range(n):
        d = depth(j)
        first = 2 ** d - 1
        k = j - first
        pos[j] = ((k + .5) / 2 ** d, d + 1)
    return pos, (.5, 0), max_d


def draw_river(ax, x, a, title, label_outlet=False):
    pos, outlet, max_d = junctions(len(x))
    for j in range(len(x)):
        x1, y1 = pos[j]
        x0, y0 = pos[parent(j)] if j > 0 else outlet
        ax.plot([x0, x1], [y0, y1], color=CODE_COLOR[x[j]], lw=5.5, solid_capstyle="round", zorder=1)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if a[j] != NOTHING:
            ax.scatter([mx], [my], s=210, facecolors="white", edgecolors=INK, linewidths=1.3,
                       linestyle="-" if a[j] == RESTORE else "--", zorder=2)
            ax.text(mx, my, "R" if a[j] == RESTORE else "E", ha="center", va="center", fontsize=6.5,
                    weight="bold", color=INK, zorder=3)
    ax.annotate("", xy=(outlet[0], -.45), xytext=outlet, arrowprops=dict(arrowstyle="-|>", color="#7b858a", lw=.8))
    if label_outlet:
        ax.text(outlet[0] + .06, -.3, "outlet", fontsize=6.5, color="#5e707c", va="center")
    ax.set_title(title, loc="left", fontsize=8)
    ax.set_xlim(-.02, 1.02)
    ax.set_ylim(-.6, max_d + 1.2)
    ax.axis("off")


def figure_replay(roll):
    ep = roll["full"]["episodes"][0]
    years = [0, 1, 2, 4, 8]
    rows = [("optimal", "Optimal policy, one treatment per year"),
            ("upstream", "Heuristic: restore the most upstream invaded reach")]
    fig, axes = plt.subplots(2, len(years), figsize=(7.2, 3.6))
    for r, (key, name) in enumerate(rows):
        run = ep[key]
        for c, t in enumerate(years):
            cum = sum(GAMMA ** k * run["costs"][k] for k in range(t))  # discounted, years before t
            draw_river(axes[r, c], run["states"][t], run["actions"][t] if t < YEARS else [0] * N,
                       f"Year {t}: cost so far {cum:.1f}", label_outlet=(c == 0))
    fig.tight_layout(h_pad=2.6, w_pad=.4, rect=(0, .06, 1, .95))
    for r, (_, name) in enumerate(rows):
        top = axes[r, 0].get_position().y1
        fig.text(.01, top + .05, name, fontsize=9.5, weight="bold", va="bottom")
    handles = [plt.Line2D([], [], color=COLORS[k], lw=5, label=k) for k in ("tamarisk", "native", "empty")]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=8,
                           label="reach treated this year: R restore, E eradicate")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5, handlelength=1.6)
    save(fig, "river")


def figure_scaling(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    n = [r["n"] for r in rows]
    ax1.semilogy(n, [r["states"] for r in rows], "o-", color="#0072B2", label=r"states $3^n$ = joint actions")
    ax1.semilogy(n, [r["budget_actions"] for r in rows], "s-", color="#E69F00", label=r"actions with budget, $2n+1$")
    ax1.semilogy(n, [r["dense_entries_free"] for r in rows], "^--", color="#CC79A7",
                 label=r"entries of $p(j\,|\,s,a)$, no budget")
    last = rows[N_MAX_FREE - 1]
    ax1.annotate(f"{last['dense_entries_free'] * 8 / 1e9:.0f} GB at n = {N_MAX_FREE}",
                 xy=(N_MAX_FREE, last["dense_entries_free"]), xytext=(N_MAX_FREE - 1.2, last["dense_entries_free"] / 3e3),
                 fontsize=7, color="#CC79A7", arrowprops=dict(arrowstyle="-", color="#CC79A7", lw=.6))
    ax1.set_xlabel("reaches $n$")
    ax1.set_title("Size of the model", loc="left")
    ax1.legend(frameon=False, fontsize=7, loc="upper left")
    ax1.grid(alpha=.25, which="major")
    ax2.semilogy(n, [r["vi_s"] + r["build_s"] for r in rows], "o-", color="#0072B2", label="value iteration, budget")
    pn = [r["n"] for r in rows if "pi_s" in r]
    ax2.semilogy(pn, [r["pi_s"] + r["build_s"] for r in rows if "pi_s" in r], "s-", color="#009E73",
                 label="policy iteration, budget")
    fn = [r["n"] for r in rows if "free_vi_s" in r]
    ax2.semilogy(fn, [r["free_vi_s"] for r in rows if "free_vi_s" in r], "^--", color="#CC79A7",
                 label="value iteration, no budget")
    ax2.set_xlabel("reaches $n$")
    ax2.set_title("Seconds to build the model and solve", loc="left")
    ax2.legend(frameon=False, fontsize=7, loc="upper left")
    ax2.grid(alpha=.25, which="major")
    for ax in (ax1, ax2):
        ax.set_xticks(n)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2)
    save(fig, "scaling")


def write_results(values, rows):
    order = ["optimal", "upstream", "downstream", "nothing", "all", "free_optimal"]
    names = {**POLICIES, "free_optimal": "Optimal with no budget"}
    lines = ["| Policy | Every reach invaded | Two headwater reaches | Outlet only | Uniform start |",
             "| :--- | ---: | ---: | ---: | ---: |"]
    for key in order:
        v = values[key]
        cells = [f"{v[encode(x0)]:.2f}" for _, x0 in STARTS.values()] + [f"{v.mean():.2f}"]
        if key == "optimal":
            cells = [f"**{c}**" for c in cells]
        lines.append(f"| {names[key]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Expected discounted cost (γ = {GAMMA}) on the {N}-reach river, computed exactly by solving "
                 "the policy-evaluation equations. The first four policies respect the budget of one treated "
                 "reach per year; the last two do not. *Uniform start* averages over all "
                 f"{3 ** N} starting states.")
    (ARTIFACTS / "results.md").write_text("\n".join(lines) + "\n")


def main():
    STATIC.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rows = scaling()
    river, vi, pi, fvi, policies, values = solve_featured()
    issues = []
    if np.abs(pi.v - vi.v).max() > 1e-4:
        issues.append("policy iteration and value iteration disagree")
    if np.abs(values["optimal"] - pi.v).max() > 1e-4:
        issues.append("greedy value-iteration policy is not near optimal")
    for key in ("upstream", "downstream", "nothing"):
        if np.any(values[key] < pi.v - 1e-6):
            issues.append(f"heuristic {key} beats the optimal policy somewhere")
    if np.any(fvi.v > pi.v + 1e-6) or np.any(values["all"] < fvi.v - 1e-6):
        issues.append("unconstrained optimum inconsistent")
    roll = rollouts(river, policies, values)
    issues += check_rollouts(roll, values)
    if issues:
        raise SystemExit("\n".join(issues))

    pos, outlet, _ = junctions(N)
    replay = {
        "n": N, "years": YEARS, "gamma": GAMMA,
        "parents": [parent(j) for j in range(N)],
        "junctions": [list(pos[j]) for j in range(N)], "outlet": list(outlet),
        "policies": POLICIES,
        "starts": {k: {"name": name, "state": x0, **roll[k]} for k, (name, x0) in STARTS.items()},
        "scaling": [{k: r[k] for k in ("n", "states", "budget_actions", "dense_entries_budget", "dense_entries_free")}
                    for r in rows],
    }
    REPLAY_DATA.write_text(json.dumps(replay, separators=(",", ":"), allow_nan=False) + "\n")
    write_results(values, rows)
    figure_style()
    figure_replay(roll)
    figure_scaling(rows)

    X = decode(N)
    full = encode([TAMARISK] * N)
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "schema_version": 1,
        "model": f"{N}-reach binary river tree, one slot per reach, Taleghan et al. 2015 parameters",
        "parameters": {"death": DEATH, "seeds_per_plant": SEEDS, "dispersal_down": DOWN, "dispersal_up": UP,
                       "eradicate_success": CHI, "restore_planting_success": BETA,
                       "action_cost": list(ACTION_COST), "invaded_reach_cost": INVADED_COST,
                       "tamarisk_slot_cost": SLOT_COST, "exogenous_tamarisk_mean": EXO_TAMARISK,
                       "exogenous_native_mean": EXO_NATIVE, "competitive_advantage": ADVANTAGE,
                       "gamma": GAMMA, "budget": 1},
        "featured": {
            "value_iteration_iterations": vi.iterations, "policy_iteration_iterations": pi.iterations,
            "max_abs_vi_minus_pi": float(np.abs(vi.v - pi.v).max()),
            "states_where_policy_treats": int(sum(any(a) for a in vi.policy)),
            "optimal_action_all_invaded": list(vi.policy[full]),
            "expected_cost": {k: {sk: float(v[encode(x0)]) for sk, (_, x0) in STARTS.items()} | {"uniform": float(v.mean())}
                              for k, v in values.items()},
            "n_states": int(len(X)),
        },
        "scaling": rows,
    }, indent=1) + "\n")


if __name__ == "__main__":
    main()
