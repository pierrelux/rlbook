#!/usr/bin/env python3
"""Rebuild the bridge-deck maintenance MDP, its figure, table, and replay data.

Run from the repository root:

    uv run python scripts/build_bridge_deck_maintenance_artifacts.py [--mustem-mat PATH]

Source of the do-nothing transition counts, in order of preference:

1. Raw NBI delimited files under ``data/nbi/`` (see ``load_raw_nbi``). The
   counts are re-estimated from them and written to
   ``artifacts/bridge_deck_maintenance/deck_transition_counts.csv``.
2. ``--mustem-mat PATH``: ``Deck/Samples/Feature_Matrix_fold_1.mat`` from
   https://github.com/HengLiu2021/MUSTEM-Model-using-NBI (needs h5py). It holds
   27 annual NBI deck ratings, 1993 to 2019, of 88,596 decks whose rating never
   went up (Liu, Nehme and Lu 2023). The counts are re-estimated from it.
3. Otherwise the counts already shipped in ``deck_transition_counts.csv``,
   which were computed from source 2.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import bridge_deck_maintenance as bdm  # noqa: E402
from bridge_deck_maintenance import (  # noqa: E402
    ACTIONS, ACTION_COST, ACTION_LABELS, COUNT_RATINGS, FAILURE_PROB, GAMMA,
    N_STATES, RATING_LABELS, RATINGS, DeckMDP, build_mdp, count_transitions,
    do_nothing_matrix, evaluate, policy_iteration, reactive_policy,
    schedule_start, scheduled_action, simulate, transition_tensor,
    value_iteration,
)

STATIC = ROOT / "_static" / "bridge_deck_maintenance"
ARTIFACTS = ROOT / "artifacts" / "bridge_deck_maintenance"
COUNTS_CSV = ARTIFACTS / "deck_transition_counts.csv"
RAW_DIR = ROOT / "data" / "nbi"
REPLAY_DATA = ROOT / "interactive" / "bridge-deck-maintenance-data.json"

MUSTEM_URL = "https://github.com/HengLiu2021/MUSTEM-Model-using-NBI"
RISK_MULTIPLIERS = (2, 5, 10, 20, 40)   # failure cost in multiples of the replacement cost
BASE_MULTIPLIER = 10                    # Saifullah et al.
REPLAY_MULTIPLIERS = (5, 10, 40)
MAX_PERIOD = 40
N_BRIDGES = 12
N_YEARS = 50
SEED = 58                               # NBI item 58 is the deck rating

ACTION_COLORS = ["#d9e1e5", "#56B4E9", "#E69F00", "#CC79A7"]
RATING_COLORS = ["#1a7a4c", "#5aae61", "#a6d96a", "#fee08b", "#f4a582", "#b2182b"]


# ------------------------------------------------------------------ data

def _year_of(path: Path) -> int | None:
    for part in [path.name] + [p.name for p in path.parents]:
        m = re.search(r"(19[89]\d|20[0-4]\d)", part)
        if m:
            return int(m.group(1))
    m = re.search(r"(\d\d)\.(txt|csv)$", path.name, re.I)
    if m:
        yy = int(m.group(1))
        return 1900 + yy if yy >= 80 else 2000 + yy
    return None


def load_raw_nbi(directory: Path):
    """Read NBI delimited files (one per state or all states, one per year).

    Expected files: the comma-delimited NBI downloads from
    https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm, e.g. ``data/nbi/NE19.txt``
    or ``data/nbi/2019/2019HwyBridgesDelimitedAllStates.txt``. The year is read
    from the file or folder name. Required columns (header row):
    STATE_CODE_001, STRUCTURE_NUMBER_008, DECK_COND_058; optional:
    YEAR_RECONSTRUCTED_106. Returns (histories, work_flags, years).
    """
    records: dict[tuple[str, str], dict[int, tuple[int | None, str]]] = {}
    years = set()
    for path in sorted(p for p in directory.rglob("*") if p.suffix.lower() in (".txt", ".csv")):
        year = _year_of(path)
        if year is None:
            print(f"skipping {path}: no year in its name", file=sys.stderr)
            continue
        with open(path, newline="", encoding="latin-1") as fh:
            reader = csv.DictReader(fh, quotechar="'")
            if not reader.fieldnames or "DECK_COND_058" not in reader.fieldnames:
                print(f"skipping {path}: no DECK_COND_058 column", file=sys.stderr)
                continue
            years.add(year)
            for row in reader:
                key = (row["STATE_CODE_001"].strip(), row["STRUCTURE_NUMBER_008"].strip())
                raw = row["DECK_COND_058"].strip()
                rating = int(raw) if raw.isdigit() else None
                records.setdefault(key, {})[year] = (rating, row.get("YEAR_RECONSTRUCTED_106", "").strip())
    if not years:
        return None
    span = list(range(min(years), max(years) + 1))
    histories, flags = [], []
    for by_year in records.values():
        h = [by_year.get(y, (None, ""))[0] for y in span]
        recon = [by_year.get(y, (None, ""))[1] for y in span]
        histories.append(h)
        flags.append([bool(recon[t] and recon[t + 1] and recon[t] != recon[t + 1]) for t in range(len(span) - 1)])
    return histories, flags, span


def load_mustem(path: Path):
    import h5py  # only needed for this source

    with h5py.File(path, "r") as f:
        # MATLAB stores (1, features, years, bridges); h5py sees it reversed.
        deck = np.asarray(f["Feature_Matrix"][:, :, -1, 0], dtype=int)
    return deck


def write_counts(C: np.ndarray, source: str):
    with open(COUNTS_CSV, "w", newline="") as fh:
        fh.write(f"# Year-to-year NBI deck rating (item 58) transitions of decks with no work. Source: {source}\n")
        fh.write("# Rows: rating this year; columns: rating next year. Rating 3 pools ratings 0-3.\n")
        w = csv.writer(fh)
        w.writerow(["from\\to"] + list(COUNT_RATINGS))
        for r, row in zip(COUNT_RATINGS, C):
            w.writerow([r] + [int(x) for x in row])


def read_counts():
    rows = [line for line in COUNTS_CSV.read_text().splitlines() if not line.startswith("#")]
    source = COUNTS_CSV.read_text().splitlines()[0].split("Source: ", 1)[1]
    table = list(csv.reader(rows))
    return np.array([[int(x) for x in r[1:]] for r in table[1:]]), source


def get_counts(mustem: Path | None):
    """Return (counts, initial rating counts or None, provenance dict)."""
    if RAW_DIR.exists():
        raw = load_raw_nbi(RAW_DIR)
        if raw is not None:
            histories, flags, span = raw
            C = count_transitions(histories, flags)
            source = f"raw NBI delimited files in data/nbi/, {span[0]}-{span[-1]}"
            write_counts(C, source)
            first = [h[0] for h in histories if h[0] is not None]
            return C, np.array(first), {"kind": "raw NBI files", "source": source, "bridges": len(histories)}
    if mustem is not None:
        deck = load_mustem(mustem)
        C = count_transitions(deck.tolist())
        source = (f"GitHub mirror {MUSTEM_URL}, Deck/Samples/Feature_Matrix_fold_1.mat: "
                  f"{deck.shape[0]} decks x {deck.shape[1]} annual NBI ratings, 1993-2019 (Liu, Nehme and Lu 2023)")
        write_counts(C, source)
        np.savetxt(ARTIFACTS / "initial_rating_counts.csv",
                   np.array([[r, int((deck[:, 0] == r).sum())] for r in COUNT_RATINGS]), fmt="%d",
                   delimiter=",", header="rating,decks in 1993", comments="")
        return C, deck[:, 0], {"kind": "GitHub mirror of NBI records", "source": source, "bridges": int(deck.shape[0])}
    C, source = read_counts()
    kind = "raw NBI files" if source.startswith("raw NBI") else "GitHub mirror of NBI records"
    return C, None, {"kind": kind, "source": source, "bridges": None}


def initial_distribution(first_ratings):
    if first_ratings is None:
        table = np.loadtxt(ARTIFACTS / "initial_rating_counts.csv", delimiter=",", skiprows=1)
        counts = {int(r): n for r, n in table}
    else:
        first_ratings = np.asarray(first_ratings)
        counts = {r: int((np.maximum(first_ratings, 3) == r).sum()) for r in COUNT_RATINGS}
        if not (ARTIFACTS / "initial_rating_counts.csv").exists() or RAW_DIR.exists():
            np.savetxt(ARTIFACTS / "initial_rating_counts.csv",
                       np.array([[r, counts[r]] for r in COUNT_RATINGS]), fmt="%d",
                       delimiter=",", header="rating,decks in first year", comments="")
    mu = np.array([counts[r] for r in RATINGS[:-1]] + [counts[4] + counts[3]], float)
    return mu / mu.sum()


# ------------------------------------------------------------------ solves

def mdp_with_risk(P0, multiplier, gamma=GAMMA):
    c = np.zeros((N_STATES, len(ACTIONS)))
    for a, name in enumerate(ACTIONS):
        for s in range(N_STATES):
            c[s, a] = ACTION_COST[a] + FAILURE_PROB[bdm.improved_state(s, name)] * multiplier * ACTION_COST[3]
    return DeckMDP(transition_tensor(P0), c, gamma)


def best_schedule(mdp, mu):
    vals = {k: float(mu @ schedule_start(mdp, k)) for k in range(1, MAX_PERIOD + 1)}
    k = min(vals, key=vals.get)
    return k, vals


def solve(P0, mu, multiplier):
    mdp = mdp_with_risk(P0, multiplier)
    v_vi, pol_vi, n_vi, hist = value_iteration(mdp, tol=1e-6)
    v, pol, trace = policy_iteration(mdp)
    k, sched = best_schedule(mdp, mu)
    v_react = evaluate(mdp, reactive_policy())
    return {
        "mdp": mdp, "v": v, "policy": pol, "pi_trace": trace,
        "vi": {"v": v_vi, "policy": pol_vi, "iterations": n_vi, "history": hist},
        "period": k, "schedule_values": sched,
        "v_schedule": schedule_start(mdp, k), "v_reactive": v_react,
        "cost": {"optimal": float(mu @ v), "schedule": sched[k], "reactive": float(mu @ v_react)},
    }


# ------------------------------------------------------------------ replay

def replay(sol, mu, multiplier):
    rng = np.random.default_rng(SEED)
    s0 = rng.choice(N_STATES, size=N_BRIDGES, p=mu)
    U = rng.random((N_YEARS, N_BRIDGES))
    mdp, pol, k = sol["mdp"], sol["policy"], sol["period"]
    runs = {}
    choosers = {
        "optimal": lambda t, s: int(pol[s]),
        "reactive": lambda t, s: int(reactive_policy()[s]),
        "schedule": lambda t, s: scheduled_action(k, t, s),
    }
    disc = mdp.gamma ** np.arange(N_YEARS)
    for name, choose in choosers.items():
        states, actions, costs = simulate(mdp, choose, s0, U)
        runs[name] = {
            "states": states.tolist(),
            "actions": actions.tolist(),
            "cum": np.round(np.cumsum(disc[:, None] * costs, axis=0).mean(axis=1), 2).tolist(),
        }
    exp = {"optimal": float(np.mean(sol["v"][s0])),
           "reactive": float(np.mean(sol["v_reactive"][s0])),
           "schedule": float(np.mean(sol["v_schedule"][s0]))}
    return {
        "multiplier": multiplier,
        "policy": [int(a) for a in pol],
        "period": k,
        "expected": {n: round(x, 2) for n, x in exp.items()},
        "start": s0.tolist(),
        "runs": runs,
    }


def write_replay(P0, mu, sols):
    data = {
        "ratings": list(RATING_LABELS),
        "actions": list(ACTION_LABELS),
        "action_cost": ACTION_COST.tolist(),
        "gamma": GAMMA,
        "years": N_YEARS,
        "bridges": N_BRIDGES,
        "base": BASE_MULTIPLIER,
        "scenarios": {str(m): replay(sols[m], mu, m) for m in REPLAY_MULTIPLIERS},
    }
    REPLAY_DATA.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    return data


# ------------------------------------------------------------------ outputs

def write_results(sols, provenance):
    base = sols[BASE_MULTIPLIER]
    lines = [
        "| Failure cost (× replacement) | Optimal action at ratings 9, 8, 7, 6, 5, ≤4 | Optimal | Best schedule | Reactive |",
        "| ---: | :--- | ---: | ---: | ---: |",
    ]
    for m in RISK_MULTIPLIERS:
        s = sols[m]
        acts = ", ".join(ACTIONS[a] for a in s["policy"])
        c = s["cost"]
        sched = f"{c['schedule']:,.0f} (every {s['period']} yr)"
        row = f"| {m} | {acts} | **{c['optimal']:,.0f}** | {sched} | {c['reactive']:,.0f} |"
        if m == BASE_MULTIPLIER:
            row = row.replace(f"| {m} |", f"| {m} (base) |", 1)
        lines.append(row)
    lines.append("")
    lines.append(
        "Expected discounted cost in USD per m² of deck (γ = 0.97), averaged over the first-year NBI ratings "
        "of the decks in the sample. *Schedule* is a minor repair every k years, with the best k, plus a "
        "replacement whenever the deck is poor; *reactive* does nothing until the deck is poor and then replaces it. "
        f"Do-nothing transitions: {provenance['kind']}.")
    (ARTIFACTS / "results.md").write_text("\n".join(lines) + "\n")


def figure_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9,
        "legend.fontsize": 8, "figure.dpi": 160, "savefig.dpi": 180,
        "svg.fonttype": "none", "svg.hashsalt": "bridge-deck-maintenance",
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


def figure(P0, sols):
    base = sols[BASE_MULTIPLIER]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.0, 1.05, 1.2]})

    # (a) do-nothing matrix
    ax = axes[0]
    ax.imshow(P0, cmap="Blues", vmin=0, vmax=1)
    for i in range(N_STATES):
        for j in range(N_STATES):
            if P0[i, j] >= 0.005:
                ax.text(j, i, f"{P0[i, j]:.2f}".replace("0.", ".") if P0[i, j] < 1 else "1",
                        ha="center", va="center", fontsize=6.4, color="white" if P0[i, j] > .6 else "#20333f")
    ax.set_xticks(range(N_STATES), RATING_LABELS)
    ax.set_yticks(range(N_STATES), RATING_LABELS)
    ax.set_xlabel("rating next year")
    ax.set_ylabel("rating this year")
    ax.set_title("(a) Do-nothing kernel (NBI)", loc="left")
    ax.tick_params(length=0)

    # (b) policy map over the failure cost
    ax = axes[1]
    grid = np.array([sols[m]["policy"] for m in RISK_MULTIPLIERS]).T
    ax.imshow(grid, cmap=ListedColormap(ACTION_COLORS), vmin=-.5, vmax=3.5, aspect="auto")
    for i in range(N_STATES):
        for j, m in enumerate(RISK_MULTIPLIERS):
            ax.text(j, i, "·mMR"[grid[i, j]] if grid[i, j] else "", ha="center", va="center", fontsize=7.5,
                    color="#20333f", weight="bold")
    ax.set_xticks(range(len(RISK_MULTIPLIERS)), [str(m) for m in RISK_MULTIPLIERS])
    ax.set_yticks(range(N_STATES), RATING_LABELS)
    ax.set_xlabel("failure cost (× replacement cost)")
    ax.set_ylabel("deck rating")
    jb = RISK_MULTIPLIERS.index(BASE_MULTIPLIER)
    ax.add_patch(plt.Rectangle((jb - .5, -.5), 1, N_STATES, fill=False, ec="#20333f", lw=1.2))
    ax.set_title("(b) Optimal action", loc="left")
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#9aa8b0", lw=.4) for c in ACTION_COLORS]
    ax.legend(handles, ["nothing", "minor (m)", "major (M)", "replace (R)"], loc="upper center",
              bbox_to_anchor=(.5, -.28), ncol=2, frameon=False, fontsize=7, handlelength=1, columnspacing=.8)

    # (c) value by rating
    ax = axes[2]
    x = np.arange(N_STATES)
    w = .27
    series = [("optimal", base["v"], "#0072B2"),
              (f"schedule (every {base['period']} yr)", base["v_schedule"], "#009E73"),
              ("reactive", base["v_reactive"], "#7b858a")]
    for i, (name, v, col) in enumerate(series):
        ax.bar(x + (i - 1) * w, v / 1000, w, color=col, label=name)
    ax.set_xticks(x, RATING_LABELS)
    ax.set_xlabel("deck rating now")
    ax.set_ylabel("thousand USD per m²")
    ax.set_title("(c) Expected discounted cost", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(base["v_reactive"].max(), base["v_schedule"].max()) / 1000 * 1.32)
    fig.tight_layout(w_pad=2.0)
    save(fig, "policy")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mustem-mat", type=Path, help="Deck Feature_Matrix_fold_1.mat from the MUSTEM repository")
    args = parser.parse_args()
    STATIC.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    C, first, provenance = get_counts(args.mustem_mat)
    P0 = do_nothing_matrix(C)
    mu = initial_distribution(first)
    sols = {m: solve(P0, mu, m) for m in sorted(set(RISK_MULTIPLIERS) | set(REPLAY_MULTIPLIERS))}

    issues = []
    if not np.allclose(P0.sum(axis=1), 1):
        issues.append("do-nothing rows do not sum to one")
    if np.any(np.tril(P0, -1) > 0):
        issues.append("a deck improves without work")
    for m, s in sols.items():
        if not np.array_equal(s["vi"]["policy"], s["policy"]):
            issues.append(f"x{m}: value and policy iteration disagree on the policy")
        if np.max(np.abs(s["vi"]["v"] - s["v"])) > 1e-4:
            issues.append(f"x{m}: value iteration is not within tolerance of policy iteration")
        residual = np.max(np.abs(bdm.q_values(s["mdp"], s["v"]).min(axis=1) - s["v"]))
        if residual > 1e-6:
            issues.append(f"x{m}: Bellman residual {residual:.2e}")
        if s["cost"]["optimal"] > min(s["cost"]["schedule"], s["cost"]["reactive"]) + 1e-6:
            issues.append(f"x{m}: optimal policy worse than a baseline")
        if np.any(s["v"] > s["v_reactive"] + 1e-6) or np.any(s["v"] > s["v_schedule"] + 1e-6):
            issues.append(f"x{m}: optimal value exceeds a baseline in some state")
    if issues:
        raise SystemExit("\n".join(issues))

    data = write_replay(P0, mu, sols)
    # The replay must follow the solver's policy and kernel support.
    for sc in data["scenarios"].values():
        run = sc["runs"]["optimal"]
        for t in range(N_YEARS):
            for b in range(N_BRIDGES):
                s, a, j = run["states"][t][b], run["actions"][t][b], run["states"][t + 1][b]
                assert a == sc["policy"][s]
                assert sols[sc["multiplier"]]["mdp"].P[a, s, j] > 0

    write_results(sols, provenance)
    figure_style()
    figure(P0, sols)

    base = sols[BASE_MULTIPLIER]
    sojourn = {RATING_LABELS[i]: round(1 / (1 - P0[i, i]), 1) for i in range(N_STATES - 1)}
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "schema_version": 1,
        "transition_source": provenance,
        "counts_ratings": list(COUNT_RATINGS),
        "counts": C.tolist(),
        "transitions_counted": int(C.sum()),
        "do_nothing_matrix": np.round(P0, 6).tolist(),
        "mean_years_in_rating": sojourn,
        "initial_distribution": np.round(mu, 6).tolist(),
        "gamma": GAMMA,
        "action_cost_usd_per_m2": dict(zip(ACTIONS, ACTION_COST.tolist())),
        "failure_probability": dict(zip(RATING_LABELS, FAILURE_PROB.tolist())),
        "base_failure_multiplier": BASE_MULTIPLIER,
        "base": {
            "policy": [ACTIONS[a] for a in base["policy"]],
            "value_by_rating": dict(zip(RATING_LABELS, np.round(base["v"], 2).tolist())),
            "value_iteration_sweeps": base["vi"]["iterations"],
            "value_iteration_tol": 1e-6,
            "policy_iteration_policies": [[ACTIONS[a] for a in p] for p, _ in base["pi_trace"]],
            "best_schedule_period": base["period"],
            "cost": {k: round(v, 2) for k, v in base["cost"].items()},
            "reactive_by_rating": dict(zip(RATING_LABELS, np.round(base["v_reactive"], 2).tolist())),
            "schedule_by_rating": dict(zip(RATING_LABELS, np.round(base["v_schedule"], 2).tolist())),
        },
        "sensitivity": {str(m): {"policy": [ACTIONS[a] for a in sols[m]["policy"]],
                                 "cost": {k: round(v, 2) for k, v in sols[m]["cost"].items()},
                                 "best_schedule_period": sols[m]["period"],
                                 "policy_iteration_steps": len(sols[m]["pi_trace"]),
                                 "value_iteration_sweeps": sols[m]["vi"]["iterations"]}
                        for m in sorted(sols)},
        "replay": {m: {"expected": sc["expected"], "final_cum": {k: r["cum"][-1] for k, r in sc["runs"].items()}}
                   for m, sc in data["scenarios"].items()},
    }, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
