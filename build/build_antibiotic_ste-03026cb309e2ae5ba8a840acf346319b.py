#!/usr/bin/env python3
"""Rebuild the antibiotic-steering solves, teaching figure, table, and replay data.

Run from the repository root: uv run python scripts/build_antibiotic_steering_artifacts.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from antibiotic_steering import (  # noqa: E402
    DRUGS, LANDSCAPES, MUTATIONS, N_GENOTYPES, backward_induction,
    best_open_loop, fitness, label, neighbours, resistant_genotype,
    step_matrix, uniform_start,
)

STATIC = ROOT / "_static" / "antibiotic_steering"
ARTIFACTS = ROOT / "artifacts" / "antibiotic_steering"
REPLAY_DATA = ROOT / "interactive" / "antibiotic-steering-data.json"
FEATURED = "AMP"
MAX_STEPS = 3
STARTS = {"uniform": uniform_start(), "wild": np.eye(N_GENOTYPES)[0]}
EDGES = [(g, h) for g in range(N_GENOTYPES) for h in neighbours(g) if g < h]
FITNESS_MAP = LinearSegmentedColormap.from_list(
    "fitness", ["#f7fbff", "#fde0c5", "#f4a261", "#d55e00", "#7f2704"])
MASS = "#0072B2"
GRAY = "#7b858a"


def layout():
    """Hasse layout of the 4-cube: column = number of mutations."""
    pos = {}
    for k in range(5):
        members = [g for g in range(N_GENOTYPES) if bin(g).count("1") == k]
        for i, g in enumerate(members):
            pos[g] = (k, (len(members) - 1) / 2 - i)
    return pos


def stage_frames(start, assignment):
    """Distributions after each substitution while each genotype's assigned drug acts.

    The drug is chosen from the genotype at the start of the stage and held
    for the whole stage, so mass is tracked separately for each drug in use.
    """
    parts = {d: start * (np.asarray(assignment) == d) for d in set(assignment)}
    steps = {d: step_matrix(d) for d in parts}
    frames, flows = [start.copy()], []
    while True:
        flow = np.zeros((N_GENOTYPES, N_GENOTYPES))
        for d in parts:
            moved = parts[d][:, None] * steps[d]
            np.fill_diagonal(moved, 0.0)
            flow += moved
            parts[d] = parts[d] @ steps[d]
        if flow.sum() < 1e-12:
            break
        frames.append(sum(parts.values()))
        flows.append(flow)
    return frames, flows


def course(start, assignments):
    stages, dist = [], start
    for assignment in assignments:
        frames, flows = stage_frames(dist, assignment)
        stages.append({
            "drugs": list(assignment),
            "frames": [[round(float(x), 6) for x in f] for f in frames],
            "flows": [[[int(g), int(h), round(float(flow[g, h]), 6)] for g, h in zip(*np.nonzero(flow > 1e-9))]
                      for flow in flows],
        })
        dist = frames[-1]
    return stages, dist


def solve_all():
    results = {}
    for final in DRUGS:
        target = resistant_genotype(final)
        entry = {"resistant": target, "by_start": {}}
        dp = {k: backward_induction(final, k) for k in range(MAX_STEPS + 1)}
        entry["dp_policy"] = {k: [[DRUGS[a] for a in row] for row in dp[k].policy]
                              for k in range(1, MAX_STEPS + 1)}
        for name, start in STARTS.items():
            rows = []
            for k in range(MAX_STEPS + 1):
                seq, p_fixed = best_open_loop(final, k, start)
                p_dp = float(start @ dp[k].values[0])
                rows.append({"steps": k, "fixed": list(seq), "p_fixed": p_fixed, "p_dp": p_dp})
            entry["by_start"][name] = rows
        results[final] = entry
    return results


def replay_data(results):
    data = {
        "drugs": [{"code": d, "name": LANDSCAPES[d][0], "fitness": list(fitness(d))} for d in DRUGS],
        "mutations": list(MUTATIONS),
        "labels": [label(g) for g in range(N_GENOTYPES)],
        "edges": EDGES,
        "featured": FEATURED,
        "scenarios": {},
    }
    for final, entry in results.items():
        scen = {"resistant": entry["resistant"], "starts": {}}
        for name, start in STARTS.items():
            per_k = []
            for row in entry["by_start"][name]:
                k = row["steps"]
                fixed_assign = [[d] * N_GENOTYPES for d in row["fixed"]]
                dp_assign = entry["dp_policy"].get(k, [])
                final_assign = [[final] * N_GENOTYPES]
                fixed_stages, fixed_end = course(start, fixed_assign + final_assign)
                dp_stages, dp_end = course(start, dp_assign + final_assign)
                for end, p in ((fixed_end, row["p_fixed"]), (dp_end, row["p_dp"])):
                    assert abs(end[entry["resistant"]] - p) < 1e-9
                per_k.append({"fixed": {"sequence": row["fixed"], "p": round(row["p_fixed"], 6), "stages": fixed_stages},
                              "dp": {"p": round(row["p_dp"], 6), "stages": dp_stages}})
            scen["starts"][name] = per_k
        data["scenarios"][final] = scen
    REPLAY_DATA.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")


def write_results(results):
    lines = [
        "| Final drug | Resistant genotype | No steering | Fixed, 1 drug | Feedback, 1 drug | Fixed, 2 drugs | Feedback, 2 drugs | Fixed, 3 drugs | Feedback, 3 drugs |",
        "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for final in DRUGS:
        rows = results[final]["by_start"]["uniform"]
        cells = [f"{LANDSCAPES[final][0]} ({final})", f"`{label(results[final]['resistant'])}`", f"{rows[0]['p_fixed']:.2f}"]
        for r in rows[1:]:
            fixed = f"{r['p_fixed']:.2f}"
            dp = f"{r['p_dp']:.2f}"
            if r["p_dp"] < r["p_fixed"] - 5e-3:
                dp = f"**{dp}**"
            cells += [fixed, dp]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Probability that the population ends at the most resistant genotype of the final drug, "
                 "when every starting genotype is equally likely. *Fixed* is the best drug sequence found by "
                 "exhaustive search; *feedback* is the backward-induction policy, which observes the genotype "
                 "before each drug. Bold marks a feedback value lower than the best fixed sequence.")
    (ARTIFACTS / "results.md").write_text("\n".join(lines) + "\n")


def figure_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9,
        "legend.fontsize": 8, "figure.dpi": 160, "savefig.dpi": 180,
        "svg.fonttype": "none", "svg.hashsalt": "antibiotic-steering",
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


def draw_cube(ax, dist, drugs, target, title, show_drugs=False):
    pos = layout()
    for g, h in EDGES:
        (x0, y0), (x1, y1) = pos[g], pos[h]
        ax.plot([1.6 * x0, 1.6 * x1], [y0, y1], color="#d3dce0", lw=.6, zorder=0)
    for g in range(N_GENOTYPES):
        x, y = pos[g]
        x *= 1.6
        f = fitness(drugs[g])
        ax.scatter([x], [y], s=210, c=[FITNESS_MAP(f[g] / f.max())], edgecolors="#4a5a63",
                   linewidths=.5, zorder=1)
        if dist[g] > 1e-4:
            # Disc area proportional to probability; a sure genotype fills the node.
            ax.scatter([x], [y], s=190 * dist[g], c=MASS, zorder=2, linewidths=0)
        if g == target:
            ax.scatter([x], [y], s=380, facecolors="none", edgecolors="#b2182b", linewidths=1.2, zorder=3)
        ax.text(x, y - (.4 if g == target else .3), label(g), ha="center", va="top", fontsize=5.8, family="monospace",
                color="#33434c")
        if show_drugs:
            ax.text(x + .3, y + .12, drugs[g], ha="left", va="bottom", fontsize=5.8, color="#7f2704")
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.set_xlim(-.6, 1.6 * 4 + .6)
    ax.set_ylim(-3.05, 2.9)
    ax.axis("off")


def figure(results):
    """Snapshots for the featured final drug with one steering drug."""
    final = FEATURED
    entry = results[final]
    target = entry["resistant"]
    row = entry["by_start"]["uniform"][1]
    start = uniform_start()
    fixed_drug = row["fixed"][0]
    dp_drugs = entry["dp_policy"][1][0]
    fixed_stages, _ = course(start, [[fixed_drug] * N_GENOTYPES, [final] * N_GENOTYPES])
    dp_stages, _ = course(start, [dp_drugs, [final] * N_GENOTYPES])

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    rows = [
        (f"Fixed sequence: {fixed_drug} whatever the genotype", f"After {fixed_drug}", fixed_stages,
         [fixed_drug] * 16, row["p_fixed"]),
        ("Feedback: the drug written next to each genotype", "After the chosen drugs", dp_stages,
         dp_drugs, row["p_dp"]),
    ]
    for r, (name, middle, stages, drugs, p) in enumerate(rows):
        panels = [
            (np.array(stages[0]["frames"][0]), drugs, "Start"),
            (np.array(stages[0]["frames"][-1]), drugs, middle),
            (np.array(stages[1]["frames"][-1]), [final] * 16, f"After {final}: P(resistant) = {p:.2f}"),
        ]
        for c, (dist, d, title) in enumerate(panels):
            draw_cube(axes[r, c], dist, d, target, title, show_drugs=(r == 1 and c == 0))
    fig.tight_layout(h_pad=2.4, w_pad=.6, rect=(0, 0, 1, .96))
    for r, (name, *_rest) in enumerate(rows):
        top = axes[r, 0].get_position().y1
        fig.text(.01, top + .045, name, fontsize=9.5, weight="bold", va="bottom")
    save(fig, "steering")


def main():
    STATIC.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    results = solve_all()
    issues = []
    for final, entry in results.items():
        for rows in entry["by_start"].values():
            for r in rows:
                if r["p_dp"] > r["p_fixed"] + 1e-9:
                    issues.append(f"{final}: feedback worse than fixed with {r['steps']} drugs")
    if issues:
        raise SystemExit("\n".join(issues))
    replay_data(results)
    write_results(results)
    figure_style()
    figure(results)
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "schema_version": 1,
        "model": "SSWM limit chain, r = 0, Mira et al. 2015 landscapes",
        "results": {final: {"resistant": label(e["resistant"]), "by_start": e["by_start"],
                            "dp_policy": e["dp_policy"]} for final, e in results.items()},
    }, indent=1) + "\n")


if __name__ == "__main__":
    main()
