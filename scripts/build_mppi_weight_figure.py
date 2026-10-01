#!/usr/bin/env python3
"""Render the introductory MPPI weighting curve with LaTeX labels.

Run: MPLCONFIGDIR=/private/tmp/mppi-mpl .venv/bin/python scripts/build_mppi_weight_figure.py
Requires Matplotlib, NumPy, LaTeX (with amsmath), and pdftocairo.
This figure is analytic; no rollouts or saved experiments are regenerated.
"""

from argparse import ArgumentParser
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INK = "#253440"
MUTED = "#5B6470"
RULE = "#CCD3DA"
BLUE = "#0072B2"
ORANGE = "#B64E00"
GREEN = "#008567"
STYLE = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": .7,
    "svg.fonttype": "path",
    "svg.hashsalt": "mppi-exponential-weights",
}


def weighting_figure():
    """Show weight ratios, not a normalized weight as a univariate function."""
    fig = plt.figure(figsize=(7.8, 4.7), facecolor="white")
    ax = fig.add_axes([.13, .24, .83, .60])
    gap = np.linspace(0, 4, 501)
    for temperature, color, style in [
        (.5, ORANGE, "--"), (1., BLUE, "-"), (2., GREEN, "-."),
    ]:
        ax.plot(gap, np.exp(-gap/temperature), color=color, ls=style, lw=1.8)

    ax.axhline(1, color=MUTED, ls=":", lw=1)
    ax.text(3.96, 1.025, r"Equal weights ($\lambda\to\infty$)",
            ha="right", color=MUTED, fontsize=11)
    ax.text(.95, .18, r"$\lambda=\frac12$", color=ORANGE, fontsize=14)
    ax.text(1.65, .23, r"$\lambda=1$", color=BLUE, fontsize=14)
    ax.text(2.80, .30, r"$\lambda=2$", color=GREEN, fontsize=14)

    half_gap = np.log(2)
    ax.plot([0, half_gap, half_gap], [.5, .5, 0],
            color=BLUE, lw=.8, ls=":", zorder=1)
    ax.scatter([half_gap], [.5], color=BLUE, edgecolor="white", lw=.7,
               s=45, zorder=5)
    ax.set(
        xlim=(0, 4), ylim=(-.015, 1.09),
        xticks=[0, half_gap, 1, 2, 3, 4],
        xticklabels=[r"$0$", r"$\log 2$", r"$1$", r"$2$", r"$3$", r"$4$"],
        yticks=[0, .25, .5, .75, 1],
        yticklabels=[r"$0$", r"$\frac14$", r"$\frac12$", r"$\frac34$", r"$1$"],
        xlabel=r"Excess trajectory cost $\Delta S=S_i-S_{\min}$",
        ylabel=r"Relative weight $w_i/w_{\mathrm{best}}$",
    )
    ax.tick_params(length=3, width=.7, color=RULE, pad=5)
    ax.xaxis.labelpad = 8
    ax.yaxis.labelpad = 8
    ax.spines[["left", "bottom"]].set_color(RULE)
    ax.get_xticklabels()[1].set_color(BLUE)

    fig.text(.13, .94, "Cost differences determine relative influence", fontsize=15)
    fig.text(.13, .078,
             r"For $\lambda=1$, an extra cost of $\log 2$ halves the weight relative to the best.",
             fontsize=11, color=MUTED)
    return fig


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    output = ROOT / "_static" / "brownian_mppi"
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig = weighting_figure()
        pdf = output / "exponential-weights.pdf"
        fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
        # Match the collocation figures: PDF conversion preserves the LaTeX
        # glyph geometry that the native Matplotlib SVG backend can distort.
        subprocess.run(
            ["pdftocairo", "-svg", str(pdf), str(output / "exponential-weights.svg")],
            check=True,
        )
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.preview_dir / "exponential-weights.png",
                        dpi=720 / fig.get_figwidth())
        plt.close(fig)
    print("Rendered exponential weighting to LaTeX-typeset PDF and SVG.")


if __name__ == "__main__":
    main()
