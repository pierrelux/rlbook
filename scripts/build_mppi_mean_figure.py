#!/usr/bin/env python3
"""Draw the two MPPI optimization problems for the scalar quadratic example.

Run: MPLCONFIGDIR=/private/tmp/mppi-mpl .venv/bin/python scripts/build_mppi_mean_figure.py
Requires Matplotlib, NumPy, LaTeX (with amsmath), and pdftocairo.
All densities are analytic; no experiments or rollouts are regenerated.
"""

from argparse import ArgumentParser
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_mppi_weight_figure import BLUE, INK, MUTED, ORANGE, ROOT, RULE, STYLE


def normal_density(v, mean, variance):
    return np.exp(-0.5 * (v - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)


def mean_figure():
    """Compare the optimal distribution with its fixed-variance approximation."""
    kappa = 2 * np.log(2)
    mean = kappa / (1 + kappa)
    variance = 1 / (1 + kappa)
    v = np.linspace(-3.1, 3.5, 1201)
    reference = normal_density(v, 0, 1)
    target = normal_density(v, mean, variance)
    fitted = normal_density(v, mean, 1)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.98, bottom=.24, top=.72, wspace=.17)
    for ax in axes:
        ax.fill_between(v, target, color=BLUE, alpha=.07, lw=0)
        ax.plot(v, target, color=BLUE, lw=1.9, zorder=3)
        ax.axvline(mean, color=BLUE, ls=":", lw=.9, zorder=1)
        ax.set(xlim=(-3, 3.5), ylim=(0, .76), xticks=[-2, 0, 2],
               yticks=[0, .2, .4, .6], xlabel=r"Candidate input $v$")
        ax.tick_params(length=3, width=.7, color=RULE, pad=5, labelsize=10)
        ax.spines[["left", "bottom"]].set_color(RULE)
        ax.xaxis.labelpad = 7

    left, right = axes
    left.set_ylabel("Probability density", labelpad=8)
    left.plot(v, reference, color=MUTED, ls="--", lw=1.7)
    left.vlines(0, 0, normal_density(0, 0, 1), color=MUTED, ls=":", lw=.9)
    left.annotate(r"Reference $p_0$", xy=(-1, normal_density(-1, 0, 1)),
                  xytext=(-2.85, .44), fontsize=11, color=MUTED,
                  arrowprops={"arrowstyle": "-", "color": MUTED, "lw": .7})
    left.text(mean + .13, .67, r"$p^\star$: cost-weighted reference",
              fontsize=10.5, color=BLUE)
    left.text(.5, -.27, "Mean moves toward target 1", transform=left.transAxes,
              fontsize=10.5, ha="center", color=MUTED)

    right.plot(v, fitted, color=ORANGE, ls="-.", lw=1.9)
    right.text(mean + .13, .67, r"$p^\star$: variance $\approx0.419$",
               fontsize=10.5, color=BLUE)
    right.annotate(r"$q_{u^+}$: variance $1$",
                   xy=(1.7, normal_density(1.7, mean, 1)), xytext=(1.15, .43),
                   fontsize=11, color=ORANGE,
                   arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": .7})
    right.text(.5, -.27, r"Same mean $u^+\approx0.581$", transform=right.transAxes,
               fontsize=10.5, ha="center", color=BLUE)

    for ax, title, objective in [
        (left, "1. Optimize the distribution",
         r"$\min_Q\;\{\mathbb{E}_Q[S(v)]+D_{\mathrm{KL}}(Q\|P_0)\}$"),
        (right, "2. Choose the Gaussian mean",
         r"$\min_m\;D_{\mathrm{KL}}(P^\star\|Q_m),\quad Q_m=\mathcal{N}(m,1)$"),
    ]:
        pos = ax.get_position()
        fig.text(pos.x0, .93, title, fontsize=13, color=INK)
        fig.text(pos.x0, .84, objective, fontsize=11.5, color=INK)

    fig.text(.08, .055,
             r"$S(v)=\frac{\kappa}{2}(v-1)^2,\quad\kappa=2\log 2,\quad\lambda=1$",
             fontsize=11, color=MUTED)
    return fig


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    output = ROOT / "_static" / "brownian_mppi"
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({**STYLE, "svg.hashsalt": "mppi-gaussian-mean",
                         "text.latex.preamble": r"\usepackage{amsmath,amssymb}"}):
        fig = mean_figure()
        pdf = output / "gaussian-mean.pdf"
        fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
        subprocess.run(
            ["pdftocairo", "-svg", str(pdf), str(output / "gaussian-mean.svg")],
            check=True,
        )
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["pdftoppm", "-singlefile", "-scale-to-x", "780", "-scale-to-y", "-1",
                 "-png", str(pdf), str(args.preview_dir / "gaussian-mean")],
                check=True,
            )
        plt.close(fig)
    print("Rendered the MPPI distribution and Gaussian-mean objectives to PDF and SVG.")


if __name__ == "__main__":
    main()
