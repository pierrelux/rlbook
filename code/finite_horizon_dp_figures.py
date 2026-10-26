"""Conceptual figures for finite-horizon dynamic programming.

The cut-and-paste figure follows one inductive step of the backward-induction
proof with concrete costs: an arbitrary control sequence is cut after its first
step, its tail is replaced by an optimal one, and the best first step pasted to
its optimal tail attains the lower bound. The cardinal-function figure compares
rounding, linear interpolation and a cubic spline as ways of reconstructing a
cost-to-go from its values at grid states.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
}

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    # DejaVu emits standard Unicode Greek glyphs when SVG text remains text.
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.dpi": 150,
    "svg.fonttype": "none",
    "svg.hashsalt": "rlbook-finite-horizon-dp-v1",
}

STAGES = (0, 1, 2, 3, 4)
STAGE_LABELS = (r"$t$", r"$t+1$", r"$t+2$", r"$\cdots$", r"$T$")
START = (0.0, 0.0)


def _stage_axis(axis: plt.Axes, title: str) -> None:
    for stage in STAGES:
        axis.axvline(stage, color="0.88", linewidth=0.8, zorder=0)
    axis.set_xlim(-0.35, 5.25)
    axis.set_ylim(-2.0, 2.1)
    axis.set_xticks(STAGES, STAGE_LABELS)
    axis.tick_params(axis="x", length=0, pad=4)
    axis.set_yticks([])
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("0.6")
    axis.set_xlabel("stage", labelpad=2, color="0.3")
    axis.set_title(title, fontweight="semibold", loc="left", pad=8)


def _path(axis: plt.Axes, points, color: str, width: float, **kwargs) -> None:
    xs, ys = zip(*points)
    axis.plot(
        xs,
        ys,
        color=color,
        linewidth=width,
        solid_capstyle="round",
        solid_joinstyle="round",
        **kwargs,
    )


def _state(axis: plt.Axes, point, color: str = "black", size: float = 4.6) -> None:
    axis.plot(
        *point,
        marker="o",
        markersize=size,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.3,
        zorder=5,
    )


def make_cut_and_paste_figure() -> plt.Figure:
    """Draw the two moves of one inductive step with concrete costs."""

    # The first step of the arbitrary sequence in Step 1 is the top candidate
    # of Step 2, so the two panels share that next state and its optimal tail.
    top_state = (1.0, 1.1)
    top_tail = [(2.0, 1.3), (3.0, 1.2), (4.0, 1.4)]

    with mpl.rc_context(STYLE):
        figure, (cut_axis, paste_axis) = plt.subplots(
            1,
            2,
            figsize=(7.4, 3.45),
            gridspec_kw={
                "left": 0.02,
                "right": 0.99,
                "bottom": 0.13,
                "top": 0.87,
                "wspace": 0.08,
            },
        )

        # Step 1: cut an arbitrary sequence after its first step.
        _stage_axis(cut_axis, "Step 1: cut after the first step")
        actual_tail = [top_state, (2.0, 0.2), (3.0, 0.75), (4.0, -0.3)]
        optimal_tail = [top_state, *top_tail]

        _path(cut_axis, [START, top_state], "0.15", 1.8)
        _path(cut_axis, actual_tail, OI["vermilion"], 1.6, linestyle=(0, (4, 2)))
        _path(cut_axis, optimal_tail, OI["blue"], 2.0)
        for point in actual_tail[1:]:
            _state(cut_axis, point, OI["vermilion"], 3.6)
        for point in optimal_tail[1:]:
            _state(cut_axis, point, OI["blue"], 3.6)
        _state(cut_axis, START)
        _state(cut_axis, top_state)

        cut_axis.plot(
            [1.0, 1.0], [-0.75, 1.95], color="0.35", linewidth=0.9, linestyle=":"
        )
        cut_axis.text(1.05, 1.95, "cut", color="0.3", fontsize=8.5, va="top")

        cut_axis.text(-0.1, -0.12, r"$\mathbf{x}$", ha="right", va="top", fontsize=10)
        cut_axis.text(
            0.9, 1.2, r"$\mathbf{x}_{t+1}$", ha="right", va="bottom", fontsize=10
        )
        cut_axis.text(
            0.4, 0.42, r"$c_t=2$", ha="left", va="top", fontsize=8.5, color="0.15"
        )
        cut_axis.text(
            4.12,
            -0.3,
            "actual tail\ncosts 9",
            color=OI["vermilion"],
            fontsize=8.5,
            va="center",
        )
        cut_axis.text(
            4.12,
            1.4,
            "optimal tail\n" + r"$J^\star_{t+1}(\mathbf{x}_{t+1})=7$",
            color=OI["blue"],
            fontsize=8.5,
            va="center",
        )
        cut_axis.text(
            0.0,
            -1.42,
            "any sequence: " + r"$2+9=11$",
            color=OI["vermilion"],
            fontsize=8.5,
            va="center",
        )
        cut_axis.text(
            0.0,
            -1.8,
            "same first step, optimal tail: " + r"$2+7=9\ \leq\ 11$",
            color=OI["blue"],
            fontsize=8.5,
            va="center",
        )

        # Step 2: compare first steps, each pasted to its optimal tail.
        _stage_axis(paste_axis, "Step 2: best first step, paste its optimal tail")
        candidates = (
            # (next state, first-step cost, optimal tail, tail cost, chosen)
            (top_state, 2, top_tail, 7, False),
            ((1.0, 0.0), 3, [(2.0, -0.2), (3.0, 0.05), (4.0, 0.15)], 4, True),
            ((1.0, -1.1), 1, [(2.0, -1.0), (3.0, -1.2), (4.0, -1.1)], 8, False),
        )
        for point, step_cost, tail, tail_cost, chosen in candidates:
            if chosen:
                step_color, tail_color = OI["green"], OI["blue"]
                step_width, tail_width = 2.4, 2.4
                label_color = tail_color
            else:
                step_color, tail_color = "0.55", OI["skyblue"]
                step_width, tail_width = 1.2, 1.2
                label_color = "0.4"
            _path(paste_axis, [START, point], step_color, step_width)
            _path(paste_axis, [point, *tail], tail_color, tail_width)
            for tail_point in tail:
                _state(paste_axis, tail_point, tail_color, 4.0 if chosen else 3.6)
            _state(paste_axis, point, step_color if chosen else "0.35")

            above = point[1] >= 0
            paste_axis.text(
                0.45 if above else 0.5,
                0.5 * point[1] + (0.1 if above else -0.12),
                str(step_cost),
                ha="right" if point[1] > 0 else "center",
                va="bottom" if above else "top",
                fontsize=8.5,
                color=step_color if chosen else "0.35",
                fontweight="semibold" if chosen else "normal",
            )
            paste_axis.text(
                4.12,
                tail[-1][1],
                rf"$J^\star_{{t+1}}={tail_cost}$",
                color=label_color,
                fontsize=8.5,
                va="center",
            )

        _state(paste_axis, START)
        paste_axis.text(
            -0.1, -0.12, r"$\mathbf{x}$", ha="right", va="top", fontsize=10
        )
        paste_axis.text(
            0.74,
            -0.1,
            r"$\boldsymbol{\pi}^\star_t(\mathbf{x})$",
            color=OI["green"],
            fontsize=9.5,
            ha="center",
            va="top",
        )
        paste_axis.text(
            2.2,
            -0.42,
            "pasted: " + r"$3+4=7$",
            color=OI["blue"],
            fontsize=8.5,
            ha="left",
            va="center",
        )
        paste_axis.text(
            0.0,
            -1.8,
            r"$J^\star_t(\mathbf{x})=\min\{2+7,\ 3+4,\ 1+8\}=7\ \leq\ 9$",
            color="0.1",
            fontsize=8.5,
            va="center",
        )

        return figure


def make_cardinal_functions_figure() -> plt.Figure:
    """Cardinal functions and reconstructions for three interpolation rules.

    The top row shows the cardinal function of one node and of its two
    neighbors. The bottom row reconstructs stored values that jump from zero to
    one, as an all-or-nothing decision table does, and shades their range.
    """
    from scipy.interpolate import CubicSpline

    grid = np.arange(10.0)
    points = np.linspace(grid[0], grid[-1], 701)
    step_values = np.where(grid < 4.5, 0.0, 1.0)

    def reconstruct(rule: str, values: np.ndarray) -> np.ndarray:
        if rule == "nearest":
            index = np.clip(np.rint(points).astype(int), 0, len(grid) - 1)
            return values[index]
        if rule == "linear":
            return np.interp(points, grid, values)
        return CubicSpline(grid, values)(points)

    titles = {"nearest": "Rounding", "linear": "Linear", "cubic": "Cubic spline"}
    with mpl.rc_context(STYLE):
        figure, axes = plt.subplots(
            2, 3, figsize=(6.6, 3.7), sharex=True, sharey="row",
            constrained_layout=True,
        )
        for column, (rule, title) in enumerate(titles.items()):
            top, bottom = axes[:, column]
            for neighbor in (3, 5):
                top.plot(points, reconstruct(rule, np.eye(len(grid))[neighbor]),
                         color="0.75", linewidth=0.8)
            unit = np.eye(len(grid))[4]
            top.plot(points, reconstruct(rule, unit), color=OI["blue"],
                     linewidth=1.8)
            top.scatter(grid, unit, color="0.3", edgecolor="white", s=22,
                        zorder=3)
            top.axhline(0.0, color="0.6", linewidth=0.6, linestyle=":")
            top.set_title(title)

            bottom.axhspan(0.0, 1.0, color="0.93", zorder=0)
            bottom.plot(points, reconstruct(rule, step_values),
                        color=OI["vermilion"], linewidth=1.8)
            bottom.scatter(grid, step_values, color="0.3", edgecolor="white",
                           s=22, zorder=3)
            bottom.set_xlabel(r"state $x$")
            for axis in (top, bottom):
                axis.spines[["top", "right"]].set_visible(False)
                axis.set_xticks(grid)
                axis.set_xticklabels([])
        axes[0, 0].set_ylabel(r"cardinal function $\ell_i(x)$")
        axes[1, 0].set_ylabel(r"reconstruction $\hat J(x)$")
        axes[0, 0].set_ylim(-0.25, 1.15)
        axes[1, 0].set_ylim(-0.2, 1.2)

    plt.close(figure)
    return figure


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    builders = {
        "cut-and-paste": make_cut_and_paste_figure,
        "cardinal-functions": make_cardinal_functions_figure,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument(
        "--figure",
        choices=tuple(builders),
        default="cut-and-paste",
        help="Select the figure to render.",
    )
    arguments = parser.parse_args()

    output_figure = builders[arguments.figure]()
    if arguments.output is None:
        plt.show()
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        save_options = {"bbox_inches": "tight", "dpi": 200}
        if arguments.output.suffix.lower() == ".svg":
            save_options["metadata"] = {"Date": None}
        # SVG element identifiers are assigned during serialization, so retain
        # the fixed hash salt while saving as well as while constructing.
        with mpl.rc_context(STYLE):
            output_figure.savefig(arguments.output, **save_options)
        if arguments.output.suffix.lower() == ".svg":
            serialized = arguments.output.read_text(encoding="utf-8")
            arguments.output.write_text(
                "\n".join(line.rstrip() for line in serialized.splitlines()) + "\n",
                encoding="utf-8",
            )
