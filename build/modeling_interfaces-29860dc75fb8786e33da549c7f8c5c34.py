"""Conceptual figure comparing three physical control interfaces.

The drawing uses the same visual grammar across three oscillatory systems while
placing the control input where it actually enters the mechanics.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle
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
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.dpi": 150,
}


def _base_axis(axis: plt.Axes, title: str) -> None:
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.18, 1.15)
    axis.set_aspect("equal")
    axis.set_title(title, fontweight="semibold", pad=7)
    axis.axis("off")


def _double_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="<->",
            mutation_scale=11,
            linewidth=1.8,
            color=color,
        )
    )


def _draw_swing(axis: plt.Axes) -> None:
    _base_axis(axis, "Swing")
    pivot = np.array([0.0, 0.86])
    angle = np.deg2rad(20.0)
    seat = pivot + 1.35 * np.array([np.sin(angle), -np.cos(angle)])
    torso = seat + np.array([-0.10, 0.46])

    axis.plot(
        [pivot[0], seat[0]],
        [pivot[1], seat[1]],
        color="0.22",
        linewidth=2.2,
    )
    axis.add_patch(Circle(pivot, 0.055, color=OI["black"]))
    axis.plot(
        [seat[0] - 0.22, seat[0] + 0.20],
        [seat[1], seat[1]],
        color=OI["black"],
        linewidth=3.0,
    )
    axis.plot(
        [seat[0], torso[0]],
        [seat[1], torso[1]],
        color=OI["vermilion"],
        linewidth=5.0,
        solid_capstyle="round",
    )
    axis.add_patch(Circle(torso + np.array([-0.02, 0.14]), 0.10, color=OI["vermilion"]))
    _double_arrow(
        axis,
        tuple(seat + np.array([0.33, 0.03])),
        tuple(torso + np.array([0.25, 0.02])),
        OI["vermilion"],
    )
    axis.text(
        0.84,
        0.12,
        "internal\nshape $u$",
        color=OI["vermilion"],
        ha="center",
        va="center",
        fontsize=8.5,
    )
    axis.text(0.0, -1.02, "create oscillation", color="0.25", ha="center")


def _draw_crane(axis: plt.Axes) -> None:
    _base_axis(axis, "Overhead crane")
    rail_y = 0.75
    trolley_x = -0.25
    angle = np.deg2rad(-17.0)
    load = np.array(
        [
            trolley_x + 1.25 * np.sin(angle),
            rail_y - 1.25 * np.cos(angle),
        ]
    )

    axis.plot([-1.05, 1.05], [rail_y, rail_y], color="0.35", linewidth=2.2)
    axis.add_patch(
        Rectangle(
            (trolley_x - 0.17, rail_y - 0.10),
            0.34,
            0.20,
            color=OI["blue"],
        )
    )
    axis.plot(
        [trolley_x, load[0]],
        [rail_y - 0.10, load[1]],
        color="0.20",
        linewidth=2.0,
    )
    axis.add_patch(Circle(load, 0.14, color=OI["orange"]))
    axis.add_patch(
        FancyArrowPatch(
            (-0.82, 0.98),
            (0.35, 0.98),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.8,
            color=OI["blue"],
        )
    )
    axis.text(-0.24, 1.04, r"pivot acceleration $u=\ddot x$", color=OI["blue"], ha="center")
    axis.text(0.0, -1.02, "suppress oscillation", color="0.25", ha="center")


def _draw_wave(axis: plt.Axes) -> None:
    _base_axis(axis, "Wave-energy flap")
    x = np.linspace(-1.18, 0.25, 160)
    surface = 0.16 + 0.10 * np.sin(2.5 * np.pi * (x + 0.20))
    axis.fill_between(x, -0.95, surface, color=OI["skyblue"], alpha=0.20)
    axis.plot(x, surface, color=OI["skyblue"], linewidth=2.2)

    hinge = np.array([0.50, -0.78])
    top = np.array([0.38, 0.45])
    axis.plot(
        [hinge[0], top[0]],
        [hinge[1], top[1]],
        color=OI["black"],
        linewidth=7.0,
        solid_capstyle="round",
    )
    axis.add_patch(Circle(hinge, 0.065, color=OI["orange"], zorder=4))
    axis.add_patch(
        Arc(
            hinge,
            0.72,
            0.72,
            angle=0.0,
            theta1=68,
            theta2=132,
            linewidth=2.0,
            color=OI["green"],
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.70, -0.48),
            (0.61, -0.36),
            arrowstyle="-|>",
            mutation_scale=10,
            color=OI["green"],
            linewidth=1.6,
        )
    )
    axis.text(0.79, -0.13, r"PTO damping $u=\rho\geq0$", color=OI["green"], ha="center")
    axis.add_patch(
        FancyArrowPatch(
            (-1.06, 0.48),
            (-0.40, 0.48),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.5,
            color=OI["skyblue"],
        )
    )
    axis.text(-0.73, 0.59, "uncommanded waves", color=OI["blue"], ha="center")
    axis.text(0.0, -1.02, "harvest oscillation", color="0.25", ha="center")


def make_overview_figure() -> plt.Figure:
    """Draw the three control interfaces on identical visual footing."""

    with mpl.rc_context(STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(8.2, 3.1), constrained_layout=True)
        _draw_swing(axes[0])
        _draw_crane(axes[1])
        _draw_wave(axes[2])
        return figure


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    arguments = parser.parse_args()

    output_figure = make_overview_figure()
    if arguments.output is None:
        plt.show()
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        output_figure.savefig(arguments.output, bbox_inches="tight", dpi=200)
