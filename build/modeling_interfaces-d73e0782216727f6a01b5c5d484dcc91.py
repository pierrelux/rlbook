"""Conceptual figures for physical interfaces and sampled dynamics.

The drawings use a common visual grammar to locate physical action channels and
to show how an interval integrator and sampling period define a discrete model.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
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
    "svg.fonttype": "none",
    "svg.hashsalt": "rlbook-modeling-interfaces-v2",
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


def _rk4_held_interval(
    initial_velocity: float,
    action: float,
    disturbance: float,
    duration: float,
    *,
    internal_step: float = 0.005,
) -> float:
    """Advance ``v_dot = -v + action + disturbance`` over one held interval."""

    step_count = int(round(duration / internal_step))
    if step_count <= 0 or not np.isclose(step_count * internal_step, duration):
        raise ValueError("duration must be a positive multiple of internal_step")
    step = duration / step_count
    velocity = float(initial_velocity)

    def rate(value: float) -> float:
        return -value + action + disturbance

    for _ in range(step_count):
        k1 = rate(velocity)
        k2 = rate(velocity + 0.5 * step * k1)
        k3 = rate(velocity + 0.5 * step * k2)
        k4 = rate(velocity + step * k3)
        velocity += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return velocity


def _sampled_velocity(period: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nodes, held disturbance samples, and RK4 state nodes."""

    node_count = int(round(1.0 / period))
    times = np.linspace(0.0, 1.0, node_count + 1)
    disturbance = np.where(
        (times[:-1] >= 0.4 - 1e-12) & (times[:-1] < 0.6 - 1e-12),
        1.5,
        0.0,
    )
    velocity = np.zeros(node_count + 1)
    for index, disturbance_value in enumerate(disturbance):
        velocity[index + 1] = _rk4_held_interval(
            velocity[index],
            action=0.6,
            disturbance=float(disturbance_value),
            duration=period,
        )
    return times, disturbance, velocity


def _continuous_velocity(time: np.ndarray) -> np.ndarray:
    """Exact response to the short continuous disturbance used in the figure."""

    time = np.asarray(time, dtype=float)
    baseline = 0.6 * (1.0 - np.exp(-time))
    pulse = np.zeros_like(time)
    during = (time >= 0.4) & (time < 0.6)
    after = time >= 0.6
    pulse[during] = 1.5 * (1.0 - np.exp(-(time[during] - 0.4)))
    pulse[after] = (
        1.5
        * (1.0 - np.exp(-0.2))
        * np.exp(-(time[after] - 0.6))
    )
    return baseline + pulse


def make_sampling_figure() -> plt.Figure:
    """Show how sampling changes a discrete map and disturbance resolution."""

    time = np.linspace(0.0, 1.0, 501)
    truth = _continuous_velocity(time)
    no_pulse = 0.6 * (1.0 - np.exp(-time))
    fine_times, fine_disturbance, fine_velocity = _sampled_velocity(0.1)
    coarse_times, coarse_disturbance, coarse_velocity = _sampled_velocity(1.0)

    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(8.2, 4.25))
        grid = figure.add_gridspec(
            3,
            2,
            height_ratios=(0.72, 0.72, 2.25),
            left=0.075,
            right=0.985,
            bottom=0.13,
            top=0.96,
            hspace=0.10,
            wspace=0.18,
        )

        pipeline = figure.add_subplot(grid[0, :])
        pipeline.set_xlim(0.0, 1.0)
        pipeline.set_ylim(0.0, 1.0)
        pipeline.axis("off")
        box = {
            "boxstyle": "round,pad=0.35",
            "facecolor": "#F4F4F2",
            "edgecolor": "0.55",
            "linewidth": 0.8,
        }
        pipeline.text(
            0.15,
            0.52,
            r"continuous model" "\n" r"$\dot v=-v+u+\xi(t)$",
            ha="center",
            va="center",
            bbox=box,
        )
        pipeline.text(
            0.50,
            0.52,
            r"sample $u$ and $\xi$" "\n" r"integrate for $\Delta t$",
            ha="center",
            va="center",
            bbox=box,
        )
        pipeline.text(
            0.85,
            0.52,
            r"discrete model" "\n" r"$v_{k+1}=F_{\Delta t}(v_k,u_k,\xi_k)$",
            ha="center",
            va="center",
            bbox=box,
        )
        for start, end in ((0.27, 0.38), (0.62, 0.73)):
            pipeline.add_patch(
                FancyArrowPatch(
                    (start, 0.52),
                    (end, 0.52),
                    transform=pipeline.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.2,
                    color="0.30",
                )
            )

        input_axes = [figure.add_subplot(grid[1, column]) for column in range(2)]
        state_axes = [
            figure.add_subplot(grid[2, 0]),
            figure.add_subplot(grid[2, 1], sharey=None),
        ]

        panel_data = (
            (
                "fine model: $\\Delta t=0.1$ s",
                fine_times,
                fine_disturbance,
                fine_velocity,
                OI["blue"],
                "o",
            ),
            (
                "coarse model: $\\Delta t=1$ s",
                coarse_times,
                coarse_disturbance,
                coarse_velocity,
                OI["vermilion"],
                "s",
            ),
        )

        for column, (title, times, disturbance, velocity, color, marker) in enumerate(
            panel_data
        ):
            input_axis = input_axes[column]
            state_axis = state_axes[column]
            input_axis.set_title(title, pad=5, fontweight="semibold")
            input_axis.axvspan(
                0.4,
                0.6,
                facecolor=OI["orange"],
                edgecolor=OI["orange"],
                alpha=0.18,
                hatch="////",
                linewidth=0.8,
            )
            held_x = np.repeat(times, 2)[1:-1]
            held_y = np.repeat(disturbance, 2)
            input_axis.plot(
                held_x,
                held_y,
                color=color,
                linestyle="-" if column == 0 else "--",
                linewidth=1.6,
            )
            sampled_values = np.r_[disturbance, 0.0]
            input_axis.plot(
                times,
                sampled_values,
                linestyle="none",
                marker=marker,
                markersize=4.3,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.0,
            )
            input_axis.set_xlim(0.0, 1.0)
            input_axis.set_ylim(-0.12, 1.82)
            input_axis.set_yticks((0.0, 1.5))
            input_axis.set_ylabel(r"$\xi_k$" if column == 0 else "")
            input_axis.tick_params(axis="x", labelbottom=False, length=0)
            input_axis.spines[["top", "right"]].set_visible(False)
            input_axis.text(
                0.50,
                1.55,
                r"true pulse $\xi(t)$",
                color=OI["orange"],
                ha="center",
                va="bottom",
                fontsize=8,
            )

            state_axis.plot(time, truth, color=OI["black"], linewidth=1.8)
            if column == 0:
                state_axis.plot(
                    times,
                    velocity,
                    color=color,
                    linewidth=1.0,
                    marker=marker,
                    markersize=3.8,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                )
                state_axis.text(
                    0.98,
                    0.51,
                    "fine nodes",
                    color=color,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                )
                state_axis.text(
                    0.03,
                    0.59,
                    r"$F_{0.1}=0.9048v+0.09516(u+\xi)$",
                    fontsize=8,
                    va="top",
                )
                input_axis.text(
                    0.98,
                    0.18,
                    "two held intervals resolve it",
                    color=color,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                )
            else:
                state_axis.plot(
                    time,
                    no_pulse,
                    color=color,
                    linestyle="--",
                    linewidth=1.6,
                )
                state_axis.plot(
                    times,
                    velocity,
                    color=color,
                    linestyle="none",
                    marker=marker,
                    markersize=4.3,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                )
                state_axis.annotate(
                    "",
                    xy=(1.0, truth[-1]),
                    xytext=(1.0, coarse_velocity[-1]),
                    arrowprops={
                        "arrowstyle": "<->",
                        "color": OI["vermilion"],
                        "linewidth": 1.1,
                    },
                )
                state_axis.text(
                    0.97,
                    0.47,
                    "miss = 0.182 m/s",
                    color=OI["vermilion"],
                    ha="right",
                    va="center",
                    fontsize=8,
                )
                state_axis.text(
                    0.03,
                    0.59,
                    r"$F_{1}=0.3679v+0.6321(u+\xi)$",
                    fontsize=8,
                    va="top",
                )
                input_axis.text(
                    0.98,
                    0.18,
                    "left sample misses it",
                    color=color,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                )
                state_axis.text(
                    0.77,
                    no_pulse[np.searchsorted(time, 0.77)] - 0.025,
                    "coarse prediction",
                    color=color,
                    ha="center",
                    va="top",
                    fontsize=8,
                )

            state_axis.text(
                0.77,
                truth[np.searchsorted(time, 0.77)] + 0.018,
                "continuous truth",
                color=OI["black"],
                ha="center",
                va="bottom",
                fontsize=8,
            )
            state_axis.set_xlim(0.0, 1.07)
            state_axis.set_ylim(0.0, 0.64)
            state_axis.set_xlabel("time (s)")
            state_axis.set_ylabel("velocity (m/s)" if column == 0 else "")
            state_axis.spines[["top", "right"]].set_visible(False)

        return figure


def make_open_loop_feedback_figure() -> plt.Figure:
    """Contrast a fixed action schedule with a state-dependent rule."""

    time = np.arange(6, dtype=float)
    open_actions = np.zeros(5)
    feedback_actions = np.zeros(5)
    open_state = np.zeros(6)
    feedback_state = np.zeros(6)
    disturbance = np.zeros(5)
    disturbance[1] = 1.0

    for index in range(5):
        feedback_actions[index] = -0.6 * feedback_state[index]
        open_state[index + 1] = (
            0.8 * open_state[index] + open_actions[index] + disturbance[index]
        )
        feedback_state[index + 1] = (
            0.8 * feedback_state[index]
            + feedback_actions[index]
            + disturbance[index]
        )

    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(8.2, 3.55))
        grid = figure.add_gridspec(
            2,
            2,
            height_ratios=(2.1, 1.0),
            left=0.075,
            right=0.985,
            bottom=0.16,
            top=0.91,
            hspace=0.12,
            wspace=0.18,
        )
        state_axes = [
            figure.add_subplot(grid[0, 0]),
            figure.add_subplot(grid[0, 1]),
        ]
        action_axes = [
            figure.add_subplot(grid[1, 0]),
            figure.add_subplot(grid[1, 1]),
        ]

        panels = (
            (
                "Open-loop plan",
                open_state,
                open_actions,
                OI["vermilion"],
                r"$u_0,\ldots,u_4$ fixed at $k=0$",
            ),
            (
                "Feedback policy",
                feedback_state,
                feedback_actions,
                OI["blue"],
                r"$u_k=-0.6x_k$ evaluated after observing $x_k$",
            ),
        )
        for column, (title, state, actions, color, subtitle) in enumerate(panels):
            state_axis = state_axes[column]
            action_axis = action_axes[column]
            state_axis.set_title(title, fontweight="semibold", pad=22)
            state_axis.text(
                0.5,
                1.06,
                subtitle,
                transform=state_axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.30",
            )
            state_axis.axhline(0.0, color="0.72", linewidth=0.8)
            state_axis.plot(
                time,
                state,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=4.2,
                markerfacecolor="white",
                markeredgewidth=1.2,
            )
            state_axis.annotate(
                "same unexpected\ndisturbance",
                xy=(2.0, 1.0),
                xytext=(1.12, 1.35),
                arrowprops={
                    "arrowstyle": "->",
                    "color": OI["orange"],
                    "linewidth": 1.1,
                },
                color=OI["orange"],
                fontsize=8,
                ha="center",
            )
            state_axis.set_xlim(-0.15, 5.15)
            state_axis.set_ylim(-0.12, 1.52)
            state_axis.set_xticks(time)
            state_axis.tick_params(axis="x", labelbottom=False)
            state_axis.set_ylabel(r"state $x_k$" if column == 0 else "")
            state_axis.spines[["top", "right"]].set_visible(False)

            if column == 0:
                state_axis.annotate(
                    "fixed actions cannot react",
                    xy=(4.0, state[4]),
                    xytext=(3.95, 1.12),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "linewidth": 1.0,
                    },
                    color=color,
                    fontsize=8,
                    ha="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
                )
            else:
                state_axis.annotate(
                    "later actions reduce the deviation",
                    xy=(3.0, state[3]),
                    xytext=(3.75, 0.64),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "linewidth": 1.0,
                    },
                    color=color,
                    fontsize=8,
                    ha="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
                )

            action_axis.axhline(0.0, color="0.45", linewidth=0.8)
            action_axis.bar(
                time[:-1],
                actions,
                width=0.46,
                color=color,
                alpha=0.82,
                edgecolor=color,
                linewidth=0.8,
            )
            action_axis.plot(
                time[:-1],
                actions,
                linestyle="none",
                marker="o",
                markersize=3.5,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.0,
            )
            action_axis.set_xlim(-0.15, 5.15)
            action_axis.set_ylim(-0.68, 0.16)
            action_axis.set_xticks(time)
            action_axis.set_xlabel("decision time $k$")
            action_axis.set_ylabel(r"action $u_k$" if column == 0 else "")
            action_axis.spines[["top", "right"]].set_visible(False)

        return figure


def make_swing_coordinates_figure() -> plt.Figure:
    """Draw the aggregate coordinates used in the reduced swing equation."""

    theta = np.deg2rad(20.0)
    alpha = np.deg2rad(13.0)
    beta = np.deg2rad(29.0)
    psi = theta + alpha
    pivot = np.array([0.0, 1.45])
    seat = pivot + 1.75 * np.array([np.sin(theta), -np.cos(theta)])
    center = pivot + 1.38 * np.array([np.sin(psi), -np.cos(psi)])
    body_direction = np.array(
        [np.sin(theta + beta), -np.cos(theta + beta)]
    )

    with mpl.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(8.2, 3.65))
        figure.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.94)
        axis.set_xlim(-1.05, 3.2)
        axis.set_ylim(-0.55, 1.78)
        axis.set_aspect("equal")
        axis.axis("off")

        axis.plot(
            [pivot[0], pivot[0]],
            [pivot[1], -0.35],
            color="0.62",
            linewidth=1.0,
            linestyle=(0, (3, 3)),
        )
        axis.text(-0.08, -0.31, "straight down", color="0.42", ha="right")
        axis.plot(
            [pivot[0], seat[0]],
            [pivot[1], seat[1]],
            color=OI["blue"],
            linewidth=2.4,
        )
        axis.plot(
            [pivot[0], center[0]],
            [pivot[1], center[1]],
            color=OI["orange"],
            linewidth=1.4,
            linestyle=(0, (4, 2)),
        )
        axis.add_patch(Circle(pivot, 0.055, color=OI["black"], zorder=5))
        axis.plot(
            [seat[0] - 0.19, seat[0] + 0.19],
            [seat[1], seat[1]],
            color=OI["blue"],
            linewidth=3.0,
        )
        axis.add_patch(
            Circle(
                center,
                0.095,
                facecolor=OI["orange"],
                edgecolor="white",
                linewidth=1.0,
                zorder=6,
            )
        )

        half_body = 0.43
        body_start = center - half_body * body_direction
        body_end = center + half_body * body_direction
        axis.plot(
            [body_start[0], body_end[0]],
            [body_start[1], body_end[1]],
            color=OI["vermilion"],
            linewidth=5.2,
            solid_capstyle="round",
            zorder=4,
        )
        head = body_end + 0.08 * body_direction
        axis.add_patch(Circle(head, 0.105, color=OI["vermilion"], zorder=5))

        local_span = 0.58
        suspension_direction = np.array([np.sin(theta), -np.cos(theta)])
        local_start = center - 0.18 * suspension_direction
        local_end = center + local_span * suspension_direction
        axis.plot(
            [local_start[0], local_end[0]],
            [local_start[1], local_end[1]],
            color="0.62",
            linewidth=0.9,
            linestyle=(0, (3, 3)),
        )

        axis.add_patch(
            Arc(
                pivot,
                0.62,
                0.62,
                theta1=-90,
                theta2=-70,
                color=OI["blue"],
                linewidth=1.5,
            )
        )
        axis.text(0.075, 1.12, r"$\theta$", color=OI["blue"], fontsize=11)
        axis.add_patch(
            Arc(
                pivot,
                1.05,
                1.05,
                theta1=-70,
                theta2=-57,
                color=OI["orange"],
                linewidth=1.5,
            )
        )
        axis.text(0.28, 1.04, r"$\alpha$", color=OI["orange"], fontsize=11)
        axis.add_patch(
            Arc(
                pivot,
                1.42,
                1.42,
                theta1=-90,
                theta2=-57,
                color=OI["green"],
                linewidth=1.2,
                linestyle=(0, (3, 2)),
            )
        )
        axis.text(0.30, 0.77, r"$\psi=\theta+\alpha$", color=OI["green"], fontsize=10)

        beta_center = center
        axis.add_patch(
            Arc(
                beta_center,
                0.68,
                0.68,
                theta1=-70,
                theta2=-41,
                color=OI["vermilion"],
                linewidth=1.5,
            )
        )
        axis.text(
            center[0] + 0.27,
            center[1] - 0.18,
            r"$\beta$",
            color=OI["vermilion"],
            fontsize=11,
        )

        midpoint = 0.54 * pivot + 0.46 * center
        axis.text(
            midpoint[0] + 0.08,
            midpoint[1] - 0.02,
            r"$\rho$",
            color=OI["orange"],
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        )
        axis.text(pivot[0] - 0.09, pivot[1] + 0.11, "pivot", ha="right")
        axis.text(seat[0] + 0.24, seat[1] - 0.02, "seat", color=OI["blue"])
        axis.text(
            center[0] + 0.10,
            center[1] + 0.12,
            "center of mass",
            color=OI["orange"],
        )

        note = (
            r"configuration: $\theta$"
            "\n"
            r"shape: $\rho,\ \alpha,\ \beta$"
            "\n"
            r"center-of-mass angle: $\psi=\theta+\alpha$"
        )
        axis.text(
            2.05,
            1.12,
            note,
            ha="left",
            va="center",
            linespacing=1.65,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#F4F4F2",
                "edgecolor": "0.72",
                "linewidth": 0.8,
            },
        )
        axis.text(
            2.05,
            0.24,
            "A squat changes the center-of-mass distance.\n"
            "A lean changes the offset and body orientation.",
            ha="left",
            va="center",
            fontsize=8.5,
            color="0.28",
            linespacing=1.45,
        )
        return figure


def make_inference_boundary_figure() -> plt.Figure:
    """Show what lies inside and outside the one-GPU teaching model."""

    with mpl.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(8.2, 4.0))
        figure.subplots_adjust(left=0.02, right=0.985, bottom=0.04, top=0.97)
        axis.set_xlim(0.0, 10.0)
        axis.set_ylim(0.0, 6.2)
        axis.axis("off")

        boundary = FancyBboxPatch(
            (1.85, 0.62),
            6.30,
            4.32,
            boxstyle="round,pad=0.10,rounding_size=0.10",
            facecolor="#F7F7F5",
            edgecolor=OI["blue"],
            linewidth=1.7,
        )
        axis.add_patch(boundary)
        axis.text(
            5.0,
            4.70,
            "system boundary: one serving process and one GPU",
            color=OI["blue"],
            fontsize=10.2,
            fontweight="semibold",
            ha="center",
        )

        process = FancyBboxPatch(
            (2.18, 0.98),
            2.35,
            3.28,
            boxstyle="round,pad=0.10",
            facecolor="white",
            edgecolor="0.66",
            linewidth=1.0,
        )
        gpu = FancyBboxPatch(
            (4.82, 0.98),
            2.98,
            3.28,
            boxstyle="round,pad=0.10",
            facecolor="#EEF6FA",
            edgecolor=OI["blue"],
            linewidth=1.1,
        )
        axis.add_patch(process)
        axis.add_patch(gpu)
        axis.text(3.35, 4.04, "serving process", ha="center", fontweight="semibold")
        axis.text(6.31, 4.04, "one GPU", ha="center", fontweight="semibold")

        inner_style = {
            "boxstyle": "round,pad=0.32",
            "facecolor": "#FFFFFF",
            "edgecolor": "0.72",
            "linewidth": 0.8,
        }
        axis.text(3.35, 3.35, "request queue", ha="center", va="center", bbox=inner_style)
        axis.text(
            3.35,
            2.48,
            "scheduler\nchooses prefill or decode",
            ha="center",
            va="center",
            linespacing=1.35,
            bbox=inner_style,
        )
        axis.text(
            3.35,
            1.48,
            "completion records\nand queue ages",
            ha="center",
            va="center",
            linespacing=1.35,
            bbox=inner_style,
        )

        axis.text(
            6.31,
            3.38,
            "language-model execution\nprefill  •  decode",
            ha="center",
            va="center",
            linespacing=1.4,
            bbox={**inner_style, "edgecolor": OI["blue"]},
        )
        axis.text(6.31, 2.47, "key–value cache", ha="center", va="center", bbox=inner_style)
        axis.text(
            6.31,
            1.48,
            "realized clock  •  power\ntemperature  •  utilization",
            ha="center",
            va="center",
            linespacing=1.35,
            bbox=inner_style,
        )

        def arrow(
            start: tuple[float, float],
            end: tuple[float, float],
            *,
            color: str = "0.34",
            connectionstyle: str = "arc3",
        ) -> None:
            axis.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.15,
                    color=color,
                    connectionstyle=connectionstyle,
                )
            )

        arrow((3.35, 3.08), (3.35, 2.80))
        arrow((4.24, 2.49), (5.00, 3.08), color=OI["blue"])
        arrow((5.00, 2.83), (4.24, 1.72), color=OI["blue"])
        arrow((6.31, 3.05), (6.31, 2.77))
        arrow((6.31, 2.19), (6.31, 1.81))

        controller_box = {
            "boxstyle": "round,pad=0.38",
            "facecolor": "#FFF6DF",
            "edgecolor": OI["orange"],
            "linewidth": 1.0,
        }
        axis.text(
            5.0,
            5.86,
            "controller",
            ha="center",
            va="center",
            fontweight="semibold",
            bbox=controller_box,
        )
        arrow((4.72, 5.68), (3.62, 4.96), color=OI["orange"])
        arrow((6.38, 4.96), (5.28, 5.68), color=OI["green"])
        axis.text(
            3.55,
            5.62,
            "action $u_t$\nrequested clock + scheduling rule",
            color=OI["orange"],
            ha="right",
            va="center",
            fontsize=8.2,
        )
        axis.text(
            6.46,
            5.63,
            "observation $y_t$\nqueues, tokens, cache, hardware telemetry",
            color=OI["green"],
            ha="left",
            va="center",
            fontsize=8.2,
        )

        axis.text(
            0.72,
            3.35,
            "arriving requests\narrival time + prompt length",
            ha="center",
            va="center",
            color=OI["vermilion"],
            fontsize=8.5,
        )
        arrow((1.42, 3.35), (2.18, 3.35), color=OI["vermilion"])
        axis.text(
            0.72,
            2.57,
            "future output length\nunknown until completion",
            ha="center",
            va="center",
            color="0.38",
            fontsize=8.0,
        )

        arrow((7.80, 3.38), (8.65, 3.38), color=OI["green"])
        axis.text(
            9.24,
            3.38,
            "generated\nresponses",
            ha="center",
            va="center",
            color=OI["green"],
            fontsize=8.5,
        )
        axis.text(
            6.30,
            0.26,
            "ambient thermal conditions",
            ha="center",
            va="center",
            color=OI["vermilion"],
            fontsize=8.5,
        )
        arrow((6.30, 0.45), (6.30, 0.98), color=OI["vermilion"])
        axis.text(
            9.22,
            1.10,
            "outside the model:\nother GPUs, network routing,\nand downstream applications",
            ha="center",
            va="center",
            color="0.42",
            fontsize=8.0,
            linespacing=1.35,
        )
        return figure


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    builders = {
        "overview": make_overview_figure,
        "sampling": make_sampling_figure,
        "open-loop-feedback": make_open_loop_feedback_figure,
        "swing-coordinates": make_swing_coordinates_figure,
        "inference-boundary": make_inference_boundary_figure,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument(
        "--figure",
        choices=tuple(builders),
        default="overview",
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
