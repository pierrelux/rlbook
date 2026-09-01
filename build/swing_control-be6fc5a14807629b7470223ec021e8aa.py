"""Structured SwingRL controller, evaluation, and visualization utilities.

The package deliberately stays thin: SwingRL supplies the physical models,
environment, controllers, rollout recorder, illustration, and animation.  This
module only fixes the comparison used in the book and gives it a small plotting
interface shared by MyST and command-line smoke tests.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from swing_rl.control import ArticulatedPumper
from swing_rl.envs import SwingEnv, make_chain_env
from swing_rl.jaxsim import rider_for
from swing_rl.physics import RewardParams, SwingParams
from swing_rl.physics.models import articulated_seated, articulated_standing
from swing_rl.viz import Rollout, SwingAnimation, record_episode


OI = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
}

FIGURE_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
}


def run_episode(
    model_factory: Callable[[SwingParams], Any],
    policy_factory: Callable[[float], ArticulatedPumper],
    seconds: float,
) -> Rollout:
    """Record one deterministic SwingRL rollout.

    The success angle is one full unwrapped rotation.  SwingRL's default task
    ends at the top of the circle, which is useful for other experiments but
    would not test the claim made in this chapter.
    """

    parameters = SwingParams()
    model = model_factory(parameters)
    environment = SwingEnv(
        swing=parameters,
        model=model,
        rider=rider_for(model, parameters),
        reward=RewardParams(success_angle=2.0 * np.pi),
        max_episode_steps=int(seconds / 0.02),
    )
    controller = policy_factory(parameters.natural_frequency)
    return record_episode(environment, controller, seed=0)


def run_chain_episode(
    policy_factory: Callable[[float], ArticulatedPumper],
    seconds: float,
) -> Rollout:
    """Record the standing controller on SwingRL's unilateral chain model."""

    parameters = SwingParams()
    environment = make_chain_env(
        body="articulated_standing",
        swing=parameters,
        reward=RewardParams(success_angle=2.0 * np.pi),
        max_episode_steps=int(seconds / 0.02),
    )
    controller = policy_factory(parameters.natural_frequency)
    return record_episode(environment, controller, seed=0)


def run_comparison() -> dict[str, Rollout]:
    """Run matched structured controllers on rod and chain suspensions."""

    return {
        "standing": run_episode(
            articulated_standing,
            ArticulatedPumper.standing,
            seconds=32.0,
        ),
        "seated": run_episode(
            articulated_seated,
            ArticulatedPumper.seated,
            seconds=60.0,
        ),
        "standing_chain": run_chain_episode(
            ArticulatedPumper.standing,
            seconds=32.0,
        ),
    }


def make_summary_figure(results: dict[str, Rollout]) -> plt.Figure:
    """Plot angle, energy, and the force required from the suspension."""

    with mpl.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(7.2, 5.8),
            constrained_layout=True,
        )
        specifications = (
            ("standing", "standing, rigid rod", OI["blue"], "-"),
            ("standing_chain", "standing, chain", OI["green"], "-."),
            ("seated", "seated, rigid rod", OI["vermilion"], "--"),
        )

        for name, label, color, style in specifications:
            rollout = results[name]
            axes[0].plot(
                rollout.times,
                np.rad2deg(rollout.thetas),
                color=color,
                linestyle=style,
                label=label,
            )
            axes[1].plot(
                rollout.times,
                rollout.energies,
                color=color,
                linestyle=style,
                label=name,
            )
            axes[2].plot(
                rollout.times,
                rollout.tensions,
                color=color,
                linestyle=style,
                label=name,
            )

        standing = results["standing"]
        rotation_level = 360.0 * float(np.sign(standing.thetas[-1]))
        axes[0].axhline(
            rotation_level,
            color="0.55",
            linewidth=0.8,
            linestyle=":",
        )
        axes[0].annotate(
            "one full rotation",
            xy=(standing.success_time, rotation_level),
            xytext=(-5, 10 if rotation_level < 0.0 else -13),
            textcoords="offset points",
            ha="right",
            color="0.35",
        )
        axes[0].set(ylabel="unwrapped angle (deg)", xlabel="time (s)")
        axes[0].legend(loc="upper left")

        axes[1].axhline(1.0, color="0.55", linewidth=0.8, linestyle=":")
        axes[1].annotate(
            "energy required to reach the top at rest",
            xy=(results["seated"].times[-1], 1.0),
            xytext=(-5, 4),
            textcoords="offset points",
            ha="right",
            color="0.35",
        )
        axes[1].set(ylabel="normalized energy", xlabel="time (s)")

        negative_indices = np.flatnonzero(standing.tensions < 0.0)
        axes[2].axhline(0.0, color="0.45", linewidth=0.8, linestyle=":")
        axes[2].fill_between(
            standing.times,
            standing.tensions,
            0.0,
            where=standing.tensions < 0.0,
            color=OI["vermilion"],
            alpha=0.13,
            linewidth=0.0,
        )
        if negative_indices.size:
            first = int(negative_indices[0])
            axes[2].annotate(
                "a chain would go slack",
                xy=(standing.times[first], standing.tensions[first]),
                xytext=(37.0, -250.0),
                textcoords="data",
                arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 0.8},
                color="0.25",
            )
        axes[2].set(ylabel="required suspension force (N)", xlabel="time (s)")

        for axis in axes:
            axis.grid(axis="y", color="0.90", linewidth=0.6)

        return figure


def make_animation(
    rollout: Rollout,
    *,
    stride: int = 20,
    fps: int = 16,
) -> tuple[FuncAnimation, SwingAnimation]:
    """Create SwingRL's actual Matplotlib illustration and animation."""

    view = SwingAnimation(
        rollout,
        stride=stride,
        figsize=(5.6, 5.6),
        show_diagnostics=False,
    )
    return view.animation(fps=fps), view


def format_metrics(results: dict[str, Rollout]) -> str:
    """Return the outcome values used in the accompanying prose."""

    lines = []
    for name in ("standing", "standing_chain", "seated"):
        rollout = results[name]
        outcome = (
            f"rotation at {rollout.success_time:.2f} s"
            if rollout.success
            else "no rotation"
        )
        lines.append(
            f"{name:<16} {outcome:<20} "
            f"peak={np.rad2deg(np.abs(rollout.thetas)).max():.2f} deg "
            f"min_force={rollout.tensions.min():.0f} N"
        )
    return "\n".join(lines)


def _smoke_test() -> None:
    results = run_comparison()
    standing = results["standing"]
    seated = results["seated"]
    chain = results["standing_chain"]

    if not standing.success or standing.success_time is None:
        raise AssertionError("standing baseline should complete one rotation")
    if seated.success:
        raise AssertionError("seated baseline should not complete one rotation")
    if chain.success:
        raise AssertionError("standing chain baseline should not complete one rotation")
    if standing.tensions.min() >= 0.0:
        raise AssertionError("standing rod trajectory should expose negative force")
    if seated.tensions.min() <= 0.0:
        raise AssertionError("seated trajectory should keep the chain taut")
    if chain.tensions.min() < -1e-8:
        raise AssertionError("unilateral chain model cannot sustain negative tension")

    figure = make_summary_figure(results)
    plt.close(figure)
    animation, view = make_animation(standing, stride=100)
    animation._draw_was_started = True
    plt.close(view.fig)
    print(format_metrics(results))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="save the trajectory figure")
    arguments = parser.parse_args()

    comparison = run_comparison()
    _smoke_test()
    if arguments.save is not None:
        output = make_summary_figure(comparison)
        arguments.save.parent.mkdir(parents=True, exist_ok=True)
        output.savefig(arguments.save, bbox_inches="tight", dpi=200)
        plt.close(output)
