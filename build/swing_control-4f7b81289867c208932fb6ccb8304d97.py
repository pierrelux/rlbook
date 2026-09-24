"""Structured SwingRL controller, evaluation, and visualization utilities.

The package deliberately stays thin: SwingRL supplies the physical models,
environment, controllers, rollout recorder, illustration, and animation.  This
module only fixes the comparison used in the book and gives it a small plotting
interface shared by MyST and command-line smoke tests.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import html
from importlib.metadata import distribution, version
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

from swing_rl.control import ArticulatedPumper
from swing_rl.envs import SwingEnv, make_chain_env
from swing_rl.jaxsim import rider_for
from swing_rl.physics import IntegratorParams, RewardParams, SwingParams
from swing_rl.physics.models import articulated_seated, articulated_standing
from swing_rl.viz import Rollout, SwingAnimation, record_episode
from swing_rl.viz.animate import (
    MARKER_STYLE,
    POLYLINE_STYLE,
    draw_swing_frame,
)


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


BOOK = {
    "paper": "#F6F7F4",
    "raised": "#FFFFFF",
    "ink": "#1B2430",
    "muted": "#5C6874",
    "rule": "#D2D9D7",
    "structure": "#2F6F8F",
    "stands": "#2E7D5B",
    "caveat": "#B8860B",
    "withdrawn": "#A83A32",
}

BOOK_FIGURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": BOOK["rule"],
    "axes.labelcolor": BOOK["ink"],
    "xtick.color": BOOK["muted"],
    "ytick.color": BOOK["muted"],
    "text.color": BOOK["ink"],
    "figure.facecolor": BOOK["paper"],
    "axes.facecolor": BOOK["paper"],
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.facecolor": BOOK["paper"],
    "savefig.bbox": "tight",
}


@dataclass(frozen=True)
class SwingScenario:
    """The matched model-audit scenario used in the modeling chapter."""

    horizon_seconds: float = 32.0
    control_interval: float = 0.02
    integrator_substeps: int = 8
    success_angle: float = 2.0 * math.pi
    success_radius_fraction: float = 0.9
    initial_angle: float = 0.0
    initial_angular_velocity: float = 0.0
    seed: int = 0
    command_ramp_seconds: float = 0.25

    @property
    def maximum_steps(self) -> int:
        return int(round(self.horizon_seconds / self.control_interval))


DEFAULT_SWING_SCENARIO = SwingScenario()

_STANDING_PRESET = ArticulatedPumper.standing(1.0)
STANDING_PUMP_CHANNELS = tuple(_STANDING_PRESET.channels)
STANDING_STARTUP_AMPLITUDE = float(_STANDING_PRESET.startup_amplitude)


def structured_standing_actions(
    observations: Any,
    time_step: Any,
    natural_frequency: float,
    scenario: SwingScenario = DEFAULT_SWING_SCENARIO,
    *,
    array_module: Any = np,
) -> Any:
    """Evaluate SwingRL's standing phase law with the shared startup ramp.

    ``array_module`` may be NumPy or ``jax.numpy``.  The channel harmonics,
    phases, and startup threshold come directly from SwingRL's
    :meth:`ArticulatedPumper.standing` preset.  This functional form lets PPO
    evaluate the same controller over a batch without maintaining Python
    controller objects inside a JAX scan.
    """

    xp = array_module
    values = xp.asarray(observations)
    psi = xp.arctan2(values[..., 4], values[..., 3])
    psi_dot = values[..., 5] * natural_frequency
    cosine_amplitude = xp.cos(psi) - xp.square(psi_dot) / (
        2.0 * natural_frequency**2
    )
    amplitude = xp.arccos(xp.clip(cosine_amplitude, -1.0, 1.0))
    phase = xp.arctan2(-psi_dot / natural_frequency, psi)
    elapsed = (time_step + 1) * scenario.control_interval
    clock = natural_frequency * elapsed
    source = xp.where(amplitude < STANDING_STARTUP_AMPLITUDE, clock, phase)
    actions = xp.stack(
        [
            gain * xp.sin(harmonic * source + phase_offset)
            for harmonic, phase_offset, gain in STANDING_PUMP_CHANNELS
        ],
        axis=-1,
    )
    if scenario.command_ramp_seconds > 0.0:
        envelope = xp.minimum(elapsed / scenario.command_ramp_seconds, 1.0)
        actions = actions * envelope
    return xp.clip(actions, -1.0, 1.0)


@dataclass
class RampedStandingController:
    """SwingRL's standing pumper with a short ramp from the neutral pose.

    The phase law remains SwingRL's :class:`ArticulatedPumper`.  The common
    ramp prevents an instantaneous initial target change from becoming the
    first chain-release event in the teaching comparison.
    """

    natural_frequency: float
    control_interval: float = 0.02
    ramp_seconds: float = 0.25
    _controller: ArticulatedPumper = field(init=False, repr=False)
    _elapsed: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        self._controller = ArticulatedPumper.standing(
            self.natural_frequency,
            dt=self.control_interval,
        )

    def reset(self) -> None:
        self._controller.reset()
        self._elapsed = 0.0

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        action = np.asarray(self._controller(observation), dtype=np.float32)
        self._elapsed += self.control_interval
        if self.ramp_seconds <= 0.0:
            return action
        scale = min(self._elapsed / self.ramp_seconds, 1.0)
        return np.asarray(scale * action, dtype=np.float32)


@dataclass(frozen=True)
class SwingTrace:
    """A SwingRL rollout with the diagnostics needed for a model audit."""

    name: str
    rollout: Rollout
    target_actions: np.ndarray
    realized_commands: np.ndarray
    seat_radius: np.ndarray
    taut: np.ndarray
    slack_time: np.ndarray
    snap_energy_loss: np.ndarray

    @property
    def times(self) -> np.ndarray:
        return self.rollout.times

    @property
    def angles(self) -> np.ndarray:
        return self.rollout.thetas

    @property
    def energies(self) -> np.ndarray:
        return self.rollout.energies

    @property
    def tensions(self) -> np.ndarray:
        return self.rollout.tensions


def make_environment(
    scenario: SwingScenario = DEFAULT_SWING_SCENARIO,
    *,
    suspension: str,
    reset_noise: float = 0.0,
) -> tuple[SwingParams, SwingEnv]:
    """Construct a matched rigid-rod or unilateral-chain SwingRL plant."""

    parameters = SwingParams()
    integrator = IntegratorParams(
        dt=scenario.control_interval,
        substeps=scenario.integrator_substeps,
    )
    reward = RewardParams(
        success_angle=scenario.success_angle,
        success_radius_fraction=scenario.success_radius_fraction,
    )
    common = {
        "swing": parameters,
        "integrator": integrator,
        "reward": reward,
        "max_episode_steps": scenario.maximum_steps,
        "reset_noise": reset_noise,
    }
    if suspension == "rigid_rod":
        model = articulated_standing(parameters)
        environment = SwingEnv(
            model=model,
            rider=rider_for(model, parameters),
            **common,
        )
    elif suspension == "unilateral_chain":
        environment = make_chain_env(
            body="articulated_standing",
            **common,
        )
    else:
        raise ValueError(
            "suspension must be 'rigid_rod' or 'unilateral_chain'"
        )
    return parameters, environment


def make_structured_controller(
    parameters: SwingParams,
    scenario: SwingScenario = DEFAULT_SWING_SCENARIO,
) -> RampedStandingController:
    """Return the shared feedback controller for the model audit."""

    return RampedStandingController(
        natural_frequency=parameters.natural_frequency,
        control_interval=scenario.control_interval,
        ramp_seconds=scenario.command_ramp_seconds,
    )


def _hybrid_diagnostics(environment: SwingEnv) -> tuple[bool, float, float]:
    rendered = environment.render_state()
    taut = bool(rendered.get("taut", True))
    state = environment._state
    slack = float(getattr(state, "slack_time", 0.0))
    loss = float(getattr(state, "energy_lost", 0.0))
    return taut, slack, loss


def simulate_controller(
    environment: SwingEnv,
    controller: Callable[[np.ndarray], np.ndarray],
    scenario: SwingScenario = DEFAULT_SWING_SCENARIO,
    *,
    name: str,
) -> SwingTrace:
    """Run one deterministic closed-loop episode and retain audit diagnostics."""

    observation, _ = environment.reset(
        seed=scenario.seed,
        options={
            "theta": scenario.initial_angle,
            "theta_dot": scenario.initial_angular_velocity,
            "noise": 0.0,
        },
    )
    if hasattr(controller, "reset"):
        controller.reset()

    rollout = Rollout(length=environment.model.nominal_length)
    rollout.append(environment)
    zero_action = np.zeros(environment.action_space.shape, dtype=np.float32)
    target_actions = [zero_action]
    realized_commands = [rollout.frames[-1].commands.copy()]
    seat_radius = [float(np.linalg.norm(rollout.frames[-1].seat))]
    taut, slack, loss = _hybrid_diagnostics(environment)
    taut_values = [taut]
    slack_values = [slack]
    loss_values = [loss]

    for _ in range(scenario.maximum_steps):
        action = np.asarray(controller(observation), dtype=np.float32)
        observation, _, terminated, truncated, info = environment.step(action)
        rollout.append(environment)
        target_actions.append(action.copy())
        realized_commands.append(rollout.frames[-1].commands.copy())
        seat_radius.append(float(np.linalg.norm(rollout.frames[-1].seat)))
        taut, slack, loss = _hybrid_diagnostics(environment)
        taut_values.append(taut)
        slack_values.append(slack)
        loss_values.append(loss)
        if terminated:
            rollout.success = True
            rollout.success_time = float(info["time"])
            break
        if truncated:
            break

    return SwingTrace(
        name=name,
        rollout=rollout,
        target_actions=np.asarray(target_actions),
        realized_commands=np.asarray(realized_commands),
        seat_radius=np.asarray(seat_radius),
        taut=np.asarray(taut_values, dtype=bool),
        slack_time=np.asarray(slack_values),
        snap_energy_loss=np.asarray(loss_values),
    )


def run_model_audit(
    scenario: SwingScenario = DEFAULT_SWING_SCENARIO,
) -> dict[str, SwingTrace]:
    """Apply the same structured feedback law to the two suspension models."""

    traces: dict[str, SwingTrace] = {}
    for suspension in ("rigid_rod", "unilateral_chain"):
        parameters, environment = make_environment(
            scenario,
            suspension=suspension,
        )
        controller = make_structured_controller(parameters, scenario)
        traces[suspension] = simulate_controller(
            environment,
            controller,
            scenario,
            name=suspension,
        )
    return traces


def mode_events(trace: SwingTrace) -> list[dict[str, Any]]:
    """Return ordered release and reattachment events from a hybrid trace."""

    indices = np.flatnonzero(trace.taut[1:] != trace.taut[:-1]) + 1
    return [
        {
            "kind": "reattachment" if trace.taut[index] else "release",
            "time_seconds": float(trace.times[index]),
            "angle_degrees": float(np.rad2deg(trace.angles[index])),
            "seat_radius_fraction": float(
                trace.seat_radius[index] / trace.rollout.length
            ),
        }
        for index in indices
    ]


def compute_audit_metrics(
    trace: SwingTrace,
    scenario: SwingScenario = DEFAULT_SWING_SCENARIO,
) -> dict[str, Any]:
    """Compute physical outcome and feasibility metrics for one trace."""

    compression = np.maximum(-trace.tensions, 0.0)
    commands = trace.realized_commands
    command_motion = (
        float(np.linalg.norm(np.diff(commands, axis=0), axis=1).sum())
        if len(commands) > 1
        else 0.0
    )
    events = mode_events(trace)
    releases = [event for event in events if event["kind"] == "release"]
    return {
        "success": bool(trace.rollout.success),
        "time_to_rotation_seconds": (
            None
            if trace.rollout.success_time is None
            else float(trace.rollout.success_time)
        ),
        "peak_absolute_angle_degrees": float(
            np.rad2deg(np.abs(trace.angles)).max()
        ),
        "minimum_tension_newtons": float(trace.tensions.min()),
        "maximum_compression_demand_newtons": float(compression.max()),
        "negative_tension_fraction": float(np.mean(trace.tensions < 0.0)),
        "negative_tension_seconds": float(
            scenario.control_interval * np.count_nonzero(trace.tensions < 0.0)
        ),
        "minimum_seat_radius_fraction": float(
            trace.seat_radius.min() / trace.rollout.length
        ),
        "slack_time_seconds": float(trace.slack_time[-1]),
        "snap_energy_loss_joules": float(trace.snap_energy_loss[-1]),
        "release_count": len(releases),
        "first_release_seconds": (
            None if not releases else float(releases[0]["time_seconds"])
        ),
        "integrated_command_motion": command_motion,
        "sample_count": int(len(trace.times)),
    }


def audit_metrics(
    traces: dict[str, SwingTrace],
    scenario: SwingScenario = DEFAULT_SWING_SCENARIO,
) -> dict[str, dict[str, Any]]:
    return {
        name: compute_audit_metrics(trace, scenario)
        for name, trace in traces.items()
    }


def _frame_index_at_time(trace: SwingTrace, time_seconds: float) -> int:
    return min(
        int(np.searchsorted(trace.times, time_seconds, side="right") - 1),
        len(trace.times) - 1,
    )


def _slack_intervals(trace: SwingTrace) -> list[tuple[float, float]]:
    padded = np.r_[False, np.logical_not(trace.taut), False].astype(int)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    intervals = []
    for start, stop in zip(starts, stops):
        right = min(stop, len(trace.times) - 1)
        intervals.append((float(trace.times[start]), float(trace.times[right])))
    return intervals


def _prepare_scene(axis: plt.Axes, trace: SwingTrace, title: str) -> dict[str, Any]:
    length = trace.rollout.length
    radius = 1.43 * length
    axis.set(xlim=(-radius, radius), ylim=(-radius, radius))
    axis.set_aspect("equal")
    axis.axis("off")
    draw_swing_frame(axis, length)
    axis.add_patch(
        plt.Circle(
            (0.0, 0.0),
            length,
            fill=False,
            linestyle=":",
            linewidth=0.7,
            edgecolor=BOOK["rule"],
            zorder=0,
        )
    )
    first = trace.rollout.frames[0]
    body_lines = [
        axis.plot([], [], **dict(POLYLINE_STYLE[style]))[0]
        for style in first.styles
    ]
    marker_dots = [
        axis.plot([], [], **dict(MARKER_STYLE[kind]))[0]
        for _, kind in first.markers
    ]
    seat_dot = axis.plot([], [], "o", ms=4, color=BOOK["ink"], zorder=7)[0]
    force_arrow = FancyArrowPatch(
        (0.0, 0.0),
        (0.0, 0.0),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.8,
        color=BOOK["stands"],
        zorder=8,
    )
    axis.add_patch(force_arrow)
    axis.text(
        0.5,
        1.02,
        title,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="semibold",
    )
    status = axis.text(
        0.02,
        0.98,
        "",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": BOOK["raised"],
            "edgecolor": BOOK["rule"],
            "alpha": 0.93,
        },
        zorder=10,
    )
    mode = axis.text(
        0.98,
        0.98,
        "",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        fontweight="semibold",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": BOOK["raised"],
            "edgecolor": BOOK["rule"],
        },
        zorder=10,
    )

    action_bars = []
    for row, label in enumerate(("squat", "lean")):
        y = 0.08 - 0.045 * row
        axis.text(
            0.02,
            y + 0.012,
            label,
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=6.5,
            color=BOOK["muted"],
        )
        axis.add_patch(
            Rectangle(
                (0.14, y),
                0.25,
                0.024,
                transform=axis.transAxes,
                facecolor=BOOK["raised"],
                edgecolor=BOOK["rule"],
                linewidth=0.7,
                zorder=9,
            )
        )
        fill = Rectangle(
            (0.14, y),
            0.125,
            0.024,
            transform=axis.transAxes,
            facecolor=BOOK["structure"],
            edgecolor="none",
            zorder=9,
        )
        axis.add_patch(fill)
        action_bars.append(fill)
    return {
        "body_lines": body_lines,
        "marker_dots": marker_dots,
        "seat_dot": seat_dot,
        "force_arrow": force_arrow,
        "status": status,
        "mode": mode,
        "action_bars": action_bars,
    }


def _update_scene(
    artists: dict[str, Any],
    trace: SwingTrace,
    index: int,
    *,
    is_chain: bool,
) -> None:
    frame = trace.rollout.frames[index]
    for line, segment in zip(artists["body_lines"], frame.polylines):
        line.set_data(segment[:, 0], segment[:, 1])
    for dot, (position, _) in zip(artists["marker_dots"], frame.markers):
        dot.set_data([position[0]], [position[1]])
    artists["seat_dot"].set_data([frame.seat[0]], [frame.seat[1]])

    action = trace.target_actions[index]
    for fill, value in zip(artists["action_bars"], action):
        fill.set_width(0.25 * float(np.clip(value + 1.0, 0.0, 2.0)) / 2.0)

    taut = bool(trace.taut[index])
    tension = float(trace.tensions[index])
    seat = np.asarray(frame.seat, dtype=float)
    seat_norm = max(float(np.linalg.norm(seat)), 1e-12)
    outward = seat / seat_norm
    if is_chain and not taut:
        artists["force_arrow"].set_positions(tuple(seat), tuple(seat))
        force_text = "pull       0 N"
    else:
        direction = -outward if tension >= 0.0 else outward
        magnitude = min(abs(tension) / 900.0, 1.0)
        end = seat + (0.18 + 0.32 * magnitude) * trace.rollout.length * direction
        artists["force_arrow"].set_positions(tuple(seat), tuple(end))
        force_color = BOOK["stands"] if tension >= 0.0 else BOOK["withdrawn"]
        artists["force_arrow"].set_color(force_color)
        force_text = f"axial {tension:+7.0f} N"

    artists["status"].set_text(
        f"t       {frame.time:5.2f} s\n"
        f"angle {np.rad2deg(frame.theta):+7.1f} deg\n"
        f"{force_text}"
    )
    if is_chain:
        mode_text = "TAUT" if taut else "SLACK"
        mode_color = BOOK["stands"] if taut else BOOK["caveat"]
    else:
        mode_text = "BIDIRECTIONAL"
        mode_color = BOOK["withdrawn"] if tension < 0.0 else BOOK["structure"]
    artists["mode"].set_text(mode_text)
    artists["mode"].set_color(mode_color)
    artists["mode"].get_bbox_patch().set_edgecolor(mode_color)


def make_model_audit_figure(
    traces: dict[str, SwingTrace],
    scenario: SwingScenario = DEFAULT_SWING_SCENARIO,
) -> plt.Figure:
    """Build the renderer-based static fallback and complete audit traces."""

    rod = traces["rigid_rod"]
    chain = traces["unilateral_chain"]
    snapshot_index = int(np.argmin(chain.seat_radius))
    snapshot_time = float(chain.times[snapshot_index])
    rod_snapshot_index = _frame_index_at_time(rod, snapshot_time)
    rod_metrics = compute_audit_metrics(rod, scenario)
    chain_metrics = compute_audit_metrics(chain, scenario)

    with mpl.rc_context(BOOK_FIGURE_STYLE):
        figure = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
        grid = figure.add_gridspec(
            3,
            2,
            height_ratios=(1.55, 0.95, 0.85),
            hspace=0.08,
            wspace=0.12,
        )
        rod_axis = figure.add_subplot(grid[0, 0])
        chain_axis = figure.add_subplot(grid[0, 1])
        rod_artists = _prepare_scene(rod_axis, rod, "bidirectional rod")
        chain_artists = _prepare_scene(chain_axis, chain, "unilateral chain")
        _update_scene(
            rod_artists,
            rod,
            rod_snapshot_index,
            is_chain=False,
        )
        _update_scene(
            chain_artists,
            chain,
            snapshot_index,
            is_chain=True,
        )
        rod_axis.text(
            0.5,
            -0.02,
            f"matched feedback law at t = {snapshot_time:.2f} s",
            transform=rod_axis.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color=BOOK["muted"],
        )
        chain_axis.text(
            0.5,
            -0.02,
            f"extension = {chain.seat_radius[snapshot_index] / chain.rollout.length:.3f} L",
            transform=chain_axis.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color=BOOK["caveat"],
        )

        angle_axis = figure.add_subplot(grid[1, :])
        angle_axis.plot(
            rod.times,
            np.rad2deg(rod.angles),
            color=BOOK["structure"],
            linewidth=1.5,
            label="bidirectional rod",
        )
        angle_axis.plot(
            chain.times,
            np.rad2deg(chain.angles),
            color=BOOK["ink"],
            linewidth=1.35,
            linestyle="--",
            label="unilateral chain",
        )
        rotation_sign = float(np.sign(rod.angles[-1])) or -1.0
        angle_axis.axhline(
            360.0 * rotation_sign,
            color=BOOK["rule"],
            linewidth=0.9,
            linestyle=":",
        )
        first_release = chain_metrics["first_release_seconds"]
        if first_release is not None:
            angle_axis.axvline(
                first_release,
                color=BOOK["caveat"],
                linewidth=0.9,
                linestyle=":",
            )
            angle_axis.text(
                first_release + 0.25,
                angle_axis.get_ylim()[1] * 0.78,
                "chain releases",
                color=BOOK["caveat"],
                fontsize=7.5,
                va="top",
            )
        if rod.rollout.success_time is not None:
            angle_axis.axvline(
                rod.rollout.success_time,
                color=BOOK["withdrawn"],
                linewidth=0.9,
                linestyle=":",
            )
        angle_axis.set(
            xlim=(0.0, scenario.horizon_seconds),
            ylabel="unwrapped angle (deg)",
            xlabel="time (s)",
        )
        angle_axis.grid(axis="y", color=BOOK["rule"], linewidth=0.55)
        angle_axis.legend(loc="upper left", ncol=2)

        compression_axis = figure.add_subplot(grid[2, 0])
        compression = np.maximum(-rod.tensions, 0.0)
        compression_axis.fill_between(
            rod.times,
            0.0,
            compression,
            color=BOOK["withdrawn"],
            alpha=0.22,
            linewidth=0.0,
        )
        compression_axis.plot(
            rod.times,
            compression,
            color=BOOK["withdrawn"],
            linewidth=1.35,
        )
        compression_axis.set(
            xlim=(0.0, scenario.horizon_seconds),
            ylim=(0.0, 1.08 * max(compression.max(), 1.0)),
            ylabel="rod compression demand (N)",
            xlabel="time (s)",
        )
        compression_axis.text(
            0.98,
            0.90,
            f"peak {rod_metrics['maximum_compression_demand_newtons']:.0f} N",
            transform=compression_axis.transAxes,
            ha="right",
            va="top",
            color=BOOK["withdrawn"],
            fontsize=7.5,
        )
        compression_axis.grid(axis="y", color=BOOK["rule"], linewidth=0.55)

        extension_axis = figure.add_subplot(grid[2, 1])
        for start, stop in _slack_intervals(chain):
            extension_axis.axvspan(
                start,
                stop,
                color=BOOK["caveat"],
                alpha=0.15,
                linewidth=0.0,
            )
        extension = chain.seat_radius / chain.rollout.length
        extension_axis.plot(
            chain.times,
            extension,
            color=BOOK["ink"],
            linewidth=1.35,
        )
        extension_axis.axhline(
            1.0,
            color=BOOK["stands"],
            linestyle=":",
            linewidth=0.9,
        )
        extension_axis.set(
            xlim=(0.0, scenario.horizon_seconds),
            ylim=(0.34, 1.05),
            ylabel="chain extension r / L",
            xlabel="time (s)",
        )
        extension_axis.text(
            0.02,
            0.10,
            f"minimum {chain_metrics['minimum_seat_radius_fraction']:.3f} L",
            transform=extension_axis.transAxes,
            ha="left",
            va="bottom",
            color=BOOK["caveat"],
            fontsize=7.5,
        )
        extension_axis.grid(axis="y", color=BOOK["rule"], linewidth=0.55)
        return figure


class SwingModelAuditAnimation:
    """Time-compressed, renderer-based comparison of the matched plants."""

    def __init__(
        self,
        traces: dict[str, SwingTrace],
        scenario: SwingScenario = DEFAULT_SWING_SCENARIO,
        *,
        fps: int = 20,
        speed: float = 2.0,
        event_hold_seconds: float = 0.55,
    ) -> None:
        self.traces = traces
        self.scenario = scenario
        self.fps = int(fps)
        self.speed = float(speed)
        self.rod = traces["rigid_rod"]
        self.chain = traces["unilateral_chain"]

        step = self.speed / self.fps
        base_times = np.arange(0.0, scenario.horizon_seconds + 0.5 * step, step)
        event_times = [
            event["time_seconds"] for event in mode_events(self.chain)
        ]
        if self.rod.rollout.success_time is not None:
            event_times.append(float(self.rod.rollout.success_time))
        holds = max(1, int(round(event_hold_seconds * self.fps)))
        timeline: list[float] = []
        timeline_points = sorted([*base_times, *event_times])
        for time_value in timeline_points:
            timeline.append(float(time_value))
            if any(abs(time_value - event) < 1e-9 for event in event_times):
                timeline.extend([float(time_value)] * holds)
        self.timeline = np.asarray(timeline)

        with mpl.rc_context(BOOK_FIGURE_STYLE):
            self.figure = plt.figure(
                figsize=(10.0, 5.625),
                constrained_layout=True,
            )
            outer = self.figure.add_gridspec(
                2,
                2,
                height_ratios=(1.65, 1.0),
                hspace=0.04,
                wspace=0.08,
            )
            self.rod_axis = self.figure.add_subplot(outer[0, 0])
            self.chain_axis = self.figure.add_subplot(outer[0, 1])
            self.rod_artists = _prepare_scene(
                self.rod_axis,
                self.rod,
                "bidirectional rod",
            )
            self.chain_artists = _prepare_scene(
                self.chain_axis,
                self.chain,
                "unilateral chain",
            )
            self.angle_axis = self.figure.add_subplot(outer[1, 0])
            right = outer[1, 1].subgridspec(2, 1, hspace=0.12)
            self.compression_axis = self.figure.add_subplot(right[0, 0])
            self.extension_axis = self.figure.add_subplot(right[1, 0])
            self._setup_diagnostics()

    def _setup_diagnostics(self) -> None:
        rod_angle = np.rad2deg(self.rod.angles)
        chain_angle = np.rad2deg(self.chain.angles)
        angle_min = min(float(rod_angle.min()), float(chain_angle.min()), -380.0)
        angle_max = max(float(rod_angle.max()), float(chain_angle.max()), 160.0)
        self.angle_axis.set(
            xlim=(0.0, self.scenario.horizon_seconds),
            ylim=(angle_min - 12.0, angle_max + 12.0),
            xlabel="time (s)",
            ylabel="unwrapped angle (deg)",
        )
        self.angle_axis.axhline(
            -360.0,
            color=BOOK["rule"],
            linestyle=":",
            linewidth=0.8,
        )
        self.rod_angle_line = self.angle_axis.plot(
            [], [], color=BOOK["structure"], linewidth=1.5
        )[0]
        self.chain_angle_line = self.angle_axis.plot(
            [], [], color=BOOK["ink"], linewidth=1.3, linestyle="--"
        )[0]
        self.angle_axis.legend(
            handles=[
                Line2D([], [], color=BOOK["structure"], label="rod"),
                Line2D([], [], color=BOOK["ink"], linestyle="--", label="chain"),
            ],
            loc="upper left",
            ncol=2,
            fontsize=7,
        )

        compression = np.maximum(-self.rod.tensions, 0.0)
        self.compression_axis.set(
            xlim=(0.0, self.scenario.horizon_seconds),
            ylim=(0.0, 1.08 * max(float(compression.max()), 1.0)),
            ylabel="compression (N)",
        )
        self.compression_line = self.compression_axis.plot(
            [], [], color=BOOK["withdrawn"], linewidth=1.35
        )[0]
        self.compression_axis.tick_params(labelbottom=False)

        self.extension_axis.set(
            xlim=(0.0, self.scenario.horizon_seconds),
            ylim=(0.34, 1.05),
            xlabel="time (s)",
            ylabel="chain r / L",
        )
        self.extension_axis.axhline(
            1.0,
            color=BOOK["stands"],
            linestyle=":",
            linewidth=0.8,
        )
        self.extension_line = self.extension_axis.plot(
            [], [], color=BOOK["ink"], linewidth=1.3
        )[0]
        self.slack_line = self.extension_axis.plot(
            [], [], color=BOOK["caveat"], linewidth=4.0, solid_capstyle="butt"
        )[0]

        for axis in (
            self.angle_axis,
            self.compression_axis,
            self.extension_axis,
        ):
            axis.grid(axis="y", color=BOOK["rule"], linewidth=0.5)
            axis.tick_params(length=2.5)
        self.cursors = [
            axis.axvline(0.0, color=BOOK["muted"], linewidth=0.7)
            for axis in (
                self.angle_axis,
                self.compression_axis,
                self.extension_axis,
            )
        ]

    def _update(self, frame_number: int) -> list[Any]:
        time_value = float(self.timeline[frame_number])
        rod_index = _frame_index_at_time(self.rod, time_value)
        chain_index = _frame_index_at_time(self.chain, time_value)
        _update_scene(
            self.rod_artists,
            self.rod,
            rod_index,
            is_chain=False,
        )
        _update_scene(
            self.chain_artists,
            self.chain,
            chain_index,
            is_chain=True,
        )

        rod_slice = slice(0, rod_index + 1)
        chain_slice = slice(0, chain_index + 1)
        self.rod_angle_line.set_data(
            self.rod.times[rod_slice],
            np.rad2deg(self.rod.angles[rod_slice]),
        )
        self.chain_angle_line.set_data(
            self.chain.times[chain_slice],
            np.rad2deg(self.chain.angles[chain_slice]),
        )
        self.compression_line.set_data(
            self.rod.times[rod_slice],
            np.maximum(-self.rod.tensions[rod_slice], 0.0),
        )
        extension = self.chain.seat_radius[chain_slice] / self.chain.rollout.length
        self.extension_line.set_data(self.chain.times[chain_slice], extension)
        slack = np.where(
            np.logical_not(self.chain.taut[chain_slice]),
            0.37,
            np.nan,
        )
        self.slack_line.set_data(self.chain.times[chain_slice], slack)
        for cursor in self.cursors:
            cursor.set_xdata([time_value, time_value])
        return []

    def animation(self) -> FuncAnimation:
        return FuncAnimation(
            self.figure,
            self._update,
            frames=len(self.timeline),
            interval=1000.0 / self.fps,
            repeat=False,
            blit=False,
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegWriter(
            fps=self.fps,
            codec="libx264",
            bitrate=2200,
            extra_args=[
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
            metadata={
                "title": "SwingRL model audit",
                "artist": "Building Up RL",
            },
        )
        animation = self.animation()
        animation.save(path, writer=writer, dpi=128)
        plt.close(self.figure)
        return path


def make_model_audit_animation(
    traces: dict[str, SwingTrace],
    scenario: SwingScenario = DEFAULT_SWING_SCENARIO,
    *,
    fps: int = 20,
    speed: float = 2.0,
) -> SwingModelAuditAnimation:
    return SwingModelAuditAnimation(
        traces,
        scenario,
        fps=fps,
        speed=speed,
    )


def model_audit_player_html(
    output_directory: Path,
    events_path: Path,
    *,
    player_id: str = "swing-model-audit-player",
    fallback_id: str = "fig-swing-model-audit-fallback",
) -> str:
    """Embed the immutable audit movie with accessible event seek buttons.

    The browser code changes only the playback position.  Simulation, event
    detection, plots, and the SwingRL rendering have already run in Python.
    The separate MyST fallback is hidden only after video metadata loads.
    """

    video_path = output_directory / "model_audit.mp4"
    poster_path = output_directory / "model_audit_poster.png"
    if not poster_path.exists():
        poster_path = output_directory / "model_audit.png"
    for path in (video_path, poster_path, events_path):
        if not path.exists():
            raise FileNotFoundError(f"SwingRL audit asset is missing: {path}")

    events = json.loads(events_path.read_text(encoding="utf-8"))
    if not isinstance(events, list) or not events:
        raise ValueError("SwingRL replay events must be a nonempty list")
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("each SwingRL replay event must be an object")
        if "label" not in event or "video_seconds" not in event:
            raise ValueError("each replay event needs a label and video_seconds")
        if not math.isfinite(float(event["video_seconds"])):
            raise ValueError("replay event times must be finite")

    video_uri = (
        "data:video/mp4;base64,"
        + base64.b64encode(video_path.read_bytes()).decode("ascii")
    )
    poster_uri = (
        "data:image/png;base64,"
        + base64.b64encode(poster_path.read_bytes()).decode("ascii")
    )
    buttons = "".join(
        f'<button type="button" disabled data-seek="{float(event["video_seconds"]):.6f}" '
        f'data-description="{html.escape(str(event["label"]), quote=True)} at '
        f'{float(event["time_seconds"]):.2f} simulated seconds">'
        f'{html.escape(str(event["label"]))}<span>'
        f'{float(event["time_seconds"]):.2f} s</span></button>'
        for event in events
    )
    fallback_json = json.dumps(str(fallback_id))
    return f"""
<section id="{html.escape(player_id, quote=True)}" class="swing-audit-player"
         aria-label="Recorded SwingRL model audit">
  <style>
    #{player_id} {{ color:#1B2430; background:#F6F7F4; border:1px solid #D2D9D7;
      border-radius:10px; padding:12px; font-family:"IBM Plex Sans",sans-serif; }}
    #{player_id} video {{ display:block; width:100%; background:#F6F7F4; }}
    #{player_id} .event-controls {{ margin-top:10px; }}
    #{player_id} .event-controls p {{ margin:0 0 7px; color:#5C6874; font-size:.86rem; }}
    #{player_id} .event-buttons {{ display:flex; flex-wrap:wrap; gap:7px; }}
    #{player_id} button {{ display:flex; gap:7px; align-items:baseline; border:1px solid #2F6F8F;
      border-radius:999px; padding:6px 10px; color:#1B2430; background:#fff; cursor:pointer;
      font:600 .78rem "IBM Plex Sans",sans-serif; }}
    #{player_id} button span {{ color:#5C6874; font:500 .76rem "IBM Plex Mono",monospace; }}
    #{player_id} button:hover {{ background:#e8f0f3; }}
    #{player_id} button:focus-visible {{ outline:3px solid #B8860B; outline-offset:2px; }}
    #{player_id} button:disabled {{ opacity:.55; cursor:wait; }}
    #{player_id} output {{ display:block; min-height:1.25em; margin-top:7px; color:#2F6F8F;
      font:500 .78rem "IBM Plex Mono",monospace; }}
  </style>
  <video controls preload="metadata" playsinline poster="{poster_uri}">
    <source src="{video_uri}" type="video/mp4">
    The recorded audit is available as an MP4 download below.
  </video>
  <nav class="event-controls" aria-label="Jump to a recorded physical event">
    <p>Jump to a recorded event</p>
    <div class="event-buttons">{buttons}</div>
    <output aria-live="polite">Use the video controls or choose an event.</output>
  </nav>
  <script>
  (() => {{
    const root = document.getElementById({json.dumps(player_id)});
    if (!root) return;
    const video = root.querySelector("video");
    const buttons = [...root.querySelectorAll("button[data-seek]")];
    const status = root.querySelector("output");
    const fallbackId = {fallback_json};
    const hideFallback = doc => {{
      if (!doc || !fallbackId) return false;
      const fallback = doc.getElementById(fallbackId);
      if (!fallback) return false;
      fallback.hidden = true;
      fallback.setAttribute("aria-hidden", "true");
      return true;
    }};
    const ready = () => {{
      buttons.forEach(button => {{ button.disabled = false; }});
      hideFallback(document);
      try {{ if (window.parent && window.parent !== window) hideFallback(window.parent.document); }}
      catch (_) {{ /* A cross-origin parent keeps the local fallback. */ }}
    }};
    buttons.forEach(button => button.addEventListener("click", () => {{
      video.pause();
      video.currentTime = Number(button.dataset.seek);
      status.textContent = button.dataset.description;
    }}));
    video.addEventListener("loadedmetadata", ready, {{ once:true }});
    if (video.readyState >= 1) ready();
  }})();
  </script>
</section>
"""


def _package_provenance(package_name: str) -> dict[str, Any]:
    """Return installed-version and direct-VCS metadata when available."""

    metadata: dict[str, Any] = {"version": version(package_name)}
    direct_url = distribution(package_name).read_text("direct_url.json")
    if direct_url:
        metadata["direct_url"] = json.loads(direct_url)
    return metadata


def _repository_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _write_trace_archive(path: Path, trace: SwingTrace) -> None:
    frames = trace.rollout.frames
    np.savez_compressed(
        path,
        time_seconds=trace.times,
        angle_radians=trace.angles,
        normalized_energy=trace.energies,
        axial_force_newtons=trace.tensions,
        target_actions=trace.target_actions,
        realized_commands=trace.realized_commands,
        seat_radius_metres=trace.seat_radius,
        taut=trace.taut,
        cumulative_slack_time_seconds=trace.slack_time,
        cumulative_snap_energy_loss_joules=trace.snap_energy_loss,
        seat_position=np.asarray([frame.seat for frame in frames]),
    )


def _write_trace_csv(path: Path, trace: SwingTrace) -> None:
    columns = np.column_stack(
        (
            trace.times,
            trace.angles,
            np.rad2deg(trace.angles),
            trace.energies,
            trace.tensions,
            trace.target_actions,
            trace.realized_commands,
            trace.seat_radius / trace.rollout.length,
            trace.taut.astype(int),
            trace.slack_time,
            trace.snap_energy_loss,
        )
    )
    header = ",".join(
        (
            "time_seconds",
            "angle_radians",
            "angle_degrees",
            "normalized_energy",
            "axial_force_newtons",
            "target_action_squat",
            "target_action_lean",
            "realized_command_squat",
            "realized_command_lean",
            "seat_radius_fraction",
            "taut",
            "cumulative_slack_time_seconds",
            "cumulative_snap_energy_loss_joules",
        )
    )
    np.savetxt(path, columns, delimiter=",", header=header, comments="")


def _results_markdown(
    metrics: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
) -> str:
    rod = metrics["rigid_rod"]
    chain = metrics["unilateral_chain"]
    rotation = (
        f"yes, {rod['time_to_rotation_seconds']:.2f} s"
        if rod["success"]
        else "no"
    )
    first_release = (
        "none"
        if chain["first_release_seconds"] is None
        else f"{chain['first_release_seconds']:.2f} s"
    )
    event_lines = "\n".join(
        f"- {event['kind'].capitalize()} at "
        f"{event['time_seconds']:.2f} s "
        f"($\\theta={event['angle_degrees']:.1f}^\\circ$)."
        for event in events
    )
    return f"""<!-- Generated by scripts/build_swing_modeling_artifacts.py. -->

| Suspension model | Full rotation | Peak $|\\theta|$ | Minimum axial force | Minimum $r/L$ | Slack time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bidirectional rigid rod | {rotation} | {rod['peak_absolute_angle_degrees']:.2f}° | {rod['minimum_tension_newtons']:.1f} N | {rod['minimum_seat_radius_fraction']:.3f} | {rod['slack_time_seconds']:.2f} s |
| Unilateral chain | no | {chain['peak_absolute_angle_degrees']:.2f}° | {chain['minimum_tension_newtons']:.1f} N | {chain['minimum_seat_radius_fraction']:.3f} | {chain['slack_time_seconds']:.2f} s |

The rod trajectory requests as much as **{rod['maximum_compression_demand_newtons']:.1f} N of compression**, which a chain cannot provide. The chain first releases at **{first_release}**, releases {chain['release_count']} times in total, and loses **{chain['snap_energy_loss_joules']:.1f} J** across reattachments.

Recorded chain mode changes:

{event_lines}
"""


def write_model_audit_artifacts(
    traces: dict[str, SwingTrace],
    *,
    static_directory: Path,
    record_directory: Path,
    scenario: SwingScenario = DEFAULT_SWING_SCENARIO,
    fps: int = 20,
    speed: float = 2.0,
) -> dict[str, Path]:
    """Write the reproducible data, static fallback, and MP4 audit replay."""

    static_directory.mkdir(parents=True, exist_ok=True)
    record_directory.mkdir(parents=True, exist_ok=True)
    metrics = audit_metrics(traces, scenario)
    chain_events = mode_events(traces["unilateral_chain"])

    figure = make_model_audit_figure(traces, scenario)
    svg_path = static_directory / "model_audit.svg"
    png_path = static_directory / "model_audit.png"
    pdf_path = static_directory / "model_audit.pdf"
    figure.savefig(svg_path)
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path)
    plt.close(figure)

    movie_path = static_directory / "model_audit.mp4"
    animation_view = make_model_audit_animation(
        traces,
        scenario,
        fps=fps,
        speed=speed,
    )
    animation_view.save(movie_path)
    poster_path = static_directory / "model_audit_poster.png"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(movie_path),
            "-frames:v",
            "1",
            str(poster_path),
        ],
        check=True,
    )

    def playback_time(simulation_time: float) -> float:
        matching = np.flatnonzero(
            np.isclose(animation_view.timeline, simulation_time, atol=1e-9)
        )
        if not matching.size:
            raise AssertionError("recorded event is missing from the movie timeline")
        return float(matching[0] / fps)

    counters = {"release": 0, "reattachment": 0}
    replay_events: list[dict[str, Any]] = []
    for event in chain_events:
        kind = str(event["kind"])
        counters[kind] += 1
        label = (
            f"chain release {counters[kind]}"
            if kind == "release"
            else f"reattachment / snap {counters[kind]}"
        )
        replay_events.append(
            {
                **event,
                "label": label,
                "video_seconds": playback_time(float(event["time_seconds"])),
            }
        )
    rod_success_time = traces["rigid_rod"].rollout.success_time
    if rod_success_time is not None:
        replay_events.append(
            {
                "kind": "rod_rotation",
                "label": "nominal rod rotation",
                "time_seconds": float(rod_success_time),
                "video_seconds": playback_time(float(rod_success_time)),
            }
        )
    replay_events.sort(key=lambda event: float(event["time_seconds"]))

    metrics_path = record_directory / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    events_path = record_directory / "events.json"
    events_path.write_text(
        json.dumps(replay_events, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, trace in traces.items():
        _write_trace_archive(record_directory / f"{name}.npz", trace)
        _write_trace_csv(record_directory / f"{name}.csv", trace)

    results_path = record_directory / "results.md"
    results_path.write_text(
        _results_markdown(metrics, chain_events),
        encoding="utf-8",
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/build_swing_modeling_artifacts.py",
        "repository_revision": _repository_revision(),
        "swing_rl": _package_provenance("swing-rl"),
        "numpy_version": version("numpy"),
        "matplotlib_version": version("matplotlib"),
        "scenario": asdict(scenario),
        "animation": {
            "frames_per_second": fps,
            "simulation_speed": speed,
            "event_hold_seconds": 0.55,
            "renderer": "swing_rl.viz.animate.draw_swing_frame",
        },
        "metrics": metrics,
        "chain_mode_events": chain_events,
        "replay_events": replay_events,
        "files": {
            "static_figure_svg": str(svg_path),
            "static_figure_png": str(png_path),
            "static_figure_pdf": str(pdf_path),
            "animation_mp4": str(movie_path),
            "animation_poster_png": str(poster_path),
            "raw_archives": [f"{name}.npz" for name in traces],
            "raw_csv": [f"{name}.csv" for name in traces],
        },
    }
    manifest_path = record_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "svg": svg_path,
        "png": png_path,
        "pdf": pdf_path,
        "mp4": movie_path,
        "poster": poster_path,
        "metrics": metrics_path,
        "events": events_path,
        "results": results_path,
        "manifest": manifest_path,
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
