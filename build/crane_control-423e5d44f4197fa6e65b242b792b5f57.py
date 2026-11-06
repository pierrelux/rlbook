"""Constrained point-to-point control for a nonlinear overhead crane.

The module keeps the plant, baselines, transcription, validation, metrics, and
visualization in one place.  This prevents the three controllers in the
textbook comparison from silently using different dynamics or input limits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
}

COLORS = {
    "direct": OI["vermilion"],
    "zv": OI["blue"],
    "collocation": OI["green"],
}
STYLES = {"direct": "--", "zv": "-.", "collocation": "-"}

PUBLICATION_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.25,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


@dataclass(frozen=True)
class CraneParameters:
    """Physical parameters, task specification, and common constraints."""

    cable_length: float = 1.20
    gravity: float = 9.81
    sway_damping: float = 0.035
    target_position: float = 4.0
    acceleration_limit: float = 1.60
    velocity_limit: float = 1.50
    sway_limit: float = np.deg2rad(15.0)
    profile_acceleration: float = 0.80
    profile_max_speed: float = 1.00
    coast_time: float = 2.0

    def validate(self) -> None:
        positive = {
            "cable_length": self.cable_length,
            "gravity": self.gravity,
            "target_position": self.target_position,
            "acceleration_limit": self.acceleration_limit,
            "velocity_limit": self.velocity_limit,
            "sway_limit": self.sway_limit,
            "profile_acceleration": self.profile_acceleration,
            "profile_max_speed": self.profile_max_speed,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.sway_damping < 0.0:
            raise ValueError("sway_damping must be nonnegative")


@dataclass(frozen=True)
class TrapezoidalMove:
    """Symmetric point-to-point command, triangular for short moves."""

    distance: float
    acceleration: float
    max_speed: float

    def __post_init__(self) -> None:
        if self.distance == 0.0:
            raise ValueError("distance must be nonzero")
        if self.acceleration <= 0.0 or self.max_speed <= 0.0:
            raise ValueError("acceleration and max_speed must be positive")

    @property
    def direction(self) -> float:
        return float(np.sign(self.distance))

    @property
    def acceleration_time(self) -> float:
        nominal = self.max_speed / self.acceleration
        if self.acceleration * nominal**2 >= abs(self.distance):
            return float(np.sqrt(abs(self.distance) / self.acceleration))
        return nominal

    @property
    def cruise_time(self) -> float:
        ta = self.acceleration_time
        peak_speed = self.acceleration * ta
        return max(0.0, abs(self.distance) / peak_speed - ta)

    @property
    def duration(self) -> float:
        return 2.0 * self.acceleration_time + self.cruise_time

    def acceleration_at(self, time: float | np.ndarray) -> np.ndarray:
        t = np.asarray(time, dtype=float)
        command = np.zeros_like(t)
        ta = self.acceleration_time
        deceleration_start = ta + self.cruise_time
        command[(t >= 0.0) & (t < ta)] = self.acceleration
        command[(t >= deceleration_start) & (t < self.duration)] = -self.acceleration
        return self.direction * command


@dataclass(frozen=True)
class ZeroVibrationShaper:
    """Two-impulse shaper for the linearized payload mode."""

    first_weight: float
    second_weight: float
    delay: float

    @classmethod
    def from_parameters(cls, params: CraneParameters) -> "ZeroVibrationShaper":
        natural_frequency = np.sqrt(params.gravity / params.cable_length)
        damping_ratio = params.sway_damping / (2.0 * natural_frequency)
        damped_frequency = natural_frequency * np.sqrt(1.0 - damping_ratio**2)
        decay = np.exp(-damping_ratio * np.pi / np.sqrt(1.0 - damping_ratio**2))
        return cls(
            first_weight=float(1.0 / (1.0 + decay)),
            second_weight=float(decay / (1.0 + decay)),
            delay=float(np.pi / damped_frequency),
        )

    def shape(self, move: TrapezoidalMove, time: float | np.ndarray) -> np.ndarray:
        t = np.asarray(time, dtype=float)
        return (
            self.first_weight * move.acceleration_at(t)
            + self.second_weight * move.acceleration_at(t - self.delay)
        )


@dataclass(frozen=True)
class CollocationSolution:
    """Node values returned by the trapezoidal direct transcription."""

    time: np.ndarray
    state: np.ndarray
    acceleration: np.ndarray
    objective: float
    max_defect: float
    success: bool
    message: str


@dataclass(frozen=True)
class CraneMetrics:
    """Metrics computed after continuous nonlinear validation."""

    terminal_position_error_m: float
    terminal_speed_m_per_s: float
    terminal_sway_deg: float
    peak_sway_deg: float
    residual_sway_deg: float
    effort_m2_per_s3: float
    max_acceleration_m_per_s2: float
    max_velocity_m_per_s: float
    max_constraint_violation: float


@dataclass(frozen=True)
class SimulationResult:
    """Continuous-time validation of one fixed acceleration command."""

    key: str
    label: str
    time: np.ndarray
    state: np.ndarray
    acceleration: np.ndarray
    command_end: float
    metrics: CraneMetrics


@dataclass(frozen=True)
class CraneComparison:
    """Nominal and mismatched replays of the same three commands."""

    parameters: CraneParameters
    mismatch_parameters: CraneParameters
    collocation: CollocationSolution
    nominal: Mapping[str, SimulationResult]
    mismatch: Mapping[str, SimulationResult]


AccelerationLaw = Callable[[float], float]


def crane_dynamics(
    state: np.ndarray,
    acceleration: float,
    params: CraneParameters,
) -> np.ndarray:
    """Return the nonlinear crane vector field.

    The angle is measured from the downward vertical.  Positive trolley
    acceleration initially makes the payload lag, so the angle becomes negative.
    """

    _, velocity, angle, angular_velocity = np.asarray(state, dtype=float)
    angle_acceleration = (
        -(params.gravity / params.cable_length) * np.sin(angle)
        - (acceleration / params.cable_length) * np.cos(angle)
        - params.sway_damping * angular_velocity
    )
    return np.array(
        [velocity, acceleration, angular_velocity, angle_acceleration],
        dtype=float,
    )


def _pack(state: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    return np.concatenate([state.ravel(), acceleration])


def _unpack(vector: np.ndarray, intervals: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = intervals + 1
    split = 4 * nodes
    return vector[:split].reshape(nodes, 4), vector[split:]


def transcription_defects(
    solution: CollocationSolution,
    params: CraneParameters,
) -> np.ndarray:
    """Evaluate all trapezoidal dynamics defects at the saved node values."""

    state = solution.state
    acceleration = solution.acceleration
    steps = np.diff(solution.time)
    defects = []
    for k, step in enumerate(steps):
        left = crane_dynamics(state[k], acceleration[k], params)
        right = crane_dynamics(state[k + 1], acceleration[k + 1], params)
        defects.append(state[k + 1] - state[k] - 0.5 * step * (left + right))
    return np.asarray(defects)


def _initial_guess(
    params: CraneParameters,
    node_time: np.ndarray,
    move: TrapezoidalMove,
    shaper: ZeroVibrationShaper,
) -> tuple[np.ndarray, np.ndarray]:
    acceleration = shaper.shape(move, node_time)
    law = lambda time: float(shaper.shape(move, time))
    result = simulate_command(
        "zv",
        "ZV shaped",
        law,
        command_end=float(node_time[-1]),
        params=params,
        final_time=float(node_time[-1]),
        sample_time=node_time,
    )
    state = result.state.copy()
    state[0] = 0.0
    state[-1] = np.array([params.target_position, 0.0, 0.0, 0.0])
    return state, acceleration


def solve_direct_collocation(
    params: CraneParameters | None = None,
    *,
    intervals: int = 28,
    horizon: float | None = None,
    max_iterations: int = 350,
) -> CollocationSolution:
    """Solve a constrained nonlinear point-to-point transcription.

    States and accelerations at every node are decision variables.  Trapezoidal
    defects enforce the nonlinear dynamics, while boundary and path constraints
    are imposed directly on the node values.
    """

    params = params or CraneParameters()
    params.validate()
    if intervals < 8:
        raise ValueError("intervals must be at least eight")

    move = TrapezoidalMove(
        params.target_position,
        params.profile_acceleration,
        params.profile_max_speed,
    )
    shaper = ZeroVibrationShaper.from_parameters(params)
    if horizon is None:
        horizon = move.duration + shaper.delay
    if horizon <= 0.0:
        raise ValueError("horizon must be positive")

    node_time = np.linspace(0.0, horizon, intervals + 1)
    step = horizon / intervals
    state_guess, acceleration_guess = _initial_guess(params, node_time, move, shaper)
    initial = _pack(state_guess, acceleration_guess)

    initial_state = np.zeros(4)
    terminal_state = np.array([params.target_position, 0.0, 0.0, 0.0])

    def objective(vector: np.ndarray) -> float:
        state, acceleration = _unpack(vector, intervals)
        sway_cost = 6.0 * state[:, 2] ** 2 + 0.15 * state[:, 3] ** 2
        effort_cost = 0.035 * acceleration**2
        slew_cost = 0.002 * np.diff(acceleration) ** 2 / step**2
        return float(step * np.sum(sway_cost + effort_cost) + step * np.sum(slew_cost))

    def equality(vector: np.ndarray) -> np.ndarray:
        state, acceleration = _unpack(vector, intervals)
        defects = np.empty((intervals, 4))
        for k in range(intervals):
            left = crane_dynamics(state[k], acceleration[k], params)
            right = crane_dynamics(state[k + 1], acceleration[k + 1], params)
            defects[k] = state[k + 1] - state[k] - 0.5 * step * (left + right)
        return np.concatenate(
            [state[0] - initial_state, defects.ravel(), state[-1] - terminal_state]
        )

    nodes = intervals + 1
    lower_state = np.tile(
        np.array([-0.25, -params.velocity_limit, -params.sway_limit, -2.0]),
        (nodes, 1),
    )
    upper_state = np.tile(
        np.array(
            [params.target_position + 0.25, params.velocity_limit, params.sway_limit, 2.0]
        ),
        (nodes, 1),
    )
    lower = _pack(lower_state, np.full(nodes, -params.acceleration_limit))
    upper = _pack(upper_state, np.full(nodes, params.acceleration_limit))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints={"type": "eq", "fun": equality},
        options={"maxiter": max_iterations, "ftol": 2e-10, "disp": False},
    )
    state, acceleration = _unpack(result.x, intervals)
    max_defect = float(np.max(np.abs(equality(result.x))))
    success = bool(result.success and max_defect < 2e-5)
    if not success:
        raise RuntimeError(
            "crane direct collocation failed: "
            f"{result.message}; maximum equality residual {max_defect:.3e}"
        )
    return CollocationSolution(
        time=node_time,
        state=state,
        acceleration=acceleration,
        objective=float(result.fun),
        max_defect=max_defect,
        success=success,
        message=str(result.message),
    )


def _compute_metrics(
    time: np.ndarray,
    state: np.ndarray,
    acceleration: np.ndarray,
    command_end: float,
    params: CraneParameters,
) -> CraneMetrics:
    after_move = time >= command_end
    if not np.any(after_move):
        after_move = np.ones_like(time, dtype=bool)
    position_violation = np.maximum(-0.25 - state[:, 0], state[:, 0] - params.target_position - 0.25)
    violations = np.concatenate(
        [
            np.maximum(np.abs(acceleration) - params.acceleration_limit, 0.0),
            np.maximum(np.abs(state[:, 1]) - params.velocity_limit, 0.0),
            np.maximum(np.abs(state[:, 2]) - params.sway_limit, 0.0),
            np.maximum(position_violation, 0.0),
        ]
    )
    return CraneMetrics(
        terminal_position_error_m=float(abs(state[-1, 0] - params.target_position)),
        terminal_speed_m_per_s=float(abs(state[-1, 1])),
        terminal_sway_deg=float(abs(np.rad2deg(state[-1, 2]))),
        peak_sway_deg=float(np.max(np.abs(np.rad2deg(state[:, 2])))),
        residual_sway_deg=float(np.max(np.abs(np.rad2deg(state[after_move, 2])))),
        effort_m2_per_s3=float(np.trapezoid(acceleration**2, time)),
        max_acceleration_m_per_s2=float(np.max(np.abs(acceleration))),
        max_velocity_m_per_s=float(np.max(np.abs(state[:, 1]))),
        max_constraint_violation=float(np.max(violations)),
    )


def simulate_command(
    key: str,
    label: str,
    acceleration_law: AccelerationLaw,
    *,
    command_end: float,
    params: CraneParameters,
    final_time: float,
    sample_period: float = 0.02,
    sample_time: np.ndarray | None = None,
) -> SimulationResult:
    """Replay a fixed command on the continuous nonlinear plant."""

    if sample_time is None:
        count = int(np.round(final_time / sample_period))
        sample_time = np.linspace(0.0, final_time, count + 1)
    else:
        sample_time = np.asarray(sample_time, dtype=float)

    def right_hand_side(time: float, state: np.ndarray) -> np.ndarray:
        acceleration = float(
            np.clip(
                acceleration_law(time),
                -params.acceleration_limit,
                params.acceleration_limit,
            )
        )
        return crane_dynamics(state, acceleration, params)

    solution = solve_ivp(
        right_hand_side,
        (0.0, final_time),
        np.zeros(4),
        t_eval=sample_time,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11,
        max_step=min(0.01, sample_period),
    )
    if not solution.success:
        raise RuntimeError(f"crane validation failed: {solution.message}")
    state = solution.y.T
    acceleration = np.array(
        [
            np.clip(
                acceleration_law(time),
                -params.acceleration_limit,
                params.acceleration_limit,
            )
            for time in sample_time
        ]
    )
    return SimulationResult(
        key=key,
        label=label,
        time=sample_time,
        state=state,
        acceleration=acceleration,
        command_end=command_end,
        metrics=_compute_metrics(sample_time, state, acceleration, command_end, params),
    )


def _command_laws(
    params: CraneParameters,
    collocation: CollocationSolution,
) -> tuple[float, Mapping[str, tuple[str, AccelerationLaw]]]:
    move = TrapezoidalMove(
        params.target_position,
        params.profile_acceleration,
        params.profile_max_speed,
    )
    shaper = ZeroVibrationShaper.from_parameters(params)

    def direct(time: float) -> float:
        return float(move.acceleration_at(time))

    def shaped(time: float) -> float:
        return float(shaper.shape(move, time))

    def optimized(time: float) -> float:
        if time > collocation.time[-1]:
            return 0.0
        return float(np.interp(time, collocation.time, collocation.acceleration))

    command_end = float(collocation.time[-1])
    laws = {
        "direct": ("Direct command", direct),
        "zv": ("ZV shaped", shaped),
        "collocation": ("Direct collocation", optimized),
    }
    return command_end, laws


def run_comparison(
    params: CraneParameters | None = None,
    *,
    intervals: int = 28,
    sample_period: float = 0.02,
    mismatch_fraction: float = 0.10,
) -> CraneComparison:
    """Design once, then validate nominally and with a longer cable."""

    params = params or CraneParameters()
    params.validate()
    collocation = solve_direct_collocation(params, intervals=intervals)
    command_end, laws = _command_laws(params, collocation)
    final_time = command_end + params.coast_time
    mismatch_params = replace(
        params,
        cable_length=params.cable_length * (1.0 + mismatch_fraction),
    )

    def replay(plant: CraneParameters) -> dict[str, SimulationResult]:
        return {
            key: simulate_command(
                key,
                label,
                law,
                command_end=command_end,
                params=plant,
                final_time=final_time,
                sample_period=sample_period,
            )
            for key, (label, law) in laws.items()
        }

    return CraneComparison(
        parameters=params,
        mismatch_parameters=mismatch_params,
        collocation=collocation,
        nominal=replay(params),
        mismatch=replay(mismatch_params),
    )


def metrics_table(comparison: CraneComparison) -> list[dict[str, float | str]]:
    """Return display-ready nominal and mismatch metrics."""

    rows: list[dict[str, float | str]] = []
    for scenario, results in (
        ("nominal", comparison.nominal),
        ("cable +10%", comparison.mismatch),
    ):
        for result in results.values():
            rows.append(
                {
                    "scenario": scenario,
                    "controller": result.label,
                    "residual_sway_deg": result.metrics.residual_sway_deg,
                    "peak_sway_deg": result.metrics.peak_sway_deg,
                    "position_error_mm": 1_000.0
                    * result.metrics.terminal_position_error_m,
                    "effort": result.metrics.effort_m2_per_s3,
                }
            )
    return rows


def make_summary_figure(comparison: CraneComparison) -> plt.Figure:
    """Plot common-scale trajectories and the model-mismatch comparison."""

    with mpl.rc_context(PUBLICATION_STYLE):
        figure = plt.figure(figsize=(5.5, 5.0), constrained_layout=True)
        grid = figure.add_gridspec(3, 1, height_ratios=(1.0, 1.15, 0.9))
        position_axis = figure.add_subplot(grid[0])
        sway_axis = figure.add_subplot(grid[1], sharex=position_axis)
        mismatch_axis = figure.add_subplot(grid[2])

        for key, result in comparison.nominal.items():
            position_axis.plot(
                result.time,
                result.state[:, 0],
                color=COLORS[key],
                linestyle=STYLES[key],
                label=result.label,
            )
            sway_axis.plot(
                result.time,
                np.rad2deg(result.state[:, 2]),
                color=COLORS[key],
                linestyle=STYLES[key],
            )
        position_axis.axhline(
            comparison.parameters.target_position,
            color="0.5",
            linestyle=":",
            linewidth=0.8,
        )
        position_axis.set_ylabel("trolley position (m)")
        position_axis.legend(loc="lower right", ncols=3)
        position_axis.tick_params(labelbottom=False)
        sway_axis.axhline(0.0, color="0.7", linewidth=0.6)
        sway_axis.axhspan(
            -np.rad2deg(comparison.parameters.sway_limit),
            np.rad2deg(comparison.parameters.sway_limit),
            color="0.94",
            zorder=-2,
        )
        sway_axis.set_ylabel(r"payload sway $\theta$ (deg)")
        sway_axis.set_xlabel("time (s)")

        locations = np.arange(3)
        width = 0.32
        for offset, (scenario, results, hatch) in enumerate(
            (
                ("nominal", comparison.nominal, ""),
                ("cable +10%", comparison.mismatch, "//"),
            )
        ):
            values = [results[key].metrics.residual_sway_deg for key in COLORS]
            mismatch_axis.bar(
                locations + (offset - 0.5) * width,
                values,
                width=width,
                facecolor=[COLORS[key] for key in COLORS],
                alpha=0.85 if offset == 0 else 0.45,
                edgecolor="0.2",
                linewidth=0.5,
                hatch=hatch,
                label=scenario,
            )
        mismatch_axis.set_xticks(
            locations,
            [comparison.nominal[key].label for key in COLORS],
        )
        mismatch_axis.set_ylabel("residual sway (deg)")
        mismatch_axis.legend(loc="upper left", ncols=2)
        mismatch_axis.grid(axis="y", color="0.9", linewidth=0.5)
        return figure


def create_animation(
    comparison: CraneComparison,
    *,
    frame_stride: int = 4,
    interval_ms: int = 35,
) -> FuncAnimation:
    """Animate the three nominal nonlinear validation trajectories."""

    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    reference = comparison.nominal["direct"]
    keys = tuple(COLORS)
    with mpl.rc_context(PUBLICATION_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharex=True, sharey=True)
        artists = []
        rail_height = 0.15
        cable_length = comparison.parameters.cable_length
        for axis, key in zip(axes, keys):
            result = comparison.nominal[key]
            axis.set_xlim(-0.5, comparison.parameters.target_position + 0.5)
            axis.set_ylim(-cable_length - 0.25, 0.42)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(result.label)
            axis.axhline(rail_height, color="0.4", linewidth=1.6)
            axis.axvline(
                comparison.parameters.target_position,
                color="0.75",
                linestyle=":",
                linewidth=0.8,
            )
            cable, = axis.plot([], [], color="0.25", linewidth=1.2)
            trolley = Rectangle((-0.12, rail_height - 0.07), 0.24, 0.14,
                                facecolor=COLORS[key], edgecolor="0.15")
            payload = Circle((0.0, 0.0), 0.09, facecolor=COLORS[key], edgecolor="0.15")
            axis.add_patch(trolley)
            axis.add_patch(payload)
            clock = axis.text(0.03, 0.04, "", transform=axis.transAxes)
            axis.set_xlabel("position (m)")
            artists.append((cable, trolley, payload, clock))
        axes[0].set_ylabel("height (m)")

        frame_indices = np.arange(0, reference.time.size, frame_stride)

        def update(frame: int):
            index = int(frame_indices[frame])
            changed = []
            for key, (cable, trolley, payload, clock) in zip(keys, artists):
                result = comparison.nominal[key]
                position, _, angle, _ = result.state[index]
                pivot_y = rail_height - 0.07
                payload_x = position + cable_length * np.sin(angle)
                payload_y = pivot_y - cable_length * np.cos(angle)
                trolley.set_x(position - 0.12)
                payload.center = (payload_x, payload_y)
                cable.set_data([position, payload_x], [pivot_y, payload_y])
                clock.set_text(f"t = {result.time[index]:.1f} s")
                changed.extend([cable, trolley, payload, clock])
            return changed

        return FuncAnimation(
            figure,
            update,
            frames=len(frame_indices),
            interval=interval_ms,
            blit=False,
            repeat=True,
        )


__all__ = [
    "CraneComparison",
    "CraneMetrics",
    "CraneParameters",
    "CollocationSolution",
    "SimulationResult",
    "TrapezoidalMove",
    "ZeroVibrationShaper",
    "crane_dynamics",
    "create_animation",
    "make_summary_figure",
    "metrics_table",
    "run_comparison",
    "simulate_command",
    "solve_direct_collocation",
    "transcription_defects",
]
