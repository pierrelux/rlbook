"""Economic model predictive control for a hinged-flap wave-energy device.

All controllers act through a nonnegative damping coefficient.  The same
projection enforces damping, PTO torque, and damping-rate limits before an
action reaches the plant.  The economic MPC additionally predicts the stroke
constraint and maximizes harvested energy over a receding horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arc, Polygon
import numpy as np
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
    "constant": OI["blue"],
    "phase": OI["vermilion"],
    "mpc": OI["green"],
}
STYLES = {"constant": "--", "phase": "-.", "mpc": "-"}

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
class WaveParameters:
    """Plant, sea-state, actuator, and economic-MPC parameters."""

    inertia: float = 1_200.0
    radiation_damping: float = 750.0
    hydrostatic_stiffness: float = 18_000.0
    wave_amplitudes: tuple[float, float, float] = (2_400.0, 1_440.0, 810.0)
    wave_frequencies: tuple[float, float, float] = (3.35, 3.85, 4.45)
    wave_phases: tuple[float, float, float] = (0.20, 1.65, -0.85)
    constant_damping: float = 900.0
    phase_damping_min: float = 120.0
    phase_damping_max: float = 1_900.0
    phase_power_scale: float = 4_000.0
    damping_max: float = 2_400.0
    damping_rate_limit: float = 5_000.0
    torque_limit: float = 2_800.0
    stroke_limit: float = 0.55
    control_period: float = 0.12
    horizon_steps: int = 18
    slew_weight: float = 2.0e-3

    def validate(self) -> None:
        positive = {
            "inertia": self.inertia,
            "radiation_damping": self.radiation_damping,
            "hydrostatic_stiffness": self.hydrostatic_stiffness,
            "damping_max": self.damping_max,
            "damping_rate_limit": self.damping_rate_limit,
            "torque_limit": self.torque_limit,
            "stroke_limit": self.stroke_limit,
            "control_period": self.control_period,
            "horizon_steps": float(self.horizon_steps),
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (
            len(self.wave_amplitudes)
            == len(self.wave_frequencies)
            == len(self.wave_phases)
            == 3
        ):
            raise ValueError("the forcing must contain exactly three components")
        if self.phase_damping_min < 0.0:
            raise ValueError("phase_damping_min must be nonnegative")
        if self.phase_damping_max > self.damping_max:
            raise ValueError("phase damping exceeds the actuator limit")


@dataclass(frozen=True)
class MPCStep:
    """One finite-horizon economic optimization result."""

    damping: float
    sequence: np.ndarray
    predicted_state: np.ndarray
    objective: float
    success: bool
    max_predicted_violation: float
    solve_time_s: float


@dataclass(frozen=True)
class WaveMetrics:
    """Closed-loop energy, motion, actuation, and feasibility metrics."""

    absorbed_energy_j: float
    peak_stroke_rad: float
    peak_pto_torque_nm: float
    damping_variation_per_s: float
    stroke_violation_rad: float
    torque_violation_nm: float
    damping_rate_violation_per_s: float
    mpc_success_fraction: float
    mean_mpc_solve_time_s: float


@dataclass(frozen=True)
class WaveResult:
    """One closed-loop replay on a common time grid."""

    key: str
    label: str
    time: np.ndarray
    angle: np.ndarray
    angular_velocity: np.ndarray
    wave_torque: np.ndarray
    damping: np.ndarray
    pto_torque: np.ndarray
    absorbed_power: np.ndarray
    cumulative_energy: np.ndarray
    metrics: WaveMetrics


@dataclass(frozen=True)
class DampingSweep:
    damping: np.ndarray
    absorbed_energy_j: np.ndarray
    peak_stroke_rad: np.ndarray
    peak_pto_torque_nm: np.ndarray


def wave_torque(time: float | np.ndarray, params: WaveParameters) -> np.ndarray:
    """Evaluate the deterministic three-frequency excitation torque."""

    t = np.asarray(time, dtype=float)
    amplitude = np.asarray(params.wave_amplitudes)
    frequency = np.asarray(params.wave_frequencies)
    phase = np.asarray(params.wave_phases)
    return np.sum(amplitude * np.sin(t[..., None] * frequency + phase), axis=-1)


def flap_dynamics(
    time: float,
    state: np.ndarray,
    damping: float,
    params: WaveParameters,
) -> np.ndarray:
    """Return angle and angular-velocity derivatives for passive PTO damping."""

    angle, velocity = np.asarray(state, dtype=float)
    excitation = float(wave_torque(time, params))
    acceleration = (
        excitation
        - (params.radiation_damping + damping) * velocity
        - params.hydrostatic_stiffness * angle
    ) / params.inertia
    return np.array([velocity, acceleration])


def rk4_step(
    time: float,
    state: np.ndarray,
    damping: float,
    params: WaveParameters,
    step: float | None = None,
) -> np.ndarray:
    """Advance the plant by one fixed control period."""

    h = params.control_period if step is None else step
    k1 = flap_dynamics(time, state, damping, params)
    k2 = flap_dynamics(time + 0.5 * h, state + 0.5 * h * k1, damping, params)
    k3 = flap_dynamics(time + 0.5 * h, state + 0.5 * h * k2, damping, params)
    k4 = flap_dynamics(time + h, state + h * k3, damping, params)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def project_damping(
    requested: float,
    velocity: float,
    previous: float,
    params: WaveParameters,
) -> float:
    """Project a request onto damping, rate, and instantaneous torque limits."""

    step_limit = params.damping_rate_limit * params.control_period
    lower = max(0.0, previous - step_limit)
    upper = min(params.damping_max, previous + step_limit)
    if abs(velocity) > 1e-10:
        upper = min(upper, params.torque_limit / abs(velocity))
    if upper < lower:
        lower = upper
    return float(np.clip(requested, lower, upper))


def constant_damping_request(_time: float, _state: np.ndarray, params: WaveParameters) -> float:
    return params.constant_damping


def phase_aware_request(time: float, state: np.ndarray, params: WaveParameters) -> float:
    """Request more damping when wave power enters the flap."""

    incoming_power = float(wave_torque(time, params)) * float(state[1])
    gate = 0.5 * (1.0 + np.tanh(incoming_power / params.phase_power_scale))
    return float(
        params.phase_damping_min
        + (params.phase_damping_max - params.phase_damping_min) * gate
    )


def predict_trajectory(
    time: float,
    state: np.ndarray,
    damping: np.ndarray,
    params: WaveParameters,
) -> np.ndarray:
    """Predict the state sequence for one candidate damping sequence."""

    damping = np.asarray(damping, dtype=float)
    prediction = np.empty((damping.size + 1, 2))
    prediction[0] = state
    for k, value in enumerate(damping):
        prediction[k + 1] = rk4_step(
            time + k * params.control_period,
            prediction[k],
            float(value),
            params,
        )
    return prediction


def _constraint_margins(
    damping: np.ndarray,
    prediction: np.ndarray,
    previous_damping: float,
    params: WaveParameters,
) -> np.ndarray:
    velocity = prediction[:-1, 1]
    stroke_margin = (params.stroke_limit - np.abs(prediction[1:, 0])) / params.stroke_limit
    torque_margin = (
        params.torque_limit - np.abs(damping * velocity)
    ) / params.torque_limit
    changes = np.diff(np.concatenate([[previous_damping], damping]))
    rate_margin = (
        params.damping_rate_limit * params.control_period - np.abs(changes)
    ) / (params.damping_rate_limit * params.control_period)
    return np.concatenate([stroke_margin, torque_margin, rate_margin])


def solve_economic_mpc(
    time: float,
    state: np.ndarray,
    previous_damping: float,
    params: WaveParameters | None = None,
    *,
    warm_start: np.ndarray | None = None,
    max_iterations: int = 45,
) -> MPCStep:
    """Maximize predicted harvested energy under plant and actuator constraints."""

    params = params or WaveParameters()
    params.validate()
    state = np.asarray(state, dtype=float)
    if state.shape != (2,) or not np.all(np.isfinite(state)):
        raise ValueError("state must have shape (2,) and contain finite values")
    horizon = params.horizon_steps
    if warm_start is None:
        initial = np.full(horizon, previous_damping, dtype=float)
    else:
        warm_start = np.asarray(warm_start, dtype=float)
        if warm_start.shape != (horizon,):
            raise ValueError("warm_start has the wrong horizon")
        initial = warm_start.copy()

    initial[0] = project_damping(initial[0], state[1], previous_damping, params)
    for k in range(1, horizon):
        change_limit = params.damping_rate_limit * params.control_period
        initial[k] = np.clip(
            initial[k],
            max(0.0, initial[k - 1] - change_limit),
            min(params.damping_max, initial[k - 1] + change_limit),
        )

    def objective(normalized: np.ndarray) -> float:
        sequence = params.damping_max * normalized
        prediction = predict_trajectory(time, state, sequence, params)
        velocity = prediction[:-1, 1]
        energy = params.control_period * np.sum(sequence * velocity**2)
        changes = np.diff(np.concatenate([[previous_damping], sequence]))
        slew = np.sum((changes / params.damping_max) ** 2)
        return float(-energy / 1_000.0 + params.slew_weight * slew)

    def constraints(normalized: np.ndarray) -> np.ndarray:
        sequence = params.damping_max * normalized
        prediction = predict_trajectory(time, state, sequence, params)
        return _constraint_margins(sequence, prediction, previous_damping, params)

    started = perf_counter()
    result = minimize(
        objective,
        initial / params.damping_max,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * horizon,
        constraints={"type": "ineq", "fun": constraints},
        options={"maxiter": max_iterations, "ftol": 2e-6, "disp": False},
    )
    elapsed = perf_counter() - started
    sequence = np.asarray(
        params.damping_max * result.x if np.all(np.isfinite(result.x)) else initial
    )
    prediction = predict_trajectory(time, state, sequence, params)
    margins = _constraint_margins(sequence, prediction, previous_damping, params)
    maximum_violation = float(max(0.0, -np.min(margins)))
    success = bool(result.success and maximum_violation <= 2e-4)
    if not success:
        sequence = initial
        prediction = predict_trajectory(time, state, sequence, params)
        margins = _constraint_margins(sequence, prediction, previous_damping, params)
        maximum_violation = float(max(0.0, -np.min(margins)))

    first = project_damping(sequence[0], state[1], previous_damping, params)
    sequence = sequence.copy()
    sequence[0] = first
    return MPCStep(
        damping=first,
        sequence=sequence,
        predicted_state=prediction,
        objective=float(objective(sequence / params.damping_max)),
        success=success,
        max_predicted_violation=maximum_violation,
        solve_time_s=elapsed,
    )


def _metrics(
    time: np.ndarray,
    angle: np.ndarray,
    velocity: np.ndarray,
    damping: np.ndarray,
    pto_torque: np.ndarray,
    cumulative_energy: np.ndarray,
    params: WaveParameters,
    successes: list[bool],
    solve_times: list[float],
) -> WaveMetrics:
    damping_rate = np.diff(damping) / params.control_period
    return WaveMetrics(
        absorbed_energy_j=float(cumulative_energy[-1]),
        peak_stroke_rad=float(np.max(np.abs(angle))),
        peak_pto_torque_nm=float(np.max(np.abs(pto_torque))),
        damping_variation_per_s=float(
            np.sum(np.abs(np.diff(damping))) / max(time[-1] - time[0], 1e-12)
        ),
        stroke_violation_rad=float(max(0.0, np.max(np.abs(angle)) - params.stroke_limit)),
        torque_violation_nm=float(
            max(0.0, np.max(np.abs(pto_torque)) - params.torque_limit)
        ),
        damping_rate_violation_per_s=float(
            max(0.0, np.max(np.abs(damping_rate), initial=0.0) - params.damping_rate_limit)
        ),
        mpc_success_fraction=float(np.mean(successes)) if successes else float("nan"),
        mean_mpc_solve_time_s=float(np.mean(solve_times)) if solve_times else float("nan"),
    )


def simulate_controller(
    key: str,
    label: str,
    params: WaveParameters | None = None,
    *,
    duration: float = 45.0,
    initial_state: tuple[float, float] = (0.0, 0.0),
) -> WaveResult:
    """Run a closed-loop controller with the same actuator projection."""

    params = params or WaveParameters()
    params.validate()
    steps = int(np.round(duration / params.control_period))
    if not np.isclose(steps * params.control_period, duration, atol=1e-9):
        raise ValueError("duration must be an integer multiple of control_period")
    time = np.linspace(0.0, duration, steps + 1)
    state = np.empty((steps + 1, 2))
    state[0] = np.asarray(initial_state, dtype=float)
    damping = np.empty(steps + 1)
    damping[0] = params.constant_damping
    successes: list[bool] = []
    solve_times: list[float] = []
    warm_start: np.ndarray | None = None

    for k in range(steps):
        now = time[k]
        previous = damping[k]
        if key == "constant":
            requested = constant_damping_request(now, state[k], params)
        elif key == "phase":
            requested = phase_aware_request(now, state[k], params)
        elif key == "mpc":
            step = solve_economic_mpc(
                now,
                state[k],
                previous,
                params,
                warm_start=warm_start,
            )
            requested = step.damping
            warm_start = np.concatenate([step.sequence[1:], step.sequence[-1:]])
            successes.append(step.success)
            solve_times.append(step.solve_time_s)
        else:
            raise ValueError("key must be 'constant', 'phase', or 'mpc'")
        applied = project_damping(requested, state[k, 1], previous, params)
        damping[k] = applied
        state[k + 1] = rk4_step(now, state[k], applied, params)
        damping[k + 1] = applied

    excitation = np.asarray(wave_torque(time, params))
    pto_torque = -damping * state[:, 1]
    absorbed_power = damping * state[:, 1] ** 2
    increments = 0.5 * params.control_period * (
        absorbed_power[:-1] + absorbed_power[1:]
    )
    cumulative_energy = np.concatenate([[0.0], np.cumsum(increments)])
    return WaveResult(
        key=key,
        label=label,
        time=time,
        angle=state[:, 0],
        angular_velocity=state[:, 1],
        wave_torque=excitation,
        damping=damping,
        pto_torque=pto_torque,
        absorbed_power=absorbed_power,
        cumulative_energy=cumulative_energy,
        metrics=_metrics(
            time,
            state[:, 0],
            state[:, 1],
            damping,
            pto_torque,
            cumulative_energy,
            params,
            successes,
            solve_times,
        ),
    )


def run_comparison(
    params: WaveParameters | None = None,
    *,
    duration: float = 45.0,
) -> dict[str, WaveResult]:
    """Compare two passive baselines with receding-horizon economic MPC."""

    params = params or WaveParameters()
    return {
        "constant": simulate_controller(
            "constant", "Constant damping", params, duration=duration
        ),
        "phase": simulate_controller(
            "phase", "Phase-aware damping", params, duration=duration
        ),
        "mpc": simulate_controller(
            "mpc", "Economic MPC", params, duration=duration
        ),
    }


def shifted_sea_state(params: WaveParameters | None = None) -> WaveParameters:
    """Return a deterministic amplitude, frequency, and phase shift."""

    params = params or WaveParameters()
    return replace(
        params,
        wave_amplitudes=tuple(1.08 * np.asarray(params.wave_amplitudes)),
        wave_frequencies=tuple(1.04 * np.asarray(params.wave_frequencies)),
        wave_phases=tuple(np.asarray(params.wave_phases) + np.array([0.35, -0.20, 0.25])),
    )


def run_damping_sweep(
    params: WaveParameters | None = None,
    *,
    damping_values: np.ndarray | None = None,
    duration: float = 45.0,
) -> DampingSweep:
    """Evaluate constant-damping requests with the common actuator projection."""

    params = params or WaveParameters()
    if damping_values is None:
        damping_values = np.linspace(100.0, 2_800.0, 16)
    damping_values = np.asarray(damping_values, dtype=float)
    if damping_values.ndim != 1 or damping_values.size < 2:
        raise ValueError("damping_values must be a one-dimensional sweep")
    results = []
    for value in damping_values:
        local = replace(params, constant_damping=float(min(value, params.damping_max)))
        results.append(
            simulate_controller(
                "constant",
                f"constant {value:.0f}",
                local,
                duration=duration,
            )
        )
    return DampingSweep(
        damping=damping_values,
        absorbed_energy_j=np.array([result.metrics.absorbed_energy_j for result in results]),
        peak_stroke_rad=np.array([result.metrics.peak_stroke_rad for result in results]),
        peak_pto_torque_nm=np.array(
            [result.metrics.peak_pto_torque_nm for result in results]
        ),
    )


def metrics_table(results: Mapping[str, WaveResult]) -> list[dict[str, float | str]]:
    """Return display-ready rows without rounding away violations."""

    return [
        {
            "controller": result.label,
            "energy_kj": result.metrics.absorbed_energy_j / 1_000.0,
            "peak_stroke_deg": np.rad2deg(result.metrics.peak_stroke_rad),
            "peak_torque_knm": result.metrics.peak_pto_torque_nm / 1_000.0,
            "damping_variation_per_s": result.metrics.damping_variation_per_s,
            "stroke_violation_deg": np.rad2deg(result.metrics.stroke_violation_rad),
            "torque_violation_nm": result.metrics.torque_violation_nm,
        }
        for result in results.values()
    ]


def make_closed_loop_figure(
    results: Mapping[str, WaveResult],
    params: WaveParameters | None = None,
) -> plt.Figure:
    """Plot motion, energy, and applied damping on shared time axes."""

    params = params or WaveParameters()
    with mpl.rc_context(PUBLICATION_STYLE):
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(5.5, 5.0),
            sharex=True,
            constrained_layout=True,
        )
        for key, result in results.items():
            axes[0].plot(
                result.time,
                np.rad2deg(result.angle),
                color=COLORS[key],
                linestyle=STYLES[key],
                label=result.label,
            )
            axes[1].plot(
                result.time,
                result.cumulative_energy / 1_000.0,
                color=COLORS[key],
                linestyle=STYLES[key],
            )
            axes[2].plot(
                result.time,
                result.damping / 1_000.0,
                color=COLORS[key],
                linestyle=STYLES[key],
            )
        limit = np.rad2deg(params.stroke_limit)
        axes[0].axhspan(-limit, limit, color="0.94", zorder=-2)
        axes[0].axhline(limit, color="0.55", linestyle=":", linewidth=0.7)
        axes[0].axhline(-limit, color="0.55", linestyle=":", linewidth=0.7)
        axes[0].set_ylabel("flap angle (deg)")
        axes[0].legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            borderaxespad=0.0,
            ncols=3,
        )
        axes[1].set_ylabel("captured energy (kJ)")
        axes[2].set_ylabel(r"damping $\rho$ (kN m s/rad)")
        axes[2].set_xlabel("time (s)")
        for axis in axes:
            axis.grid(axis="y", color="0.91", linewidth=0.5)
        return figure


def make_tradeoff_figure(
    results: Mapping[str, WaveResult],
    sweep: DampingSweep,
    params: WaveParameters | None = None,
) -> plt.Figure:
    """Plot harvested energy against peak stroke for the matched controllers."""

    params = params or WaveParameters()
    with mpl.rc_context(PUBLICATION_STYLE):
        figure, axis = plt.subplots(figsize=(5.5, 2.9), constrained_layout=True)
        axis.plot(
            np.rad2deg(sweep.peak_stroke_rad),
            sweep.absorbed_energy_j / 1_000.0,
            color="0.55",
            marker="o",
            markersize=3,
            linewidth=0.9,
            label="constant-damping sweep",
        )
        for key, result in results.items():
            axis.scatter(
                np.rad2deg(result.metrics.peak_stroke_rad),
                result.metrics.absorbed_energy_j / 1_000.0,
                color=COLORS[key],
                marker={"constant": "s", "phase": "^", "mpc": "D"}[key],
                s=34,
                zorder=3,
            )
            axis.annotate(
                result.label,
                (
                    np.rad2deg(result.metrics.peak_stroke_rad),
                    result.metrics.absorbed_energy_j / 1_000.0,
                ),
                xytext=(4, 3),
                textcoords="offset points",
                color=COLORS[key],
                fontsize=7,
            )
        axis.axvline(
            np.rad2deg(params.stroke_limit),
            color="0.35",
            linestyle=":",
            linewidth=0.8,
        )
        axis.text(
            np.rad2deg(params.stroke_limit),
            axis.get_ylim()[0],
            " stroke limit",
            color="0.35",
            va="bottom",
            fontsize=7,
        )
        axis.set_xlabel("peak flap stroke (deg)")
        axis.set_ylabel("captured energy (kJ)")
        axis.legend(loc="best")
        axis.grid(color="0.91", linewidth=0.5)
        return figure


def create_animation(
    results: Mapping[str, WaveResult],
    params: WaveParameters | None = None,
    *,
    frame_stride: int = 2,
    interval_ms: int = 40,
) -> FuncAnimation:
    """Animate the MPC flap beside its accumulated-energy trace."""

    params = params or WaveParameters()
    result = results["mpc"]
    frames = np.arange(0, result.time.size, frame_stride)
    with mpl.rc_context(PUBLICATION_STYLE):
        figure, (device_axis, energy_axis) = plt.subplots(
            1,
            2,
            figsize=(5.5, 2.5),
            gridspec_kw={"width_ratios": (0.9, 1.6)},
            constrained_layout=True,
        )
        device_axis.set_xlim(-1.2, 1.2)
        device_axis.set_ylim(-0.25, 1.7)
        device_axis.set_aspect("equal")
        device_axis.axis("off")
        device_axis.axhline(0.0, color=OI["blue"], linewidth=1.4)
        flap = Polygon(
            [[-0.05, 0.0], [0.05, 0.0], [0.08, 1.25], [-0.08, 1.25]],
            closed=True,
            facecolor=OI["green"],
            edgecolor="0.15",
        )
        device_axis.add_patch(flap)
        arc = Arc((0.0, 0.0), 0.55, 0.55, theta1=0, theta2=0, color="0.35")
        device_axis.add_patch(arc)
        clock = device_axis.text(0.02, 0.94, "", transform=device_axis.transAxes)
        damping_text = device_axis.text(0.02, 0.86, "", transform=device_axis.transAxes)

        energy_axis.set_xlim(result.time[0], result.time[-1])
        energy_axis.set_ylim(0.0, 1.05 * result.cumulative_energy[-1] / 1_000.0)
        energy_axis.set_xlabel("time (s)")
        energy_axis.set_ylabel("captured energy (kJ)")
        energy_axis.grid(color="0.91", linewidth=0.5)
        energy_line, = energy_axis.plot([], [], color=OI["green"])
        energy_point, = energy_axis.plot([], [], marker="D", color=OI["green"])

        base = np.array([[-0.05, 0.0], [0.05, 0.0], [0.08, 1.25], [-0.08, 1.25]])

        def update(frame: int):
            index = int(frames[frame])
            angle = result.angle[index]
            rotation = np.array(
                [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
            )
            flap.set_xy(base @ rotation)
            degrees = np.rad2deg(angle)
            if degrees >= 0.0:
                arc.theta1, arc.theta2 = 90.0 - degrees, 90.0
            else:
                arc.theta1, arc.theta2 = 90.0, 90.0 - degrees
            clock.set_text(f"t = {result.time[index]:.1f} s")
            damping_text.set_text(rf"$\rho$ = {result.damping[index] / 1000:.2f} kN m s/rad")
            energy_line.set_data(
                result.time[: index + 1],
                result.cumulative_energy[: index + 1] / 1_000.0,
            )
            energy_point.set_data(
                [result.time[index]],
                [result.cumulative_energy[index] / 1_000.0],
            )
            return flap, arc, clock, damping_text, energy_line, energy_point

        return FuncAnimation(
            figure,
            update,
            frames=len(frames),
            interval=interval_ms,
            blit=False,
            repeat=True,
        )


__all__ = [
    "DampingSweep",
    "MPCStep",
    "WaveMetrics",
    "WaveParameters",
    "WaveResult",
    "constant_damping_request",
    "create_animation",
    "flap_dynamics",
    "make_closed_loop_figure",
    "make_tradeoff_figure",
    "metrics_table",
    "phase_aware_request",
    "predict_trajectory",
    "project_damping",
    "rk4_step",
    "run_comparison",
    "run_damping_sweep",
    "shifted_sea_state",
    "simulate_controller",
    "solve_economic_mpc",
    "wave_torque",
]
