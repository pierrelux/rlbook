"""Sampled inertial sensing and feedback for a one-axis camera gimbal.

The experiment holds the plant, disturbances, noise realization, and saturated
PD law fixed while changing the state estimator.  The plant is integrated at
1 kHz and the sensor-controller loop runs at 100 Hz.  Browser code never
advances this model; it only replays arrays produced here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
import numpy as np


EstimatorKind = Literal["accelerometer", "gyro", "complementary"]
ESTIMATOR_ORDER: tuple[EstimatorKind, ...] = (
    "accelerometer",
    "gyro",
    "complementary",
)
ESTIMATOR_LABELS: Mapping[EstimatorKind, str] = {
    "accelerometer": "Accelerometer as state",
    "gyro": "Integrated gyroscope",
    "complementary": "Complementary observer",
}

PAPER = "#F6F7F4"
INK = "#1B2430"
TEAL = "#2F6F8F"
MUTED = "#5C6874"
RULE = "#D2D9D7"
STANDS = "#2E7D5B"
CAVEAT = "#B8860B"

METHOD_STYLES: Mapping[EstimatorKind, tuple[str, str]] = {
    "accelerometer": (CAVEAT, "--"),
    "gyro": (CAVEAT, ":"),
    "complementary": (STANDS, "-"),
}

FIGURE_STYLE = {
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.65,
    "lines.linewidth": 1.45,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "svg.fonttype": "none",
}


@dataclass(frozen=True)
class GimbalParameters:
    """Physical, sensing, observer, and feedback parameters."""

    inertia_kg_m2: float = 0.018
    damping_nm_s_per_rad: float = 0.025
    torque_limit_nm: float = 0.18
    gravity_mps2: float = 9.81
    controller_kp_nm_per_rad: float = 0.9
    controller_kd_nm_s_per_rad: float = 0.12
    observer_angle_gain_per_s: float = 0.7
    observer_bias_gain_per_s2: float = 0.08
    gyro_noise_std_rad_s: float = float(np.deg2rad(0.15))
    accelerometer_noise_std_mps2: float = 0.04
    bias_random_walk_std_rad_s_sqrt_s: float = float(np.deg2rad(0.02))

    def validate(self) -> None:
        positive = {
            "inertia_kg_m2": self.inertia_kg_m2,
            "damping_nm_s_per_rad": self.damping_nm_s_per_rad,
            "torque_limit_nm": self.torque_limit_nm,
            "gravity_mps2": self.gravity_mps2,
            "controller_kp_nm_per_rad": self.controller_kp_nm_per_rad,
            "controller_kd_nm_s_per_rad": self.controller_kd_nm_s_per_rad,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative = {
            "observer_angle_gain_per_s": self.observer_angle_gain_per_s,
            "observer_bias_gain_per_s2": self.observer_bias_gain_per_s2,
            "gyro_noise_std_rad_s": self.gyro_noise_std_rad_s,
            "accelerometer_noise_std_mps2": self.accelerometer_noise_std_mps2,
            "bias_random_walk_std_rad_s_sqrt_s": (
                self.bias_random_walk_std_rad_s_sqrt_s
            ),
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class GimbalScenario:
    """One reproducible disturbance and sampling scenario."""

    duration_s: float = 10.0
    sensor_period_s: float = 0.01
    integration_step_s: float = 0.001
    initial_angle_rad: float = float(np.deg2rad(8.0))
    initial_angular_velocity_rad_s: float = 0.0
    initial_gyro_bias_rad_s: float = float(np.deg2rad(0.8))
    tap_start_s: float = 1.50
    tap_end_s: float = 1.62
    tap_torque_nm: float = 0.14
    acceleration_start_s: float = 4.0
    acceleration_ramp_s: float = 0.1
    acceleration_plateau_end_s: float = 4.5
    acceleration_end_s: float = 4.6
    acceleration_mps2: float = 3.0
    seed: int = 11

    def validate(self) -> None:
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        for name, value in {
            "sensor_period_s": self.sensor_period_s,
            "integration_step_s": self.integration_step_s,
        }.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        ratio = self.sensor_period_s / self.integration_step_s
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-12):
            raise ValueError("sensor_period_s must be a multiple of integration_step_s")
        horizon_ratio = self.duration_s / self.sensor_period_s
        if not np.isclose(horizon_ratio, round(horizon_ratio), rtol=0.0, atol=1e-12):
            raise ValueError("duration_s must be a multiple of sensor_period_s")
        if not (
            0.0 <= self.tap_start_s < self.tap_end_s <= self.duration_s
            and 0.0 <= self.acceleration_start_s
            < self.acceleration_start_s + self.acceleration_ramp_s
            <= self.acceleration_plateau_end_s
            < self.acceleration_end_s
            <= self.duration_s
        ):
            raise ValueError("disturbance event times are inconsistent")
        if not np.isclose(
            self.acceleration_end_s - self.acceleration_plateau_end_s,
            self.acceleration_ramp_s,
        ):
            raise ValueError("acceleration rise and fall ramps must have equal duration")

    @property
    def control_steps(self) -> int:
        return int(round(self.duration_s / self.sensor_period_s))

    @property
    def integration_substeps(self) -> int:
        return int(round(self.sensor_period_s / self.integration_step_s))


@dataclass(frozen=True)
class NoiseRealization:
    """Common noise draws and gyro-bias path for matched comparisons."""

    gyro_noise_rad_s: np.ndarray
    accelerometer_noise_mps2: np.ndarray
    gyro_bias_rad_s: np.ndarray


@dataclass(frozen=True)
class GimbalObservation:
    """One sampled IMU observation."""

    gyro_rad_s: float
    accelerometer_x_mps2: float
    accelerometer_y_mps2: float

    @property
    def accelerometer_angle_rad(self) -> float:
        return float(
            wrap_angle(
                np.arctan2(
                    self.accelerometer_x_mps2,
                    self.accelerometer_y_mps2,
                )
            )
        )


@dataclass(frozen=True)
class EstimatorMemory:
    """Recursive state retained by an estimator between sensor samples."""

    angle_rad: float
    bias_rad_s: float
    initialized: bool = True


@dataclass(frozen=True)
class GimbalMetrics:
    """Closed-loop regulation and actuation diagnostics."""

    rms_angle_deg: float
    peak_acceleration_window_deg: float
    final_abs_angle_deg: float
    normalized_torque_effort: float
    saturation_fraction: float
    regulation_score: float
    estimator_rmse_deg: float
    final_bias_true_deg_s: float
    final_bias_estimate_deg_s: float
    peak_torque_nm: float


@dataclass(frozen=True)
class GimbalRollout:
    """One estimator and controller replay on the common scenario."""

    estimator: EstimatorKind
    label: str
    time_s: np.ndarray
    state: np.ndarray
    observation: np.ndarray
    estimate: np.ndarray
    torque_nm: np.ndarray
    base_acceleration_mps2: np.ndarray
    disturbance_torque_nm: np.ndarray
    metrics: GimbalMetrics


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle to ``[-pi, pi)``."""

    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def base_acceleration(time_s: float, scenario: GimbalScenario) -> float:
    """Raised-cosine lateral acceleration pulse."""

    start = scenario.acceleration_start_s
    ramp = scenario.acceleration_ramp_s
    plateau_end = scenario.acceleration_plateau_end_s
    end = scenario.acceleration_end_s
    amplitude = scenario.acceleration_mps2
    if start <= time_s < start + ramp:
        phase = np.pi * (time_s - start) / ramp
        return float(0.5 * amplitude * (1.0 - np.cos(phase)))
    if start + ramp <= time_s < plateau_end:
        return float(amplitude)
    if plateau_end <= time_s < end:
        phase = np.pi * (time_s - plateau_end) / ramp
        return float(0.5 * amplitude * (1.0 + np.cos(phase)))
    return 0.0


def disturbance_torque(time_s: float, scenario: GimbalScenario) -> float:
    """Short mechanical tap applied to every plant."""

    if scenario.tap_start_s <= time_s < scenario.tap_end_s:
        return float(scenario.tap_torque_nm)
    return 0.0


def continuous_dynamics(
    time_s: float,
    state: np.ndarray,
    torque_nm: float,
    parameters: GimbalParameters,
    scenario: GimbalScenario,
) -> np.ndarray:
    """Continuous one-axis rigid-gimbal dynamics."""

    angle_rad, angular_velocity_rad_s = np.asarray(state, dtype=float)
    acceleration = (
        torque_nm
        - parameters.damping_nm_s_per_rad * angular_velocity_rad_s
        + disturbance_torque(time_s, scenario)
    ) / parameters.inertia_kg_m2
    return np.array([angular_velocity_rad_s, acceleration], dtype=float)


def _rk4_control_period(
    time_s: float,
    state: np.ndarray,
    torque_nm: float,
    parameters: GimbalParameters,
    scenario: GimbalScenario,
) -> np.ndarray:
    step = scenario.integration_step_s
    result = np.asarray(state, dtype=float).copy()
    for substep in range(scenario.integration_substeps):
        local_time = time_s + substep * step
        k1 = continuous_dynamics(local_time, result, torque_nm, parameters, scenario)
        k2 = continuous_dynamics(
            local_time + 0.5 * step,
            result + 0.5 * step * k1,
            torque_nm,
            parameters,
            scenario,
        )
        k3 = continuous_dynamics(
            local_time + 0.5 * step,
            result + 0.5 * step * k2,
            torque_nm,
            parameters,
            scenario,
        )
        k4 = continuous_dynamics(
            local_time + step,
            result + step * k3,
            torque_nm,
            parameters,
            scenario,
        )
        result = result + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return result


def generate_noise(
    parameters: GimbalParameters,
    scenario: GimbalScenario,
) -> NoiseRealization:
    """Generate one bias path and sensor-noise draw for every controller."""

    parameters.validate()
    scenario.validate()
    generator = np.random.default_rng(scenario.seed)
    count = scenario.control_steps
    gyro_noise = generator.normal(0.0, parameters.gyro_noise_std_rad_s, count)
    accelerometer_noise = generator.normal(
        0.0,
        parameters.accelerometer_noise_std_mps2,
        (count, 2),
    )
    bias_innovations = generator.normal(0.0, 1.0, count)
    bias = np.empty(count + 1, dtype=float)
    bias[0] = scenario.initial_gyro_bias_rad_s
    diffusion = (
        parameters.bias_random_walk_std_rad_s_sqrt_s
        * np.sqrt(scenario.sensor_period_s)
    )
    for index in range(count):
        bias[index + 1] = bias[index] + diffusion * bias_innovations[index]
    return NoiseRealization(gyro_noise, accelerometer_noise, bias)


def sample_observation(
    state: np.ndarray,
    gyro_bias_rad_s: float,
    lateral_acceleration_mps2: float,
    gyro_noise_rad_s: float,
    accelerometer_noise_mps2: np.ndarray,
    parameters: GimbalParameters,
) -> GimbalObservation:
    """Sample a gyro and two-axis accelerometer in the camera frame."""

    angle_rad, angular_velocity_rad_s = np.asarray(state, dtype=float)
    noise = np.asarray(accelerometer_noise_mps2, dtype=float)
    if noise.shape != (2,):
        raise ValueError("accelerometer_noise_mps2 must have shape (2,)")
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    accelerometer_x = (
        cosine * lateral_acceleration_mps2
        + sine * parameters.gravity_mps2
        + noise[0]
    )
    accelerometer_y = (
        -sine * lateral_acceleration_mps2
        + cosine * parameters.gravity_mps2
        + noise[1]
    )
    return GimbalObservation(
        gyro_rad_s=float(angular_velocity_rad_s + gyro_bias_rad_s + gyro_noise_rad_s),
        accelerometer_x_mps2=float(accelerometer_x),
        accelerometer_y_mps2=float(accelerometer_y),
    )


def update_estimator(
    estimator: EstimatorKind,
    memory: EstimatorMemory | None,
    observation: GimbalObservation,
    parameters: GimbalParameters,
    scenario: GimbalScenario,
) -> tuple[EstimatorMemory, np.ndarray]:
    """Update one transparent attitude estimator and return ``[angle, rate, bias]``."""

    accelerometer_angle = observation.accelerometer_angle_rad
    if memory is None:
        memory = EstimatorMemory(accelerometer_angle, 0.0)
        return memory, np.array(
            [accelerometer_angle, observation.gyro_rad_s, 0.0], dtype=float
        )

    if estimator == "accelerometer":
        updated = EstimatorMemory(accelerometer_angle, 0.0)
        estimate = np.array(
            [accelerometer_angle, observation.gyro_rad_s, 0.0], dtype=float
        )
    elif estimator == "gyro":
        angle = float(
            wrap_angle(memory.angle_rad + scenario.sensor_period_s * observation.gyro_rad_s)
        )
        updated = EstimatorMemory(angle, 0.0)
        estimate = np.array([angle, observation.gyro_rad_s, 0.0], dtype=float)
    elif estimator == "complementary":
        innovation = float(wrap_angle(accelerometer_angle - memory.angle_rad))
        angle = float(
            wrap_angle(
                memory.angle_rad
                + scenario.sensor_period_s
                * (
                    observation.gyro_rad_s
                    - memory.bias_rad_s
                    + parameters.observer_angle_gain_per_s * innovation
                )
            )
        )
        bias = float(
            memory.bias_rad_s
            - parameters.observer_bias_gain_per_s2
            * scenario.sensor_period_s
            * innovation
        )
        updated = EstimatorMemory(angle, bias)
        estimate = np.array(
            [angle, observation.gyro_rad_s - bias, bias], dtype=float
        )
    else:
        raise ValueError(f"unknown estimator: {estimator}")
    return updated, estimate


def saturated_pd(estimate: np.ndarray, parameters: GimbalParameters) -> float:
    """Apply the common saturated feedback law to an estimated state."""

    angle_rad, angular_velocity_rad_s = np.asarray(estimate, dtype=float)[:2]
    requested = -(
        parameters.controller_kp_nm_per_rad * angle_rad
        + parameters.controller_kd_nm_s_per_rad * angular_velocity_rad_s
    )
    return float(
        np.clip(requested, -parameters.torque_limit_nm, parameters.torque_limit_nm)
    )


def compute_metrics(
    time_s: np.ndarray,
    state: np.ndarray,
    estimate: np.ndarray,
    torque_nm: np.ndarray,
    parameters: GimbalParameters,
    scenario: GimbalScenario,
) -> GimbalMetrics:
    """Compute diagnostics from the true simulated state."""

    angle = np.asarray(wrap_angle(state[:, 0]), dtype=float)
    angle_deg = np.rad2deg(angle)
    acceleration_window = (time_s >= scenario.acceleration_start_s) & (
        time_s <= scenario.acceleration_end_s + 0.9
    )
    normalized_effort = float(np.mean((torque_nm / parameters.torque_limit_nm) ** 2))
    regulation_score = float(
        np.mean(
            (angle[:-1] / np.deg2rad(5.0)) ** 2
            + 0.05 * (torque_nm / parameters.torque_limit_nm) ** 2
        )
    )
    estimate_error = np.asarray(
        wrap_angle(estimate[:, 0] - state[:-1, 0]), dtype=float
    )
    return GimbalMetrics(
        rms_angle_deg=float(np.sqrt(np.mean(angle_deg**2))),
        peak_acceleration_window_deg=float(
            np.max(np.abs(angle_deg[acceleration_window]))
        ),
        final_abs_angle_deg=float(abs(angle_deg[-1])),
        normalized_torque_effort=normalized_effort,
        saturation_fraction=float(
            np.mean(
                np.isclose(
                    np.abs(torque_nm),
                    parameters.torque_limit_nm,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        ),
        regulation_score=regulation_score,
        estimator_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(estimate_error**2)))),
        final_bias_true_deg_s=float(np.rad2deg(state[-1, 2])),
        final_bias_estimate_deg_s=float(np.rad2deg(estimate[-1, 2])),
        peak_torque_nm=float(np.max(np.abs(torque_nm))),
    )


def simulate_closed_loop(
    estimator: EstimatorKind,
    parameters: GimbalParameters | None = None,
    scenario: GimbalScenario | None = None,
    noise: NoiseRealization | None = None,
) -> GimbalRollout:
    """Simulate one estimator with the common plant and feedback law."""

    parameters = parameters or GimbalParameters()
    scenario = scenario or GimbalScenario()
    parameters.validate()
    scenario.validate()
    if estimator not in ESTIMATOR_ORDER:
        raise ValueError(f"unknown estimator: {estimator}")
    noise = noise or generate_noise(parameters, scenario)
    count = scenario.control_steps
    if (
        noise.gyro_noise_rad_s.shape != (count,)
        or noise.accelerometer_noise_mps2.shape != (count, 2)
        or noise.gyro_bias_rad_s.shape != (count + 1,)
    ):
        raise ValueError("noise realization does not match the scenario horizon")

    time_s = np.arange(count + 1, dtype=float) * scenario.sensor_period_s
    state = np.empty((count + 1, 3), dtype=float)
    state[0] = (
        scenario.initial_angle_rad,
        scenario.initial_angular_velocity_rad_s,
        noise.gyro_bias_rad_s[0],
    )
    observation = np.empty((count, 3), dtype=float)
    estimate = np.empty((count, 3), dtype=float)
    torque = np.empty(count, dtype=float)
    lateral_acceleration = np.array(
        [base_acceleration(value, scenario) for value in time_s], dtype=float
    )
    tap = np.array(
        [disturbance_torque(value, scenario) for value in time_s], dtype=float
    )
    memory: EstimatorMemory | None = None

    for index in range(count):
        sampled = sample_observation(
            state[index, :2],
            noise.gyro_bias_rad_s[index],
            lateral_acceleration[index],
            noise.gyro_noise_rad_s[index],
            noise.accelerometer_noise_mps2[index],
            parameters,
        )
        memory, estimate[index] = update_estimator(
            estimator, memory, sampled, parameters, scenario
        )
        torque[index] = saturated_pd(estimate[index], parameters)
        state[index + 1, :2] = _rk4_control_period(
            time_s[index],
            state[index, :2],
            torque[index],
            parameters,
            scenario,
        )
        state[index + 1, 2] = noise.gyro_bias_rad_s[index + 1]
        observation[index] = (
            sampled.gyro_rad_s,
            sampled.accelerometer_x_mps2,
            sampled.accelerometer_y_mps2,
        )

    metrics = compute_metrics(
        time_s, state, estimate, torque, parameters, scenario
    )
    return GimbalRollout(
        estimator=estimator,
        label=ESTIMATOR_LABELS[estimator],
        time_s=time_s,
        state=state,
        observation=observation,
        estimate=estimate,
        torque_nm=torque,
        base_acceleration_mps2=lateral_acceleration,
        disturbance_torque_nm=tap,
        metrics=metrics,
    )


def run_comparison(
    parameters: GimbalParameters | None = None,
    scenario: GimbalScenario | None = None,
) -> dict[EstimatorKind, GimbalRollout]:
    """Run all estimators with one common random realization."""

    parameters = parameters or GimbalParameters()
    scenario = scenario or GimbalScenario()
    noise = generate_noise(parameters, scenario)
    return {
        estimator: simulate_closed_loop(estimator, parameters, scenario, noise)
        for estimator in ESTIMATOR_ORDER
    }


def _draw_gimbal_snapshot(
    axis: plt.Axes,
    rollout: GimbalRollout,
    index: int,
) -> None:
    angle = float(rollout.state[index, 0])
    estimate_index = min(index, rollout.estimate.shape[0] - 1)
    estimate = float(rollout.estimate[estimate_index, 0])
    color, line_style = METHOD_STYLES[rollout.estimator]
    pivot = np.array([0.0, 0.18])

    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-0.68, 0.86)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.axhline(pivot[1], color=TEAL, linewidth=0.8, alpha=0.75)
    axis.text(
        -0.95,
        0.70,
        "world horizon",
        color=TEAL,
        fontsize=6.8,
        va="bottom",
    )

    axis.add_patch(
        Rectangle(
            (-0.34, -0.57),
            0.68,
            0.10,
            facecolor="#E8ECEB",
            edgecolor=RULE,
            linewidth=0.8,
        )
    )
    axis.plot([0.0, pivot[0]], [-0.47, pivot[1]], color=MUTED, linewidth=2.0)
    axis.scatter([pivot[0]], [pivot[1]], s=26, color=PAPER, edgecolor=INK, zorder=8)

    body = np.array(
        [[-0.48, -0.13], [0.34, -0.13], [0.34, 0.13], [-0.48, 0.13]]
    )
    lens = np.array([[0.34, -0.085], [0.55, -0.055], [0.55, 0.055], [0.34, 0.085]])
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    body = body @ rotation.T + pivot
    lens = lens @ rotation.T + pivot
    axis.add_patch(
        Polygon(body, closed=True, facecolor=INK, edgecolor=INK, linewidth=0.7, zorder=5)
    )
    axis.add_patch(
        Polygon(lens, closed=True, facecolor=TEAL, edgecolor=INK, linewidth=0.6, zorder=6)
    )
    estimate_tip = pivot + 0.72 * np.array([np.cos(estimate), np.sin(estimate)])
    axis.plot(
        [pivot[0], estimate_tip[0]],
        [pivot[1], estimate_tip[1]],
        color=color,
        linestyle=line_style,
        linewidth=1.4,
        zorder=7,
    )

    if rollout.base_acceleration_mps2[index] > 0.1:
        axis.add_patch(
            FancyArrowPatch(
                (-0.6, -0.39),
                (0.62, -0.39),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.1,
                color=CAVEAT,
            )
        )
        axis.text(
            0.0,
            -0.34,
            "$a_x=3$ m/s²",
            color=CAVEAT,
            fontsize=7.0,
            ha="center",
            va="bottom",
        )

    axis.set_title(rollout.label, color=INK, pad=2.0)
    axis.text(
        0.0,
        -0.65,
        f"true {np.rad2deg(angle):+.1f}°   estimate {np.rad2deg(estimate):+.1f}°",
        color=color,
        fontsize=7.0,
        ha="center",
        va="bottom",
    )


def make_summary_figure(
    results: Mapping[EstimatorKind, GimbalRollout],
    snapshot_time_s: float = 4.35,
) -> plt.Figure:
    """Create the vector static fallback from the full-resolution trajectories."""

    missing = [name for name in ESTIMATOR_ORDER if name not in results]
    if missing:
        raise ValueError(f"missing estimator results: {missing}")
    reference = results[ESTIMATOR_ORDER[0]]
    snapshot_index = int(np.argmin(np.abs(reference.time_s - snapshot_time_s)))

    with mpl.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.2, 4.25), constrained_layout=True)
        grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 1.15))
        for column, estimator in enumerate(ESTIMATOR_ORDER):
            _draw_gimbal_snapshot(
                figure.add_subplot(grid[0, column]),
                results[estimator],
                snapshot_index,
            )

        axis = figure.add_subplot(grid[1, :])
        axis.axhline(0.0, color=TEAL, linewidth=0.8, zorder=0)
        axis.axvspan(4.0, 4.6, color=CAVEAT, alpha=0.13, linewidth=0.0)
        axis.axvline(1.50, color=MUTED, linestyle=(0, (2, 2)), linewidth=0.8)
        for estimator in ESTIMATOR_ORDER:
            rollout = results[estimator]
            color, line_style = METHOD_STYLES[estimator]
            axis.plot(
                rollout.time_s,
                np.rad2deg(wrap_angle(rollout.state[:, 0])),
                color=color,
                linestyle=line_style,
            )
        axis.annotate(
            "mechanical tap",
            xy=(1.50, 0.0),
            xytext=(1.72, 10.5),
            color=MUTED,
            fontsize=7.2,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.7},
        )
        axis.text(
            4.30,
            10.5,
            "lateral acceleration",
            color=CAVEAT,
            fontsize=7.2,
            ha="center",
        )
        axis.annotate(
            "accelerometer as state",
            xy=(
                4.43,
                np.rad2deg(
                    results["accelerometer"].state[
                        int(np.argmin(np.abs(results["accelerometer"].time_s - 4.43))),
                        0,
                    ]
                ),
            ),
            xytext=(5.0, -18.5),
            color=CAVEAT,
            fontsize=7.3,
            arrowprops={"arrowstyle": "-", "color": CAVEAT, "linewidth": 0.7},
        )
        axis.annotate(
            "complementary observer",
            xy=(
                5.1,
                np.rad2deg(
                    results["complementary"].state[
                        int(np.argmin(np.abs(results["complementary"].time_s - 5.1))),
                        0,
                    ]
                ),
            ),
            xytext=(5.55, -9.0),
            color=STANDS,
            fontsize=7.3,
            arrowprops={"arrowstyle": "-", "color": STANDS, "linewidth": 0.7},
        )
        axis.annotate(
            "integrated gyro",
            xy=(10.0, np.rad2deg(results["gyro"].state[-1, 0])),
            xytext=(8.05, -15.0),
            color=CAVEAT,
            fontsize=7.3,
            arrowprops={"arrowstyle": "-", "color": CAVEAT, "linewidth": 0.7},
        )
        axis.set_xlim(0.0, 10.0)
        axis.set_ylim(-23.0, 14.0)
        axis.set_xlabel("time (s)")
        axis.set_ylabel("true camera angle (degrees)")
        axis.spines["left"].set_color(RULE)
        axis.spines["bottom"].set_color(RULE)
        axis.tick_params(colors=MUTED)
        axis.xaxis.label.set_color(INK)
        axis.yaxis.label.set_color(INK)
        figure.suptitle(
            "The same controller acts on three state estimates",
            color=INK,
            fontsize=13,
            fontfamily="serif",
            fontweight="normal",
        )
        return figure


def comparison_to_artifact(
    results: Mapping[EstimatorKind, GimbalRollout],
    parameters: GimbalParameters,
    scenario: GimbalScenario,
    frame_stride: int = 4,
) -> dict[str, object]:
    """Serialize a compact, causal browser replay plus full-resolution metrics."""

    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    runs: dict[str, object] = {}
    for estimator in ESTIMATOR_ORDER:
        rollout = results[estimator]
        final_state_index = rollout.time_s.size - 1
        indices = np.arange(0, rollout.time_s.size, frame_stride, dtype=int)
        if indices[-1] != final_state_index:
            indices = np.concatenate([indices, [final_state_index]])
        accelerometer_angle = np.rad2deg(
            np.arctan2(rollout.observation[:, 1], rollout.observation[:, 2])
        )
        runs[estimator] = {
            "label": rollout.label,
            "style": {
                "color": METHOD_STYLES[estimator][0],
                "dash": METHOD_STYLES[estimator][1],
            },
            "frames": [
                {
                    "time_s": float(rollout.time_s[state_index]),
                    "true_angle_deg": float(
                        np.rad2deg(wrap_angle(rollout.state[state_index, 0]))
                    ),
                    "estimated_angle_deg": float(
                        np.rad2deg(wrap_angle(rollout.estimate[sample_index, 0]))
                    ),
                    "true_bias_deg_s": float(
                        np.rad2deg(rollout.state[state_index, 2])
                    ),
                    "estimated_bias_deg_s": float(
                        np.rad2deg(rollout.estimate[sample_index, 2])
                    ),
                    "torque_nm": float(rollout.torque_nm[sample_index]),
                    "base_acceleration_mps2": float(
                        rollout.base_acceleration_mps2[state_index]
                    ),
                    "tap_torque_nm": float(
                        rollout.disturbance_torque_nm[state_index]
                    ),
                    "accelerometer_angle_deg": float(
                        accelerometer_angle[sample_index]
                    ),
                }
                for state_index in indices
                for sample_index in [
                    min(state_index, rollout.torque_nm.size - 1)
                ]
            ],
            "metrics": asdict(rollout.metrics),
        }
    return {
        "schema_version": 1,
        "title": "Partial observation in camera stabilization",
        "description": (
            "Recorded Python trajectories for three estimators under one plant, "
            "controller, disturbance sequence, and random realization."
        ),
        "parameters": asdict(parameters),
        "scenario": asdict(scenario),
        "frame_stride": frame_stride,
        "playback_fps": 25,
        "runs": runs,
    }


def save_artifact(artifact: Mapping[str, object], destination: str | Path) -> Path:
    """Write a deterministic JSON replay artifact."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def metrics_rows(
    results: Mapping[EstimatorKind, GimbalRollout],
) -> list[dict[str, object]]:
    """Return one flat metrics row per estimator."""

    return [
        {
            "estimator": estimator,
            "label": results[estimator].label,
            **asdict(results[estimator].metrics),
        }
        for estimator in ESTIMATOR_ORDER
    ]


__all__ = [
    "ESTIMATOR_LABELS",
    "ESTIMATOR_ORDER",
    "EstimatorKind",
    "EstimatorMemory",
    "GimbalMetrics",
    "GimbalObservation",
    "GimbalParameters",
    "GimbalRollout",
    "GimbalScenario",
    "NoiseRealization",
    "base_acceleration",
    "comparison_to_artifact",
    "compute_metrics",
    "continuous_dynamics",
    "disturbance_torque",
    "generate_noise",
    "make_summary_figure",
    "metrics_rows",
    "run_comparison",
    "sample_observation",
    "saturated_pd",
    "save_artifact",
    "simulate_closed_loop",
    "update_estimator",
    "wrap_angle",
]
