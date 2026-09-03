"""Differential-drag phasing for a three-satellite CubeSat formation.

The planning model is a 180-step linear program with one daily high-drag duty
fraction per spacecraft.  The optimized, immutable plan is then replayed in a
separate nonlinear orbital model with altitude- and time-varying density.  All
phase coordinates are unwrapped and measured relative to an all-low-drag
reference orbit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any

import numpy as np
from scipy.optimize import linprog


DAY_SECONDS = 86_400.0
RAD_TO_DEG = 180.0 / np.pi
STATE_COMPONENTS = (
    "phase_deg",
    "relative_rate_deg_per_day",
    "extra_altitude_loss_km",
)
SATELLITE_NAMES = ("satellite_1", "satellite_2", "satellite_3")

REFERENCE_ALPHA_DEG_PER_DAY2 = 0.05445030597065867
REFERENCE_D_KM_PER_DAY = 0.045157234506305814


def _readonly_array(
    values: np.ndarray | tuple[float, ...] | list[float],
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, received {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CubeSatParameters:
    """Fixed physical constants and numerical tolerances for the scenario."""

    satellite_count: int = 3
    horizon_days: int = 180
    interval_days: float = 1.0
    earth_radius_km: float = 6_378.137
    initial_altitude_km: float = 475.0
    gravitational_parameter_m3_s2: float = 3.986004418e14
    reference_density_kg_m3: float = 3.0e-13
    low_drag_ballistic_coefficient_kg_m2: float = 60.0
    high_drag_ballistic_coefficient_kg_m2: float = 20.0
    density_mean_factor: float = 0.90
    density_amplitude: float = 0.15
    density_period_days: float = 26.0
    density_phase_rad: float = np.pi / 6.0
    density_scale_height_km: float = 60.0
    initial_phase_deg: tuple[float, float, float] = (-0.5, 0.0, 0.5)
    target_cyclic_gaps_deg: tuple[float, float, float] = (120.0, 120.0, -240.0)
    gap_tolerance_deg: float = 0.1
    cyclic_rate_tolerance_deg_per_day: float = 0.002
    primary_lock_tolerance_km: float = 1.0e-9

    def __post_init__(self) -> None:
        if self.satellite_count != 3:
            raise ValueError("this scenario requires exactly three satellites")
        if self.horizon_days != 180 or self.interval_days != 1.0:
            raise ValueError("this scenario requires 180 one-day control intervals")
        positive = (
            self.earth_radius_km,
            self.initial_altitude_km,
            self.gravitational_parameter_m3_s2,
            self.reference_density_kg_m3,
            self.low_drag_ballistic_coefficient_kg_m2,
            self.high_drag_ballistic_coefficient_kg_m2,
            self.density_period_days,
            self.density_scale_height_km,
            self.gap_tolerance_deg,
            self.cyclic_rate_tolerance_deg_per_day,
            self.primary_lock_tolerance_km,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("physical constants and tolerances must be finite and positive")
        if self.high_drag_ballistic_coefficient_kg_m2 >= self.low_drag_ballistic_coefficient_kg_m2:
            raise ValueError("the high-drag ballistic coefficient must be smaller")
        if self.density_mean_factor <= abs(self.density_amplitude):
            raise ValueError("the density modulation must remain positive")

    @property
    def initial_semi_major_axis_m(self) -> float:
        return (self.earth_radius_km + self.initial_altitude_km) * 1_000.0

    @property
    def initial_mean_motion_rad_s(self) -> float:
        return float(
            np.sqrt(
                self.gravitational_parameter_m3_s2
                / self.initial_semi_major_axis_m**3
            )
        )

    @property
    def differential_inverse_ballistic_coefficient_m2_kg(self) -> float:
        return (
            1.0 / self.high_drag_ballistic_coefficient_kg_m2
            - 1.0 / self.low_drag_ballistic_coefficient_kg_m2
        )

    @property
    def d_km_per_day(self) -> float:
        """Extra daily altitude loss caused by a full day in high drag."""

        extra_decay_m_s = (
            self.reference_density_kg_m3
            * self.differential_inverse_ballistic_coefficient_m2_kg
            * np.sqrt(
                self.gravitational_parameter_m3_s2
                * self.initial_semi_major_axis_m
            )
        )
        return float(extra_decay_m_s * DAY_SECONDS / 1_000.0)

    @property
    def alpha_deg_per_day2(self) -> float:
        """Relative angular acceleration caused by a full day in high drag."""

        extra_decay_m_s = self.d_km_per_day * 1_000.0 / DAY_SECONDS
        angular_acceleration_rad_s2 = (
            1.5
            * self.initial_mean_motion_rad_s
            / self.initial_semi_major_axis_m
            * extra_decay_m_s
        )
        return float(angular_acceleration_rad_s2 * RAD_TO_DEG * DAY_SECONDS**2)

    @property
    def linear_A(self) -> np.ndarray:
        return _readonly_array(
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            shape=(3, 3),
        )

    @property
    def linear_B(self) -> np.ndarray:
        alpha = self.alpha_deg_per_day2
        return _readonly_array(
            [0.5 * alpha, alpha, self.d_km_per_day],
            shape=(3,),
        )

    @property
    def gap_matrix(self) -> np.ndarray:
        """Map satellite values to the three directed cyclic differences."""

        return _readonly_array(
            [[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 0.0, -1.0]],
            shape=(3, 3),
        )


@dataclass(frozen=True)
class CubeSatPlan:
    """Primary and TV-refined daily plans returned by the two LP solves."""

    daily_high_drag_fraction: np.ndarray
    primary_daily_high_drag_fraction: np.ndarray
    primary_max_final_extra_loss_km: float
    refined_max_final_extra_loss_km: float
    primary_total_variation: float
    refined_total_variation: float
    primary_solver_status: int
    refinement_solver_status: int
    primary_solver_message: str
    refinement_solver_message: str
    plan_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        refined = _readonly_array(self.daily_high_drag_fraction, shape=(180, 3))
        primary = _readonly_array(
            self.primary_daily_high_drag_fraction,
            shape=(180, 3),
        )
        object.__setattr__(self, "daily_high_drag_fraction", refined)
        object.__setattr__(self, "primary_daily_high_drag_fraction", primary)
        digest = hashlib.sha256()
        digest.update(b"cubesat-differential-drag-plan-v1\0")
        digest.update(np.ascontiguousarray(refined, dtype="<f8").tobytes())
        object.__setattr__(self, "plan_sha256", digest.hexdigest())


@dataclass(frozen=True)
class LinearRollout:
    """Daily rollout of the three-state linear planning model."""

    plan_sha256: str
    time_days: np.ndarray
    state: np.ndarray
    cyclic_gaps_deg: np.ndarray
    cyclic_relative_rates_deg_per_day: np.ndarray
    altitude_km: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_days", _readonly_array(self.time_days, shape=(181,)))
        object.__setattr__(self, "state", _readonly_array(self.state, shape=(181, 3, 3)))
        object.__setattr__(
            self,
            "cyclic_gaps_deg",
            _readonly_array(self.cyclic_gaps_deg, shape=(181, 3)),
        )
        object.__setattr__(
            self,
            "cyclic_relative_rates_deg_per_day",
            _readonly_array(self.cyclic_relative_rates_deg_per_day, shape=(181, 3)),
        )
        object.__setattr__(
            self,
            "altitude_km",
            _readonly_array(self.altitude_km, shape=(181, 3)),
        )


@dataclass(frozen=True)
class NonlinearReplay:
    """Variable-density orbital replay sampled at a fixed sub-daily step."""

    plan_sha256: str
    step_hours: float
    time_days: np.ndarray
    state: np.ndarray
    cyclic_gaps_deg: np.ndarray
    cyclic_relative_rates_deg_per_day: np.ndarray
    altitude_km: np.ndarray
    reference_altitude_km: np.ndarray
    density_kg_m3: np.ndarray
    reference_density_kg_m3: np.ndarray

    def __post_init__(self) -> None:
        samples = len(self.time_days)
        object.__setattr__(
            self,
            "time_days",
            _readonly_array(self.time_days, shape=(samples,)),
        )
        for name in (
            "cyclic_gaps_deg",
            "cyclic_relative_rates_deg_per_day",
            "altitude_km",
            "density_kg_m3",
        ):
            object.__setattr__(
                self,
                name,
                _readonly_array(getattr(self, name), shape=(samples, 3)),
            )
        object.__setattr__(
            self,
            "state",
            _readonly_array(self.state, shape=(samples, 3, 3)),
        )
        for name in ("reference_altitude_km", "reference_density_kg_m3"):
            object.__setattr__(
                self,
                name,
                _readonly_array(getattr(self, name), shape=(samples,)),
            )


@dataclass(frozen=True)
class ReplayResolutionCheck:
    """Maximum hourly-versus-reference replay differences at common times."""

    coarse_step_hours: float
    fine_step_hours: float
    max_phase_delta_deg: float
    max_relative_rate_delta_deg_per_day: float
    max_extra_loss_delta_km: float
    max_altitude_delta_km: float
    max_density_delta_kg_m3: float
    terminal_gap_delta_deg: float
    terminal_cyclic_rate_delta_deg_per_day: float


@dataclass(frozen=True)
class CubeSatMetrics:
    """Scalar and per-satellite diagnostics used by tests and artifacts."""

    plan_sha256: str
    alpha_deg_per_day2: float
    d_km_per_day: float
    primary_max_final_extra_loss_km: float
    refined_max_final_extra_loss_km: float
    primary_total_variation: float
    refined_total_variation: float
    action_min: float
    action_max: float
    equivalent_high_drag_days: np.ndarray
    max_nominal_dynamics_residual: float
    nominal_final_extra_loss_km: np.ndarray
    nominal_terminal_cyclic_gaps_deg: np.ndarray
    nominal_terminal_gap_error_deg: np.ndarray
    nominal_max_gap_error_deg: float
    nominal_terminal_cyclic_rates_deg_per_day: np.ndarray
    nominal_max_cyclic_rate_deg_per_day: float
    nonlinear_final_extra_loss_km: np.ndarray
    nonlinear_final_altitude_km: np.ndarray
    nonlinear_terminal_cyclic_gaps_deg: np.ndarray
    nonlinear_terminal_gap_error_deg: np.ndarray
    nonlinear_max_gap_error_deg: float
    nonlinear_terminal_cyclic_rates_deg_per_day: np.ndarray
    nonlinear_max_cyclic_rate_deg_per_day: float
    nonlinear_min_altitude_km: float
    nonlinear_reference_final_altitude_km: float
    density_min_kg_m3: float
    density_max_kg_m3: float
    replay_max_phase_delta_deg: float
    replay_max_relative_rate_delta_deg_per_day: float
    replay_max_extra_loss_delta_km: float
    replay_max_altitude_delta_km: float

    def __post_init__(self) -> None:
        for name in (
            "equivalent_high_drag_days",
            "nominal_final_extra_loss_km",
            "nominal_terminal_cyclic_gaps_deg",
            "nominal_terminal_gap_error_deg",
            "nominal_terminal_cyclic_rates_deg_per_day",
            "nonlinear_final_extra_loss_km",
            "nonlinear_final_altitude_km",
            "nonlinear_terminal_cyclic_gaps_deg",
            "nonlinear_terminal_gap_error_deg",
            "nonlinear_terminal_cyclic_rates_deg_per_day",
        ):
            object.__setattr__(
                self,
                name,
                _readonly_array(getattr(self, name), shape=(3,)),
            )


@dataclass(frozen=True)
class CubeSatScenario:
    """Complete deterministic solve and independent replay bundle."""

    parameters: CubeSatParameters
    plan: CubeSatPlan
    nominal: LinearRollout
    nonlinear: NonlinearReplay
    resolution_check: ReplayResolutionCheck
    metrics: CubeSatMetrics


def _append_absolute_constraint(
    rows: list[np.ndarray],
    bounds: list[float],
    row: np.ndarray,
    constant: float,
    target: float,
    tolerance: float,
) -> None:
    rows.append(row)
    bounds.append(target + tolerance - constant)
    rows.append(-row)
    bounds.append(-target + tolerance + constant)


def _terminal_inequalities(
    params: CubeSatParameters,
    variable_count: int,
) -> tuple[list[np.ndarray], list[float]]:
    intervals = params.horizon_days
    action_count = intervals * params.satellite_count
    alpha = params.alpha_deg_per_day2
    phase_weights = alpha * (intervals - np.arange(intervals) - 0.5)
    gap_matrix = params.gap_matrix
    initial_phase = np.asarray(params.initial_phase_deg)
    target_gap = np.asarray(params.target_cyclic_gaps_deg)
    rows: list[np.ndarray] = []
    bounds: list[float] = []

    for gap_index in range(3):
        row = np.zeros(variable_count)
        row[:action_count] = np.outer(
            phase_weights,
            gap_matrix[gap_index],
        ).ravel()
        _append_absolute_constraint(
            rows,
            bounds,
            row,
            float(gap_matrix[gap_index] @ initial_phase),
            float(target_gap[gap_index]),
            params.gap_tolerance_deg,
        )

    for gap_index in range(3):
        row = np.zeros(variable_count)
        row[:action_count] = np.tile(alpha * gap_matrix[gap_index], intervals)
        _append_absolute_constraint(
            rows,
            bounds,
            row,
            0.0,
            0.0,
            params.cyclic_rate_tolerance_deg_per_day,
        )

    return rows, bounds


def _total_variation(action: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(action, axis=0))))


def solve_lexicographic_plan(
    params: CubeSatParameters | None = None,
) -> CubeSatPlan:
    """Minimize maximum extra loss, then total variation at that optimum."""

    params = params or CubeSatParameters()
    intervals = params.horizon_days
    satellites = params.satellite_count
    action_count = intervals * satellites
    d_km = params.d_km_per_day

    primary_variable_count = action_count + 1
    primary_epigraph_index = action_count
    primary_objective = np.zeros(primary_variable_count)
    primary_objective[primary_epigraph_index] = 1.0
    primary_rows, primary_bounds = _terminal_inequalities(
        params,
        primary_variable_count,
    )
    for satellite in range(satellites):
        row = np.zeros(primary_variable_count)
        row[satellite:action_count:satellites] = d_km
        row[primary_epigraph_index] = -1.0
        primary_rows.append(row)
        primary_bounds.append(0.0)

    primary_result = linprog(
        primary_objective,
        A_ub=np.asarray(primary_rows),
        b_ub=np.asarray(primary_bounds),
        bounds=[(0.0, 1.0)] * action_count + [(0.0, None)],
        method="highs",
    )
    if not primary_result.success:
        raise RuntimeError(f"primary CubeSat LP failed: {primary_result.message}")
    primary_action = np.clip(
        primary_result.x[:action_count].reshape(intervals, satellites),
        0.0,
        1.0,
    )
    primary_optimum = float(primary_result.fun)

    variation_count = (intervals - 1) * satellites
    refined_variable_count = action_count + variation_count
    refined_objective = np.concatenate(
        [np.zeros(action_count), np.ones(variation_count)]
    )
    refined_rows, refined_bounds = _terminal_inequalities(
        params,
        refined_variable_count,
    )
    for satellite in range(satellites):
        row = np.zeros(refined_variable_count)
        row[satellite:action_count:satellites] = d_km
        refined_rows.append(row)
        refined_bounds.append(primary_optimum + params.primary_lock_tolerance_km)

    for day in range(intervals - 1):
        for satellite in range(satellites):
            variation_index = action_count + day * satellites + satellite
            forward = np.zeros(refined_variable_count)
            forward[(day + 1) * satellites + satellite] = 1.0
            forward[day * satellites + satellite] = -1.0
            forward[variation_index] = -1.0
            refined_rows.append(forward)
            refined_bounds.append(0.0)

            backward = np.zeros(refined_variable_count)
            backward[day * satellites + satellite] = 1.0
            backward[(day + 1) * satellites + satellite] = -1.0
            backward[variation_index] = -1.0
            refined_rows.append(backward)
            refined_bounds.append(0.0)

    refined_result = linprog(
        refined_objective,
        A_ub=np.asarray(refined_rows),
        b_ub=np.asarray(refined_bounds),
        bounds=[(0.0, 1.0)] * action_count
        + [(0.0, None)] * variation_count,
        method="highs",
    )
    if not refined_result.success:
        raise RuntimeError(f"CubeSat TV refinement failed: {refined_result.message}")
    refined_action = np.clip(
        refined_result.x[:action_count].reshape(intervals, satellites),
        0.0,
        1.0,
    )
    refined_loss = d_km * np.sum(refined_action, axis=0)

    return CubeSatPlan(
        daily_high_drag_fraction=refined_action,
        primary_daily_high_drag_fraction=primary_action,
        primary_max_final_extra_loss_km=primary_optimum,
        refined_max_final_extra_loss_km=float(np.max(refined_loss)),
        primary_total_variation=_total_variation(primary_action),
        refined_total_variation=_total_variation(refined_action),
        primary_solver_status=int(primary_result.status),
        refinement_solver_status=int(refined_result.status),
        primary_solver_message=str(primary_result.message),
        refinement_solver_message=str(refined_result.message),
    )


def rollout_nominal(
    plan: CubeSatPlan,
    params: CubeSatParameters | None = None,
) -> LinearRollout:
    """Roll out the exact discrete dynamics used by the planning LP."""

    params = params or CubeSatParameters()
    state = np.zeros((params.horizon_days + 1, 3, 3))
    state[0, :, 0] = np.asarray(params.initial_phase_deg)
    matrix = params.linear_A
    control_vector = params.linear_B
    for day, action in enumerate(plan.daily_high_drag_fraction):
        state[day + 1] = state[day] @ matrix.T + action[:, None] * control_vector

    gap_matrix = params.gap_matrix
    cyclic_gaps = state[:, :, 0] @ gap_matrix.T
    cyclic_rates = state[:, :, 1] @ gap_matrix.T
    altitude = params.initial_altitude_km - state[:, :, 2]
    return LinearRollout(
        plan_sha256=plan.plan_sha256,
        time_days=np.arange(params.horizon_days + 1, dtype=float),
        state=state,
        cyclic_gaps_deg=cyclic_gaps,
        cyclic_relative_rates_deg_per_day=cyclic_rates,
        altitude_km=altitude,
    )


def nominal_dynamics_residual(
    plan: CubeSatPlan,
    rollout: LinearRollout,
    params: CubeSatParameters | None = None,
) -> np.ndarray:
    """Return every residual of the discrete planning dynamics."""

    params = params or CubeSatParameters()
    predicted = (
        rollout.state[:-1] @ params.linear_A.T
        + plan.daily_high_drag_fraction[:, :, None] * params.linear_B
    )
    return rollout.state[1:] - predicted


def atmospheric_density_kg_m3(
    altitude_km: float | np.ndarray,
    time_days: float | np.ndarray,
    params: CubeSatParameters | None = None,
) -> np.ndarray:
    """Evaluate the deterministic variable-density replay model."""

    params = params or CubeSatParameters()
    altitude = np.asarray(altitude_km, dtype=float)
    time = np.asarray(time_days, dtype=float)
    temporal_factor = params.density_mean_factor + params.density_amplitude * np.sin(
        2.0 * np.pi * time / params.density_period_days + params.density_phase_rad
    )
    altitude_factor = np.exp(
        (params.initial_altitude_km - altitude) / params.density_scale_height_km
    )
    return params.reference_density_kg_m3 * temporal_factor * altitude_factor


def _nonlinear_derivative(
    time_seconds: float,
    augmented_state: np.ndarray,
    action: np.ndarray,
    params: CubeSatParameters,
) -> np.ndarray:
    reference_axis = augmented_state[0]
    controlled_axes = augmented_state[1:4]
    reference_altitude = reference_axis / 1_000.0 - params.earth_radius_km
    controlled_altitude = controlled_axes / 1_000.0 - params.earth_radius_km
    time_days = time_seconds / DAY_SECONDS
    reference_density = atmospheric_density_kg_m3(
        reference_altitude,
        time_days,
        params,
    )
    controlled_density = atmospheric_density_kg_m3(
        controlled_altitude,
        time_days,
        params,
    )
    inverse_ballistic = (
        (1.0 - action) / params.low_drag_ballistic_coefficient_kg_m2
        + action / params.high_drag_ballistic_coefficient_kg_m2
    )

    derivative = np.empty(7)
    derivative[0] = (
        -reference_density
        / params.low_drag_ballistic_coefficient_kg_m2
        * np.sqrt(params.gravitational_parameter_m3_s2 * reference_axis)
    )
    derivative[1:4] = (
        -controlled_density
        * inverse_ballistic
        * np.sqrt(params.gravitational_parameter_m3_s2 * controlled_axes)
    )
    reference_motion = np.sqrt(
        params.gravitational_parameter_m3_s2 / reference_axis**3
    )
    controlled_motion = np.sqrt(
        params.gravitational_parameter_m3_s2 / controlled_axes**3
    )
    derivative[4:7] = controlled_motion - reference_motion
    return derivative


def _rk4_orbit_step(
    time_seconds: float,
    augmented_state: np.ndarray,
    action: np.ndarray,
    step_seconds: float,
    params: CubeSatParameters,
) -> np.ndarray:
    k1 = _nonlinear_derivative(time_seconds, augmented_state, action, params)
    k2 = _nonlinear_derivative(
        time_seconds + 0.5 * step_seconds,
        augmented_state + 0.5 * step_seconds * k1,
        action,
        params,
    )
    k3 = _nonlinear_derivative(
        time_seconds + 0.5 * step_seconds,
        augmented_state + 0.5 * step_seconds * k2,
        action,
        params,
    )
    k4 = _nonlinear_derivative(
        time_seconds + step_seconds,
        augmented_state + step_seconds * k3,
        action,
        params,
    )
    return augmented_state + step_seconds * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def replay_variable_density(
    plan: CubeSatPlan,
    params: CubeSatParameters | None = None,
    *,
    step_hours: float = 1.0,
) -> NonlinearReplay:
    """Replay the immutable daily plan with a variable-density RK4 model."""

    params = params or CubeSatParameters()
    steps_per_day = int(round(24.0 / step_hours))
    if step_hours <= 0.0 or not np.isclose(steps_per_day * step_hours, 24.0):
        raise ValueError("step_hours must divide one day exactly")
    step_seconds = step_hours * 3_600.0
    total_steps = params.horizon_days * steps_per_day
    time_seconds = np.arange(total_steps + 1, dtype=float) * step_seconds

    augmented = np.zeros((total_steps + 1, 7))
    augmented[0, :4] = params.initial_semi_major_axis_m
    augmented[0, 4:7] = np.deg2rad(np.asarray(params.initial_phase_deg))
    for step in range(total_steps):
        day = step // steps_per_day
        augmented[step + 1] = _rk4_orbit_step(
            time_seconds[step],
            augmented[step],
            plan.daily_high_drag_fraction[day],
            step_seconds,
            params,
        )

    time_days = time_seconds / DAY_SECONDS
    reference_axis = augmented[:, 0]
    controlled_axes = augmented[:, 1:4]
    reference_altitude = reference_axis / 1_000.0 - params.earth_radius_km
    altitude = controlled_axes / 1_000.0 - params.earth_radius_km
    relative_phase = np.rad2deg(augmented[:, 4:7])
    reference_motion = np.sqrt(
        params.gravitational_parameter_m3_s2 / reference_axis**3
    )[:, None]
    controlled_motion = np.sqrt(
        params.gravitational_parameter_m3_s2 / controlled_axes**3
    )
    relative_rate = (controlled_motion - reference_motion) * RAD_TO_DEG * DAY_SECONDS
    extra_loss = reference_altitude[:, None] - altitude

    state = np.empty((total_steps + 1, 3, 3))
    state[:, :, 0] = relative_phase
    state[:, :, 1] = relative_rate
    state[:, :, 2] = extra_loss
    gap_matrix = params.gap_matrix
    density = atmospheric_density_kg_m3(altitude, time_days[:, None], params)
    reference_density = atmospheric_density_kg_m3(
        reference_altitude,
        time_days,
        params,
    )
    return NonlinearReplay(
        plan_sha256=plan.plan_sha256,
        step_hours=step_hours,
        time_days=time_days,
        state=state,
        cyclic_gaps_deg=relative_phase @ gap_matrix.T,
        cyclic_relative_rates_deg_per_day=relative_rate @ gap_matrix.T,
        altitude_km=altitude,
        reference_altitude_km=reference_altitude,
        density_kg_m3=density,
        reference_density_kg_m3=reference_density,
    )


def compare_replay_resolutions(
    coarse: NonlinearReplay,
    fine: NonlinearReplay,
) -> ReplayResolutionCheck:
    """Compare a replay with a finer replay at the coarse sample times."""

    if coarse.plan_sha256 != fine.plan_sha256:
        raise ValueError("replays were not generated from the same immutable plan")
    ratio = int(round(coarse.step_hours / fine.step_hours))
    if ratio < 1 or not np.isclose(ratio * fine.step_hours, coarse.step_hours):
        raise ValueError("the fine replay step must divide the coarse replay step")
    fine_indices = np.arange(len(coarse.time_days)) * ratio
    if fine_indices[-1] >= len(fine.time_days):
        raise ValueError("the replay horizons do not match")
    fine_state = fine.state[fine_indices]
    fine_altitude = fine.altitude_km[fine_indices]
    fine_density = fine.density_kg_m3[fine_indices]
    return ReplayResolutionCheck(
        coarse_step_hours=coarse.step_hours,
        fine_step_hours=fine.step_hours,
        max_phase_delta_deg=float(np.max(np.abs(coarse.state[:, :, 0] - fine_state[:, :, 0]))),
        max_relative_rate_delta_deg_per_day=float(
            np.max(np.abs(coarse.state[:, :, 1] - fine_state[:, :, 1]))
        ),
        max_extra_loss_delta_km=float(
            np.max(np.abs(coarse.state[:, :, 2] - fine_state[:, :, 2]))
        ),
        max_altitude_delta_km=float(np.max(np.abs(coarse.altitude_km - fine_altitude))),
        max_density_delta_kg_m3=float(
            np.max(np.abs(coarse.density_kg_m3 - fine_density))
        ),
        terminal_gap_delta_deg=float(
            np.max(np.abs(coarse.cyclic_gaps_deg[-1] - fine.cyclic_gaps_deg[-1]))
        ),
        terminal_cyclic_rate_delta_deg_per_day=float(
            np.max(
                np.abs(
                    coarse.cyclic_relative_rates_deg_per_day[-1]
                    - fine.cyclic_relative_rates_deg_per_day[-1]
                )
            )
        ),
    )


def compute_metrics(
    params: CubeSatParameters,
    plan: CubeSatPlan,
    nominal: LinearRollout,
    nonlinear: NonlinearReplay,
    resolution_check: ReplayResolutionCheck,
) -> CubeSatMetrics:
    """Collect all deterministic planning and replay diagnostics."""

    target = np.asarray(params.target_cyclic_gaps_deg)
    nominal_gap_error = nominal.cyclic_gaps_deg[-1] - target
    nonlinear_gap_error = nonlinear.cyclic_gaps_deg[-1] - target
    return CubeSatMetrics(
        plan_sha256=plan.plan_sha256,
        alpha_deg_per_day2=params.alpha_deg_per_day2,
        d_km_per_day=params.d_km_per_day,
        primary_max_final_extra_loss_km=plan.primary_max_final_extra_loss_km,
        refined_max_final_extra_loss_km=plan.refined_max_final_extra_loss_km,
        primary_total_variation=plan.primary_total_variation,
        refined_total_variation=plan.refined_total_variation,
        action_min=float(np.min(plan.daily_high_drag_fraction)),
        action_max=float(np.max(plan.daily_high_drag_fraction)),
        equivalent_high_drag_days=np.sum(plan.daily_high_drag_fraction, axis=0),
        max_nominal_dynamics_residual=float(
            np.max(np.abs(nominal_dynamics_residual(plan, nominal, params)))
        ),
        nominal_final_extra_loss_km=nominal.state[-1, :, 2],
        nominal_terminal_cyclic_gaps_deg=nominal.cyclic_gaps_deg[-1],
        nominal_terminal_gap_error_deg=nominal_gap_error,
        nominal_max_gap_error_deg=float(np.max(np.abs(nominal_gap_error))),
        nominal_terminal_cyclic_rates_deg_per_day=(
            nominal.cyclic_relative_rates_deg_per_day[-1]
        ),
        nominal_max_cyclic_rate_deg_per_day=float(
            np.max(np.abs(nominal.cyclic_relative_rates_deg_per_day[-1]))
        ),
        nonlinear_final_extra_loss_km=nonlinear.state[-1, :, 2],
        nonlinear_final_altitude_km=nonlinear.altitude_km[-1],
        nonlinear_terminal_cyclic_gaps_deg=nonlinear.cyclic_gaps_deg[-1],
        nonlinear_terminal_gap_error_deg=nonlinear_gap_error,
        nonlinear_max_gap_error_deg=float(np.max(np.abs(nonlinear_gap_error))),
        nonlinear_terminal_cyclic_rates_deg_per_day=(
            nonlinear.cyclic_relative_rates_deg_per_day[-1]
        ),
        nonlinear_max_cyclic_rate_deg_per_day=float(
            np.max(np.abs(nonlinear.cyclic_relative_rates_deg_per_day[-1]))
        ),
        nonlinear_min_altitude_km=float(np.min(nonlinear.altitude_km)),
        nonlinear_reference_final_altitude_km=float(nonlinear.reference_altitude_km[-1]),
        density_min_kg_m3=float(
            min(np.min(nonlinear.density_kg_m3), np.min(nonlinear.reference_density_kg_m3))
        ),
        density_max_kg_m3=float(
            max(np.max(nonlinear.density_kg_m3), np.max(nonlinear.reference_density_kg_m3))
        ),
        replay_max_phase_delta_deg=resolution_check.max_phase_delta_deg,
        replay_max_relative_rate_delta_deg_per_day=(
            resolution_check.max_relative_rate_delta_deg_per_day
        ),
        replay_max_extra_loss_delta_km=resolution_check.max_extra_loss_delta_km,
        replay_max_altitude_delta_km=resolution_check.max_altitude_delta_km,
    )


def run_scenario(params: CubeSatParameters | None = None) -> CubeSatScenario:
    """Solve, roll out, replay, refine the time step, and compute metrics."""

    params = params or CubeSatParameters()
    plan = solve_lexicographic_plan(params)
    nominal = rollout_nominal(plan, params)
    nonlinear = replay_variable_density(plan, params, step_hours=1.0)
    reference = replay_variable_density(plan, params, step_hours=0.5)
    resolution_check = compare_replay_resolutions(nonlinear, reference)
    metrics = compute_metrics(params, plan, nominal, nonlinear, resolution_check)
    return CubeSatScenario(
        parameters=params,
        plan=plan,
        nominal=nominal,
        nonlinear=nonlinear,
        resolution_check=resolution_check,
        metrics=metrics,
    )


def validation_checks(scenario: CubeSatScenario) -> dict[str, bool]:
    """Return named acceptance checks for tests and artifact manifests."""

    params = scenario.parameters
    plan = scenario.plan
    metrics = scenario.metrics
    replay_arrays = (
        scenario.nonlinear.state,
        scenario.nonlinear.altitude_km,
        scenario.nonlinear.density_kg_m3,
    )
    return {
        "derived_coefficients": bool(
            np.isclose(
                metrics.alpha_deg_per_day2,
                REFERENCE_ALPHA_DEG_PER_DAY2,
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.isclose(
                metrics.d_km_per_day,
                REFERENCE_D_KM_PER_DAY,
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        "solver_status": bool(
            plan.primary_solver_status == 0 and plan.refinement_solver_status == 0
        ),
        "nominal_dynamics": metrics.max_nominal_dynamics_residual < 1.0e-11,
        "action_bounds": bool(
            metrics.action_min >= -1.0e-10
            and metrics.action_max <= 1.0 + 1.0e-10
        ),
        "nominal_terminal_gaps": (
            metrics.nominal_max_gap_error_deg
            <= params.gap_tolerance_deg + 1.0e-7
        ),
        "nominal_terminal_rates": (
            metrics.nominal_max_cyclic_rate_deg_per_day
            <= params.cyclic_rate_tolerance_deg_per_day + 1.0e-7
        ),
        "plan_identity": bool(
            scenario.nominal.plan_sha256
            == scenario.nonlinear.plan_sha256
            == plan.plan_sha256
        ),
        "replay_finite": all(np.all(np.isfinite(array)) for array in replay_arrays),
        "replay_resolution": bool(
            metrics.replay_max_phase_delta_deg < 1.0e-5
            and metrics.replay_max_relative_rate_delta_deg_per_day < 1.0e-7
            and metrics.replay_max_extra_loss_delta_km < 1.0e-7
            and metrics.replay_max_altitude_delta_km < 1.0e-7
            and scenario.resolution_check.terminal_gap_delta_deg < 0.02
        ),
        "density_variation": bool(
            metrics.density_min_kg_m3 > 0.0
            and metrics.density_max_kg_m3 / metrics.density_min_kg_m3 > 1.1
        ),
        "altitude_safe": bool(
            metrics.nonlinear_min_altitude_km > 450.0
            and np.all(np.diff(scenario.nonlinear.altitude_km, axis=0) <= 1.0e-10)
        ),
        "extra_loss_nonnegative": bool(np.min(scenario.nonlinear.state[:, :, 2]) >= -1.0e-9),
        "primary_objective_locked": bool(
            plan.refined_max_final_extra_loss_km
            <= plan.primary_max_final_extra_loss_km
            + params.primary_lock_tolerance_km
            + 5.0e-9
        ),
        "tv_refinement": bool(
            plan.refined_total_variation < plan.primary_total_variation - 1.0e-6
        ),
        "nonlinear_stress_test_visible": metrics.nonlinear_max_gap_error_deg >= 2.0,
    }


def assert_valid_scenario(scenario: CubeSatScenario) -> None:
    """Raise with all failed named checks if the scenario is invalid."""

    failed = [name for name, passed in validation_checks(scenario).items() if not passed]
    if failed:
        raise AssertionError("CubeSat validation failed: " + ", ".join(failed))


def metrics_as_dict(metrics: CubeSatMetrics) -> dict[str, Any]:
    """Convert metrics to JSON-compatible scalars and lists."""

    result = asdict(metrics)
    for name, value in tuple(result.items()):
        if isinstance(value, np.ndarray):
            result[name] = value.tolist()
        elif isinstance(value, np.generic):
            result[name] = value.item()
    return result


__all__ = [
    "CubeSatMetrics",
    "CubeSatParameters",
    "CubeSatPlan",
    "CubeSatScenario",
    "DAY_SECONDS",
    "LinearRollout",
    "NonlinearReplay",
    "ReplayResolutionCheck",
    "SATELLITE_NAMES",
    "STATE_COMPONENTS",
    "assert_valid_scenario",
    "atmospheric_density_kg_m3",
    "compare_replay_resolutions",
    "compute_metrics",
    "metrics_as_dict",
    "nominal_dynamics_residual",
    "replay_variable_density",
    "rollout_nominal",
    "run_scenario",
    "solve_lexicographic_plan",
    "validation_checks",
]
