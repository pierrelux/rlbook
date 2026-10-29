"""Control and learning methods for the inference-serving case study.

The detailed simulator in :mod:`inference_serving` is used for frequency
control.  Scheduling dynamic programming and fitted Q iteration use a smaller
finite-state abstraction whose assumptions are explicit in
``SchedulingMDP.description``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.optimize import linprog, minimize
from sklearn.ensemble import ExtraTreesRegressor

from inference_serving import (
    ClockController,
    MPCDiagnostics,
    PerformanceProfile,
    Request,
    Scheduler,
    ServingObservation,
    ServingPlant,
    ServingResult,
    chunked_prefill_scheduler,
    reactive_clock_controller,
    sample_and_hold_clock_controller,
    simulate,
)


SCHEDULING_ACTIONS = ("prefill", "decode", "idle")
SchedulingActionName = Literal["prefill", "decode", "idle"]


@dataclass(frozen=True)
class OpenLoopPlan:
    """A continuous fluid solution and its supported-clock implementation."""

    time_s: np.ndarray
    continuous_clock_mhz: np.ndarray
    applied_clock_mhz: np.ndarray
    predicted_backlog_s: np.ndarray
    objective: float
    optimization_method: str
    success: bool
    message: str
    control_period_s: float
    workload_checksum: str
    profile_status: str


@dataclass(frozen=True)
class SchedulingMDP:
    """Finite queueing abstraction used for exact DP and offline RL."""

    states: np.ndarray
    transitions: np.ndarray
    stage_cost: np.ndarray
    valid_actions: np.ndarray
    drop_probability: np.ndarray
    arrival_probability: float
    prefill_completion_probability: float
    decode_completion_probability: float
    action_energy: np.ndarray
    gamma: float
    decision_period_s: float
    description: str

    def validate(self) -> None:
        number_states = self.states.shape[0]
        number_actions = len(SCHEDULING_ACTIONS)
        if self.states.shape != (245, 3):
            raise ValueError("the approved reduced model must have 245 states")
        if self.transitions.shape != (number_states, number_actions, number_states):
            raise ValueError("transition tensor has an incompatible shape")
        if self.stage_cost.shape != (number_states, number_actions):
            raise ValueError("stage_cost has an incompatible shape")
        if self.valid_actions.shape != self.stage_cost.shape:
            raise ValueError("valid_actions has an incompatible shape")
        probabilities = self.transitions.sum(axis=2)
        if not np.allclose(probabilities[self.valid_actions], 1.0, atol=1e-12):
            raise ValueError("valid transition rows must sum to one")
        if np.any(self.transitions < -1e-14):
            raise ValueError("transition probabilities must be nonnegative")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie strictly between zero and one")
        if self.decision_period_s <= 0.0:
            raise ValueError("decision_period_s must be positive")


@dataclass(frozen=True)
class DPSolution:
    value: np.ndarray
    policy: np.ndarray
    policy_labels: tuple[SchedulingActionName, ...]
    iterations: int
    bellman_residual: float
    mdp: SchedulingMDP


@dataclass(frozen=True)
class TransitionDataset:
    state: np.ndarray
    action: np.ndarray
    cost: np.ndarray
    next_state: np.ndarray
    behavior: str
    coverage_fraction: float
    random_seed: int


@dataclass(frozen=True)
class FQISolution:
    model: ExtraTreesRegressor
    q_values: np.ndarray
    policy: np.ndarray
    policy_labels: tuple[SchedulingActionName, ...]
    dataset: TransitionDataset
    sweeps: int
    random_seed: int
    policy_disagreement_fraction: float | None = None


@dataclass(frozen=True)
class PolicyEvaluation:
    mean_discounted_return: float
    mean_queue_length: float
    mean_waiting_time_s: float
    mean_decode_stalls: float
    mean_dropped_arrivals: float
    episodes: int
    horizon_steps: int

    def as_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


def _request_work_seconds(request: Request, profile: PerformanceProfile) -> float:
    maximum_clock = profile.maximum_clock_mhz
    return (
        request.prompt_tokens / profile.rate("prefill", maximum_clock)
        + request.output_tokens / profile.rate("decode", maximum_clock)
    )


def _arrival_work_grid(
    workload: Sequence[Request],
    profile: PerformanceProfile,
    horizon_steps: int,
    control_period_s: float,
) -> np.ndarray:
    work = np.zeros(horizon_steps, dtype=float)
    for request in workload:
        index = int(np.floor(request.arrival_time_s / control_period_s))
        if 0 <= index < horizon_steps:
            work[index] += _request_work_seconds(request, profile)
    return work


def _fluid_rollout(
    clock_mhz: np.ndarray,
    arrival_work_s: np.ndarray,
    plant: ServingPlant,
    control_period_s: float,
    *,
    initial_backlog_s: float = 0.0,
    initial_temperature_c: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profile = plant.profile
    maximum_clock = profile.maximum_clock_mhz
    maximum_power = max(
        profile.power("prefill", maximum_clock),
        profile.power("decode", maximum_clock),
    )
    service_scale = np.interp(
        clock_mhz,
        profile.clock_mhz,
        0.5
        * (
            profile.prefill_tokens_per_s / profile.prefill_tokens_per_s[-1]
            + profile.decode_tokens_per_s / profile.decode_tokens_per_s[-1]
        ),
    )
    power = np.interp(
        clock_mhz,
        profile.clock_mhz,
        np.maximum(profile.prefill_power_w, profile.decode_power_w),
    )
    backlog = np.empty(clock_mhz.size + 1, dtype=float)
    temperature = np.empty(clock_mhz.size + 1, dtype=float)
    backlog[0] = max(0.0, initial_backlog_s)
    temperature[0] = (
        plant.ambient_temperature_c
        if initial_temperature_c is None
        else initial_temperature_c
    )
    thermal_decay = np.exp(-control_period_s / plant.thermal_time_constant_s)
    for index in range(clock_mhz.size):
        backlog[index + 1] = max(
            0.0,
            backlog[index]
            + arrival_work_s[index]
            - control_period_s * service_scale[index],
        )
        equilibrium = (
            plant.ambient_temperature_c
            + plant.thermal_resistance_c_per_w * power[index]
        )
        temperature[index + 1] = equilibrium + (
            temperature[index] - equilibrium
        ) * thermal_decay
    return backlog, temperature, power / maximum_power


def optimize_open_loop(
    workload: Sequence[Request],
    plant: ServingPlant,
    scheduler: Scheduler = chunked_prefill_scheduler,
    *,
    horizon_s: float = 60.0,
    control_period_s: float = 1.0,
) -> OpenLoopPlan:
    """Optimize a nominal fluid frequency schedule, then quantize downward."""

    del scheduler  # The fluid plan assumes the fixed chunked scheduler named by the caller.
    plant.validate()
    if horizon_s <= 0.0 or control_period_s <= 0.0:
        raise ValueError("horizon and control period must be positive")
    horizon_steps = int(np.ceil(horizon_s / control_period_s))
    arrivals = _arrival_work_grid(workload, plant.profile, horizon_steps, control_period_s)
    profile = plant.profile
    minimum = profile.minimum_clock_mhz
    maximum = profile.maximum_clock_mhz
    service_levels = 0.5 * (
        profile.prefill_tokens_per_s / profile.prefill_tokens_per_s[-1]
        + profile.decode_tokens_per_s / profile.decode_tokens_per_s[-1]
    )
    minimum_service = float(service_levels[0])
    maximum_power = np.maximum(profile.prefill_power_w, profile.decode_power_w)
    normalized_power = maximum_power / maximum_power[-1]
    power_slope = (normalized_power[-1] - normalized_power[0]) / (
        1.0 - minimum_service
    )

    # Variables are n normalized service levels followed by n end-of-period
    # backlogs.  The inequalities implement b_{k+1} >= b_k + w_k - h u_k.
    number_variables = 2 * horizon_steps
    linear_cost = np.zeros(number_variables, dtype=float)
    linear_cost[:horizon_steps] = power_slope
    linear_cost[horizon_steps:] = 20.0
    linear_cost[-1] += 20.0
    constraint = np.zeros((horizon_steps, number_variables), dtype=float)
    bound = -arrivals.copy()
    for index in range(horizon_steps):
        constraint[index, index] = -control_period_s
        constraint[index, horizon_steps + index] = -1.0
        if index > 0:
            constraint[index, horizon_steps + index - 1] = 1.0
    solution = linprog(
        linear_cost,
        A_ub=constraint,
        b_ub=bound,
        bounds=[(minimum_service, 1.0)] * horizon_steps
        + [(0.0, None)] * horizon_steps,
        method="highs",
    )
    if solution.success:
        optimized_service = np.asarray(solution.x[:horizon_steps], dtype=float)
    else:
        optimized_service = np.ones(horizon_steps, dtype=float)
    continuous = np.interp(optimized_service, service_levels, profile.clock_mhz)
    applied = np.array(
        [plant.profile.quantize_clock(value, downward=True) for value in continuous]
    )
    predicted_backlog, _, _ = _fluid_rollout(
        applied,
        arrivals,
        plant,
        control_period_s,
    )
    from inference_serving import workload_checksum

    objective_value = float(solution.fun) if solution.success else float("nan")
    return OpenLoopPlan(
        time_s=np.arange(horizon_steps, dtype=float) * control_period_s,
        continuous_clock_mhz=continuous,
        applied_clock_mhz=applied,
        predicted_backlog_s=predicted_backlog,
        objective=objective_value,
        optimization_method="HiGHS linear program",
        success=bool(solution.success and np.all(np.isfinite(continuous))),
        message=str(solution.message),
        control_period_s=control_period_s,
        workload_checksum=workload_checksum(workload),
        profile_status=plant.profile.profile_status,
    )


class OpenLoopClockController:
    """Clock controller that replays a precomputed plan without feedback."""

    def __init__(self, plan: OpenLoopPlan):
        self.plan = plan
        self.__name__ = "open_loop_clock"

    def __call__(self, observation: ServingObservation) -> float:
        index = min(
            int(observation.time_s // self.plan.control_period_s),
            self.plan.applied_clock_mhz.size - 1,
        )
        return float(self.plan.applied_clock_mhz[index])


def open_loop_clock_controller(plan: OpenLoopPlan) -> ClockController:
    return OpenLoopClockController(plan)


def mpc_latency_proxies(
    prefill_tokens: float,
    active_decode_requests: float,
    clock_mhz: float,
    profile: PerformanceProfile,
) -> tuple[float, float]:
    """Return separate TTFT and TPOT delay proxies for the reduced MPC model."""

    ttft_proxy = max(0.0, prefill_tokens) / max(
        profile.rate("prefill", clock_mhz), 1e-12
    )
    tpot_proxy = max(0.0, active_decode_requests) / max(
        profile.rate("decode", clock_mhz), 1e-12
    )
    return float(ttft_proxy), float(tpot_proxy)


def _mpc_fluid_rollout(
    clock_mhz: np.ndarray,
    *,
    initial_prefill_tokens: float,
    initial_decode_tokens: float,
    initial_active_decode: float,
    prompt_arrivals: np.ndarray,
    mean_prompt_tokens: float,
    mean_output_tokens: float,
    plant: ServingPlant,
    control_period_s: float,
    initial_temperature_c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict separate phase workloads under the alternating scheduler surrogate."""

    steps = clock_mhz.size
    profile = plant.profile
    prefill = np.empty(steps + 1, dtype=float)
    decode = np.empty(steps + 1, dtype=float)
    active = np.empty(steps + 1, dtype=float)
    temperature = np.empty(steps + 1, dtype=float)
    normalized_power = np.empty(steps, dtype=float)
    ttft_proxy = np.empty(steps, dtype=float)
    tpot_proxy = np.empty(steps, dtype=float)
    prefill[0] = max(0.0, initial_prefill_tokens)
    decode[0] = max(0.0, initial_decode_tokens)
    active[0] = max(0.0, initial_active_decode)
    temperature[0] = initial_temperature_c
    maximum_power = max(profile.prefill_power_w[-1], profile.decode_power_w[-1])
    decay = np.exp(-control_period_s / plant.thermal_time_constant_s)
    for index, clock in enumerate(clock_mhz):
        available_prefill = prefill[index] + prompt_arrivals[index]
        if available_prefill > 1e-12 and decode[index] > 1e-12:
            prefill_share = 0.5
            decode_share = 0.5
        elif available_prefill > 1e-12:
            prefill_share = 1.0
            decode_share = 0.0
        elif decode[index] > 1e-12:
            prefill_share = 0.0
            decode_share = 1.0
        else:
            prefill_share = 0.0
            decode_share = 0.0
        prefill_capacity = (
            profile.rate("prefill", float(clock))
            * control_period_s
            * prefill_share
        )
        served_prefill = min(available_prefill, prefill_capacity)
        completed_prefill = served_prefill / max(mean_prompt_tokens, 1.0)
        prefill[index + 1] = max(0.0, available_prefill - served_prefill)
        available_decode = decode[index] + completed_prefill * mean_output_tokens
        decode_capacity = (
            profile.rate("decode", float(clock))
            * control_period_s
            * decode_share
        )
        served_decode = min(available_decode, decode_capacity)
        completed_decode = served_decode / max(mean_output_tokens, 1.0)
        decode[index + 1] = max(0.0, available_decode - served_decode)
        active[index + 1] = max(
            0.0,
            active[index] + completed_prefill - completed_decode,
        )
        ttft_proxy[index], tpot_proxy[index] = mpc_latency_proxies(
            prefill[index + 1],
            active[index + 1],
            float(clock),
            profile,
        )
        if prefill_share > 0.0 and decode_share > 0.0:
            power = 0.5 * (
                profile.power("prefill", float(clock))
                + profile.power("decode", float(clock))
            )
        elif prefill_share > 0.0:
            power = profile.power("prefill", float(clock))
        elif decode_share > 0.0:
            power = profile.power("decode", float(clock))
        else:
            power = profile.power("idle", float(clock))
        normalized_power[index] = power / maximum_power
        equilibrium = (
            plant.ambient_temperature_c
            + plant.thermal_resistance_c_per_w * power
        )
        temperature[index + 1] = equilibrium + (
            temperature[index] - equilibrium
        ) * decay
    return (
        prefill,
        decode,
        active,
        temperature,
        normalized_power,
        ttft_proxy,
        tpot_proxy,
    )


class _MPCClockController:
    def __init__(
        self,
        workload: Sequence[Request],
        plant: ServingPlant,
        *,
        horizon_s: float,
        control_period_s: float,
        solve_time_limit_s: float,
    ):
        self.plant = plant
        self.horizon_steps = int(np.ceil(horizon_s / control_period_s))
        self.control_period_s = control_period_s
        self.solve_time_limit_s = solve_time_limit_s
        self.mean_request_work_s = float(
            np.mean([_request_work_seconds(request, plant.profile) for request in workload])
        )
        self.mean_prompt_tokens = float(np.mean([request.prompt_tokens for request in workload]))
        self.mean_output_tokens = float(np.mean([request.output_tokens for request in workload]))
        self.arrival_times_seen: list[float] = []
        self.previous_arrived_count = 0
        self.current_clock = plant.profile.minimum_clock_mhz
        self.last_control_index = -1
        self.solve_times: list[float] = []
        self.successful_solves = 0
        self.fallback_count = 0
        self.plans_by_step: dict[int, tuple[float, ...]] = {}
        self.__name__ = "mpc_clock"

    def _arrival_rate(self, observation: ServingObservation) -> float:
        new_arrivals = observation.arrived_requests - self.previous_arrived_count
        if new_arrivals > 0:
            self.arrival_times_seen.extend([observation.time_s] * new_arrivals)
            self.previous_arrived_count = observation.arrived_requests
        lower = observation.time_s - 30.0
        self.arrival_times_seen = [value for value in self.arrival_times_seen if value >= lower]
        observed_window = min(30.0, max(observation.time_s, self.control_period_s))
        return len(self.arrival_times_seen) / observed_window

    def _solve(self, observation: ServingObservation, arrival_rate: float) -> tuple[np.ndarray, bool, float]:
        profile = self.plant.profile
        minimum = profile.minimum_clock_mhz
        maximum = profile.maximum_clock_mhz
        initial_prefill_tokens = observation.prefill_remaining_tokens
        initial_decode_tokens = observation.decode_active * self.mean_output_tokens
        prompt_arrivals = np.full(
            self.horizon_steps,
            arrival_rate * self.mean_prompt_tokens * self.control_period_s,
        )
        initial = np.full(self.horizon_steps, self.current_clock)
        maximum_power = max(profile.prefill_power_w[-1], profile.decode_power_w[-1])

        def objective(sequence: np.ndarray) -> float:
            (
                prefill_tokens,
                decode_tokens,
                active_decode,
                temperature,
                normalized_power,
                ttft_proxy,
                tpot_proxy,
            ) = _mpc_fluid_rollout(
                sequence,
                initial_prefill_tokens=initial_prefill_tokens,
                initial_decode_tokens=initial_decode_tokens,
                initial_active_decode=float(observation.decode_active),
                prompt_arrivals=prompt_arrivals,
                mean_prompt_tokens=self.mean_prompt_tokens,
                mean_output_tokens=self.mean_output_tokens,
                plant=self.plant,
                control_period_s=self.control_period_s,
                initial_temperature_c=observation.temperature_c,
            )
            normalized_frequency = (sequence - minimum) / (maximum - minimum)
            previous_normalized = (self.current_clock - minimum) / (maximum - minimum)
            changes = np.diff(
                np.concatenate([[previous_normalized], normalized_frequency])
            )
            ttft_slack = np.maximum(0.0, ttft_proxy - profile.ttft_slo_s) / max(
                profile.ttft_slo_s, 1e-12
            )
            tpot_slack = np.maximum(
                0.0,
                tpot_proxy - profile.tpot_slo_s,
            ) / max(profile.tpot_slo_s, 1e-12)
            power_slack = np.maximum(
                0.0,
                normalized_power - self.plant.power_limit_w / maximum_power,
            )
            thermal_slack = np.maximum(0.0, temperature[1:] - self.plant.thermal_limit_c)
            terminal_backlog = (
                prefill_tokens[-1] / profile.prefill_tokens_per_s[-1]
                + decode_tokens[-1] / profile.decode_tokens_per_s[-1]
            )
            return float(
                np.sum(normalized_power)
                + 20.0 * np.sum(ttft_slack**2)
                + 10.0 * np.sum(tpot_slack**2)
                + 0.05 * np.sum(changes**2)
                + 1_000.0 * np.sum(power_slack**2 + thermal_slack**2)
                + 20.0 * terminal_backlog**2
            )

        start = perf_counter()
        solution = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(minimum, maximum)] * self.horizon_steps,
            options={"maxiter": 50, "ftol": 1e-6, "disp": False},
        )
        elapsed = perf_counter() - start
        candidate = np.asarray(solution.x, dtype=float)
        success = bool(
            solution.success
            and np.all(np.isfinite(candidate))
            and elapsed <= self.solve_time_limit_s
        )
        return candidate, success, elapsed

    def __call__(self, observation: ServingObservation) -> float:
        control_index = int(observation.time_s // self.control_period_s)
        if control_index == self.last_control_index:
            return self.current_clock
        self.last_control_index = control_index
        arrival_rate = self._arrival_rate(observation)
        candidate, success, elapsed = self._solve(observation, arrival_rate)
        self.solve_times.append(elapsed)
        if success:
            applied = np.array(
                [
                    self.plant.profile.quantize_clock(value, downward=True)
                    for value in candidate
                ]
            )
            self.current_clock = float(applied[0])
            self.successful_solves += 1
            self.plans_by_step[observation.step_index] = tuple(float(x) for x in applied)
        else:
            self.current_clock = reactive_clock_controller(observation)
            self.fallback_count += 1
            self.plans_by_step[observation.step_index] = (float(self.current_clock),)
        return self.current_clock

    def diagnostics(self) -> MPCDiagnostics:
        return MPCDiagnostics(
            solve_times_s=np.asarray(self.solve_times, dtype=float),
            successful_solves=self.successful_solves,
            fallback_count=self.fallback_count,
            control_period_s=self.control_period_s,
            horizon_steps=self.horizon_steps,
        )


def run_mpc(
    workload: Sequence[Request],
    plant: ServingPlant,
    scheduler: Scheduler = chunked_prefill_scheduler,
    *,
    horizon_s: float = 10.0,
    control_period_s: float = 1.0,
    solve_time_limit_s: float = 0.8,
    seed: int = 0,
) -> ServingResult:
    """Run receding-horizon frequency control on the detailed plant."""

    controller = _MPCClockController(
        workload,
        plant,
        horizon_s=horizon_s,
        control_period_s=control_period_s,
        solve_time_limit_s=solve_time_limit_s,
    )
    return simulate(
        workload,
        plant,
        scheduler,
        controller,
        seed,
        controller_name="MPC",
        scheduler_name="512-token interleaved chunked-prefill surrogate",
    )


def shift_largest_burst(
    workload: Sequence[Request],
    *,
    window_s: float = 10.0,
    shift_s: float = 20.0,
) -> tuple[Request, ...]:
    """Move the largest post-shift work burst earlier without changing requests."""

    if not workload:
        return ()
    if window_s <= 0.0 or shift_s <= 0.0:
        raise ValueError("window_s and shift_s must be positive")
    latest = max(request.arrival_time_s for request in workload)
    edges = np.arange(0.0, latest + window_s, window_s)
    if edges.size < 2:
        return tuple(workload)
    weights = np.array([request.prompt_tokens + request.output_tokens for request in workload])
    arrivals = np.array([request.arrival_time_s for request in workload])
    bin_index = np.clip(np.digitize(arrivals, edges) - 1, 0, edges.size - 2)
    totals = np.bincount(bin_index, weights=weights, minlength=edges.size - 1)
    eligible = np.where(edges[:-1] >= shift_s)[0]
    if not eligible.size:
        return tuple(workload)
    target_bin = int(eligible[np.argmax(totals[eligible])])
    shifted: list[Request] = []
    for request, index in zip(workload, bin_index):
        arrival = request.arrival_time_s - shift_s if index == target_bin else request.arrival_time_s
        shifted.append(replace(request, arrival_time_s=max(0.0, arrival)))
    shifted.sort(key=lambda request: (request.arrival_time_s, request.request_id))
    return tuple(shifted)


def _state_index(p: int, d: int, age: int) -> int:
    return (p * 7 + d) * 5 + age


def _binomial_probabilities(number: int, probability: float) -> np.ndarray:
    from math import comb

    return np.array(
        [
            comb(number, completed)
            * probability**completed
            * (1.0 - probability) ** (number - completed)
            for completed in range(number + 1)
        ],
        dtype=float,
    )


def make_scheduling_mdp(
    *,
    arrival_probability: float = 0.35,
    prefill_completion_probability: float = 0.22,
    decode_completion_probability: float = 0.28,
    gamma: float = 0.99,
    decision_period_s: float = 0.1,
    action_energy: Sequence[float] = (0.85, 1.0, 0.25),
) -> SchedulingMDP:
    """Construct the approved 245-state queueing abstraction."""

    if not 0.0 <= arrival_probability <= 1.0:
        raise ValueError("arrival_probability must lie in [0, 1]")
    if not 0.0 <= prefill_completion_probability <= 1.0:
        raise ValueError("prefill_completion_probability must lie in [0, 1]")
    if not 0.0 <= decode_completion_probability <= 1.0:
        raise ValueError("decode_completion_probability must lie in [0, 1]")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between zero and one")
    if decision_period_s <= 0.0:
        raise ValueError("decision_period_s must be positive")
    energy = np.asarray(action_energy, dtype=float)
    if energy.shape != (3,) or np.any(energy < 0.0):
        raise ValueError("action_energy must contain three nonnegative values")

    states = np.array(
        [(p, d, age) for p in range(7) for d in range(7) for age in range(5)],
        dtype=int,
    )
    number_states = states.shape[0]
    transitions = np.zeros((number_states, 3, number_states), dtype=float)
    costs = np.full((number_states, 3), np.inf, dtype=float)
    valid = np.zeros((number_states, 3), dtype=bool)
    drops = np.zeros((number_states, 3), dtype=float)
    maximum_energy = max(float(np.max(energy)), 1e-12)

    for state_index, (p_value, d_value, age_value) in enumerate(states):
        for action in range(3):
            is_valid = (
                action == 2
                or (action == 0 and p_value > 0 and d_value < 6)
                or (action == 1 and d_value > 0)
            )
            if not is_valid:
                continue
            valid[state_index, action] = True
            p_initial = int(p_value)
            d_initial = int(d_value)
            if action == 0:
                service_outcomes = (
                    (
                        p_initial - 1,
                        d_initial + 1,
                        prefill_completion_probability,
                        True,
                    ),
                    (
                        p_initial,
                        d_initial,
                        1.0 - prefill_completion_probability,
                        False,
                    ),
                )
            elif action == 1:
                service_outcomes = tuple(
                    (
                        p_initial,
                        d_initial - completed,
                        probability,
                        completed > 0,
                    )
                    for completed, probability in enumerate(
                        _binomial_probabilities(
                            d_initial, decode_completion_probability
                        )
                    )
                )
            else:
                service_outcomes = ((p_initial, d_initial, 1.0, False),)
            for p_base, d_after, service_probability, service_completed in service_outcomes:
                for arrival, arrival_prob in (
                    (0, 1.0 - arrival_probability),
                    (1, arrival_probability),
                ):
                    dropped = int(arrival == 1 and p_base >= 6)
                    p_after = min(6, p_base + arrival)
                    if p_after == 0:
                        age_after = 0
                    elif action == 0 and service_completed:
                        age_after = max(0, int(age_value) - 1)
                    else:
                        age_after = min(4, int(age_value) + 1)
                    next_index = _state_index(p_after, d_after, age_after)
                    probability = service_probability * arrival_prob
                    transitions[state_index, action, next_index] += probability
                    drops[state_index, action] += probability * dropped
            costs[state_index, action] = (
                p_value
                + d_value
                + 4.0 * (age_value == 4)
                + 2.0 * d_value * (action != 1)
                + 10.0 * drops[state_index, action]
                + 0.1 * energy[action] / maximum_energy
            )

    mdp = SchedulingMDP(
        states=states,
        transitions=transitions,
        stage_cost=costs,
        valid_actions=valid,
        drop_probability=drops,
        arrival_probability=float(arrival_probability),
        prefill_completion_probability=float(prefill_completion_probability),
        decode_completion_probability=float(decode_completion_probability),
        action_energy=energy,
        gamma=float(gamma),
        decision_period_s=float(decision_period_s),
        description=(
            "Exactly solved reduced abstraction with capped prefill and decode counts; "
            "empirical Bernoulli phase completions; it is not an exact reproduction "
            "of vLLM."
        ),
    )
    mdp.validate()
    return mdp


def scheduling_mdp_from_trace(
    workload: Sequence[Request],
    profile: PerformanceProfile,
    *,
    decision_period_s: float = 0.1,
    gamma: float = 0.99,
) -> SchedulingMDP:
    """Estimate the transition probabilities used by the reduced MDP."""

    if not workload:
        raise ValueError("workload must contain at least one request")
    span = max(workload[-1].arrival_time_s - workload[0].arrival_time_s, decision_period_s)
    arrival_rate = len(workload) / span
    arrival_probability = 1.0 - np.exp(-arrival_rate * decision_period_s)
    median_clock = float(profile.clock_mhz[len(profile.clock_mhz) // 2])
    mean_prompt = float(np.mean([request.prompt_tokens for request in workload]))
    mean_output = float(np.mean([request.output_tokens for request in workload]))
    prefill_completion_rate = profile.rate("prefill", median_clock) / max(mean_prompt, 1.0)
    completion_rate = profile.rate("decode", median_clock) / max(mean_output, 1.0)
    prefill_completion_probability = 1.0 - np.exp(
        -prefill_completion_rate * decision_period_s
    )
    completion_probability = 1.0 - np.exp(-completion_rate * decision_period_s)
    action_energy = np.array(
        [
            profile.power("prefill", median_clock),
            profile.power("decode", median_clock),
            profile.power("idle", median_clock),
        ]
    ) * decision_period_s
    return make_scheduling_mdp(
        arrival_probability=float(np.clip(arrival_probability, 0.01, 0.95)),
        prefill_completion_probability=float(
            np.clip(prefill_completion_probability, 0.01, 0.95)
        ),
        decode_completion_probability=float(np.clip(completion_probability, 0.01, 0.95)),
        gamma=gamma,
        decision_period_s=decision_period_s,
        action_energy=action_energy,
    )


def solve_scheduling_mdp(
    mdp: SchedulingMDP | None = None,
    *,
    tolerance: float = 1e-10,
    maximum_iterations: int = 100_000,
) -> DPSolution:
    """Solve the reduced discounted scheduling model by value iteration."""

    model = make_scheduling_mdp() if mdp is None else mdp
    model.validate()
    if tolerance <= 0.0 or maximum_iterations <= 0:
        raise ValueError("tolerance and maximum_iterations must be positive")
    value = np.zeros(model.states.shape[0], dtype=float)
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        continuation = np.einsum("sak,k->sa", model.transitions, value)
        q_values = model.stage_cost + model.gamma * continuation
        q_values = np.where(model.valid_actions, q_values, np.inf)
        updated = np.min(q_values, axis=1)
        if np.max(np.abs(updated - value)) <= tolerance:
            value = updated
            break
        value = updated
    continuation = np.einsum("sak,k->sa", model.transitions, value)
    q_values = np.where(
        model.valid_actions,
        model.stage_cost + model.gamma * continuation,
        np.inf,
    )
    bellman_update = np.min(q_values, axis=1)
    residual = float(np.max(np.abs(bellman_update - value)))
    policy = np.argmin(q_values, axis=1).astype(int)
    labels = tuple(SCHEDULING_ACTIONS[int(action)] for action in policy)
    return DPSolution(
        value=value,
        policy=policy,
        policy_labels=labels,
        iterations=iterations,
        bellman_residual=residual,
        mdp=model,
    )


def _sample_next_state(
    probabilities: np.ndarray,
    random: np.random.Generator,
) -> int:
    return int(random.choice(probabilities.size, p=probabilities))


def generate_transition_dataset(
    mdp: SchedulingMDP,
    *,
    behavior: Literal["uniform", "decode_priority"] = "uniform",
    number_transitions: int = 50_000,
    exploration_probability: float = 0.05,
    random_seed: int = 0,
) -> TransitionDataset:
    """Generate broad or heuristic logged transitions from the same MDP."""

    mdp.validate()
    if number_transitions <= 0:
        raise ValueError("number_transitions must be positive")
    if not 0.0 <= exploration_probability <= 1.0:
        raise ValueError("exploration_probability must lie in [0, 1]")
    random = np.random.default_rng(random_seed)
    states = np.empty(number_transitions, dtype=int)
    actions = np.empty(number_transitions, dtype=int)
    costs = np.empty(number_transitions, dtype=float)
    next_states = np.empty(number_transitions, dtype=int)
    state = _state_index(0, 0, 0)
    for index in range(number_transitions):
        if behavior == "uniform":
            state = int(random.integers(mdp.states.shape[0]))
            choices = np.flatnonzero(mdp.valid_actions[state])
            action = int(random.choice(choices))
        elif behavior == "decode_priority":
            if index % 300 == 0:
                state = _state_index(0, 0, 0)
            choices = np.flatnonzero(mdp.valid_actions[state])
            if random.random() < exploration_probability:
                action = int(random.choice(choices))
            else:
                p_value, d_value, _ = mdp.states[state]
                if d_value > 0:
                    action = 1
                elif p_value > 0 and mdp.valid_actions[state, 0]:
                    action = 0
                else:
                    action = 2
        else:
            raise ValueError(f"unknown behavior policy: {behavior}")
        next_state = _sample_next_state(mdp.transitions[state, action], random)
        states[index] = state
        actions[index] = action
        costs[index] = mdp.stage_cost[state, action]
        next_states[index] = next_state
        state = next_state
    visited = np.unique(states * 3 + actions).size
    valid_pairs = int(np.sum(mdp.valid_actions))
    return TransitionDataset(
        state=states,
        action=actions,
        cost=costs,
        next_state=next_states,
        behavior=behavior,
        coverage_fraction=visited / valid_pairs,
        random_seed=random_seed,
    )


def _features(mdp: SchedulingMDP, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    state_values = mdp.states[np.asarray(states, dtype=int)].astype(float)
    state_values[:, 0] /= 6.0
    state_values[:, 1] /= 6.0
    state_values[:, 2] /= 4.0
    one_hot = np.eye(3, dtype=float)[np.asarray(actions, dtype=int)]
    return np.column_stack([state_values, one_hot])


def fit_fqi(
    mdp: SchedulingMDP,
    dataset: TransitionDataset | None = None,
    *,
    behavior: Literal["uniform", "decode_priority"] = "uniform",
    number_transitions: int = 50_000,
    sweeps: int = 50,
    number_trees: int = 200,
    maximum_depth: int = 12,
    minimum_leaf_size: int = 2,
    exploration_probability: float = 0.05,
    random_seed: int = 0,
    reference_policy: np.ndarray | None = None,
) -> FQISolution:
    """Fit a joint state-action Extra-Trees approximation of the cost-to-go."""

    mdp.validate()
    if sweeps <= 0 or number_trees <= 0:
        raise ValueError("sweeps and number_trees must be positive")
    samples = dataset or generate_transition_dataset(
        mdp,
        behavior=behavior,
        number_transitions=number_transitions,
        exploration_probability=exploration_probability,
        random_seed=random_seed,
    )
    sample_features = _features(mdp, samples.state, samples.action)
    pair_id = samples.state * 3 + samples.action
    unique_pair, inverse = np.unique(pair_id, return_inverse=True)
    unique_states = unique_pair // 3
    unique_actions = unique_pair % 3
    unique_features = _features(mdp, unique_states, unique_actions)
    counts = np.bincount(inverse).astype(float)
    model: ExtraTreesRegressor | None = None
    target = samples.cost.copy()

    all_states = np.repeat(np.arange(mdp.states.shape[0]), 3)
    all_actions = np.tile(np.arange(3), mdp.states.shape[0])
    all_features = _features(mdp, all_states, all_actions)
    for sweep in range(sweeps):
        aggregated_target = np.bincount(inverse, weights=target) / counts
        model = ExtraTreesRegressor(
            n_estimators=number_trees,
            max_depth=maximum_depth,
            min_samples_leaf=minimum_leaf_size,
            random_state=random_seed + sweep,
            n_jobs=1,
        )
        model.fit(unique_features, aggregated_target, sample_weight=counts)
        predicted = model.predict(all_features).reshape(mdp.states.shape[0], 3)
        predicted = np.where(mdp.valid_actions, predicted, np.inf)
        next_minimum = np.min(predicted[samples.next_state], axis=1)
        target = samples.cost + mdp.gamma * next_minimum
    assert model is not None
    q_values = model.predict(all_features).reshape(mdp.states.shape[0], 3)
    q_values = np.where(mdp.valid_actions, q_values, np.inf)
    policy = np.argmin(q_values, axis=1).astype(int)
    labels = tuple(SCHEDULING_ACTIONS[int(action)] for action in policy)
    disagreement = None
    if reference_policy is not None:
        reference = np.asarray(reference_policy, dtype=int)
        if reference.shape != policy.shape:
            raise ValueError("reference_policy has an incompatible shape")
        disagreement = float(np.mean(policy != reference))
    return FQISolution(
        model=model,
        q_values=q_values,
        policy=policy,
        policy_labels=labels,
        dataset=samples,
        sweeps=sweeps,
        random_seed=random_seed,
        policy_disagreement_fraction=disagreement,
    )


def _reduced_policy_trajectory(
    mdp: SchedulingMDP,
    policy: Sequence[int],
    *,
    label: str,
    steps: int = 120,
    random_seed: int = 29,
) -> dict[str, object]:
    random = np.random.default_rng(random_seed)
    policy_array = np.asarray(policy, dtype=int)
    state = _state_index(0, 0, 0)
    completed = 0
    energy = 0.0
    time_values: list[float] = []
    prefill_values: list[int] = []
    decode_values: list[int] = []
    completed_values: list[int] = []
    power_values: list[float] = []
    energy_values: list[float] = []
    phase_values: list[str] = []
    for step in range(steps):
        action = int(policy_array[state])
        p_value, d_value, _ = mdp.states[state]
        next_state = _sample_next_state(mdp.transitions[state, action], random)
        _, next_decode, _ = mdp.states[next_state]
        if action == 1:
            completed += max(0, int(d_value) - int(next_decode))
        power = float(mdp.action_energy[action])
        energy += power
        next_prefill, next_decode, _ = mdp.states[next_state]
        time_values.append((step + 1) * 0.1)
        prefill_values.append(int(next_prefill))
        decode_values.append(int(next_decode))
        completed_values.append(completed)
        power_values.append(power)
        energy_values.append(energy)
        phase_values.append(SCHEDULING_ACTIONS[action])
        state = next_state
    return {
        "controller_name": label,
        "scheduler_name": label,
        "time_s": time_values,
        "prefill_queue": prefill_values,
        "decode_active": decode_values,
        "completed_requests": completed_values,
        "kv_tokens": [value * 2_048 for value in decode_values],
        "temperature_c": [25.0 + 4.0 * value for value in power_values],
        "power_w": [25.0 + 35.0 * value for value in power_values],
        "requested_clock_mhz": [1_200.0] * steps,
        "realized_clock_mhz": [1_200.0] * steps,
        "energy_j": energy_values,
        "phase": phase_values,
        "metrics": {},
    }


def build_textbook_results(
    animation_workload: Sequence[Request],
    evaluation_workload: Sequence[Request],
    plant: ServingPlant,
    *,
    load_dilation: float = 1.0,
    fqi_transitions: int = 50_000,
    fqi_sweeps: int = 50,
    fqi_trees: int = 200,
    evaluation_episodes: int = 2_000,
    evaluation_horizon_steps: int = 300,
) -> dict[str, object]:
    """Run the deterministic experiment suite consumed by the MyST chapters."""

    from inference_serving import maximum_clock_controller, reactive_clock_controller

    modeling_result = simulate(
        animation_workload,
        plant,
        chunked_prefill_scheduler,
        maximum_clock_controller,
        controller_name="Maximum clock",
        scheduler_name="512-token interleaved chunked-prefill surrogate",
    )
    frequency_workload = tuple(
        request for request in evaluation_workload if request.arrival_time_s <= 60.0
    )
    if not frequency_workload:
        frequency_workload = tuple(evaluation_workload[:1])
    plan = optimize_open_loop(frequency_workload, plant)
    plan_controller = open_loop_clock_controller(plan)
    open_loop_nominal = simulate(
        frequency_workload,
        plant,
        chunked_prefill_scheduler,
        plan_controller,
        controller_name="Offline plan",
        scheduler_name="512-token interleaved chunked-prefill surrogate",
    )
    shifted_workload = shift_largest_burst(frequency_workload)
    open_loop_shifted = simulate(
        shifted_workload,
        plant,
        chunked_prefill_scheduler,
        open_loop_clock_controller(plan),
        controller_name="Same offline plan, shifted arrivals",
        scheduler_name="512-token interleaved chunked-prefill surrogate",
    )
    maximum_result = simulate(
        shifted_workload,
        plant,
        chunked_prefill_scheduler,
        maximum_clock_controller,
        controller_name="Maximum clock",
        scheduler_name="512-token interleaved chunked-prefill surrogate",
    )
    reactive_result = simulate(
        shifted_workload,
        plant,
        chunked_prefill_scheduler,
        sample_and_hold_clock_controller(
            reactive_clock_controller,
            period_s=1.0,
        ),
        controller_name="Reactive governor",
        scheduler_name="512-token interleaved chunked-prefill surrogate",
    )
    mpc_result = run_mpc(shifted_workload, plant)

    mdp = scheduling_mdp_from_trace(evaluation_workload, plant.profile)
    exact = solve_scheduling_mdp(mdp)
    broad = fit_fqi(
        mdp,
        behavior="uniform",
        number_transitions=fqi_transitions,
        sweeps=fqi_sweeps,
        number_trees=fqi_trees,
        random_seed=0,
        reference_policy=exact.policy,
    )
    narrow = fit_fqi(
        mdp,
        behavior="decode_priority",
        number_transitions=fqi_transitions,
        sweeps=fqi_sweeps,
        number_trees=fqi_trees,
        exploration_probability=0.05,
        random_seed=0,
        reference_policy=exact.policy,
    )
    exact_evaluation = evaluate_scheduling_policy(
        mdp,
        exact.policy,
        episodes=evaluation_episodes,
        horizon_steps=evaluation_horizon_steps,
    )
    broad_evaluation = evaluate_scheduling_policy(
        mdp,
        broad.policy,
        episodes=evaluation_episodes,
        horizon_steps=evaluation_horizon_steps,
    )
    narrow_evaluation = evaluate_scheduling_policy(
        mdp,
        narrow.policy,
        episodes=evaluation_episodes,
        horizon_steps=evaluation_horizon_steps,
    )
    stride = max(1, int(round(0.5 / plant.time_step_s)))
    moved_request_ids = {
        original.request_id
        for original in frequency_workload
        for shifted in shifted_workload
        if shifted.request_id == original.request_id
        and abs(shifted.arrival_time_s - original.arrival_time_s) > 1e-9
    }

    def matched_latency(result: ServingResult) -> tuple[float, float]:
        values = np.array(
            [
                record.first_token_time_s - record.arrival_time_s
                for record in result.request_records
                if record.request_id in moved_request_ids
                and record.first_token_time_s is not None
            ],
            dtype=float,
        )
        return float(np.mean(values)), float(np.percentile(values, 95))

    def queued_at(result: ServingResult, time_s: float) -> float:
        index = int(np.searchsorted(result.time_s, time_s, side="right") - 1)
        if index < 0:
            return 0.0
        return float(result.prefill_queue[index] + result.decode_active[index])

    nominal_matched_mean, nominal_matched_p95 = matched_latency(open_loop_nominal)
    shifted_matched_mean, shifted_matched_p95 = matched_latency(open_loop_shifted)
    nominal_open_payload = open_loop_nominal.as_dict(stride=stride)
    shifted_open_payload = open_loop_shifted.as_dict(stride=stride)
    nominal_open_payload["metrics"].update(
        {
            "matched_moved_burst_mean_ttft_s": nominal_matched_mean,
            "matched_moved_burst_p95_ttft_s": nominal_matched_p95,
            "queued_requests_at_30_s": queued_at(open_loop_nominal, 30.0),
        }
    )
    shifted_open_payload["metrics"].update(
        {
            "matched_moved_burst_mean_ttft_s": shifted_matched_mean,
            "matched_moved_burst_p95_ttft_s": shifted_matched_p95,
            "queued_requests_at_30_s": queued_at(open_loop_shifted, 30.0),
        }
    )
    exact_slices = {
        str(age): policy_slice(exact.policy, oldest_age_bin=age).tolist()
        for age in range(5)
    }
    broad_slices = {
        str(age): policy_slice(broad.policy, oldest_age_bin=age).tolist()
        for age in range(5)
    }
    narrow_slices = {
        str(age): policy_slice(narrow.policy, oldest_age_bin=age).tolist()
        for age in range(5)
    }

    def coverage_slices(dataset: TransitionDataset) -> dict[str, list[list[int]]]:
        counts = np.bincount(dataset.state, minlength=mdp.states.shape[0])
        return {
            str(age): np.array(
                [
                    [counts[_state_index(p, d, age)] for d in range(7)]
                    for p in range(7)
                ],
                dtype=int,
            ).tolist()
            for age in range(5)
        }

    return {
        "metadata": {
            "profile_status": plant.profile.profile_status,
            "profile_source": plant.profile.source_label,
            "load_dilation": float(load_dilation),
            "frequency_experiment_source_window_s": 60.0,
            "scheduling_trace_source_window_s": 300.0,
            "warning": plant.profile.manifest.get("warning", ""),
        },
        "modeling": modeling_result.as_dict(stride=stride),
        "open_loop": {
            "controllers": {
                "nominal": nominal_open_payload,
                "shifted": shifted_open_payload,
            },
            "mismatch_metrics": {
                "moved_request_count": len(moved_request_ids),
                "arrival_shift_s": -20.0,
                "matched_mean_ttft_delta_s": (
                    shifted_matched_mean - nominal_matched_mean
                ),
                "matched_p95_ttft_delta_s": shifted_matched_p95 - nominal_matched_p95,
                "queued_requests_at_30_s_delta": (
                    queued_at(open_loop_shifted, 30.0)
                    - queued_at(open_loop_nominal, 30.0)
                ),
                "peak_queue_at_minimum_clock_delta": (
                    open_loop_shifted.metrics.peak_queued_requests_at_minimum_clock
                    - open_loop_nominal.metrics.peak_queued_requests_at_minimum_clock
                ),
            },
            "plan": {
                "time_s": plan.time_s.tolist(),
                "continuous_clock_mhz": plan.continuous_clock_mhz.tolist(),
                "applied_clock_mhz": plan.applied_clock_mhz.tolist(),
                "predicted_backlog_s": plan.predicted_backlog_s.tolist(),
                "objective": plan.objective,
                "optimization_method": plan.optimization_method,
                "success": plan.success,
                "message": plan.message,
            },
        },
        "mpc": {
            "controllers": {
                "maximum_clock": maximum_result.as_dict(stride=stride),
                "reactive": reactive_result.as_dict(stride=stride),
                "offline": open_loop_shifted.as_dict(stride=stride),
                "mpc": mpc_result.as_dict(stride=stride),
            }
        },
        "scheduling": {
            "controllers": {
                "exact_dp": _reduced_policy_trajectory(mdp, exact.policy, label="Exact DP"),
                "decode_priority": _reduced_policy_trajectory(
                    mdp,
                    np.array(
                        [
                            1 if d > 0 else (0 if p > 0 and d < 6 else 2)
                            for p, d, _ in mdp.states
                        ]
                    ),
                    label="Decode-priority heuristic",
                ),
            },
            "bellman_residual": exact.bellman_residual,
            "iterations": exact.iterations,
            "exact_policy_evaluation": exact_evaluation.as_dict(),
            "policy_slice_age_4": policy_slice(exact.policy, oldest_age_bin=4).tolist(),
            "policy_slices": exact_slices,
            "description": mdp.description,
            "calibration": {
                "arrival_probability": mdp.arrival_probability,
                "prefill_completion_probability": mdp.prefill_completion_probability,
                "decode_completion_probability": mdp.decode_completion_probability,
                "action_energy_j": mdp.action_energy.tolist(),
                "decision_period_s": mdp.decision_period_s,
            },
        },
        "fqi": {
            "controllers": {
                "exact_dp": _reduced_policy_trajectory(mdp, exact.policy, label="Exact DP"),
                "broad_fqi": _reduced_policy_trajectory(
                    mdp, broad.policy, label="FQI, broad coverage"
                ),
                "narrow_fqi": _reduced_policy_trajectory(
                    mdp, narrow.policy, label="FQI, narrow log"
                ),
            },
            "metrics": {
                "exact_dp": exact_evaluation.as_dict(),
                "broad_fqi": {
                    **broad_evaluation.as_dict(),
                    "coverage_fraction": broad.dataset.coverage_fraction,
                    "policy_disagreement_fraction": broad.policy_disagreement_fraction,
                },
                "narrow_fqi": {
                    **narrow_evaluation.as_dict(),
                    "coverage_fraction": narrow.dataset.coverage_fraction,
                    "policy_disagreement_fraction": narrow.policy_disagreement_fraction,
                },
            },
            "policy_slices": {
                "exact_dp": exact_slices,
                "broad_fqi": broad_slices,
                "narrow_fqi": narrow_slices,
            },
            "coverage_slices": {
                "broad_fqi": coverage_slices(broad.dataset),
                "narrow_fqi": coverage_slices(narrow.dataset),
            },
            "protocol": {
                "transitions_per_dataset": fqi_transitions,
                "sweeps": fqi_sweeps,
                "trees": fqi_trees,
                "maximum_depth": 12,
                "minimum_leaf_size": 2,
                "random_seed": 0,
                "evaluation_episodes": evaluation_episodes,
                "evaluation_horizon_steps": evaluation_horizon_steps,
            },
        },
    }


def evaluate_scheduling_policy(
    mdp: SchedulingMDP,
    policy: Sequence[int],
    *,
    episodes: int = 2_000,
    horizon_steps: int = 300,
    random_seed: int = 17,
) -> PolicyEvaluation:
    """Evaluate one stationary policy on fixed random-number streams."""

    policy_array = np.asarray(policy, dtype=int)
    if policy_array.shape != (mdp.states.shape[0],):
        raise ValueError("policy has an incompatible shape")
    if episodes <= 0 or horizon_steps <= 0:
        raise ValueError("episodes and horizon_steps must be positive")
    if np.any(~mdp.valid_actions[np.arange(mdp.states.shape[0]), policy_array]):
        raise ValueError("policy selects an invalid action")
    random = np.random.default_rng(random_seed)
    discounted_returns = np.zeros(episodes, dtype=float)
    queue_totals = np.zeros(episodes, dtype=float)
    stall_totals = np.zeros(episodes, dtype=float)
    drop_totals = np.zeros(episodes, dtype=float)
    completion_totals = np.zeros(episodes, dtype=float)
    for episode in range(episodes):
        state = _state_index(0, 0, 0)
        discount = 1.0
        for _ in range(horizon_steps):
            action = int(policy_array[state])
            p_value, d_value, _ = mdp.states[state]
            discounted_returns[episode] -= discount * mdp.stage_cost[state, action]
            queue_totals[episode] += p_value + d_value
            stall_totals[episode] += float(d_value > 0 and action != 1)
            drop_totals[episode] += mdp.drop_probability[state, action]
            next_state = _sample_next_state(mdp.transitions[state, action], random)
            if action == 1:
                next_decode = int(mdp.states[next_state, 1])
                completion_totals[episode] += max(0, int(d_value) - next_decode)
            state = next_state
            discount *= mdp.gamma
    mean_waiting = np.mean(
        queue_totals * mdp.decision_period_s / np.maximum(completion_totals, 1.0)
    )
    return PolicyEvaluation(
        mean_discounted_return=float(np.mean(discounted_returns)),
        mean_queue_length=float(np.mean(queue_totals) / horizon_steps),
        mean_waiting_time_s=float(mean_waiting),
        mean_decode_stalls=float(np.mean(stall_totals)),
        mean_dropped_arrivals=float(np.mean(drop_totals)),
        episodes=episodes,
        horizon_steps=horizon_steps,
    )


def policy_slice(
    policy: Sequence[int],
    *,
    oldest_age_bin: int,
) -> np.ndarray:
    """Return a 7 by 7 action table for a fixed oldest-request age."""

    if not 0 <= oldest_age_bin <= 4:
        raise ValueError("oldest_age_bin must lie in [0, 4]")
    policy_array = np.asarray(policy, dtype=int)
    return np.array(
        [
            [policy_array[_state_index(p, d, oldest_age_bin)] for d in range(7)]
            for p in range(7)
        ],
        dtype=int,
    )


__all__ = [
    "DPSolution",
    "FQISolution",
    "OpenLoopClockController",
    "OpenLoopPlan",
    "PolicyEvaluation",
    "SCHEDULING_ACTIONS",
    "SchedulingMDP",
    "TransitionDataset",
    "build_textbook_results",
    "evaluate_scheduling_policy",
    "fit_fqi",
    "generate_transition_dataset",
    "make_scheduling_mdp",
    "mpc_latency_proxies",
    "open_loop_clock_controller",
    "optimize_open_loop",
    "policy_slice",
    "run_mpc",
    "scheduling_mdp_from_trace",
    "shift_largest_burst",
    "solve_scheduling_mdp",
]
