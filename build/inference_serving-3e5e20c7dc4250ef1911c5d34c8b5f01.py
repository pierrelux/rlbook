"""A calibrated queueing plant for language-model inference serving.

The simulator separates the physical plant from its controllers.  A scheduler
chooses which phase receives the next slice of GPU service, while a clock
controller requests a graphics frequency.  Both controllers receive a
``ServingObservation`` that deliberately omits every request's eventual output
length.  The simulator retains those lengths as hidden disturbances so that it
can decide when decode requests complete.

The committed profile shipped with the book is an engineering proxy.  Its
manifest marks it as unmeasured, and :func:`load_profile` preserves that status
in every result.  A measured L4 profile can replace the CSV without changing
the public interfaces in this module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np


Phase = Literal["prefill", "decode", "idle"]


@dataclass(frozen=True)
class Request:
    """One request from an arrival trace.

    ``output_tokens`` is part of the exogenous workload.  It is never copied
    into :class:`ServingObservation`, so a controller cannot use it before the
    request reaches end of sequence.
    """

    request_id: int
    arrival_time_s: float
    prompt_tokens: int
    output_tokens: int
    original_timestamp: str = ""

    def validate(self) -> None:
        if self.request_id < 0:
            raise ValueError("request_id must be nonnegative")
        if not np.isfinite(self.arrival_time_s) or self.arrival_time_s < 0.0:
            raise ValueError("arrival_time_s must be finite and nonnegative")
        if self.prompt_tokens <= 0 or self.output_tokens <= 0:
            raise ValueError("token counts must be positive")


@dataclass(frozen=True)
class PerformanceProfile:
    """Clock-indexed service rates and electrical power measurements."""

    clock_mhz: np.ndarray
    prefill_tokens_per_s: np.ndarray
    decode_tokens_per_s: np.ndarray
    idle_power_w: np.ndarray
    prefill_power_w: np.ndarray
    decode_power_w: np.ndarray
    baseline_ttft_s: float
    baseline_tpot_s: float
    profile_status: str
    source_label: str
    manifest: Mapping[str, object]

    def validate(self) -> None:
        fields = (
            self.clock_mhz,
            self.prefill_tokens_per_s,
            self.decode_tokens_per_s,
            self.idle_power_w,
            self.prefill_power_w,
            self.decode_power_w,
        )
        sizes = {np.asarray(field).size for field in fields}
        if len(sizes) != 1 or next(iter(sizes)) < 2:
            raise ValueError("profile columns must have the same length, at least two")
        for field in fields:
            values = np.asarray(field, dtype=float)
            if values.ndim != 1 or np.any(~np.isfinite(values)):
                raise ValueError("profile columns must be finite one-dimensional arrays")
            if np.any(values <= 0.0):
                raise ValueError("profile values must be positive")
        if np.any(np.diff(self.clock_mhz) <= 0.0):
            raise ValueError("clock levels must be strictly increasing")
        if self.baseline_ttft_s <= 0.0 or self.baseline_tpot_s <= 0.0:
            raise ValueError("baseline latency values must be positive")

    @property
    def is_measured(self) -> bool:
        return self.profile_status == "measured_l4"

    @property
    def minimum_clock_mhz(self) -> float:
        return float(self.clock_mhz[0])

    @property
    def maximum_clock_mhz(self) -> float:
        return float(self.clock_mhz[-1])

    @property
    def ttft_slo_s(self) -> float:
        return 2.0 * self.baseline_ttft_s

    @property
    def tpot_slo_s(self) -> float:
        return 1.5 * self.baseline_tpot_s

    def quantize_clock(self, requested_mhz: float, *, downward: bool = True) -> float:
        """Map a requested clock onto a supported clock level."""

        requested = float(np.clip(requested_mhz, self.clock_mhz[0], self.clock_mhz[-1]))
        if downward:
            eligible = self.clock_mhz[self.clock_mhz <= requested + 1e-12]
            return float(eligible[-1])
        index = int(np.argmin(np.abs(self.clock_mhz - requested)))
        return float(self.clock_mhz[index])

    def rate(self, phase: Phase, clock_mhz: float) -> float:
        if phase == "prefill":
            values = self.prefill_tokens_per_s
        elif phase == "decode":
            values = self.decode_tokens_per_s
        else:
            return 0.0
        return float(np.interp(clock_mhz, self.clock_mhz, values))

    def power(self, phase: Phase, clock_mhz: float) -> float:
        if phase == "prefill":
            values = self.prefill_power_w
        elif phase == "decode":
            values = self.decode_power_w
        else:
            values = self.idle_power_w
        return float(np.interp(clock_mhz, self.clock_mhz, values))


@dataclass(frozen=True)
class ServingPlant:
    """Plant limits and thermal parameters shared by all controllers."""

    profile: PerformanceProfile
    time_step_s: float = 0.1
    kv_capacity_tokens: float = 131_072.0
    decode_kv_reserve_tokens_per_active: float = 1_024.0
    maximum_active_requests: int = 64
    ambient_temperature_c: float = 25.0
    thermal_time_constant_s: float = 35.0
    thermal_resistance_c_per_w: float = 0.55
    power_limit_w: float = 64.8
    thermal_limit_c: float = 75.0
    reporting_horizon_s: float = 60.0
    maximum_simulation_time_s: float = 3_600.0

    def validate(self) -> None:
        self.profile.validate()
        positive = {
            "time_step_s": self.time_step_s,
            "kv_capacity_tokens": self.kv_capacity_tokens,
            "decode_kv_reserve_tokens_per_active": self.decode_kv_reserve_tokens_per_active,
            "maximum_active_requests": float(self.maximum_active_requests),
            "thermal_time_constant_s": self.thermal_time_constant_s,
            "thermal_resistance_c_per_w": self.thermal_resistance_c_per_w,
            "power_limit_w": self.power_limit_w,
            "thermal_limit_c": self.thermal_limit_c,
            "reporting_horizon_s": self.reporting_horizon_s,
            "maximum_simulation_time_s": self.maximum_simulation_time_s,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.thermal_limit_c <= self.ambient_temperature_c:
            raise ValueError("thermal_limit_c must exceed ambient_temperature_c")


@dataclass(frozen=True)
class ScheduleAction:
    """Scheduling decision for one simulator step."""

    phase: Phase
    maximum_prefill_tokens: float = np.inf

    def validate(self) -> None:
        if self.phase not in {"prefill", "decode", "idle"}:
            raise ValueError(f"unknown scheduling phase: {self.phase}")
        if self.maximum_prefill_tokens <= 0.0:
            raise ValueError("maximum_prefill_tokens must be positive")


@dataclass(frozen=True)
class ServingObservation:
    """Information available to online scheduling and clock controllers."""

    time_s: float
    step_index: int
    prefill_queue: int
    decode_active: int
    prefill_remaining_tokens: float
    generated_decode_tokens: float
    oldest_prefill_age_s: float
    kv_tokens: float
    kv_capacity_tokens: float
    temperature_c: float
    previous_clock_mhz: float
    clock_levels_mhz: tuple[float, ...]
    arrived_requests: int
    completed_requests: int


@dataclass(frozen=True)
class RequestRecord:
    """Completed or censored timing record produced after simulation."""

    request_id: int
    arrival_time_s: float
    prefill_start_s: float | None
    first_token_time_s: float | None
    completion_time_s: float | None
    prompt_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ServingMetrics:
    completed_fraction: float
    mean_ttft_s: float
    p95_ttft_s: float
    mean_tpot_s: float
    p95_tpot_s: float
    mean_latency_s: float
    output_throughput_tokens_per_s: float
    energy_j: float
    energy_per_output_token_j: float
    peak_power_w: float
    peak_temperature_c: float
    peak_kv_tokens: float
    ttft_violation_rate: float
    tpot_violation_rate: float
    power_violation_w: float
    thermal_violation_c: float
    kv_violation_tokens: float
    decode_stall_fraction: float
    peak_queued_requests: float
    peak_queued_requests_at_minimum_clock: float
    unfinished_requests_at_reporting_horizon: float
    unfinished_requests_at_end: float
    reporting_horizon_s: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class MPCDiagnostics:
    solve_times_s: np.ndarray
    successful_solves: int
    fallback_count: int
    control_period_s: float
    horizon_steps: int


@dataclass(frozen=True)
class ServingResult:
    """One plant replay on a fixed time grid."""

    controller_name: str
    scheduler_name: str
    time_s: np.ndarray
    prefill_queue: np.ndarray
    decode_active: np.ndarray
    completed_requests: np.ndarray
    kv_tokens: np.ndarray
    temperature_c: np.ndarray
    power_w: np.ndarray
    requested_clock_mhz: np.ndarray
    realized_clock_mhz: np.ndarray
    energy_j: np.ndarray
    phase: tuple[Phase, ...]
    cumulative_prefill_tokens: np.ndarray
    cumulative_decode_tokens: np.ndarray
    request_records: tuple[RequestRecord, ...]
    metrics: ServingMetrics
    profile_status: str
    workload_checksum: str
    planned_clock_mhz: tuple[tuple[float, ...], ...] = ()
    mpc_diagnostics: MPCDiagnostics | None = None

    def as_dict(self, *, stride: int = 1) -> dict[str, object]:
        """Return a JSON-compatible representation for figures and artifacts."""

        if stride <= 0:
            raise ValueError("stride must be positive")
        indices = np.arange(0, self.time_s.size, stride, dtype=int)
        if indices.size and indices[-1] != self.time_s.size - 1:
            indices = np.append(indices, self.time_s.size - 1)

        def values(array: np.ndarray) -> list[float]:
            return np.asarray(array)[indices].astype(float).tolist()

        payload: dict[str, object] = {
            "controller_name": self.controller_name,
            "scheduler_name": self.scheduler_name,
            "time_s": values(self.time_s),
            "prefill_queue": values(self.prefill_queue),
            "decode_active": values(self.decode_active),
            "completed_requests": values(self.completed_requests),
            "kv_tokens": values(self.kv_tokens),
            "temperature_c": values(self.temperature_c),
            "power_w": values(self.power_w),
            "requested_clock_mhz": values(self.requested_clock_mhz),
            "realized_clock_mhz": values(self.realized_clock_mhz),
            "energy_j": values(self.energy_j),
            "phase": [self.phase[int(i)] for i in indices],
            "metrics": self.metrics.as_dict(),
            "profile_status": self.profile_status,
            "workload_checksum": self.workload_checksum,
            "requests": [record.__dict__ for record in self.request_records],
        }
        if self.planned_clock_mhz:
            payload["planned_clock_mhz"] = [
                list(self.planned_clock_mhz[int(i)]) for i in indices
            ]
        if self.mpc_diagnostics is not None:
            solve_times = self.mpc_diagnostics.solve_times_s
            payload["mpc_diagnostics"] = {
                "successful_solves": self.mpc_diagnostics.successful_solves,
                "fallback_count": self.mpc_diagnostics.fallback_count,
                "control_period_s": self.mpc_diagnostics.control_period_s,
                "horizon_steps": self.mpc_diagnostics.horizon_steps,
                "mean_solve_time_s": (
                    float(np.mean(solve_times)) if solve_times.size else 0.0
                ),
                "maximum_solve_time_s": (
                    float(np.max(solve_times)) if solve_times.size else 0.0
                ),
            }
        return payload


@dataclass
class _RuntimeRequest:
    request: Request
    prefill_remaining: float
    prompt_processed: float = 0.0
    generated: float = 0.0
    prefill_start_s: float | None = None
    first_token_time_s: float | None = None
    completion_time_s: float | None = None


Scheduler = Callable[[ServingObservation], ScheduleAction]
ClockController = Callable[[ServingObservation], float]


def _manifest_path(profile_path: Path) -> Path:
    candidate = profile_path.with_name("profile_manifest.json")
    return candidate if candidate.exists() else profile_path.with_suffix(".json")


def load_profile(path: str | Path) -> PerformanceProfile:
    """Load a measured or explicitly provisional service profile."""

    profile_path = Path(path)
    with profile_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty service profile: {profile_path}")
    aggregate_columns = {
        "clock_mhz",
        "prefill_tokens_per_s",
        "decode_tokens_per_s",
        "idle_power_w",
        "prefill_power_w",
        "decode_power_w",
    }
    manifest_path = _manifest_path(profile_path)
    manifest: dict[str, object] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = str(manifest.get("profile_status", rows[0].get("profile_status", "unknown")))
    if aggregate_columns.issubset(rows[0]):
        clock = np.array([float(row["clock_mhz"]) for row in rows])
        prefill_rate = np.array([float(row["prefill_tokens_per_s"]) for row in rows])
        decode_rate = np.array([float(row["decode_tokens_per_s"]) for row in rows])
        idle_power = np.array([float(row["idle_power_w"]) for row in rows])
        prefill_power = np.array([float(row["prefill_power_w"]) for row in rows])
        decode_power = np.array([float(row["decode_power_w"]) for row in rows])
    else:
        (
            clock,
            prefill_rate,
            decode_rate,
            idle_power,
            prefill_power,
            decode_power,
        ) = _aggregate_raw_profile(rows, manifest)
    profile = PerformanceProfile(
        clock_mhz=clock,
        prefill_tokens_per_s=prefill_rate,
        decode_tokens_per_s=decode_rate,
        idle_power_w=idle_power,
        prefill_power_w=prefill_power,
        decode_power_w=decode_power,
        baseline_ttft_s=float(manifest.get("baseline_ttft_s", 0.18)),
        baseline_tpot_s=float(manifest.get("baseline_tpot_s", 0.006)),
        profile_status=status,
        source_label=str(manifest.get("source_label", "unspecified profile")),
        manifest=manifest,
    )
    profile.validate()
    return profile


def _aggregate_raw_profile(
    rows: Sequence[Mapping[str, str]],
    manifest: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce the profiler's per-request schema to clock-indexed medians."""

    required = {
        "phase",
        "prompt_tokens",
        "output_tokens",
        "concurrency",
        "requested_clock_mhz",
        "realized_clock_mhz",
        "repeat",
        "ttft_s",
        "tpot_s",
        "total_s",
        "energy_j",
        "mean_power_w",
        "peak_power_w",
        "peak_temp_c",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            "profile is neither an aggregate table nor the profiler schema; "
            f"missing columns: {sorted(missing)}"
        )
    grouped: dict[tuple[float, str, int, int, int, int], list[Mapping[str, str]]] = {}
    for row in rows:
        phase = row["phase"].strip().lower()
        if phase not in {"prefill", "decode", "idle"}:
            continue
        key = (
            float(row["requested_clock_mhz"]),
            phase,
            int(row["prompt_tokens"]),
            int(row["output_tokens"]),
            int(row["concurrency"]),
            int(row["repeat"]),
        )
        grouped.setdefault(key, []).append(row)
    requested_clocks = sorted({key[0] for key in grouped})
    if len(requested_clocks) < 2:
        raise ValueError("raw profile must contain at least two requested clocks")
    phase_rates: dict[tuple[float, str], list[float]] = {}
    phase_power: dict[tuple[float, str], list[float]] = {}
    for (clock, phase, prompt, output, concurrency, _repeat), group in grouped.items():
        elapsed = max(float(row["total_s"]) for row in group)
        if elapsed <= 0.0:
            raise ValueError("raw profile total_s must be positive")
        if phase == "prefill":
            tokens = prompt * concurrency
        elif phase == "decode":
            tokens = output * concurrency
        else:
            tokens = 0
        if phase != "idle":
            phase_rates.setdefault((clock, phase), []).append(tokens / elapsed)
        phase_power.setdefault((clock, phase), []).append(
            float(np.median([float(row["mean_power_w"]) for row in group]))
        )
    for clock in requested_clocks:
        for phase in ("prefill", "decode"):
            if (clock, phase) not in phase_rates:
                raise ValueError(f"raw profile lacks {phase} measurements at {clock:g} MHz")
    prefill_rate = np.array(
        [np.median(phase_rates[(clock, "prefill")]) for clock in requested_clocks], dtype=float
    )
    decode_rate = np.array(
        [np.median(phase_rates[(clock, "decode")]) for clock in requested_clocks], dtype=float
    )
    prefill_power = np.array(
        [np.median(phase_power[(clock, "prefill")]) for clock in requested_clocks], dtype=float
    )
    decode_power = np.array(
        [np.median(phase_power[(clock, "decode")]) for clock in requested_clocks], dtype=float
    )
    if all((clock, "idle") in phase_power for clock in requested_clocks):
        idle_power = np.array(
            [np.median(phase_power[(clock, "idle")]) for clock in requested_clocks],
            dtype=float,
        )
    elif "idle_power_w" in manifest:
        idle_power = np.full(len(requested_clocks), float(manifest["idle_power_w"]))
    else:
        idle_power = np.full(
            len(requested_clocks),
            0.45 * min(float(np.min(prefill_power)), float(np.min(decode_power))),
        )
    realized_clock = np.array(
        [
            np.median(
                [
                    float(row["realized_clock_mhz"])
                    for key, group in grouped.items()
                    if key[0] == requested
                    for row in group
                ]
            )
            for requested in requested_clocks
        ],
        dtype=float,
    )
    order = np.argsort(realized_clock)
    return (
        realized_clock[order],
        prefill_rate[order],
        decode_rate[order],
        idle_power[order],
        prefill_power[order],
        decode_power[order],
    )


def _parse_timestamp(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    return datetime.fromisoformat(cleaned)


def load_workload(
    path: str | Path,
    *,
    maximum_elapsed_s: float | None = None,
) -> tuple[Request, ...]:
    """Load an attributed Azure-format request trace.

    The source timestamp and token counts are preserved.  Simulation time is
    measured relative to the first retained request.
    """

    trace_path = Path(path)
    with trace_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty workload trace: {trace_path}")
    columns = set(rows[0])
    expected = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
    if not expected.issubset(columns):
        raise ValueError(f"trace must contain Azure columns {sorted(expected)}")
    start = _parse_timestamp(rows[0]["TIMESTAMP"])
    requests: list[Request] = []
    for row in rows:
        timestamp = _parse_timestamp(row["TIMESTAMP"])
        elapsed = (timestamp - start).total_seconds()
        if maximum_elapsed_s is not None and elapsed > maximum_elapsed_s + 1e-12:
            continue
        request = Request(
            request_id=len(requests),
            arrival_time_s=float(elapsed),
            prompt_tokens=int(row["ContextTokens"]),
            output_tokens=int(row["GeneratedTokens"]),
            original_timestamp=row["TIMESTAMP"],
        )
        request.validate()
        requests.append(request)
    if not requests:
        raise ValueError("no workload rows remain after filtering")
    return tuple(requests)


def workload_checksum(workload: Sequence[Request]) -> str:
    """Return a stable digest of arrival times and token counts."""

    canonical = "\n".join(
        f"{r.request_id},{r.arrival_time_s:.9f},{r.prompt_tokens},{r.output_tokens}"
        for r in workload
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_offered_load(
    workload: Sequence[Request],
    profile: PerformanceProfile,
    *,
    target_utilization: float = 0.8,
) -> tuple[tuple[Request, ...], float]:
    """Dilate arrival times once using isolated maximum-clock service times."""

    if not 0.0 < target_utilization < 1.0:
        raise ValueError("target_utilization must lie strictly between zero and one")
    if not workload:
        raise ValueError("workload must contain at least one request")
    span = max(workload[-1].arrival_time_s - workload[0].arrival_time_s, 1e-9)
    max_clock = profile.maximum_clock_mhz
    isolated_service = sum(
        request.prompt_tokens / profile.rate("prefill", max_clock)
        + request.output_tokens / profile.rate("decode", max_clock)
        for request in workload
    )
    rho_max = isolated_service / span
    dilation = max(1.0, rho_max / target_utilization)
    normalized = tuple(
        Request(
            request_id=request.request_id,
            arrival_time_s=(request.arrival_time_s - workload[0].arrival_time_s) * dilation,
            prompt_tokens=request.prompt_tokens,
            output_tokens=request.output_tokens,
            original_timestamp=request.original_timestamp,
        )
        for request in workload
    )
    return normalized, float(dilation)


def maximum_clock_controller(observation: ServingObservation) -> float:
    return observation.clock_levels_mhz[-1]


def fixed_clock_controller(clock_mhz: float) -> ClockController:
    def controller(_observation: ServingObservation) -> float:
        return float(clock_mhz)

    controller.__name__ = f"fixed_{clock_mhz:g}_mhz"
    return controller


def static_batch_scheduler(observation: ServingObservation) -> ScheduleAction:
    """Finish queued prefills before returning to decode."""

    if observation.prefill_queue:
        return ScheduleAction("prefill")
    if observation.decode_active:
        return ScheduleAction("decode")
    return ScheduleAction("idle")


def continuous_batch_scheduler(observation: ServingObservation) -> ScheduleAction:
    """Give active decodes strict priority, then admit another prompt."""

    if observation.decode_active:
        return ScheduleAction("decode")
    if observation.prefill_queue:
        return ScheduleAction("prefill")
    return ScheduleAction("idle")


def chunked_prefill_scheduler(observation: ServingObservation) -> ScheduleAction:
    """Interleave at most 512 prefill tokens with active decode service."""

    if not observation.decode_active and observation.prefill_queue:
        return ScheduleAction("prefill", maximum_prefill_tokens=512.0)
    kv_pressure = observation.kv_tokens / observation.kv_capacity_tokens
    if observation.decode_active and (
        kv_pressure >= 0.7 or observation.step_index % 2 == 1
    ):
        return ScheduleAction("decode")
    if observation.prefill_queue:
        return ScheduleAction("prefill", maximum_prefill_tokens=512.0)
    if observation.decode_active:
        return ScheduleAction("decode")
    return ScheduleAction("idle")


def reactive_clock_controller(observation: ServingObservation) -> float:
    """A fixed queue-and-temperature governor with one-step hysteresis."""

    clocks = observation.clock_levels_mhz
    previous_index = int(np.argmin(np.abs(np.asarray(clocks) - observation.previous_clock_mhz)))
    pressure = observation.prefill_queue + 2 * observation.decode_active
    if observation.temperature_c >= 72.0:
        index = max(0, previous_index - 1)
    elif pressure >= 8 or observation.oldest_prefill_age_s >= 1.0:
        index = min(len(clocks) - 1, previous_index + 1)
    elif pressure <= 1:
        index = max(0, previous_index - 1)
    else:
        index = previous_index
    return float(clocks[index])


class SampleAndHoldClockController:
    """Apply a feedback law at a fixed cadence and hold between updates."""

    def __init__(self, controller: ClockController, period_s: float = 1.0):
        if period_s <= 0.0:
            raise ValueError("period_s must be positive")
        self.controller = controller
        self.period_s = float(period_s)
        self.current_clock: float | None = None
        self.last_control_index = -1
        self.update_times_s: list[float] = []
        self.__name__ = f"sampled_{getattr(controller, '__name__', 'controller')}"

    def __call__(self, observation: ServingObservation) -> float:
        control_index = int(np.floor((observation.time_s + 1e-12) / self.period_s))
        if self.current_clock is None or control_index != self.last_control_index:
            self.current_clock = float(self.controller(observation))
            self.last_control_index = control_index
            self.update_times_s.append(observation.time_s)
        return self.current_clock


def sample_and_hold_clock_controller(
    controller: ClockController,
    *,
    period_s: float = 1.0,
) -> SampleAndHoldClockController:
    return SampleAndHoldClockController(controller, period_s=period_s)


def _make_observation(
    *,
    time_s: float,
    step_index: int,
    prefill: Sequence[_RuntimeRequest],
    active: Sequence[_RuntimeRequest],
    completed_count: int,
    arrived_count: int,
    kv_tokens: float,
    temperature_c: float,
    previous_clock_mhz: float,
    plant: ServingPlant,
) -> ServingObservation:
    oldest_age = 0.0
    if prefill:
        oldest_age = max(0.0, time_s - min(item.request.arrival_time_s for item in prefill))
    return ServingObservation(
        time_s=time_s,
        step_index=step_index,
        prefill_queue=len(prefill),
        decode_active=len(active),
        prefill_remaining_tokens=float(sum(item.prefill_remaining for item in prefill)),
        generated_decode_tokens=float(sum(item.generated for item in active)),
        oldest_prefill_age_s=oldest_age,
        kv_tokens=float(kv_tokens),
        kv_capacity_tokens=plant.kv_capacity_tokens,
        temperature_c=float(temperature_c),
        previous_clock_mhz=float(previous_clock_mhz),
        clock_levels_mhz=tuple(float(value) for value in plant.profile.clock_mhz),
        arrived_requests=arrived_count,
        completed_requests=completed_count,
    )


def _realized_clock(
    requested_clock_mhz: float,
    phase: Phase,
    temperature_c: float,
    plant: ServingPlant,
) -> float:
    profile = plant.profile
    realized = profile.quantize_clock(requested_clock_mhz, downward=True)
    levels = profile.clock_mhz
    if temperature_c >= plant.thermal_limit_c:
        index = int(np.where(levels == realized)[0][0])
        realized = float(levels[max(0, index - 1)])
    while profile.power(phase, realized) > plant.power_limit_w + 1e-12:
        index = int(np.where(levels == realized)[0][0])
        if index == 0:
            break
        realized = float(levels[index - 1])
    return realized


def _serve_prefill(
    queue: list[_RuntimeRequest],
    active: list[_RuntimeRequest],
    budget: float,
    now_s: float,
    plant: ServingPlant,
) -> tuple[float, float]:
    if not queue or budget <= 0.0:
        return 0.0, 0.0
    item = queue[0]
    if item.prefill_start_s is None:
        item.prefill_start_s = now_s
    occupied = sum(x.prompt_processed + x.generated for x in queue + active)
    reserved_for_decode = (
        len(active) + 1
    ) * plant.decode_kv_reserve_tokens_per_active
    free_kv = max(0.0, plant.kv_capacity_tokens - occupied - reserved_for_decode)
    if len(active) >= plant.maximum_active_requests and item.prefill_remaining <= budget:
        budget = max(0.0, item.prefill_remaining - 1e-9)
    served = min(budget, item.prefill_remaining, free_kv)
    item.prefill_remaining -= served
    item.prompt_processed += served
    if item.prefill_remaining <= 1e-8 and len(active) < plant.maximum_active_requests:
        item.prefill_remaining = 0.0
        queue.pop(0)
        active.append(item)
    return float(served), float(served)


def _serve_decode(
    active: list[_RuntimeRequest],
    completed: list[_RuntimeRequest],
    budget: float,
    now_s: float,
    end_s: float,
    plant: ServingPlant,
) -> tuple[float, float]:
    if not active or budget <= 0.0:
        return 0.0, 0.0
    occupied = sum(x.prompt_processed + x.generated for x in active)
    free_kv = max(0.0, plant.kv_capacity_tokens - occupied)
    remaining_budget = min(float(budget), free_kv)
    served_total = 0.0
    while remaining_budget > 1e-10 and active:
        share = remaining_budget / len(active)
        progress = 0.0
        completed_this_round: list[_RuntimeRequest] = []
        for item in list(active):
            remaining = item.request.output_tokens - item.generated
            served = min(share, remaining)
            if served <= 0.0:
                continue
            if item.first_token_time_s is None:
                item.first_token_time_s = end_s
            item.generated += served
            served_total += served
            remaining_budget -= served
            progress += served
            if item.generated >= item.request.output_tokens - 1e-8:
                item.generated = float(item.request.output_tokens)
                item.completion_time_s = end_s
                completed_this_round.append(item)
        for item in completed_this_round:
            active.remove(item)
            completed.append(item)
        if progress <= 1e-12:
            break
    return float(served_total), float(served_total)


def _thermal_step(temperature_c: float, power_w: float, plant: ServingPlant) -> float:
    decay = np.exp(-plant.time_step_s / plant.thermal_time_constant_s)
    equilibrium = plant.ambient_temperature_c + plant.thermal_resistance_c_per_w * power_w
    return float(equilibrium + (temperature_c - equilibrium) * decay)


def _metrics(
    records: Sequence[RequestRecord],
    time_s: np.ndarray,
    power_w: np.ndarray,
    temperature_c: np.ndarray,
    kv_tokens: np.ndarray,
    energy_j: np.ndarray,
    decode_active: np.ndarray,
    prefill_queue: np.ndarray,
    realized_clock_mhz: np.ndarray,
    phase: Sequence[Phase],
    plant: ServingPlant,
) -> ServingMetrics:
    completed = [record for record in records if record.completion_time_s is not None]
    ttft = np.array(
        [record.first_token_time_s - record.arrival_time_s for record in completed],
        dtype=float,
    )
    tpot = np.array(
        [
            (record.completion_time_s - record.first_token_time_s)
            / max(1, record.output_tokens - 1)
            for record in completed
        ],
        dtype=float,
    )
    latency = np.array(
        [record.completion_time_s - record.arrival_time_s for record in completed],
        dtype=float,
    )
    output_tokens = sum(record.output_tokens for record in completed)
    duration = max(float(time_s[-1] - time_s[0]), plant.time_step_s)
    stalls = np.array(
        [active > 0 and selected != "decode" for active, selected in zip(decode_active, phase)],
        dtype=float,
    )
    active_steps = np.asarray(decode_active) > 0
    queued = np.asarray(prefill_queue, dtype=float) + np.asarray(decode_active, dtype=float)
    minimum_clock_mask = realized_clock_mhz <= plant.profile.minimum_clock_mhz + 1e-9
    horizon = plant.reporting_horizon_s
    arrived_at_horizon = sum(record.arrival_time_s <= horizon for record in records)
    completed_at_horizon = sum(
        record.completion_time_s is not None and record.completion_time_s <= horizon
        for record in records
    )

    def mean_or_nan(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size else float("nan")

    def percentile_or_nan(values: np.ndarray) -> float:
        return float(np.percentile(values, 95)) if values.size else float("nan")

    total_energy = float(energy_j[-1]) if energy_j.size else 0.0
    return ServingMetrics(
        completed_fraction=len(completed) / max(1, len(records)),
        mean_ttft_s=mean_or_nan(ttft),
        p95_ttft_s=percentile_or_nan(ttft),
        mean_tpot_s=mean_or_nan(tpot),
        p95_tpot_s=percentile_or_nan(tpot),
        mean_latency_s=mean_or_nan(latency),
        output_throughput_tokens_per_s=output_tokens / duration,
        energy_j=total_energy,
        energy_per_output_token_j=total_energy / max(1, output_tokens),
        peak_power_w=float(np.max(power_w, initial=0.0)),
        peak_temperature_c=float(np.max(temperature_c, initial=plant.ambient_temperature_c)),
        peak_kv_tokens=float(np.max(kv_tokens, initial=0.0)),
        ttft_violation_rate=float(np.mean(ttft > plant.profile.ttft_slo_s)) if ttft.size else float("nan"),
        tpot_violation_rate=float(np.mean(tpot > plant.profile.tpot_slo_s)) if tpot.size else float("nan"),
        power_violation_w=max(0.0, float(np.max(power_w, initial=0.0)) - plant.power_limit_w),
        thermal_violation_c=max(
            0.0,
            float(np.max(temperature_c, initial=plant.ambient_temperature_c))
            - plant.thermal_limit_c,
        ),
        kv_violation_tokens=max(
            0.0,
            float(np.max(kv_tokens, initial=0.0)) - plant.kv_capacity_tokens,
        ),
        decode_stall_fraction=(
            float(np.sum(stalls[active_steps]) / np.sum(active_steps))
            if np.any(active_steps)
            else 0.0
        ),
        peak_queued_requests=float(np.max(queued, initial=0.0)),
        peak_queued_requests_at_minimum_clock=(
            float(np.max(queued[minimum_clock_mask], initial=0.0))
            if np.any(minimum_clock_mask)
            else 0.0
        ),
        unfinished_requests_at_reporting_horizon=float(
            arrived_at_horizon - completed_at_horizon
        ),
        unfinished_requests_at_end=float(len(records) - len(completed)),
        reporting_horizon_s=float(horizon),
    )


def simulate(
    workload: Sequence[Request],
    plant: ServingPlant,
    scheduler: Scheduler,
    clock_controller: ClockController,
    seed: int = 0,
    *,
    controller_name: str | None = None,
    scheduler_name: str | None = None,
) -> ServingResult:
    """Simulate one immutable workload under supplied feedback laws."""

    del seed  # The current plant is deterministic; the argument fixes the public API.
    plant.validate()
    requests = tuple(workload)
    if not requests:
        raise ValueError("workload must contain at least one request")
    for request in requests:
        request.validate()
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("request identifiers must be unique")
    if any(
        right.arrival_time_s < left.arrival_time_s
        for left, right in zip(requests, requests[1:])
    ):
        raise ValueError("workload must be sorted by arrival_time_s")

    runtime = [
        _RuntimeRequest(request=request, prefill_remaining=float(request.prompt_tokens))
        for request in requests
    ]
    pending_index = 0
    prefill: list[_RuntimeRequest] = []
    active: list[_RuntimeRequest] = []
    completed: list[_RuntimeRequest] = []
    temperature = plant.ambient_temperature_c
    previous_clock = plant.profile.minimum_clock_mhz
    cumulative_energy = 0.0
    cumulative_prefill = 0.0
    cumulative_decode = 0.0

    time_values: list[float] = []
    prefill_values: list[int] = []
    decode_values: list[int] = []
    completed_values: list[int] = []
    kv_values: list[float] = []
    temperature_values: list[float] = []
    power_values: list[float] = []
    requested_values: list[float] = []
    realized_values: list[float] = []
    energy_values: list[float] = []
    phase_values: list[Phase] = []
    cumulative_prefill_values: list[float] = []
    cumulative_decode_values: list[float] = []

    step_index = 0
    time_s = 0.0
    while time_s <= plant.maximum_simulation_time_s + 1e-12:
        while (
            pending_index < len(runtime)
            and runtime[pending_index].request.arrival_time_s <= time_s + 1e-12
        ):
            prefill.append(runtime[pending_index])
            pending_index += 1

        kv_before = sum(item.prompt_processed + item.generated for item in prefill + active)
        observation = _make_observation(
            time_s=time_s,
            step_index=step_index,
            prefill=prefill,
            active=active,
            completed_count=len(completed),
            arrived_count=pending_index,
            kv_tokens=kv_before,
            temperature_c=temperature,
            previous_clock_mhz=previous_clock,
            plant=plant,
        )
        action = scheduler(observation)
        action.validate()
        requested_clock = float(clock_controller(observation))
        if not np.isfinite(requested_clock):
            raise ValueError("clock controller returned a non-finite request")
        phase: Phase = action.phase
        if phase == "prefill" and not prefill:
            phase = "decode" if active else "idle"
        if phase == "decode" and not active:
            phase = "prefill" if prefill else "idle"
        realized_clock = _realized_clock(requested_clock, phase, temperature, plant)
        budget = plant.profile.rate(phase, realized_clock) * plant.time_step_s
        if phase == "prefill":
            budget = min(budget, action.maximum_prefill_tokens)
            served, _ = _serve_prefill(
                prefill,
                active,
                budget,
                time_s,
                plant,
            )
            cumulative_prefill += served
            if served <= 1e-12:
                phase = "idle"
        elif phase == "decode":
            served, _ = _serve_decode(
                active,
                completed,
                budget,
                time_s,
                time_s + plant.time_step_s,
                plant,
            )
            cumulative_decode += served
            if served <= 1e-12:
                phase = "idle"

        power = plant.profile.power(phase, realized_clock)
        temperature = _thermal_step(temperature, power, plant)
        cumulative_energy += power * plant.time_step_s
        kv_after = sum(item.prompt_processed + item.generated for item in prefill + active)

        time_values.append(time_s + plant.time_step_s)
        prefill_values.append(len(prefill))
        decode_values.append(len(active))
        completed_values.append(len(completed))
        kv_values.append(kv_after)
        temperature_values.append(temperature)
        power_values.append(power)
        requested_values.append(requested_clock)
        realized_values.append(realized_clock)
        energy_values.append(cumulative_energy)
        phase_values.append(phase)
        cumulative_prefill_values.append(cumulative_prefill)
        cumulative_decode_values.append(cumulative_decode)

        previous_clock = realized_clock
        time_s += plant.time_step_s
        step_index += 1
        if pending_index == len(runtime) and not prefill and not active:
            break

    records = tuple(
        RequestRecord(
            request_id=item.request.request_id,
            arrival_time_s=item.request.arrival_time_s,
            prefill_start_s=item.prefill_start_s,
            first_token_time_s=item.first_token_time_s,
            completion_time_s=item.completion_time_s,
            prompt_tokens=item.request.prompt_tokens,
            output_tokens=item.request.output_tokens,
        )
        for item in runtime
    )
    arrays = {
        "time_s": np.asarray(time_values, dtype=float),
        "prefill_queue": np.asarray(prefill_values, dtype=int),
        "decode_active": np.asarray(decode_values, dtype=int),
        "completed_requests": np.asarray(completed_values, dtype=int),
        "kv_tokens": np.asarray(kv_values, dtype=float),
        "temperature_c": np.asarray(temperature_values, dtype=float),
        "power_w": np.asarray(power_values, dtype=float),
        "requested_clock_mhz": np.asarray(requested_values, dtype=float),
        "realized_clock_mhz": np.asarray(realized_values, dtype=float),
        "energy_j": np.asarray(energy_values, dtype=float),
        "cumulative_prefill_tokens": np.asarray(cumulative_prefill_values, dtype=float),
        "cumulative_decode_tokens": np.asarray(cumulative_decode_values, dtype=float),
    }
    metrics = _metrics(
        records,
        arrays["time_s"],
        arrays["power_w"],
        arrays["temperature_c"],
        arrays["kv_tokens"],
        arrays["energy_j"],
        arrays["decode_active"],
        arrays["prefill_queue"],
        arrays["realized_clock_mhz"],
        phase_values,
        plant,
    )

    plans_by_step = getattr(clock_controller, "plans_by_step", {})
    planned = tuple(
        tuple(float(value) for value in plans_by_step.get(index, ()))
        for index in range(len(time_values))
    )
    diagnostics = None
    if hasattr(clock_controller, "diagnostics"):
        diagnostics = clock_controller.diagnostics()
    return ServingResult(
        controller_name=controller_name or getattr(clock_controller, "__name__", "controller"),
        scheduler_name=scheduler_name or getattr(scheduler, "__name__", "scheduler"),
        phase=tuple(phase_values),
        request_records=records,
        metrics=metrics,
        profile_status=plant.profile.profile_status,
        workload_checksum=workload_checksum(requests),
        planned_clock_mhz=planned,
        mpc_diagnostics=diagnostics,
        **arrays,
    )


def compute_metrics(result: ServingResult, plant: ServingPlant) -> ServingMetrics:
    """Recompute metrics from a stored trajectory and its plant limits."""

    return _metrics(
        result.request_records,
        result.time_s,
        result.power_w,
        result.temperature_c,
        result.kv_tokens,
        result.energy_j,
        result.decode_active,
        result.prefill_queue,
        result.realized_clock_mhz,
        result.phase,
        plant,
    )


def result_summary(result: ServingResult) -> dict[str, object]:
    """Return compact provenance and metric fields for CSV/JSON manifests."""

    return {
        "controller": result.controller_name,
        "scheduler": result.scheduler_name,
        "profile_status": result.profile_status,
        "workload_checksum": result.workload_checksum,
        **result.metrics.as_dict(),
    }


__all__ = [
    "ClockController",
    "MPCDiagnostics",
    "PerformanceProfile",
    "Request",
    "RequestRecord",
    "ScheduleAction",
    "SampleAndHoldClockController",
    "Scheduler",
    "ServingMetrics",
    "ServingObservation",
    "ServingPlant",
    "ServingResult",
    "chunked_prefill_scheduler",
    "compute_metrics",
    "continuous_batch_scheduler",
    "fixed_clock_controller",
    "load_profile",
    "load_workload",
    "maximum_clock_controller",
    "normalize_offered_load",
    "reactive_clock_controller",
    "result_summary",
    "sample_and_hold_clock_controller",
    "simulate",
    "static_batch_scheduler",
    "workload_checksum",
]
