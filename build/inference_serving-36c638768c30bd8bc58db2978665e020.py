"""A calibrated queueing plant for language-model inference serving.

The simulator separates the physical plant from its controllers.  A scheduler
chooses which phase receives the next slice of GPU service, while a clock
controller requests a graphics frequency.  Both controllers receive a
``ServingObservation`` that deliberately omits every request's eventual output
length.  The simulator retains those lengths as hidden disturbances so that it
can decide when decode requests complete.

The committed profile shipped with the book is a measured NVIDIA L4 profile.
Its manifest records and validates the raw observations, aggregation protocol,
hardware, model revision, and container digest.  :func:`load_profile` carries
that provenance into every result; engineering-proxy bundles remain supported
for isolated software tests but are never silently presented as measurements.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

import numpy as np


Phase = Literal["prefill", "decode", "interleaved", "idle"]

_PROFILE_SCHEMA_VERSION = 2
_MEASURED_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_MEASURED_MODEL_REVISION = "acbd96531cda22292a3ceaa67e984955d3965282"
_MEASURED_VLLM_IMAGE = "vllm/vllm-openai:v0.28.0"
_AGGREGATE_PROFILE_COLUMNS = (
    "clock_mhz",
    "prefill_tokens_per_s",
    "decode_tokens_per_s",
    "idle_power_w",
    "prefill_power_w",
    "decode_power_w",
)
_RAW_PROFILE_COLUMNS = (
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
    "telemetry_sample_count",
    "telemetry_fallback_used",
    "request_index",
    "completion_tokens",
    "prompt_tokens_observed",
)
_TELEMETRY_COLUMNS = (
    "elapsed_s",
    "utc",
    "phase",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "power_w",
    "temperature_c",
    "utilization_percent",
    "memory_used_mib",
)
_EXPECTED_PROFILE_CONDITIONS = {
    ("prefill", prompt, 1, concurrency)
    for prompt in (128, 512, 2_048, 4_096)
    for concurrency in (1, 4, 8)
} | {
    ("decode", prompt, 128, concurrency)
    for prompt in (128, 1_024)
    for concurrency in (1, 4, 8)
}


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
    """Service and power measurements indexed by requested profile clock."""

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
    measurement_validated: bool = False
    profile_csv_sha256: str = ""
    manifest_sha256: str = ""
    realized_clock_median_mhz: np.ndarray | None = None

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
        if self.is_measured and self.realized_clock_median_mhz is None:
            raise ValueError(
                "a validated measured profile must carry realized-clock medians"
            )
        realized_clocks = self.realized_clock_levels_mhz
        if (
            realized_clocks.shape != np.asarray(self.clock_mhz).shape
            or np.any(~np.isfinite(realized_clocks))
            or np.any(realized_clocks <= 0.0)
        ):
            raise ValueError(
                "realized clock medians must be finite, positive, and aligned "
                "with requested profile levels"
            )
        if np.any(np.diff(realized_clocks) <= 0.0):
            raise ValueError("realized clock medians must be strictly increasing")
        if np.any(np.diff(self.prefill_tokens_per_s) <= 0.0):
            raise ValueError(
                "prefill service rate must increase strictly with clock; "
                "refit or reject the noisy profile before interpolation"
            )
        if np.any(np.diff(self.decode_tokens_per_s) <= 0.0):
            raise ValueError(
                "decode service rate must increase strictly with clock; "
                "refit or reject the noisy profile before interpolation"
            )
        if self.baseline_ttft_s <= 0.0 or self.baseline_tpot_s <= 0.0:
            raise ValueError("baseline latency values must be positive")
        if self.profile_status == "measured_l4" and not self.measurement_validated:
            raise ValueError(
                "a measured_l4 profile must pass complete bundle validation"
            )

    @property
    def is_measured(self) -> bool:
        return self.profile_status == "measured_l4" and self.measurement_validated

    @property
    def minimum_clock_mhz(self) -> float:
        return float(self.clock_mhz[0])

    @property
    def maximum_clock_mhz(self) -> float:
        return float(self.clock_mhz[-1])

    @property
    def realized_clock_levels_mhz(self) -> np.ndarray:
        """Median observed clocks paired with the requested profile levels.

        Validated measurements carry the batch-balanced medians recorded in the
        manifest.  Provisional profiles use an identity response, which keeps
        their earlier requested-equals-realized interpretation explicit.
        """

        if self.realized_clock_median_mhz is None:
            return np.asarray(self.clock_mhz, dtype=float)
        return np.asarray(self.realized_clock_median_mhz, dtype=float)

    @property
    def minimum_realized_clock_mhz(self) -> float:
        return float(self.realized_clock_levels_mhz[0])

    def realized_clock_for_level(self, profile_clock_mhz: float) -> float:
        """Return the observed median paired with an exact profile row."""

        matches = np.flatnonzero(
            np.isclose(
                np.asarray(self.clock_mhz, dtype=float),
                float(profile_clock_mhz),
                rtol=0.0,
                atol=1e-9,
            )
        )
        if matches.size != 1:
            raise ValueError("realized clock lookup requires a supported profile level")
        return float(self.realized_clock_levels_mhz[int(matches[0])])

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
    previous_power_w: float
    clock_levels_mhz: tuple[float, ...]
    arrived_requests: int
    completed_requests: int
    realized_clock_levels_mhz: tuple[float, ...] = ()


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
    solve_time_limit_s: float


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
    planned_clock_start_time_s: tuple[float | None, ...] = ()
    plan_control_period_s: float | None = None
    mpc_diagnostics: MPCDiagnostics | None = None

    def as_dict(self, *, stride: int = 1) -> dict[str, object]:
        """Return a JSON-compatible representation for figures and artifacts."""

        if stride <= 0:
            raise ValueError("stride must be positive")
        indices = np.arange(0, self.time_s.size, stride, dtype=int)
        if indices.size and indices[-1] != self.time_s.size - 1:
            indices = np.append(indices, self.time_s.size - 1)
        if self.planned_clock_mhz:
            plan_updates = np.array(
                [index for index, plan in enumerate(self.planned_clock_mhz) if plan],
                dtype=int,
            )
            if plan_updates.size:
                indices = np.unique(np.concatenate([indices, plan_updates]))

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
            if self.planned_clock_start_time_s:
                payload["planned_clock_start_time_s"] = [
                    self.planned_clock_start_time_s[int(i)] for i in indices
                ]
            if self.plan_control_period_s is not None:
                payload["plan_dt_s"] = float(self.plan_control_period_s)
        if self.mpc_diagnostics is not None:
            payload["mpc_diagnostics"] = {
                "successful_solves": self.mpc_diagnostics.successful_solves,
                "fallback_count": self.mpc_diagnostics.fallback_count,
                "control_period_s": self.mpc_diagnostics.control_period_s,
                "horizon_steps": self.mpc_diagnostics.horizon_steps,
                "solve_time_deadline_s": self.mpc_diagnostics.solve_time_limit_s,
                # Exact wall-clock timings vary by machine and are deliberately
                # omitted from the frozen evidential artifact.
                "accepted_solves_met_deadline": True,
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


@dataclass(frozen=True)
class _ClockRealization:
    """Clock row used by the plant and the corresponding observed response."""

    applied_profile_clock_mhz: float
    observed_clock_mhz: float


Scheduler = Callable[[ServingObservation], ScheduleAction]
ClockController = Callable[[ServingObservation], float]


def _manifest_path(profile_path: Path) -> Path:
    candidate = profile_path.with_name("profile_manifest.json")
    return candidate if candidate.exists() else profile_path.with_suffix(".json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv_checked(
    path: Path,
    expected_columns: Sequence[str],
) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"measured profile bundle is missing required file: {path.name}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise ValueError(
                f"measured profile file {path.name} has an unexpected schema; "
                f"expected {list(expected_columns)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"measured profile file is empty: {path.name}")
    return rows


def _manifest_sequence(
    manifest: Mapping[str, object], name: str
) -> Sequence[object]:
    value = manifest.get(name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"measured profile manifest field {name!r} must be a sequence")
    return value


def _manifest_realized_clock_medians(
    manifest: Mapping[str, object],
    requested_clocks_mhz: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Read the batch-balanced realized response for requested profile levels."""

    selection = manifest.get("clock_profile_selection")
    if not isinstance(selection, Mapping):
        raise ValueError(
            "measured profile manifest lacks clock_profile_selection metadata"
        )
    raw_mapping = selection.get("realized_clock_median_mhz_by_requested")
    if not isinstance(raw_mapping, Mapping):
        raise ValueError(
            "measured profile manifest lacks batch-balanced realized-clock medians"
        )

    parsed: list[tuple[float, float]] = []
    for raw_requested, raw_realized in raw_mapping.items():
        try:
            requested = float(raw_requested)
            realized = float(raw_realized)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "measured realized-clock mapping must contain numeric values"
            ) from error
        if (
            not np.isfinite(requested)
            or not np.isfinite(realized)
            or requested <= 0.0
            or realized <= 0.0
        ):
            raise ValueError(
                "measured realized-clock mapping must contain finite positive values"
            )
        parsed.append((requested, realized))

    medians: list[float] = []
    for requested in np.asarray(requested_clocks_mhz, dtype=float):
        matches = [
            realized
            for declared_requested, realized in parsed
            if np.isclose(
                declared_requested,
                requested,
                rtol=0.0,
                atol=1e-9,
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                "measured profile manifest must declare exactly one realized-clock "
                f"median for requested level {requested:g} MHz"
            )
        medians.append(matches[0])
    return np.asarray(medians, dtype=float)


def _validate_measured_profile_bundle(
    profile_path: Path,
    rows: Sequence[Mapping[str, str]],
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> np.ndarray:
    """Fail closed unless a measured profile is a complete pinned bundle."""

    if manifest.get("status") != "complete":
        raise ValueError("measured profile manifest status must be 'complete'")
    if manifest.get("profile_status") != "measured_l4":
        raise ValueError("complete measured manifest must declare profile_status 'measured_l4'")
    if manifest.get("schema_version") != _PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported measured profile schema_version: {manifest.get('schema_version')!r}"
        )
    if tuple(manifest.get("profile_columns", ())) != _AGGREGATE_PROFILE_COLUMNS:
        raise ValueError("measured profile manifest does not declare the expected aggregate schema")
    if tuple(manifest.get("raw_profile_columns", ())) != _RAW_PROFILE_COLUMNS:
        raise ValueError("measured profile manifest does not declare the expected raw schema")
    if tuple(rows[0]) != _AGGREGATE_PROFILE_COLUMNS:
        raise ValueError("measured aggregate profile has an unexpected column schema")

    gpu = manifest.get("gpu")
    if not isinstance(gpu, Mapping) or "nvidia l4" not in str(gpu.get("name", "")).lower():
        raise ValueError("measured profile hardware metadata must identify an NVIDIA L4")
    if manifest.get("model") != _MEASURED_MODEL:
        raise ValueError("measured profile model metadata does not match the pinned model")
    if manifest.get("model_revision") != _MEASURED_MODEL_REVISION:
        raise ValueError("measured profile model revision does not match the pinned revision")
    if manifest.get("vllm_image") != _MEASURED_VLLM_IMAGE:
        raise ValueError("measured profile container does not match the pinned vLLM image")
    image_digest = str(manifest.get("vllm_image_digest", ""))
    if re.search(r"(?:^|@)sha256:[0-9a-fA-F]{64}$", image_digest) is None:
        raise ValueError("measured profile must record a resolved vLLM sha256 digest")
    if manifest.get("vllm_prefix_caching_enabled") is not False:
        raise ValueError("measured profile must explicitly disable vLLM prefix caching")
    server_arguments = _manifest_sequence(manifest, "vllm_server_arguments")
    if "--no-enable-prefix-caching" not in server_arguments:
        raise ValueError(
            "measured profile vLLM arguments must include "
            "--no-enable-prefix-caching"
        )

    measured_clocks = np.asarray(
        _manifest_sequence(manifest, "selected_graphics_clocks_mhz"), dtype=float
    )
    if (
        measured_clocks.ndim != 1
        or measured_clocks.size < 4
        or np.any(~np.isfinite(measured_clocks))
        or np.any(measured_clocks <= 0.0)
        or np.any(np.diff(measured_clocks) <= 0.0)
    ):
        raise ValueError(
            "measured profile sweep must contain at least four ordered clock levels"
        )
    modeled_clocks = np.asarray(
        [float(row["clock_mhz"]) for row in rows], dtype=float
    )
    if (
        modeled_clocks.ndim != 1
        or modeled_clocks.size < 4
        or np.any(~np.isfinite(modeled_clocks))
        or np.any(modeled_clocks <= 0.0)
        or np.any(np.diff(modeled_clocks) <= 0.0)
    ):
        raise ValueError(
            "measured aggregate profile must contain at least four ordered clock levels"
        )
    measured_clock_for_modeled: list[float] = []
    for clock in modeled_clocks:
        matches = np.flatnonzero(
            np.isclose(measured_clocks, clock, rtol=0.0, atol=1e-9)
        )
        if matches.size != 1:
            raise ValueError(
                "measured aggregate clock levels must be an ordered subset of the "
                "completed measurement sweep"
            )
        measured_clock_for_modeled.append(float(measured_clocks[int(matches[0])]))
    if "modeled_graphics_clocks_mhz" in manifest:
        declared_modeled = np.asarray(
            _manifest_sequence(manifest, "modeled_graphics_clocks_mhz"), dtype=float
        )
        if (
            declared_modeled.shape != modeled_clocks.shape
            or not np.allclose(
                declared_modeled,
                modeled_clocks,
                rtol=0.0,
                atol=1e-9,
            )
        ):
            raise ValueError(
                "measured aggregate clock levels do not match "
                "modeled_graphics_clocks_mhz"
            )
    if manifest.get("measured_repeats_per_condition") != 5:
        raise ValueError("measured profile must record five repetitions per condition")
    if manifest.get("warmup_batches_per_condition") != 1:
        raise ValueError("measured profile must record one warmup batch per condition")

    declared_conditions = set()
    for raw_condition in _manifest_sequence(manifest, "conditions"):
        if not isinstance(raw_condition, Mapping):
            raise ValueError("measured profile conditions must be mappings")
        declared_conditions.add(
            (
                str(raw_condition.get("phase", "")),
                int(raw_condition.get("prompt_tokens", 0)),
                int(raw_condition.get("output_tokens", 0)),
                int(raw_condition.get("concurrency", 0)),
            )
        )
    if declared_conditions != _EXPECTED_PROFILE_CONDITIONS:
        raise ValueError("measured profile manifest does not match the fixed condition matrix")

    bundle_paths = {
        "l4_profile.csv": profile_path,
        "l4_profile_all_requested.csv": profile_path.with_name(
            "l4_profile_all_requested.csv"
        ),
        "l4_profile_raw.csv": profile_path.with_name("l4_profile_raw.csv"),
        "l4_telemetry.csv": profile_path.with_name("l4_telemetry.csv"),
    }
    checksums = manifest.get("sha256")
    if not isinstance(checksums, Mapping):
        raise ValueError("measured profile manifest must contain file checksums")
    for name in bundle_paths:
        if name not in checksums:
            raise ValueError(f"measured profile manifest lacks a valid checksum for {name}")
    for raw_name, raw_expected in checksums.items():
        name = str(raw_name)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
            raise ValueError("measured profile manifest contains an unsafe checksum file name")
        expected = str(raw_expected)
        if re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
            raise ValueError(f"measured profile manifest lacks a valid checksum for {name}")
        bundle_path = profile_path.with_name(name)
        if not bundle_path.is_file() or _sha256(bundle_path) != expected.lower():
            raise ValueError(f"measured profile checksum mismatch for {name}")

    full_profile_rows = _read_csv_checked(
        bundle_paths["l4_profile_all_requested.csv"],
        _AGGREGATE_PROFILE_COLUMNS,
    )
    full_profile_clocks = np.asarray(
        [float(row["clock_mhz"]) for row in full_profile_rows], dtype=float
    )
    if (
        full_profile_clocks.shape != measured_clocks.shape
        or not np.allclose(
            full_profile_clocks,
            measured_clocks,
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise ValueError(
            "full aggregate profile clocks do not match the completed measurement sweep"
        )
    full_profile_by_clock: dict[float, np.ndarray] = {}
    for measured_clock, row in zip(measured_clocks, full_profile_rows):
        values = np.asarray(
            [float(row[column]) for column in _AGGREGATE_PROFILE_COLUMNS],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("full aggregate profile values must be finite and positive")
        full_profile_by_clock[float(measured_clock)] = values
    for row, measured_clock in zip(rows, measured_clock_for_modeled):
        modeled_values = np.asarray(
            [float(row[column]) for column in _AGGREGATE_PROFILE_COLUMNS],
            dtype=float,
        )
        if not np.allclose(
            modeled_values,
            full_profile_by_clock[measured_clock],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "modeled aggregate rows must match the full aggregate profile"
            )

    raw_rows = _read_csv_checked(bundle_paths["l4_profile_raw.csv"], _RAW_PROFILE_COLUMNS)
    telemetry_rows = _read_csv_checked(
        bundle_paths["l4_telemetry.csv"], _TELEMETRY_COLUMNS
    )
    if int(manifest.get("row_count", -1)) != len(raw_rows):
        raise ValueError("measured raw profile row_count does not match the manifest")
    if int(manifest.get("telemetry_row_count", -1)) != len(telemetry_rows):
        raise ValueError("measured telemetry row_count does not match the manifest")

    expected_repeats = set(range(5))
    measured_clock_values = {float(clock) for clock in measured_clocks}
    realized_by_batch: dict[
        tuple[float, str, int, int, int, int], list[float]
    ] = {}
    telemetry_by_batch: dict[
        tuple[float, str, int, int, int, int], tuple[int, bool]
    ] = {}
    matrix: dict[tuple[float, str, int, int, int], dict[int, int]] = {}
    for row in raw_rows:
        requested = float(row["requested_clock_mhz"])
        realized = float(row["realized_clock_mhz"])
        repeat = int(row["repeat"])
        if (
            not np.isfinite(requested)
            or not np.isfinite(realized)
            or requested <= 0.0
            or realized <= 0.0
        ):
            raise ValueError("measured requested/realized clocks must be finite and positive")
        if requested not in measured_clock_values:
            raise ValueError("raw profile contains a requested clock absent from the manifest")
        condition = (
            str(row["phase"]),
            int(row["prompt_tokens"]),
            int(row["output_tokens"]),
            int(row["concurrency"]),
        )
        if condition not in _EXPECTED_PROFILE_CONDITIONS:
            raise ValueError("raw profile contains a condition outside the fixed matrix")
        key = (requested, *condition)
        repetitions = matrix.setdefault(key, {})
        repetitions[repeat] = repetitions.get(repeat, 0) + 1
        batch_key = (*key, repeat)
        realized_by_batch.setdefault(batch_key, []).append(realized)
        telemetry_sample_count = int(row["telemetry_sample_count"])
        fallback_text = row["telemetry_fallback_used"].strip().lower()
        if telemetry_sample_count <= 0 or fallback_text not in {"false", "true"}:
            raise ValueError("raw profile contains invalid telemetry fallback metadata")
        telemetry_metadata = (
            telemetry_sample_count,
            fallback_text == "true",
        )
        previous_telemetry_metadata = telemetry_by_batch.setdefault(
            batch_key,
            telemetry_metadata,
        )
        if previous_telemetry_metadata != telemetry_metadata:
            raise ValueError(
                "raw profile telemetry fallback metadata must be constant within a batch"
            )
    batch_realized_by_requested: dict[float, list[float]] = {
        float(clock): [] for clock in measured_clocks
    }
    for requested in measured_clocks:
        requested_value = float(requested)
        for condition in _EXPECTED_PROFILE_CONDITIONS:
            repetitions = matrix.get((requested_value, *condition), {})
            if set(repetitions) != expected_repeats:
                raise ValueError("raw profile repetition matrix is incomplete")
            concurrency = condition[-1]
            if any(count != concurrency for count in repetitions.values()):
                raise ValueError("raw profile request multiplicity does not match concurrency")
            for repeat in expected_repeats:
                batch_realized_by_requested[requested_value].append(
                    float(
                        np.median(
                            realized_by_batch[
                                (requested_value, *condition, repeat)
                            ]
                        )
                    )
                )
    fallback_batch_count = sum(
        int(fallback_used)
        for _sample_count, fallback_used in telemetry_by_batch.values()
    )
    if int(manifest.get("telemetry_fallback_batch_count", -1)) != fallback_batch_count:
        raise ValueError(
            "measured telemetry_fallback_batch_count does not match the raw profile"
        )
    realized_medians = np.array(
        [
            np.median(batch_realized_by_requested[clock])
            for clock in measured_clock_for_modeled
        ],
        dtype=float,
    )
    if np.any(np.diff(realized_medians) <= 0.0):
        raise ValueError(
            "modeled realized clocks must be strictly increasing after batch-balanced aggregation"
        )
    declared_realized_medians = _manifest_realized_clock_medians(
        manifest,
        modeled_clocks,
    )
    if not np.allclose(
        declared_realized_medians,
        realized_medians,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "manifest realized-clock medians do not match batch-balanced raw measurements"
        )
    for date_field in ("measured_on", "completed_utc"):
        value = manifest.get(date_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"measured profile manifest lacks {date_field}")
    return declared_realized_medians


def load_profile(path: str | Path) -> PerformanceProfile:
    """Load a measured or explicitly provisional service profile."""

    profile_path = Path(path)
    with profile_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty service profile: {profile_path}")
    aggregate_columns = set(_AGGREGATE_PROFILE_COLUMNS)
    manifest_path = _manifest_path(profile_path)
    manifest: dict[str, object] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = str(manifest.get("profile_status", rows[0].get("profile_status", "unknown")))
    measurement_validated = False
    realized_clock_medians: np.ndarray | None = None
    if status == "measured_l4":
        if not manifest_path.exists():
            raise ValueError("measured_l4 profile is missing profile_manifest.json")
        realized_clock_medians = _validate_measured_profile_bundle(
            profile_path,
            rows,
            manifest_path,
            manifest,
        )
        measurement_validated = True
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
        measurement_validated=measurement_validated,
        profile_csv_sha256=_sha256(profile_path),
        manifest_sha256=_sha256(manifest_path) if manifest_path.exists() else "",
        realized_clock_median_mhz=realized_clock_medians,
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
    observed_levels = observation.realized_clock_levels_mhz or clocks
    previous_index = int(
        np.argmin(
            np.abs(np.asarray(observed_levels) - observation.previous_clock_mhz)
        )
    )
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
    previous_power_w: float,
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
        previous_power_w=float(previous_power_w),
        clock_levels_mhz=tuple(float(value) for value in plant.profile.clock_mhz),
        arrived_requests=arrived_count,
        completed_requests=completed_count,
        realized_clock_levels_mhz=tuple(
            float(value) for value in plant.profile.realized_clock_levels_mhz
        ),
    )


def _realized_clock(
    requested_clock_mhz: float,
    phase: Phase,
    temperature_c: float,
    plant: ServingPlant,
) -> _ClockRealization:
    """Resolve a request to the profile row and its observed clock response.

    Thermal protection is a modeled intervention: it selects the next-lower
    requested profile row before service and power are evaluated.  Validated L4
    rows were already measured under the experimental hardware power cap, so no
    second software power-cap downshift is applied to them.  Proxy profiles keep
    the earlier modeled power governor.
    """

    profile = plant.profile
    applied = profile.quantize_clock(requested_clock_mhz, downward=True)
    levels = profile.clock_mhz
    if temperature_c >= plant.thermal_limit_c:
        index = int(np.where(levels == applied)[0][0])
        applied = float(levels[max(0, index - 1)])
    if not profile.is_measured:
        while profile.power(phase, applied) > plant.power_limit_w + 1e-12:
            index = int(np.where(levels == applied)[0][0])
            if index == 0:
                break
            applied = float(levels[index - 1])
    return _ClockRealization(
        applied_profile_clock_mhz=applied,
        observed_clock_mhz=profile.realized_clock_for_level(applied),
    )


def _serve_prefill(
    queue: list[_RuntimeRequest],
    active: list[_RuntimeRequest],
    budget: float,
    now_s: float,
    plant: ServingPlant,
) -> tuple[float, float]:
    if not queue or budget <= 0.0:
        return 0.0, 0.0
    remaining_budget = float(budget)
    served_total = 0.0
    while queue and remaining_budget > 1e-10:
        item = queue[0]
        if item.prefill_start_s is None:
            item.prefill_start_s = now_s
        occupied = sum(x.prompt_processed + x.generated for x in queue + active)
        reserved_for_decode = (
            len(active) + 1
        ) * plant.decode_kv_reserve_tokens_per_active
        free_kv = max(
            0.0,
            plant.kv_capacity_tokens - occupied - reserved_for_decode,
        )
        local_budget = remaining_budget
        if (
            len(active) >= plant.maximum_active_requests
            and item.prefill_remaining <= local_budget
        ):
            local_budget = max(0.0, item.prefill_remaining - 1e-9)
        served = min(local_budget, item.prefill_remaining, free_kv)
        if served <= 1e-12:
            break
        item.prefill_remaining -= served
        item.prompt_processed += served
        served_total += served
        remaining_budget -= served
        if item.prefill_remaining <= 1e-8 and len(active) < plant.maximum_active_requests:
            item.prefill_remaining = 0.0
            queue.pop(0)
            active.append(item)
        else:
            break
    return float(served_total), float(served_total)


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
    cumulative_decode_tokens: np.ndarray,
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
    decode_increment = np.diff(np.concatenate([[0.0], cumulative_decode_tokens]))
    stalls = np.asarray(decode_active > 0, dtype=bool) & (decode_increment <= 1e-12)
    active_steps = np.asarray(decode_active) > 0
    queued = np.asarray(prefill_queue, dtype=float) + np.asarray(decode_active, dtype=float)
    minimum_clock_mask = (
        realized_clock_mhz <= plant.profile.minimum_realized_clock_mhz + 1e-9
    )
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
    previous_clock = plant.profile.minimum_realized_clock_mhz
    previous_power = plant.profile.power(
        "idle", plant.profile.minimum_clock_mhz
    )
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
            previous_power_w=previous_power,
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
        clock_realization = _realized_clock(
            requested_clock,
            phase,
            temperature,
            plant,
        )
        applied_clock = clock_realization.applied_profile_clock_mhz
        realized_clock = clock_realization.observed_clock_mhz
        prefill_service_time = 0.0
        decode_service_time = 0.0
        if phase == "prefill":
            prefill_rate = plant.profile.rate("prefill", applied_clock)
            budget = min(
                prefill_rate * plant.time_step_s,
                action.maximum_prefill_tokens,
            )
            served, _ = _serve_prefill(
                prefill,
                active,
                budget,
                time_s,
                plant,
            )
            cumulative_prefill += served
            prefill_service_time = served / max(prefill_rate, 1e-12)
            remaining_time = max(0.0, plant.time_step_s - prefill_service_time)
            if active and remaining_time > 1e-12:
                decode_budget = (
                    plant.profile.rate("decode", applied_clock) * remaining_time
                )
                decode_served, _ = _serve_decode(
                    active,
                    completed,
                    decode_budget,
                    time_s,
                    time_s + plant.time_step_s,
                    plant,
                )
                cumulative_decode += decode_served
                decode_service_time = (
                    decode_served
                    / max(plant.profile.rate("decode", applied_clock), 1e-12)
                )
            if served > 1e-12 and decode_service_time > 1e-12:
                phase = "interleaved"
            elif served > 1e-12:
                phase = "prefill"
            elif decode_service_time > 1e-12:
                phase = "decode"
            else:
                phase = "idle"
        elif phase == "decode":
            decode_rate = plant.profile.rate("decode", applied_clock)
            budget = decode_rate * plant.time_step_s
            served, _ = _serve_decode(
                active,
                completed,
                budget,
                time_s,
                time_s + plant.time_step_s,
                plant,
            )
            cumulative_decode += served
            decode_service_time = served / max(decode_rate, 1e-12)
            if served <= 1e-12:
                phase = "idle"

        idle_time = max(
            0.0,
            plant.time_step_s - prefill_service_time - decode_service_time,
        )
        step_energy = (
            plant.profile.power("prefill", applied_clock) * prefill_service_time
            + plant.profile.power("decode", applied_clock) * decode_service_time
            + plant.profile.power("idle", applied_clock) * idle_time
        )
        power = step_energy / plant.time_step_s
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
        previous_power = power
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
        arrays["cumulative_decode_tokens"],
        phase_values,
        plant,
    )

    plans_by_step = getattr(clock_controller, "plans_by_step", {})
    plan_start_times_by_step = getattr(
        clock_controller, "plan_start_times_by_step", {}
    )
    planned = tuple(
        tuple(float(value) for value in plans_by_step.get(index, ()))
        for index in range(len(time_values))
    )
    planned_start_times = tuple(
        (
            float(plan_start_times_by_step[index])
            if index in plan_start_times_by_step
            else None
        )
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
        planned_clock_start_time_s=planned_start_times,
        plan_control_period_s=getattr(clock_controller, "plan_dt_s", None),
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
        result.cumulative_decode_tokens,
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
