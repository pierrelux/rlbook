#!/usr/bin/env python3
"""Fit thermal models to an isolated L4 identification bundle.

This script deliberately accepts only the dedicated ``thermal-identification``
schema.  It never reads the earlier inference-profile telemetry.  Model
parameters are estimated from ``training_a`` and ``training_b`` only; the
``validation`` sequence is evaluated once after the final training fit.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares


SCHEMA_VERSION = 2
OUTPUT_SCHEMA_VERSION = 2
TELEMETRY_FILENAME = "l4_thermal_telemetry.csv"
TRAINING_SEQUENCES = ("training_a", "training_b")
VALIDATION_SEQUENCE = "validation"
SEQUENCE_ORDER = TRAINING_SEQUENCES + (VALIDATION_SEQUENCE,)
TRAINING_DURATION_CEILINGS_S = {40: 120.0, 46: 105.0, 52: 90.0, 58: 75.0, 64: 60.0}
EXPECTED_TRAINING_POWERS_W = frozenset(TRAINING_DURATION_CEILINGS_S)
EXPECTED_VALIDATION_POWERS_W = frozenset((43, 49, 55, 61))
REQUIRED_TELEMETRY_COLUMNS = (
    "elapsed_s",
    "utc",
    "phase",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "power_w",
    "temperature_c",
    "utilization_percent",
    "memory_used_mib",
    "split",
    "sequence",
    "block_id",
    "block_role",
    "requested_power_limit_w",
    "requested_clock_mhz",
    "workload_phase",
)


class ThermalFitError(RuntimeError):
    """Raised when an identification bundle or model fit is not trustworthy."""


@dataclass(frozen=True)
class FitConfig:
    """Numerical settings fixed before validation is inspected."""

    bin_width_s: float = 1.0
    multistart_count: int = 10
    max_nfev: int = 2_500
    random_seed: int = 20260902
    residual_autocorrelation_lags_s: tuple[int, ...] = (1, 5, 10, 30)

    def __post_init__(self) -> None:
        if self.bin_width_s <= 0.0:
            raise ValueError("bin_width_s must be positive")
        if self.multistart_count < 1:
            raise ValueError("multistart_count must be at least one")
        if self.max_nfev < 50:
            raise ValueError("max_nfev must be at least 50")
        if any(lag <= 0 for lag in self.residual_autocorrelation_lags_s):
            raise ValueError("residual autocorrelation lags must be positive")


@dataclass(frozen=True)
class SequenceData:
    """One independently initialized, uniformly sampled thermal pulse."""

    name: str
    split: str
    time_s: np.ndarray
    temperature_c: np.ndarray
    measured_power_w: np.ndarray
    raw_row_count: int
    repeat: str = ""
    role: str = ""
    workload_phase: str = ""
    requested_power_limit_w: float | None = None
    scheduled_duration_s: float | None = None

    def __post_init__(self) -> None:
        time_s = np.asarray(self.time_s, dtype=float)
        temperature_c = np.asarray(self.temperature_c, dtype=float)
        measured_power_w = np.asarray(self.measured_power_w, dtype=float)
        if time_s.ndim != 1 or temperature_c.ndim != 1 or measured_power_w.ndim != 1:
            raise ValueError("thermal sequence arrays must be one-dimensional")
        if len(time_s) != len(temperature_c):
            raise ValueError("temperature and time arrays must have equal length")
        if len(measured_power_w) != len(time_s) - 1:
            raise ValueError("power must contain one value per state transition")
        if len(time_s) < 3:
            raise ValueError("a thermal sequence needs at least three temperature states")
        if not (
            np.all(np.isfinite(time_s))
            and np.all(np.isfinite(temperature_c))
            and np.all(np.isfinite(measured_power_w))
        ):
            raise ValueError("thermal sequence arrays must be finite")
        if not np.all(np.diff(time_s) > 0.0):
            raise ValueError("thermal sequence time must be strictly increasing")
        if not np.all(measured_power_w > 0.0):
            raise ValueError("measured power must be positive")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "temperature_c", temperature_c)
        object.__setattr__(self, "measured_power_w", measured_power_w)


@dataclass(frozen=True)
class ScheduledPulse:
    """Immutable pulse metadata copied from the validated acquisition schedule."""

    block_id: str
    repeat: str
    split: str
    role: str
    requested_power_limit_w: float
    duration_s: float
    workload_phase: str


@dataclass(frozen=True)
class ThermalBundle:
    """Validated manifest, source hash, and prespecified data split."""

    manifest_path: Path
    telemetry_path: Path
    telemetry_sha256: str
    manifest: Mapping[str, Any]
    pulses: Mapping[str, SequenceData]
    scheduled_pulses: Mapping[str, ScheduledPulse]
    pulse_ids_by_sequence: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class _Candidate:
    transformed_parameters: np.ndarray
    cost: float
    success: bool
    nfev: int
    message: str
    jacobian: np.ndarray


@dataclass(frozen=True)
class ModelFit:
    model: str
    parameters: Mapping[str, float]
    transformed_parameters: np.ndarray
    objective_sum_squared_c: float
    diagnostics: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ThermalFitError(message)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(row: Mapping[str, str], column: str, row_number: int) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ThermalFitError(
            f"Telemetry row {row_number} has invalid {column!r}."
        ) from error
    _require(math.isfinite(value), f"Telemetry row {row_number} has non-finite {column!r}.")
    return value


def _validate_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, ScheduledPulse], dict[str, tuple[str, ...]]]:
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "Unsupported thermal schema.")
    _require(
        manifest.get("mode") == "thermal-identification",
        "Manifest is not a thermal-identification run.",
    )
    _require(
        manifest.get("protocol") == "cold-start-pulses-v2",
        "Manifest is not the cold-start pulse protocol.",
    )
    _require(manifest.get("status") == "complete", "Thermal acquisition is not complete.")
    _require(
        manifest.get("telemetry_columns") == list(REQUIRED_TELEMETRY_COLUMNS),
        "Manifest telemetry columns do not match the required schema.",
    )
    schedule = manifest.get("schedule")
    _require(isinstance(schedule, list), "Thermal manifest has no schedule.")
    _require(
        [item.get("sequence") for item in schedule] == list(SEQUENCE_ORDER),
        "Thermal split must remain training_a, training_b, validation in that order.",
    )
    scheduled_pulses: dict[str, ScheduledPulse] = {}
    pulse_ids_by_sequence: dict[str, tuple[str, ...]] = {}
    all_block_ids: list[str] = []
    for item, sequence_name in zip(schedule, SEQUENCE_ORDER):
        expected_split = "training" if sequence_name in TRAINING_SEQUENCES else "validation"
        _require(item.get("split") == expected_split, f"Incorrect split for {sequence_name}.")
        _require(
            item.get("requires_cooldown_before_every_pulse") is True,
            f"{sequence_name} lacks its per-pulse cooldown declaration.",
        )
        blocks = item.get("blocks")
        _require(isinstance(blocks, list) and blocks, f"{sequence_name} has no scheduled blocks.")
        identifiers: list[str] = []
        for block in blocks:
            block_id = block.get("block_id")
            _require(isinstance(block_id, str) and block_id, "Scheduled block lacks an identifier.")
            _require(block.get("sequence") == sequence_name, f"Block {block_id} changed sequence.")
            _require(block.get("split") == expected_split, f"Block {block_id} changed split.")
            role = block.get("role")
            _require(isinstance(role, str) and role, f"Block {block_id} has no role.")
            duration = block.get("duration_s")
            _require(
                isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0,
                f"Block {block_id} has invalid duration.",
            )
            requested_power = block.get("requested_power_limit_w")
            _require(
                isinstance(requested_power, (int, float))
                and math.isfinite(requested_power)
                and requested_power > 0.0,
                f"Block {block_id} has invalid requested power.",
            )
            condition = block.get("condition")
            _require(isinstance(condition, Mapping), f"Block {block_id} has no condition.")
            workload_phase = condition.get("phase")
            _require(
                workload_phase in ("prefill", "decode"),
                f"Block {block_id} has invalid workload phase.",
            )
            identifiers.append(block_id)
            all_block_ids.append(block_id)
            scheduled_pulses[block_id] = ScheduledPulse(
                block_id=block_id,
                repeat=sequence_name,
                split=expected_split,
                role=role,
                requested_power_limit_w=float(requested_power),
                duration_s=float(duration),
                workload_phase=str(workload_phase),
            )
        pulse_ids_by_sequence[sequence_name] = tuple(identifiers)
    _require(
        len(all_block_ids) == len(set(all_block_ids)),
        "Thermal schedule block identifiers are not unique.",
    )
    completed = manifest.get("completed_block_ids")
    _require(isinstance(completed, list), "Manifest has no completed-block list.")
    _require(
        completed == all_block_ids,
        "Completed blocks do not exactly match the prespecified schedule.",
    )
    for repeat in TRAINING_SEQUENCES:
        pulses = [scheduled_pulses[block_id] for block_id in pulse_ids_by_sequence[repeat]]
        _require(len(pulses) == 5, f"{repeat} must contain five training pulses.")
        _require(
            all(
                pulse.role == "training_pulse" and pulse.workload_phase == "decode"
                for pulse in pulses
            ),
            f"{repeat} must contain only fixed decode training pulses.",
        )
        by_power = {int(pulse.requested_power_limit_w): pulse for pulse in pulses}
        _require(
            set(by_power) == EXPECTED_TRAINING_POWERS_W,
            f"{repeat} does not contain every training power exactly once.",
        )
        _require(
            all(
                by_power[power].duration_s <= TRAINING_DURATION_CEILINGS_S[power]
                for power in EXPECTED_TRAINING_POWERS_W
            ),
            f"{repeat} exceeds a training pulse-duration ceiling.",
        )
    training_orders = [
        tuple(
            int(scheduled_pulses[block_id].requested_power_limit_w)
            for block_id in pulse_ids_by_sequence[repeat]
        )
        for repeat in TRAINING_SEQUENCES
    ]
    _require(training_orders[0] != training_orders[1], "Training pulse orders must differ.")
    for power in EXPECTED_TRAINING_POWERS_W:
        durations = [
            next(
                scheduled_pulses[block_id].duration_s
                for block_id in pulse_ids_by_sequence[repeat]
                if int(scheduled_pulses[block_id].requested_power_limit_w) == power
            )
            for repeat in TRAINING_SEQUENCES
        ]
        _require(durations[0] != durations[1], "Training repeat durations must differ by power.")

    validation_pulses = [
        scheduled_pulses[block_id]
        for block_id in pulse_ids_by_sequence[VALIDATION_SEQUENCE]
    ]
    intermediate = [pulse for pulse in validation_pulses if pulse.role == "intermediate_cap"]
    _require(
        len(intermediate) == 4
        and {int(pulse.requested_power_limit_w) for pulse in intermediate}
        == EXPECTED_VALIDATION_POWERS_W,
        "Validation intermediate-cap pulses changed.",
    )
    _require(
        all(pulse.duration_s <= 90.0 for pulse in validation_pulses),
        "Validation pulse duration exceeds 90 seconds.",
    )
    matched = [pulse for pulse in validation_pulses if pulse.role == "workload_transfer"]
    _require(
        len(matched) == 2
        and {(pulse.requested_power_limit_w, pulse.duration_s) for pulse in matched}
        == {(55.0, 60.0)}
        and {pulse.workload_phase for pulse in matched} == {"prefill", "decode"},
        "Validation lacks the matched 55 W prefill/decode pulse pair.",
    )
    _require(
        manifest.get("training_power_limits_w") == [40, 46, 52, 58, 64]
        and manifest.get("validation_power_limits_w") == [43, 49, 55, 61],
        "Manifest power-level declarations changed.",
    )
    cooldown = manifest.get("cooldown_protocol")
    _require(
        isinstance(cooldown, Mapping) and cooldown.get("before_every_pulse") is True,
        "Manifest lacks the cold-start cooldown protocol.",
    )
    cooldown_events = manifest.get("cooldown_events")
    _require(isinstance(cooldown_events, list), "Manifest has no cooldown-event record.")
    _require(
        [event.get("before_block_id") for event in cooldown_events] == all_block_ids
        and all(event.get("status") == "complete" for event in cooldown_events),
        "Completed cold-start events do not match every scheduled pulse.",
    )
    protocol = manifest.get("fit_protocol")
    _require(isinstance(protocol, Mapping), "Manifest has no fit protocol.")
    _require(
        protocol.get("acquisition_only") is True,
        "Thermal manifest is not marked as acquisition-only.",
    )
    _require(
        protocol.get("training_sequences") == list(TRAINING_SEQUENCES),
        "Manifest training sequences changed.",
    )
    _require(
        protocol.get("untouched_validation_sequence") == VALIDATION_SEQUENCE,
        "Manifest does not preserve the untouched validation sequence.",
    )
    _require(
        protocol.get("fit_input") == "measured power rather than requested cap",
        "Manifest does not require measured power for fitting.",
    )
    temperature_fit = protocol.get("temperature_fit")
    _require(
        isinstance(temperature_fit, str)
        and "do not finite-difference" in temperature_fit,
        "Manifest does not preserve level-based integer-temperature fitting.",
    )
    return scheduled_pulses, pulse_ids_by_sequence


def _time_weighted_power(times: np.ndarray, powers: np.ndarray, left: float, right: float) -> float:
    interior = (times > left) & (times < right)
    integration_times = np.concatenate(([left], times[interior], [right]))
    integration_power = np.interp(integration_times, times, powers)
    return float(np.trapezoid(integration_power, integration_times) / (right - left))


def _bin_pulse(
    pulse: ScheduledPulse,
    rows: Sequence[Mapping[str, Any]],
    *,
    bin_width_s: float,
    telemetry_period_s: float,
) -> SequenceData:
    name = pulse.block_id
    times = np.asarray([row["elapsed_s"] for row in rows], dtype=float)
    temperatures = np.asarray([row["temperature_c"] for row in rows], dtype=float)
    powers = np.asarray([row["power_w"] for row in rows], dtype=float)
    _require(len(rows) >= 3, f"{name} has too little telemetry.")
    _require(np.all(np.diff(times) > 0.0), f"{name} telemetry time is not strictly increasing.")
    maximum_gap = float(np.max(np.diff(times)))
    allowed_gap = max(0.5, 4.0 * telemetry_period_s)
    _require(
        maximum_gap <= allowed_gap,
        f"{name} telemetry has a {maximum_gap:.3f} s gap; limit is {allowed_gap:.3f} s.",
    )
    complete_intervals = int(math.floor((times[-1] - times[0]) / bin_width_s))
    _require(complete_intervals >= 2, f"{name} has fewer than two complete one-second bins.")
    grid = times[0] + bin_width_s * np.arange(complete_intervals + 1, dtype=float)
    grid_temperatures: list[float] = []
    tolerance = max(0.51 * bin_width_s, 3.0 * telemetry_period_s)
    for point in grid:
        insertion = int(np.searchsorted(times, point))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(times)]
        nearest = min(candidates, key=lambda index: abs(times[index] - point))
        distance = abs(times[nearest] - point)
        _require(distance <= tolerance, f"{name} has no temperature sample near t={point:.3f} s.")
        grid_temperatures.append(float(temperatures[nearest]))
    binned_power = np.asarray(
        [
            _time_weighted_power(times, powers, grid[index], grid[index + 1])
            for index in range(complete_intervals)
        ],
        dtype=float,
    )
    relative_time = grid - grid[0]
    return SequenceData(
        name=name,
        split=pulse.split,
        time_s=relative_time,
        temperature_c=np.asarray(grid_temperatures, dtype=float),
        measured_power_w=binned_power,
        raw_row_count=len(rows),
        repeat=pulse.repeat,
        role=pulse.role,
        workload_phase=pulse.workload_phase,
        requested_power_limit_w=pulse.requested_power_limit_w,
        scheduled_duration_s=pulse.duration_s,
    )


def load_thermal_bundle(
    manifest_path: str | Path,
    *,
    config: FitConfig = FitConfig(),
) -> ThermalBundle:
    """Load, validate, and one-second-bin a dedicated thermal acquisition."""

    path = Path(manifest_path).resolve()
    _require(path.name == "thermal_manifest.json", "Expected a thermal_manifest.json path.")
    _require(path.is_file(), f"Thermal manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ThermalFitError(f"Could not read thermal manifest: {error}") from error
    _require(isinstance(manifest, Mapping), "Thermal manifest root must be an object.")
    scheduled_pulses, pulse_ids_by_sequence = _validate_manifest(manifest)
    telemetry_path = path.parent / TELEMETRY_FILENAME
    _require(telemetry_path.is_file(), f"Missing {TELEMETRY_FILENAME}.")
    checksums = manifest.get("sha256")
    _require(isinstance(checksums, Mapping), "Manifest has no artifact checksums.")
    expected_checksum = checksums.get(TELEMETRY_FILENAME)
    _require(
        isinstance(expected_checksum, str) and len(expected_checksum) == 64,
        "Manifest has no valid telemetry checksum.",
    )
    actual_checksum = _sha256(telemetry_path)
    _require(actual_checksum == expected_checksum, "Thermal telemetry checksum mismatch.")

    parsed_rows: list[dict[str, Any]] = []
    with telemetry_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require(
            reader.fieldnames == list(REQUIRED_TELEMETRY_COLUMNS),
            "Thermal telemetry header does not match the required schema.",
        )
        for row_number, row in enumerate(reader, start=2):
            elapsed = _finite_number(row, "elapsed_s", row_number)
            power = _finite_number(row, "power_w", row_number)
            temperature = _finite_number(row, "temperature_c", row_number)
            _require(power > 0.0, f"Telemetry row {row_number} has non-positive measured power.")
            _require(
                math.isclose(temperature, round(temperature), abs_tol=1e-9),
                f"Telemetry row {row_number} is not an integer-valued temperature reading.",
            )
            parsed_rows.append(
                {
                    **row,
                    "elapsed_s": elapsed,
                    "power_w": power,
                    "temperature_c": temperature,
                }
            )
    _require(parsed_rows, "Thermal telemetry is empty.")
    expected_count = manifest.get("telemetry_row_count")
    _require(
        isinstance(expected_count, int) and expected_count == len(parsed_rows),
        "Manifest telemetry row count does not match the CSV.",
    )
    elapsed = np.asarray([row["elapsed_s"] for row in parsed_rows])
    _require(np.all(np.diff(elapsed) > 0.0), "Global telemetry time is not strictly increasing.")

    telemetry_period = manifest.get("telemetry_period_s")
    _require(
        isinstance(telemetry_period, (int, float))
        and math.isfinite(telemetry_period)
        and telemetry_period > 0.0,
        "Manifest telemetry period is invalid.",
    )
    modeled_rows = [row for row in parsed_rows if row["split"] in ("training", "validation")]
    scheduled_ids = set(scheduled_pulses)
    _require(
        all(row["block_id"] in scheduled_ids for row in modeled_rows),
        "Modeled telemetry contains a block absent from the pulse schedule.",
    )
    observed_order: list[str] = []
    for row in modeled_rows:
        if not observed_order or observed_order[-1] != row["block_id"]:
            observed_order.append(str(row["block_id"]))
    expected_order = [
        block_id
        for sequence_name in SEQUENCE_ORDER
        for block_id in pulse_ids_by_sequence[sequence_name]
    ]
    _require(
        observed_order == expected_order,
        "Modeled telemetry pulse order differs from the manifest.",
    )
    pulses: dict[str, SequenceData] = {}
    for block_id in expected_order:
        pulse = scheduled_pulses[block_id]
        selected = [row for row in modeled_rows if row["block_id"] == block_id]
        _require(selected, f"No modeled telemetry for pulse {block_id}.")
        _require(
            all(
                row["split"] == pulse.split
                and row["sequence"] == pulse.repeat
                and row["block_role"] == pulse.role
                and row["workload_phase"] == pulse.workload_phase
                and math.isclose(
                    float(row["requested_power_limit_w"]),
                    pulse.requested_power_limit_w,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for row in selected
            ),
            f"Telemetry metadata for pulse {block_id} differs from its schedule.",
        )
        span = float(selected[-1]["elapsed_s"] - selected[0]["elapsed_s"])
        tolerance = max(2.0, 5.0 * float(telemetry_period))
        _require(
            span >= pulse.duration_s - tolerance
            and span <= pulse.duration_s + 30.0,
            f"Pulse {block_id} telemetry duration does not match its schedule.",
        )
        pulses[block_id] = _bin_pulse(
            pulse,
            selected,
            bin_width_s=config.bin_width_s,
            telemetry_period_s=float(telemetry_period),
        )
    allowed_pairs = {
        ("conditioning", sequence_name) for sequence_name in SEQUENCE_ORDER
    } | {
        ("", ""),
        ("conditioning", "finalization"),
        ("training", "training_a"),
        ("training", "training_b"),
        ("validation", "validation"),
    }
    unexpected_pairs = {
        (str(row["split"]), str(row["sequence"])) for row in parsed_rows
    } - allowed_pairs
    _require(
        not unexpected_pairs,
        f"Unexpected telemetry split/sequence labels: {unexpected_pairs}",
    )
    unlabeled = [
        row for row in parsed_rows if row["split"] == "" and row["sequence"] == ""
    ]
    _require(
        all(row["phase"] == "initialization" for row in unlabeled),
        "Only initialization telemetry may omit split and sequence labels.",
    )
    conditioning = [row for row in parsed_rows if row["split"] == "conditioning"]
    for block_id in expected_order:
        cooldown_id = f"cooldown_before_{block_id}"
        _require(
            any(
                row["sequence"] == scheduled_pulses[block_id].repeat
                and row["block_id"] == cooldown_id
                and row["block_role"] == "cooldown"
                for row in conditioning
            ),
            f"Pulse {block_id} has no recorded conditioning cooldown telemetry.",
        )
    return ThermalBundle(
        manifest_path=path,
        telemetry_path=telemetry_path,
        telemetry_sha256=actual_checksum,
        manifest=manifest,
        pulses=pulses,
        scheduled_pulses=scheduled_pulses,
        pulse_ids_by_sequence=pulse_ids_by_sequence,
    )


def simulate_one_state_rc(
    parameters: Mapping[str, float], sequence: SequenceData
) -> np.ndarray:
    """Free-run an exact-discrete one-state RC model."""

    ambient = float(parameters["ambient_temperature_c"])
    resistance = float(parameters["thermal_resistance_c_per_w"])
    time_constant = float(parameters["thermal_time_constant_s"])
    if ambient <= 0.0 or resistance <= 0.0 or time_constant <= 0.0:
        raise ValueError("one-state RC parameters must be positive")
    predicted = np.empty_like(sequence.temperature_c)
    predicted[0] = sequence.temperature_c[0]
    for index, power in enumerate(sequence.measured_power_w):
        dt = sequence.time_s[index + 1] - sequence.time_s[index]
        decay = math.exp(-dt / time_constant)
        equilibrium = ambient + resistance * power
        predicted[index + 1] = decay * predicted[index] + (1.0 - decay) * equilibrium
    return predicted


def _two_state_discretization(
    parameters: Mapping[str, float], dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ambient = float(parameters["ambient_temperature_c"])
    junction_capacity = float(parameters["junction_capacitance_j_per_c"])
    sink_capacity = float(parameters["sink_capacitance_j_per_c"])
    junction_sink_resistance = float(parameters["junction_sink_resistance_c_per_w"])
    sink_ambient_resistance = float(parameters["sink_ambient_resistance_c_per_w"])
    values = (
        ambient,
        junction_capacity,
        sink_capacity,
        junction_sink_resistance,
        sink_ambient_resistance,
    )
    if any(value <= 0.0 for value in values):
        raise ValueError("two-state junction/sink RC parameters must be positive")
    junction_conductance = 1.0 / junction_sink_resistance
    ambient_conductance = 1.0 / sink_ambient_resistance
    continuous = np.asarray(
        [
            [
                -junction_conductance / junction_capacity,
                junction_conductance / junction_capacity,
            ],
            [
                junction_conductance / sink_capacity,
                -(junction_conductance + ambient_conductance) / sink_capacity,
            ],
        ],
        dtype=float,
    )
    power_input = np.asarray([1.0 / junction_capacity, 0.0], dtype=float)
    ambient_input = np.asarray(
        [0.0, ambient * ambient_conductance / sink_capacity], dtype=float
    )
    augmented = np.zeros((4, 4), dtype=float)
    augmented[:2, :2] = continuous
    augmented[:2, 2] = power_input
    augmented[:2, 3] = ambient_input
    discrete = expm(augmented * dt)
    return discrete[:2, :2], discrete[:2, 2], discrete[:2, 3]


def simulate_two_state_rc(
    parameters: Mapping[str, float], sequence: SequenceData
) -> np.ndarray:
    """Free-run a positive two-state junction/sink RC model."""

    predicted = np.empty_like(sequence.temperature_c)
    state = np.asarray([sequence.temperature_c[0], sequence.temperature_c[0]], dtype=float)
    predicted[0] = state[0]
    previous_dt: float | None = None
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for index, power in enumerate(sequence.measured_power_w):
        dt = float(sequence.time_s[index + 1] - sequence.time_s[index])
        if matrices is None or previous_dt is None or not math.isclose(dt, previous_dt):
            matrices = _two_state_discretization(parameters, dt)
            previous_dt = dt
        transition, power_gain, offset = matrices
        state = transition @ state + power_gain * power + offset
        predicted[index + 1] = state[0]
    return predicted


def _one_state_parameters(transformed: np.ndarray) -> dict[str, float]:
    ambient, log_resistance, log_time_constant = transformed
    resistance = math.exp(float(log_resistance))
    time_constant = math.exp(float(log_time_constant))
    return {
        "ambient_temperature_c": float(ambient),
        "thermal_resistance_c_per_w": resistance,
        "thermal_time_constant_s": time_constant,
        "thermal_capacitance_j_per_c": time_constant / resistance,
    }


def _two_state_parameters(transformed: np.ndarray) -> dict[str, float]:
    ambient, log_c_junction, log_c_sink, log_r_junction_sink, log_r_sink_ambient = transformed
    return {
        "ambient_temperature_c": float(ambient),
        "junction_capacitance_j_per_c": math.exp(float(log_c_junction)),
        "sink_capacitance_j_per_c": math.exp(float(log_c_sink)),
        "junction_sink_resistance_c_per_w": math.exp(float(log_r_junction_sink)),
        "sink_ambient_resistance_c_per_w": math.exp(float(log_r_sink_ambient)),
    }


def _concatenated_residuals(
    transformed: np.ndarray,
    sequences: Sequence[SequenceData],
    parameter_decoder: Callable[[np.ndarray], Mapping[str, float]],
    simulator: Callable[[Mapping[str, float], SequenceData], np.ndarray],
) -> np.ndarray:
    parameters = parameter_decoder(transformed)
    return np.concatenate(
        [simulator(parameters, sequence)[1:] - sequence.temperature_c[1:] for sequence in sequences]
    )


def _candidate_diagnostics(
    candidates: Sequence[_Candidate], best: _Candidate, lower: np.ndarray, upper: np.ndarray
) -> dict[str, Any]:
    singular_values = np.linalg.svd(best.jacobian, compute_uv=False)
    tolerance = (
        singular_values[0] * max(best.jacobian.shape) * np.finfo(float).eps
        if singular_values.size
        else math.inf
    )
    rank = int(np.sum(singular_values > tolerance)) if singular_values.size else 0
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else math.inf
    )
    margin = np.minimum(best.transformed_parameters - lower, upper - best.transformed_parameters)
    near = [
        candidate
        for candidate in candidates
        if candidate.success and candidate.cost <= best.cost * 1.01 + 1e-12
    ]
    spread = np.ptp(
        np.stack([candidate.transformed_parameters for candidate in near]), axis=0
    ) if len(near) > 1 else np.zeros_like(best.transformed_parameters)
    return {
        "optimizer_success": best.success,
        "optimizer_message": best.message,
        "optimizer_evaluations": best.nfev,
        "successful_multistarts": sum(candidate.success for candidate in candidates),
        "near_optimal_multistarts_within_one_percent": len(near),
        "jacobian_numerical_rank": rank,
        "parameter_count": int(best.jacobian.shape[1]),
        "jacobian_singular_values": [float(value) for value in singular_values],
        "jacobian_condition_number": condition if math.isfinite(condition) else None,
        "minimum_transformed_bound_margin": float(np.min(margin)),
        "near_optimal_transformed_parameter_spread": [float(value) for value in spread],
        "locally_identifiable": bool(
            rank == best.jacobian.shape[1]
            and math.isfinite(condition)
            and condition < 1.0e8
            and float(np.min(margin)) > 1.0e-4
        ),
    }


def _run_multistart(
    residual: Callable[[np.ndarray], np.ndarray],
    starts: Sequence[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_nfev: int,
) -> tuple[_Candidate, list[_Candidate]]:
    candidates: list[_Candidate] = []
    for start in starts:
        clipped = np.clip(np.asarray(start, dtype=float), lower + 1e-9, upper - 1e-9)
        result = least_squares(
            residual,
            clipped,
            bounds=(lower, upper),
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            x_scale="jac",
        )
        value = float(np.dot(result.fun, result.fun))
        if math.isfinite(value) and np.all(np.isfinite(result.x)):
            candidates.append(
                _Candidate(
                    transformed_parameters=np.asarray(result.x, dtype=float),
                    cost=value,
                    success=bool(result.success),
                    nfev=int(result.nfev),
                    message=str(result.message),
                    jacobian=np.asarray(result.jac, dtype=float),
                )
            )
    successful = [candidate for candidate in candidates if candidate.success]
    if not successful:
        raise ThermalFitError("No multistart optimization converged successfully.")
    return min(successful, key=lambda candidate: candidate.cost), candidates


def _training_summary(sequences: Sequence[SequenceData]) -> tuple[float, float, float]:
    temperatures = np.concatenate([sequence.temperature_c for sequence in sequences])
    powers = np.concatenate([sequence.measured_power_w for sequence in sequences])
    return float(np.min(temperatures)), float(np.median(temperatures)), float(np.ptp(powers))


def fit_one_state_rc(
    sequences: Sequence[SequenceData], *, config: FitConfig = FitConfig()
) -> ModelFit:
    """Fit an exact-discrete positive one-state RC model to training data."""

    _require(bool(sequences), "One-state fitting received no training sequences.")
    _require(
        all(sequence.split == "training" for sequence in sequences),
        "One-state fitting accepts training sequences only.",
    )
    minimum_temperature, median_temperature, power_span = _training_summary(sequences)
    _require(power_span >= 1.0, "Training measured power lacks sufficient excitation.")
    lower = np.asarray([0.01, math.log(1.0e-4), math.log(0.05)])
    upper = np.asarray([100.0, math.log(10.0), math.log(1.0e5)])
    powers = np.concatenate([sequence.measured_power_w for sequence in sequences])
    temperatures = np.concatenate([sequence.temperature_c for sequence in sequences])
    resistance_guess = float(np.clip(np.ptp(temperatures) / max(power_span, 1.0), 0.02, 1.0))
    ambient_guess = float(
        np.clip(median_temperature - resistance_guess * np.median(powers), 5.0, 90.0)
    )
    deterministic = [
        np.asarray([ambient, math.log(resistance), math.log(time_constant)])
        for ambient in (ambient_guess, max(5.0, minimum_temperature - 15.0))
        for resistance in (resistance_guess, max(0.01, resistance_guess / 2.0))
        for time_constant in (2.0, 20.0, 200.0)
    ]
    rng = np.random.default_rng(config.random_seed + len(sequences))
    starts = deterministic[: config.multistart_count]
    while len(starts) < config.multistart_count:
        starts.append(
            np.asarray(
                [
                    rng.uniform(5.0, 85.0),
                    rng.uniform(math.log(0.005), math.log(2.0)),
                    rng.uniform(math.log(0.2), math.log(2_000.0)),
                ]
            )
        )
    residual = lambda transformed: _concatenated_residuals(
        transformed,
        sequences,
        _one_state_parameters,
        simulate_one_state_rc,
    )
    best, candidates = _run_multistart(
        residual, starts, lower, upper, max_nfev=config.max_nfev
    )
    parameters = _one_state_parameters(best.transformed_parameters)
    diagnostics = _candidate_diagnostics(candidates, best, lower, upper)
    pole = -1.0 / parameters["thermal_time_constant_s"]
    diagnostics.update(
        {
            "continuous_poles_per_s": [pole],
            "discrete_poles_at_one_second": [math.exp(pole)],
            "asymptotically_stable": pole < 0.0,
            "positive_thermal_parameters": True,
            "measured_power_span_w": power_span,
        }
    )
    return ModelFit(
        model="one_state_rc",
        parameters=parameters,
        transformed_parameters=best.transformed_parameters,
        objective_sum_squared_c=best.cost,
        diagnostics=diagnostics,
    )


def _two_state_starts(
    sequences: Sequence[SequenceData], config: FitConfig
) -> list[np.ndarray]:
    minimum_temperature, median_temperature, power_span = _training_summary(sequences)
    powers = np.concatenate([sequence.measured_power_w for sequence in sequences])
    total_resistance = float(
        np.clip(
            np.ptp(np.concatenate([sequence.temperature_c for sequence in sequences]))
            / max(power_span, 1.0),
            0.04,
            1.5,
        )
    )
    ambient = float(np.clip(median_temperature - total_resistance * np.median(powers), 5, 90))
    physical = [
        (ambient, 20.0, 200.0, 0.05, max(0.02, total_resistance - 0.05)),
        (ambient, 50.0, 500.0, 0.02, max(0.02, total_resistance - 0.02)),
        (max(5.0, minimum_temperature - 15.0), 10.0, 1_000.0, 0.1, 0.3),
        (ambient, 100.0, 100.0, 0.1, 0.1),
    ]
    starts = [
        np.asarray([value[0], *(math.log(component) for component in value[1:])])
        for value in physical
    ][: config.multistart_count]
    rng = np.random.default_rng(config.random_seed + 100 + len(sequences))
    while len(starts) < config.multistart_count:
        starts.append(
            np.asarray(
                [
                    rng.uniform(5.0, 85.0),
                    rng.uniform(math.log(0.5), math.log(2_000.0)),
                    rng.uniform(math.log(1.0), math.log(20_000.0)),
                    rng.uniform(math.log(0.001), math.log(2.0)),
                    rng.uniform(math.log(0.005), math.log(5.0)),
                ]
            )
        )
    return starts


def _two_state_continuous_matrix(parameters: Mapping[str, float]) -> np.ndarray:
    c_junction = parameters["junction_capacitance_j_per_c"]
    c_sink = parameters["sink_capacitance_j_per_c"]
    r_junction_sink = parameters["junction_sink_resistance_c_per_w"]
    r_sink_ambient = parameters["sink_ambient_resistance_c_per_w"]
    g_junction_sink = 1.0 / r_junction_sink
    g_sink_ambient = 1.0 / r_sink_ambient
    return np.asarray(
        [
            [-g_junction_sink / c_junction, g_junction_sink / c_junction],
            [
                g_junction_sink / c_sink,
                -(g_junction_sink + g_sink_ambient) / c_sink,
            ],
        ]
    )


def fit_two_state_rc(
    sequences: Sequence[SequenceData], *, config: FitConfig = FitConfig()
) -> ModelFit:
    """Fit a positive two-state junction/sink RC model to training data."""

    _require(bool(sequences), "Two-state fitting received no training sequences.")
    _require(
        all(sequence.split == "training" for sequence in sequences),
        "Two-state fitting accepts training sequences only.",
    )
    _, _, power_span = _training_summary(sequences)
    _require(power_span >= 1.0, "Training measured power lacks sufficient excitation.")
    lower = np.asarray(
        [0.01, math.log(1.0e-2), math.log(1.0e-2), math.log(1.0e-5), math.log(1.0e-5)]
    )
    upper = np.asarray(
        [100.0, math.log(1.0e7), math.log(1.0e7), math.log(20.0), math.log(20.0)]
    )
    starts = _two_state_starts(sequences, config)
    residual = lambda transformed: _concatenated_residuals(
        transformed,
        sequences,
        _two_state_parameters,
        simulate_two_state_rc,
    )
    best, candidates = _run_multistart(
        residual, starts, lower, upper, max_nfev=config.max_nfev
    )
    parameters = _two_state_parameters(best.transformed_parameters)
    continuous = _two_state_continuous_matrix(parameters)
    poles = np.linalg.eigvals(continuous)
    real_poles = np.real(poles)
    modal_time_constants = sorted(float(-1.0 / pole) for pole in real_poles)
    diagnostics = _candidate_diagnostics(candidates, best, lower, upper)
    diagnostics.update(
        {
            "continuous_poles_per_s": [float(value) for value in sorted(real_poles)],
            "discrete_poles_at_one_second": [
                float(math.exp(value)) for value in sorted(real_poles)
            ],
            "modal_time_constants_s": modal_time_constants,
            "modal_time_constant_ratio": (
                modal_time_constants[-1] / modal_time_constants[0]
            ),
            "asymptotically_stable": bool(np.all(real_poles < 0.0)),
            "positive_thermal_parameters": True,
            "measured_power_span_w": power_span,
        }
    )
    return ModelFit(
        model="two_state_junction_sink_rc",
        parameters=parameters,
        transformed_parameters=best.transformed_parameters,
        objective_sum_squared_c=best.cost,
        diagnostics=diagnostics,
    )


def _autocorrelation(residual: np.ndarray, lag: int) -> float | None:
    if lag >= len(residual):
        return None
    left = residual[:-lag]
    right = residual[lag:]
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def trajectory_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    config: FitConfig = FitConfig(),
) -> dict[str, Any]:
    """Return free-run trajectory errors without differentiating temperature."""

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("observed and predicted temperatures must be equal-length vectors")
    residual = predicted - observed
    absolute = np.abs(residual)
    peak_error = float(np.max(predicted) - np.max(observed))
    return {
        "sample_count": int(len(observed)),
        "rmse_c": float(np.sqrt(np.mean(residual**2))),
        "mae_c": float(np.mean(absolute)),
        "maximum_absolute_error_c": float(np.max(absolute)),
        "peak_temperature_error_c": peak_error,
        "absolute_peak_temperature_error_c": abs(peak_error),
        "mean_residual_c": float(np.mean(residual)),
        "residual_autocorrelation": {
            f"lag_{lag}_s": _autocorrelation(
                residual, max(1, int(round(lag / config.bin_width_s)))
            )
            for lag in config.residual_autocorrelation_lags_s
        },
    }


def _simulator_for_fit(
    fit: ModelFit,
) -> Callable[[Mapping[str, float], SequenceData], np.ndarray]:
    return simulate_one_state_rc if fit.model == "one_state_rc" else simulate_two_state_rc


def _aggregate_trajectory_metrics(
    pulses: Sequence[SequenceData],
    predictions: Mapping[str, np.ndarray],
    *,
    config: FitConfig,
) -> dict[str, Any]:
    residuals = [
        predictions[pulse.name][1:] - pulse.temperature_c[1:] for pulse in pulses
    ]
    combined = np.concatenate(residuals)
    absolute = np.abs(combined)
    observed_peaks = np.asarray([np.max(pulse.temperature_c) for pulse in pulses])
    predicted_peaks = np.asarray([np.max(predictions[pulse.name]) for pulse in pulses])
    autocorrelation: dict[str, float | None] = {}
    for lag_s in config.residual_autocorrelation_lags_s:
        lag = max(1, int(round(lag_s / config.bin_width_s)))
        left_parts = [residual[:-lag] for residual in residuals if len(residual) > lag]
        right_parts = [residual[lag:] for residual in residuals if len(residual) > lag]
        if not left_parts:
            value = None
        else:
            left = np.concatenate(left_parts)
            right = np.concatenate(right_parts)
            value = (
                None
                if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12
                else float(np.corrcoef(left, right)[0, 1])
            )
        autocorrelation[f"lag_{lag_s}_s"] = value
    peak_errors = predicted_peaks - observed_peaks
    return {
        "pulse_count": len(pulses),
        "sample_count": int(len(combined)),
        "rmse_c": float(np.sqrt(np.mean(combined**2))),
        "mae_c": float(np.mean(absolute)),
        "maximum_absolute_error_c": float(np.max(absolute)),
        "peak_temperature_error_c": float(np.max(predicted_peaks) - np.max(observed_peaks)),
        "absolute_peak_temperature_error_c": float(
            abs(np.max(predicted_peaks) - np.max(observed_peaks))
        ),
        "mean_absolute_per_pulse_peak_error_c": float(np.mean(np.abs(peak_errors))),
        "maximum_absolute_per_pulse_peak_error_c": float(np.max(np.abs(peak_errors))),
        "mean_residual_c": float(np.mean(combined)),
        "residual_autocorrelation": autocorrelation,
    }


def _evaluate_pulses(
    fit: ModelFit,
    pulses: Sequence[SequenceData],
    *,
    config: FitConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    simulator = _simulator_for_fit(fit)
    predictions: dict[str, np.ndarray] = {}
    per_pulse: dict[str, Any] = {}
    for pulse in pulses:
        predicted = simulator(fit.parameters, pulse)
        predictions[pulse.name] = predicted
        per_pulse[pulse.name] = {
            "repeat": pulse.repeat,
            "role": pulse.role,
            "workload_phase": pulse.workload_phase,
            "requested_power_limit_w": pulse.requested_power_limit_w,
            "scheduled_duration_s": pulse.scheduled_duration_s,
            "metrics": trajectory_metrics(
                pulse.temperature_c[1:], predicted[1:], config=config
            ),
        }
    return (
        {
            "per_pulse": per_pulse,
            "aggregate": _aggregate_trajectory_metrics(
                pulses, predictions, config=config
            ),
        },
        predictions,
    )


def _matched_validation_pair(
    bundle: ThermalBundle,
    validation_report: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    matched = [
        bundle.pulses[block_id]
        for block_id in bundle.pulse_ids_by_sequence[VALIDATION_SEQUENCE]
        if bundle.scheduled_pulses[block_id].role == "workload_transfer"
    ]
    by_phase = {pulse.workload_phase: pulse for pulse in matched}
    _require(
        set(by_phase) == {"prefill", "decode"},
        "Validated manifest lost its matched workload-transfer pair.",
    )
    prefill = by_phase["prefill"]
    decode = by_phase["decode"]

    def summary(pulse: SequenceData) -> dict[str, Any]:
        predicted = predictions[pulse.name]
        return {
            "block_id": pulse.name,
            "observed_peak_temperature_c": float(np.max(pulse.temperature_c)),
            "predicted_peak_temperature_c": float(np.max(predicted)),
            "observed_temperature_rise_c": float(
                pulse.temperature_c[-1] - pulse.temperature_c[0]
            ),
            "predicted_temperature_rise_c": float(predicted[-1] - predicted[0]),
            "metrics": validation_report["per_pulse"][pulse.name]["metrics"],
        }

    prefill_summary = summary(prefill)
    decode_summary = summary(decode)
    return {
        "matching_rule": "role=workload_transfer, requested cap=55 W, duration=60 s",
        "prefill": prefill_summary,
        "decode": decode_summary,
        "observed_prefill_minus_decode_peak_c": (
            prefill_summary["observed_peak_temperature_c"]
            - decode_summary["observed_peak_temperature_c"]
        ),
        "predicted_prefill_minus_decode_peak_c": (
            prefill_summary["predicted_peak_temperature_c"]
            - decode_summary["predicted_peak_temperature_c"]
        ),
    }


def _fit_model(
    model: str, sequences: Sequence[SequenceData], *, config: FitConfig
) -> ModelFit:
    if model == "one_state_rc":
        return fit_one_state_rc(sequences, config=config)
    if model == "two_state_junction_sink_rc":
        return fit_two_state_rc(sequences, config=config)
    raise ValueError(f"Unknown thermal model {model!r}")


def _fit_to_json(fit: ModelFit) -> dict[str, Any]:
    return {
        "parameters": dict(fit.parameters),
        "objective_sum_squared_c": fit.objective_sum_squared_c,
        "diagnostics": dict(fit.diagnostics),
    }


def build_fit_report(
    bundle: ThermalBundle, *, config: FitConfig = FitConfig()
) -> dict[str, Any]:
    """Fit on training only, then perform exactly one validation evaluation."""

    training_by_repeat = {
        repeat: [
            bundle.pulses[block_id]
            for block_id in bundle.pulse_ids_by_sequence[repeat]
        ]
        for repeat in TRAINING_SEQUENCES
    }
    training = [pulse for repeat in TRAINING_SEQUENCES for pulse in training_by_repeat[repeat]]
    _require(
        all(sequence.split == "training" for sequence in training),
        "Internal split error: validation entered the fitting collection.",
    )
    validation = [
        bundle.pulses[block_id]
        for block_id in bundle.pulse_ids_by_sequence[VALIDATION_SEQUENCE]
    ]
    models: dict[str, Any] = {}
    for model in ("one_state_rc", "two_state_junction_sink_rc"):
        folds = []
        for fit_name, held_name in (
            ("training_a", "training_b"),
            ("training_b", "training_a"),
        ):
            fit_pulses = training_by_repeat[fit_name]
            held_pulses = training_by_repeat[held_name]
            fold_fit = _fit_model(model, fit_pulses, config=config)
            fit_evaluation, _ = _evaluate_pulses(
                fold_fit, fit_pulses, config=config
            )
            held_evaluation, _ = _evaluate_pulses(
                fold_fit, held_pulses, config=config
            )
            folds.append(
                {
                    "fit_training_repeats": [fit_name],
                    "fit_pulse_ids": [pulse.name for pulse in fit_pulses],
                    "held_out_training_repeat": held_name,
                    "held_out_pulse_ids": [pulse.name for pulse in held_pulses],
                    "fit": _fit_to_json(fold_fit),
                    "fit_repeat_evaluation": fit_evaluation,
                    "held_out_repeat_evaluation": held_evaluation,
                }
            )
        final_fit = _fit_model(model, training, config=config)
        training_evaluation, training_predictions = _evaluate_pulses(
            final_fit, training, config=config
        )
        training_evaluation["by_repeat"] = {
            repeat: _aggregate_trajectory_metrics(
                training_by_repeat[repeat], training_predictions, config=config
            )
            for repeat in TRAINING_SEQUENCES
        }
        # This is the sole pass in which validation temperatures are scored.
        validation_evaluation, validation_predictions = _evaluate_pulses(
            final_fit, validation, config=config
        )
        validation_evaluation["evaluation_passes"] = 1
        validation_evaluation["matched_55w_prefill_decode"] = _matched_validation_pair(
            bundle, validation_evaluation, validation_predictions
        )
        models[model] = {
            "leave_one_training_repeat_out": folds,
            "final_training_fit": _fit_to_json(final_fit),
            "training_evaluation": training_evaluation,
            "validation_evaluation": validation_evaluation,
        }
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "method": (
            "one-second measured-power bins; exact-discrete free-run trajectory fitting "
            "to integer-valued temperature observations; no finite differences"
        ),
        "source": {
            "manifest": bundle.manifest_path.name,
            "telemetry": bundle.telemetry_path.name,
            "telemetry_sha256": bundle.telemetry_sha256,
            "source_manifest_schema_version": bundle.manifest.get("schema_version"),
            "source_git_revision": bundle.manifest.get("git_revision"),
        },
        "split": {
            "training_repeats": list(TRAINING_SEQUENCES),
            "untouched_validation_sequence": VALIDATION_SEQUENCE,
            "validation_used_for_model_selection": False,
        },
        "preprocessing": {
            "bin_width_s": config.bin_width_s,
            "power_statistic": "time-weighted mean of measured power",
            "temperature_statistic": "nearest integer-valued sensor observation at bin boundary",
            "finite_differencing": False,
            "pulse_initialization": (
                "each independently cooled pulse starts from its own first observed "
                "junction temperature; the two-state sink starts at that same temperature"
            ),
            "pulse_samples": {
                name: {
                    "repeat": pulse.repeat,
                    "role": pulse.role,
                    "workload_phase": pulse.workload_phase,
                    "requested_power_limit_w": pulse.requested_power_limit_w,
                    "raw_rows": pulse.raw_row_count,
                    "binned_temperature_states": len(pulse.temperature_c),
                    "measured_power_intervals": len(pulse.measured_power_w),
                    "duration_s": float(pulse.time_s[-1]),
                    "measured_power_min_w": float(np.min(pulse.measured_power_w)),
                    "measured_power_max_w": float(np.max(pulse.measured_power_w)),
                }
                for name, pulse in bundle.pulses.items()
            },
        },
        "models": models,
    }


def write_fit_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Atomically write a finite, deterministic JSON report."""

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
    temporary.replace(path)
    return path


def fit_thermal_bundle(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    config: FitConfig = FitConfig(),
) -> dict[str, Any]:
    """Validate a bundle, fit both models, score validation once, and export JSON."""

    bundle = load_thermal_bundle(manifest_path, config=config)
    report = build_fit_report(bundle, config=config)
    write_fit_report(report, output_path)
    return report


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a completed thermal_manifest.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the immutable JSON fit report.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    try:
        report = fit_thermal_bundle(options.manifest, options.output)
    except (OSError, ValueError, ThermalFitError) as error:
        raise SystemExit(f"Thermal fitting failed closed: {error}") from error
    summaries = []
    for name, model in report["models"].items():
        metrics = model["validation_evaluation"]["aggregate"]
        summaries.append(f"{name}: validation RMSE {metrics['rmse_c']:.3f} C")
    print(f"Wrote {options.output.resolve()} ({'; '.join(summaries)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
