#!/usr/bin/env python3
"""Fit the pre-registered L4 prefill/decode phase-confirmation experiment.

The acquisition contract contains four training pulses and one untouched 55 W
decode/prefill validation pair.  This module fits two one-state RC models to
``phase_training`` only:

* a power-only model; and
* a phase-gain model with ``P_eff = P * (1 + beta * I_prefill)``.

Validation temperatures are evaluated only after both final fits are fixed.
The script is offline-only and never launches or controls cloud hardware.
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

from fit_inference_thermal import (
    FitConfig,
    ModelFit,
    REQUIRED_TELEMETRY_COLUMNS,
    ScheduledPulse,
    SequenceData,
    ThermalFitError,
    _aggregate_trajectory_metrics,
    _candidate_diagnostics,
    _one_state_parameters,
    _run_multistart,
    _time_weighted_power,
    fit_one_state_rc,
    simulate_one_state_rc,
    trajectory_metrics,
)


SOURCE_SCHEMA_VERSION = 2
REPORT_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "thermal_phase_manifest.json"
TELEMETRY_FILENAME = "l4_thermal_phase_telemetry.csv"
REQUESTS_FILENAME = "l4_thermal_phase_requests.csv"
TRAINING_SEQUENCE = "phase_training"
VALIDATION_SEQUENCE = "phase_validation"
SEQUENCE_ORDER = (TRAINING_SEQUENCE, VALIDATION_SEQUENCE)

REQUIRED_REQUEST_COLUMNS = (
    "split",
    "sequence",
    "block_id",
    "workload_phase",
    "prompt_tokens",
    "output_tokens",
    "concurrency",
    "requested_power_limit_w",
    "requested_clock_mhz",
    "batch_index",
    "request_index",
    "prompt_tokens_observed",
    "completion_tokens",
    "ttft_s",
    "tpot_s",
    "total_s",
    "start_elapsed_s",
    "end_elapsed_s",
)

_DECODE_CONDITION = {
    "phase": "decode",
    "prompt_tokens": 128,
    "output_tokens": 32,
    "concurrency": 8,
}
_PREFILL_CONDITION = {
    "phase": "prefill",
    "prompt_tokens": 4096,
    "output_tokens": 1,
    "concurrency": 8,
}
_EXPECTED_BLOCKS: tuple[dict[str, Any], ...] = (
    {
        "block_id": "phase_training_pulse_00",
        "sequence": TRAINING_SEQUENCE,
        "split": "training",
        "role": "phase_training",
        "requested_power_limit_w": 46.0,
        "duration_s": 75.0,
        "condition": _DECODE_CONDITION,
    },
    {
        "block_id": "phase_training_pulse_01",
        "sequence": TRAINING_SEQUENCE,
        "split": "training",
        "role": "phase_training",
        "requested_power_limit_w": 61.0,
        "duration_s": 45.0,
        "condition": _PREFILL_CONDITION,
    },
    {
        "block_id": "phase_training_pulse_02",
        "sequence": TRAINING_SEQUENCE,
        "split": "training",
        "role": "phase_training",
        "requested_power_limit_w": 46.0,
        "duration_s": 75.0,
        "condition": _PREFILL_CONDITION,
    },
    {
        "block_id": "phase_training_pulse_03",
        "sequence": TRAINING_SEQUENCE,
        "split": "training",
        "role": "phase_training",
        "requested_power_limit_w": 61.0,
        "duration_s": 45.0,
        "condition": _DECODE_CONDITION,
    },
    {
        "block_id": "phase_validation_pulse_00",
        "sequence": VALIDATION_SEQUENCE,
        "split": "validation",
        "role": "phase_validation",
        "requested_power_limit_w": 55.0,
        "duration_s": 60.0,
        "condition": _DECODE_CONDITION,
    },
    {
        "block_id": "phase_validation_pulse_01",
        "sequence": VALIDATION_SEQUENCE,
        "split": "validation",
        "role": "phase_validation",
        "requested_power_limit_w": 55.0,
        "duration_s": 60.0,
        "condition": _PREFILL_CONDITION,
    },
)
_EXPECTED_BY_ID = {block["block_id"]: block for block in _EXPECTED_BLOCKS}

# The gain is optimized as log(1 + beta), so effective power stays positive.
_MINIMUM_PREFILL_POWER_GAIN = 0.25
_MAXIMUM_PREFILL_POWER_GAIN = 3.0
_AUTHORIZED_CLOUD = {
    "project": "potent-arcade-491015-g7",
    "zone": "us-central1-a",
    "provisioning_model": "STANDARD",
    "machine_type": "g2-standard-8",
}


@dataclass(frozen=True)
class PhaseThermalBundle:
    """Validated phase-confirmation data, including raw and one-second pulses."""

    manifest_path: Path
    telemetry_path: Path
    requests_path: Path
    telemetry_sha256: str
    requests_sha256: str
    manifest: Mapping[str, Any]
    pulses: Mapping[str, SequenceData]
    raw_pulse_rows: Mapping[str, tuple[Mapping[str, Any], ...]]
    request_rows: tuple[Mapping[str, Any], ...]
    pulse_ids_by_sequence: Mapping[str, tuple[str, ...]]


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
            f"Row {row_number} has invalid {column!r}."
        ) from error
    _require(math.isfinite(value), f"Row {row_number} has non-finite {column!r}.")
    return value


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-9)
    except (TypeError, ValueError):
        return False


def _validate_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, ScheduledPulse], dict[str, tuple[str, ...]]]:
    """Fail closed unless the acquisition matches the pre-registered schedule."""

    _require(
        manifest.get("schema_version") == SOURCE_SCHEMA_VERSION,
        "Unsupported thermal phase schema.",
    )
    _require(
        manifest.get("mode") == "thermal-phase-identification",
        "Manifest is not a thermal-phase-identification run.",
    )
    _require(
        manifest.get("protocol") == "cold-start-phase-pairs-v1",
        "Manifest is not the pre-registered cold-start phase-pair protocol.",
    )
    _require(manifest.get("status") == "complete", "Phase acquisition is not complete.")
    _require(
        manifest.get("telemetry_columns") == list(REQUIRED_TELEMETRY_COLUMNS),
        "Manifest telemetry columns do not match the required schema.",
    )
    _require(
        manifest.get("request_columns") == list(REQUIRED_REQUEST_COLUMNS),
        "Manifest request columns do not match the required schema.",
    )
    cloud = manifest.get("cloud")
    _require(isinstance(cloud, Mapping), "Phase manifest has no cloud provenance.")
    _require(
        all(cloud.get(field) == value for field, value in _AUTHORIZED_CLOUD.items()),
        "Phase acquisition cloud provenance differs from the authorized run.",
    )
    gpu = manifest.get("gpu")
    _require(
        isinstance(gpu, Mapping) and gpu.get("name") == "NVIDIA L4",
        "Phase acquisition did not use an NVIDIA L4.",
    )
    _require(
        _same_number(manifest.get("requested_graphics_clock_mhz"), 2040.0)
        and float(manifest.get("selected_memory_clock_mhz", 0.0)) > 0.0
        and float(manifest.get("cooldown_graphics_clock_mhz", 0.0)) > 0.0,
        "Phase acquisition clock controls differ from the pre-registration.",
    )
    safety = manifest.get("safety_protocol")
    _require(isinstance(safety, Mapping), "Phase manifest has no safety protocol.")
    _require(
        safety.get("independent_of_request_loop") is True
        and _same_number(safety.get("safe_down_temperature_c"), 77.0)
        and _same_number(safety.get("abort_temperature_c"), 79.0)
        and _same_number(safety.get("safe_power_limit_w"), 40.0),
        "Phase acquisition safety limits differ from the pre-registration.",
    )
    fit_protocol = manifest.get("fit_protocol")
    _require(isinstance(fit_protocol, Mapping), "Phase manifest has no fit protocol.")
    _require(
        fit_protocol.get("acquisition_only") is True
        and fit_protocol.get("training_sequences") == [TRAINING_SEQUENCE]
        and fit_protocol.get("untouched_validation_sequence") == VALIDATION_SEQUENCE
        and fit_protocol.get("validation_evaluation_passes") == 1
        and fit_protocol.get("fit_input") == "measured power rather than requested cap"
        and fit_protocol.get("prespecified_phase_model")
        == "P_eff = P * (1 + beta * I_prefill)",
        "Phase fit protocol differs from the pre-registration.",
    )

    schedule = manifest.get("schedule")
    _require(isinstance(schedule, list), "Phase manifest has no schedule.")
    _require(
        [item.get("sequence") for item in schedule] == list(SEQUENCE_ORDER),
        "Phase sequence order changed from the pre-registration.",
    )
    expected_by_sequence = {
        sequence: [block for block in _EXPECTED_BLOCKS if block["sequence"] == sequence]
        for sequence in SEQUENCE_ORDER
    }
    scheduled_pulses: dict[str, ScheduledPulse] = {}
    pulse_ids_by_sequence: dict[str, tuple[str, ...]] = {}
    observed_ids: list[str] = []
    for sequence_item, sequence in zip(schedule, SEQUENCE_ORDER, strict=True):
        expected_split = "training" if sequence == TRAINING_SEQUENCE else "validation"
        _require(
            sequence_item.get("split") == expected_split,
            f"Sequence {sequence} has the wrong split.",
        )
        _require(
            sequence_item.get("requires_cooldown_before_every_pulse") is True,
            f"Sequence {sequence} does not require cooldown before every pulse.",
        )
        blocks = sequence_item.get("blocks")
        expected = expected_by_sequence[sequence]
        _require(
            isinstance(blocks, list) and len(blocks) == len(expected),
            f"Sequence {sequence} changed pulse count.",
        )
        sequence_ids: list[str] = []
        for observed, fixed in zip(blocks, expected, strict=True):
            block_id = fixed["block_id"]
            _require(observed.get("block_id") == block_id, f"Expected block {block_id}.")
            for field in ("sequence", "split", "role"):
                _require(
                    observed.get(field) == fixed[field],
                    f"Block {block_id} changed {field}.",
                )
            for field in ("requested_power_limit_w", "duration_s"):
                _require(
                    _same_number(observed.get(field), fixed[field]),
                    f"Block {block_id} changed {field}.",
                )
            condition = observed.get("condition")
            _require(isinstance(condition, Mapping), f"Block {block_id} has no condition.")
            for field, value in fixed["condition"].items():
                _require(
                    condition.get(field) == value,
                    f"Block {block_id} changed condition {field}.",
                )
            sequence_ids.append(block_id)
            observed_ids.append(block_id)
            scheduled_pulses[block_id] = ScheduledPulse(
                block_id=block_id,
                repeat=sequence,
                split=expected_split,
                role=str(fixed["role"]),
                requested_power_limit_w=float(fixed["requested_power_limit_w"]),
                duration_s=float(fixed["duration_s"]),
                workload_phase=str(fixed["condition"]["phase"]),
            )
        pulse_ids_by_sequence[sequence] = tuple(sequence_ids)

    _require(
        manifest.get("completed_block_ids") == observed_ids,
        "Completed blocks do not exactly match the pre-registered schedule.",
    )
    cooldown_protocol = manifest.get("cooldown_protocol")
    _require(
        isinstance(cooldown_protocol, Mapping)
        and cooldown_protocol.get("before_every_pulse") is True
        and _same_number(cooldown_protocol.get("power_limit_w"), 40.0)
        and cooldown_protocol.get("workload") == "idle"
        and _same_number(cooldown_protocol.get("target_temperature_c"), 52.0)
        and _same_number(cooldown_protocol.get("stability_band_c"), 1.0)
        and _same_number(cooldown_protocol.get("stability_window_s"), 60.0)
        and _same_number(cooldown_protocol.get("timeout_s"), 600.0),
        "Manifest cooldown protocol differs from the pre-registration.",
    )
    cooldown_events = manifest.get("cooldown_events")
    _require(isinstance(cooldown_events, list), "Manifest has no cooldown events.")
    _require(
        [event.get("before_block_id") for event in cooldown_events] == observed_ids,
        "Cooldown events do not exactly match the phase pulses.",
    )
    _require(
        all(event.get("status") == "complete" for event in cooldown_events),
        "At least one phase-pulse cooldown is incomplete.",
    )
    for event in cooldown_events:
        _require(
            _same_number(event.get("target_temperature_c"), 52.0)
            and _same_number(event.get("stability_band_c"), 1.0)
            and _same_number(event.get("stability_window_s"), 60.0)
            and float(event.get("window_max_temperature_c", math.inf)) <= 52.0
            and (
                float(event.get("window_max_temperature_c", math.inf))
                - float(event.get("window_min_temperature_c", -math.inf))
                <= 1.0
            )
            and float(event.get("final_temperature_c", math.inf)) <= 52.0,
            f"Cooldown before {event.get('before_block_id')} lacks stable cold-start evidence.",
        )

    checkpoints = manifest.get("block_checkpoints")
    _require(
        isinstance(checkpoints, list) and len(checkpoints) == len(observed_ids),
        "Phase manifest does not contain one checkpoint per pulse.",
    )
    abort_temperature = float(safety["abort_temperature_c"])
    for block_id, checkpoint in zip(observed_ids, checkpoints, strict=True):
        scheduled = scheduled_pulses[block_id]
        _require(
            checkpoint.get("status") == "complete"
            and checkpoint.get("block", {}).get("block_id") == block_id
            and int(checkpoint.get("block_telemetry_rows", 0)) > 0,
            f"Checkpoint for {block_id} is incomplete or inconsistent.",
        )
        _require(
            abs(float(checkpoint.get("actual_duration_s", math.inf)) - scheduled.duration_s)
            <= 1.0,
            f"Checkpoint duration for {block_id} differs from the schedule.",
        )
        _require(
            float(checkpoint.get("maximum_temperature_c", abort_temperature))
            < abort_temperature,
            f"Block {block_id} reached the abort temperature.",
        )
    return scheduled_pulses, pulse_ids_by_sequence


def _parse_telemetry(path: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require(
            reader.fieldnames == list(REQUIRED_TELEMETRY_COLUMNS),
            "Thermal phase telemetry header does not match the required schema.",
        )
        for row_number, row in enumerate(reader, start=2):
            parsed = dict(row)
            for column in (
                "elapsed_s",
                "graphics_clock_mhz",
                "memory_clock_mhz",
                "power_w",
                "temperature_c",
                "utilization_percent",
                "memory_used_mib",
            ):
                parsed[column] = _finite_number(row, column, row_number)
            elapsed = parsed["elapsed_s"]
            power = parsed["power_w"]
            temperature = parsed["temperature_c"]
            _require(power > 0.0, f"Telemetry row {row_number} has non-positive power.")
            _require(
                parsed["graphics_clock_mhz"] > 0.0
                and parsed["memory_clock_mhz"] > 0.0
                and 0.0 <= parsed["utilization_percent"] <= 100.0
                and parsed["memory_used_mib"] >= 0.0,
                f"Telemetry row {row_number} has invalid realized controls.",
            )
            _require(
                math.isclose(temperature, round(temperature), abs_tol=1.0e-9),
                f"Telemetry row {row_number} is not an integer temperature reading.",
            )
            rows.append(parsed)
    _require(rows, "Thermal phase telemetry is empty.")
    _require(
        manifest.get("telemetry_row_count") == len(rows),
        "Manifest telemetry row count does not match the CSV.",
    )
    elapsed = np.asarray([row["elapsed_s"] for row in rows], dtype=float)
    _require(np.all(np.diff(elapsed) > 0.0), "Telemetry time is not strictly increasing.")
    return rows


def _parse_requests(
    path: Path,
    manifest: Mapping[str, Any],
    scheduled_pulses: Mapping[str, ScheduledPulse],
) -> list[dict[str, Any]]:
    numeric_columns = REQUIRED_REQUEST_COLUMNS[4:]
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require(
            reader.fieldnames == list(REQUIRED_REQUEST_COLUMNS),
            "Thermal phase request header does not match the required schema.",
        )
        for row_number, row in enumerate(reader, start=2):
            parsed: dict[str, Any] = dict(row)
            for column in numeric_columns:
                parsed[column] = _finite_number(row, column, row_number)
            _require(
                parsed["end_elapsed_s"] >= parsed["start_elapsed_s"],
                f"Request row {row_number} ends before it starts.",
            )
            _require(
                parsed["start_elapsed_s"] >= 0.0
                and parsed["ttft_s"] >= 0.0
                and parsed["tpot_s"] >= 0.0
                and parsed["total_s"] > 0.0
                and math.isclose(
                    parsed["total_s"],
                    parsed["end_elapsed_s"] - parsed["start_elapsed_s"],
                    rel_tol=0.0,
                    abs_tol=1.0e-5,
                ),
                f"Request row {row_number} has inconsistent timing.",
            )
            for column in ("batch_index", "request_index"):
                _require(
                    parsed[column] >= 0.0
                    and math.isclose(parsed[column], round(parsed[column]), abs_tol=1.0e-9),
                    f"Request row {row_number} has invalid {column}.",
                )
            block_id = str(parsed["block_id"])
            _require(block_id in scheduled_pulses, f"Request row {row_number} has unknown block.")
            pulse = scheduled_pulses[block_id]
            fixed = _EXPECTED_BY_ID[block_id]
            _require(
                parsed["split"] == pulse.split
                and parsed["sequence"] == pulse.repeat
                and parsed["workload_phase"] == pulse.workload_phase
                and _same_number(
                    parsed["requested_power_limit_w"], pulse.requested_power_limit_w
                ),
                f"Request row {row_number} metadata differs from block {block_id}.",
            )
            condition = fixed["condition"]
            _require(
                _same_number(parsed["prompt_tokens"], condition["prompt_tokens"])
                and _same_number(parsed["output_tokens"], condition["output_tokens"])
                and _same_number(parsed["concurrency"], condition["concurrency"])
                and _same_number(
                    parsed["prompt_tokens_observed"], condition["prompt_tokens"]
                )
                and _same_number(
                    parsed["completion_tokens"], condition["output_tokens"]
                ),
                f"Request row {row_number} changed the pre-registered workload.",
            )
            rows.append(parsed)
    _require(rows, "Thermal phase request log is empty.")
    _require(
        manifest.get("request_row_count") == len(rows),
        "Manifest request row count does not match the CSV.",
    )
    observed = {str(row["block_id"]) for row in rows}
    _require(
        observed == set(scheduled_pulses),
        "At least one phase pulse has no completed request.",
    )
    identities = [
        (str(row["block_id"]), int(row["batch_index"]), int(row["request_index"]))
        for row in rows
    ]
    _require(
        len(identities) == len(set(identities)),
        "Thermal phase request log contains duplicate request identities.",
    )
    for block_id in scheduled_pulses:
        batches: dict[int, set[int]] = {}
        for row in rows:
            if row["block_id"] == block_id:
                batches.setdefault(int(row["batch_index"]), set()).add(
                    int(row["request_index"])
                )
        _require(
            sorted(batches) == list(range(len(batches)))
            and all(indices == set(range(8)) for indices in batches.values()),
            f"Request batches for {block_id} are incomplete or out of sequence.",
        )
    return rows


def _interval_coverage(
    intervals: Sequence[tuple[float, float]], left: float, right: float
) -> float:
    clipped = sorted(
        (max(start, left), min(end, right))
        for start, end in intervals
        if end > left and start < right
    )
    if not clipped:
        return 0.0
    covered = 0.0
    start, end = clipped[0]
    for next_start, next_end in clipped[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            covered += end - start
            start, end = next_start, next_end
    return covered + end - start


def load_phase_thermal_bundle(
    manifest_path: str | Path,
    *,
    config: FitConfig = FitConfig(),
) -> PhaseThermalBundle:
    """Validate and one-second-bin a completed phase-confirmation bundle."""

    _require(
        math.isclose(config.bin_width_s, 1.0, rel_tol=0.0, abs_tol=1.0e-12),
        "The pre-registered phase fit requires one-second bins.",
    )
    path = Path(manifest_path).resolve()
    _require(path.name == MANIFEST_FILENAME, f"Expected {MANIFEST_FILENAME}.")
    _require(path.is_file(), f"Phase manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ThermalFitError(f"Could not read phase manifest: {error}") from error
    _require(isinstance(manifest, Mapping), "Phase manifest root must be an object.")
    scheduled_pulses, pulse_ids_by_sequence = _validate_manifest(manifest)

    telemetry_path = path.parent / TELEMETRY_FILENAME
    requests_path = path.parent / REQUESTS_FILENAME
    _require(telemetry_path.is_file(), f"Missing {TELEMETRY_FILENAME}.")
    _require(requests_path.is_file(), f"Missing {REQUESTS_FILENAME}.")
    checksums = manifest.get("sha256")
    _require(isinstance(checksums, Mapping), "Phase manifest has no checksums.")
    telemetry_hash = _sha256(telemetry_path)
    requests_hash = _sha256(requests_path)
    _require(
        checksums.get(TELEMETRY_FILENAME) == telemetry_hash,
        "Thermal phase telemetry checksum mismatch.",
    )
    _require(
        checksums.get(REQUESTS_FILENAME) == requests_hash,
        "Thermal phase request checksum mismatch.",
    )
    for filename, expected_hash in checksums.items():
        _require(
            isinstance(filename, str)
            and Path(filename).name == filename
            and isinstance(expected_hash, str)
            and len(expected_hash) == 64,
            f"Unsafe or invalid checksum entry {filename!r}.",
        )
        artifact = path.parent / filename
        _require(artifact.is_file(), f"Checksummed artifact is missing: {filename}.")
        _require(_sha256(artifact) == expected_hash, f"Checksum mismatch for {filename}.")

    rows = _parse_telemetry(telemetry_path, manifest)
    request_rows = _parse_requests(requests_path, manifest, scheduled_pulses)
    telemetry_period = manifest.get("telemetry_period_s")
    _require(
        isinstance(telemetry_period, (int, float))
        and math.isfinite(telemetry_period)
        and telemetry_period > 0.0,
        "Manifest telemetry period is invalid.",
    )
    abort_temperature = float(manifest["safety_protocol"]["abort_temperature_c"])
    _require(
        max(float(row["temperature_c"]) for row in rows) < abort_temperature,
        "Thermal phase telemetry reached the abort temperature.",
    )
    modeled_rows = [row for row in rows if row["split"] in ("training", "validation")]
    _require(modeled_rows, "Phase bundle has no modeled telemetry.")
    _require(
        all(str(row["block_id"]) in scheduled_pulses for row in modeled_rows),
        "Modeled telemetry contains an unscheduled block.",
    )
    observed_order: list[str] = []
    for row in modeled_rows:
        block_id = str(row["block_id"])
        if not observed_order or observed_order[-1] != block_id:
            observed_order.append(block_id)
    expected_order = [block["block_id"] for block in _EXPECTED_BLOCKS]
    _require(
        observed_order == expected_order,
        "Modeled telemetry pulse order differs from the pre-registration.",
    )

    pulses: dict[str, SequenceData] = {}
    raw_pulse_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
    checkpoints = {
        checkpoint["block"]["block_id"]: checkpoint
        for checkpoint in manifest["block_checkpoints"]
    }
    requested_clock = float(manifest["requested_graphics_clock_mhz"])
    for block_id in expected_order:
        pulse = scheduled_pulses[block_id]
        checkpoint = checkpoints[block_id]
        checkpoint_start = float(checkpoint["started_elapsed_s"])
        checkpoint_end = float(checkpoint["ended_elapsed_s"])
        selected = [
            row
            for row in modeled_rows
            if row["block_id"] == block_id
            and checkpoint_start <= float(row["elapsed_s"]) <= checkpoint_end
        ]
        _require(selected, f"No telemetry for block {block_id}.")
        _require(
            all(
                row["split"] == pulse.split
                and row["sequence"] == pulse.repeat
                and row["block_role"] == pulse.role
                and row["workload_phase"] == pulse.workload_phase
                and _same_number(
                    row["requested_power_limit_w"], pulse.requested_power_limit_w
                )
                and _same_number(row["requested_clock_mhz"], requested_clock)
                for row in selected
            ),
            f"Telemetry metadata differs from block {block_id}.",
        )
        times = np.asarray([row["elapsed_s"] for row in selected], dtype=float)
        _require(np.all(np.diff(times) > 0.0), f"Block {block_id} time is not increasing.")
        maximum_gap = float(np.max(np.diff(times)))
        allowed_gap = max(0.5, 4.0 * float(telemetry_period))
        _require(
            maximum_gap <= allowed_gap,
            f"Block {block_id} telemetry has a {maximum_gap:.3f} s gap.",
        )
        span = float(times[-1] - times[0])
        tolerance = max(1.0, 2.0 * float(telemetry_period))
        _require(
            abs(span - pulse.duration_s) <= tolerance,
            f"Block {block_id} telemetry duration changed.",
        )
        _require(
            int(checkpoint["block_telemetry_rows"]) == len(selected)
            and float(checkpoint["started_elapsed_s"]) <= times[0] + tolerance
            and float(checkpoint["ended_elapsed_s"]) >= times[-1] - tolerance
            and math.isclose(
                float(checkpoint["maximum_temperature_c"]),
                max(float(row["temperature_c"]) for row in selected),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ),
            f"Checkpoint evidence differs from telemetry for {block_id}.",
        )
        utilization = np.asarray(
            [float(row["utilization_percent"]) for row in selected], dtype=float
        )
        _require(
            float(np.mean(utilization)) >= 80.0,
            f"Block {block_id} lacks sustained GPU utilization.",
        )
        block_requests = [row for row in request_rows if row["block_id"] == block_id]
        _require(
            all(_same_number(row["requested_clock_mhz"], requested_clock) for row in block_requests),
            f"Request clocks for {block_id} differ from the pre-registration.",
        )
        covered = _interval_coverage(
            [
                (float(row["start_elapsed_s"]), float(row["end_elapsed_s"]))
                for row in block_requests
            ],
            float(times[0]),
            float(times[-1]),
        )
        _require(
            covered / span >= 0.95,
            f"Completed requests cover too little of block {block_id}.",
        )
        pulses[block_id] = _bin_pulse(
            pulse,
            selected,
            telemetry_period_s=float(telemetry_period),
        )
        raw_pulse_rows[block_id] = tuple(selected)

    validation_pulses = [
        pulses[block_id] for block_id in pulse_ids_by_sequence[VALIDATION_SEQUENCE]
    ]
    _require(
        len({len(pulse.time_s) for pulse in validation_pulses}) == 1,
        "The validation pair does not share an identical one-second horizon.",
    )

    events_by_id = {
        str(event["before_block_id"]): event for event in manifest["cooldown_events"]
    }
    for block_id, pulse in scheduled_pulses.items():
        cooldown_id = f"cooldown_before_{block_id}"
        cooldown_rows = [
            row
            for row in rows
            if row["split"] == "conditioning"
            and row["sequence"] == pulse.repeat
            and row["block_id"] == cooldown_id
            and row["block_role"] == "cooldown"
        ]
        _require(cooldown_rows, f"Block {block_id} has no labeled cooldown telemetry.")
        event = events_by_id[block_id]
        started = float(event["started_elapsed_s"])
        completed = float(event["completed_elapsed_s"])
        _require(
            started < completed <= float(raw_pulse_rows[block_id][0]["elapsed_s"]),
            f"Cooldown timing is invalid before {block_id}.",
        )
        stability_window = float(event["stability_window_s"])
        window = [
            row
            for row in cooldown_rows
            if completed - stability_window <= float(row["elapsed_s"]) <= completed
        ]
        _require(window, f"Cooldown before {block_id} has no stability-window telemetry.")
        window_span = float(window[-1]["elapsed_s"] - window[0]["elapsed_s"])
        temperatures = [float(row["temperature_c"]) for row in window]
        cooldown_clock = float(manifest["cooldown_graphics_clock_mhz"])
        _require(
            window_span >= stability_window - max(0.2, 2.0 * float(telemetry_period))
            and max(temperatures) <= float(event["target_temperature_c"])
            and max(temperatures) - min(temperatures)
            <= float(event["stability_band_c"])
            and all(
                _same_number(row["requested_power_limit_w"], 40.0)
                and _same_number(row["requested_clock_mhz"], cooldown_clock)
                and row["workload_phase"] == "idle"
                for row in window
            )
            and math.isclose(
                min(temperatures),
                float(event["window_min_temperature_c"]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            and math.isclose(
                max(temperatures),
                float(event["window_max_temperature_c"]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            and math.isclose(
                temperatures[-1],
                float(event["final_temperature_c"]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ),
            f"Cooldown before {block_id} does not contain the required cold stable window.",
        )
    return PhaseThermalBundle(
        manifest_path=path,
        telemetry_path=telemetry_path,
        requests_path=requests_path,
        telemetry_sha256=telemetry_hash,
        requests_sha256=requests_hash,
        manifest=manifest,
        pulses=pulses,
        raw_pulse_rows=raw_pulse_rows,
        request_rows=tuple(request_rows),
        pulse_ids_by_sequence=pulse_ids_by_sequence,
    )


def _bin_pulse(
    pulse: ScheduledPulse,
    rows: Sequence[Mapping[str, Any]],
    *,
    telemetry_period_s: float,
) -> SequenceData:
    """Apply the pre-registered one-second level-based reduction."""

    times = np.asarray([row["elapsed_s"] for row in rows], dtype=float)
    temperatures = np.asarray([row["temperature_c"] for row in rows], dtype=float)
    powers = np.asarray([row["power_w"] for row in rows], dtype=float)
    complete_intervals = int(math.floor(times[-1] - times[0]))
    _require(complete_intervals >= 2, f"Block {pulse.block_id} has too little telemetry.")
    grid = times[0] + np.arange(complete_intervals + 1, dtype=float)
    tolerance = max(0.51, 3.0 * telemetry_period_s)
    binned_temperature: list[float] = []
    for point in grid:
        insertion = int(np.searchsorted(times, point))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(times)]
        nearest = min(candidates, key=lambda index: abs(times[index] - point))
        _require(
            abs(times[nearest] - point) <= tolerance,
            f"Block {pulse.block_id} has no temperature sample near a bin boundary.",
        )
        binned_temperature.append(float(temperatures[nearest]))
    binned_power = np.asarray(
        [
            _time_weighted_power(times, powers, grid[index], grid[index + 1])
            for index in range(complete_intervals)
        ],
        dtype=float,
    )
    return SequenceData(
        name=pulse.block_id,
        split=pulse.split,
        time_s=grid - grid[0],
        temperature_c=np.asarray(binned_temperature, dtype=float),
        measured_power_w=binned_power,
        raw_row_count=len(rows),
        repeat=pulse.repeat,
        role=pulse.role,
        workload_phase=pulse.workload_phase,
        requested_power_limit_w=pulse.requested_power_limit_w,
        scheduled_duration_s=pulse.duration_s,
    )


def _phase_gain_parameters(transformed: np.ndarray) -> dict[str, float]:
    base = _one_state_parameters(np.asarray(transformed[:3], dtype=float))
    gain = math.exp(float(transformed[3]))
    return {
        **base,
        "beta": gain - 1.0,
        "prefill_power_gain": gain,
    }


def simulate_one_state_phase_gain(
    parameters: Mapping[str, float], sequence: SequenceData
) -> np.ndarray:
    """Free-run the one-state RC model with a multiplicative prefill gain."""

    phase = sequence.workload_phase
    if phase not in ("decode", "prefill"):
        raise ValueError(f"Unsupported workload phase {phase!r}.")
    ambient = float(parameters["ambient_temperature_c"])
    resistance = float(parameters["thermal_resistance_c_per_w"])
    time_constant = float(parameters["thermal_time_constant_s"])
    beta = float(parameters["beta"])
    gain = 1.0 + beta if phase == "prefill" else 1.0
    if ambient <= 0.0 or resistance <= 0.0 or time_constant <= 0.0 or gain <= 0.0:
        raise ValueError("Phase-gain RC parameters must imply positive effective power.")
    predicted = np.empty_like(sequence.temperature_c)
    predicted[0] = sequence.temperature_c[0]
    for index, measured_power in enumerate(sequence.measured_power_w):
        dt = float(sequence.time_s[index + 1] - sequence.time_s[index])
        decay = math.exp(-dt / time_constant)
        equilibrium = ambient + resistance * gain * float(measured_power)
        predicted[index + 1] = (
            decay * predicted[index] + (1.0 - decay) * equilibrium
        )
    return predicted


def fit_one_state_phase_gain(
    sequences: Sequence[SequenceData],
    *,
    config: FitConfig = FitConfig(),
) -> ModelFit:
    """Fit the four-parameter phase-gain RC model to training pulses only."""

    _require(bool(sequences), "Phase-gain fitting received no training pulses.")
    _require(
        all(sequence.split == "training" for sequence in sequences),
        "Phase-gain fitting accepts training pulses only.",
    )
    _require(
        {sequence.workload_phase for sequence in sequences} == {"decode", "prefill"},
        "Phase-gain fitting needs both decode and prefill training pulses.",
    )
    for phase in ("decode", "prefill"):
        powers = {
            int(round(float(sequence.requested_power_limit_w)))
            for sequence in sequences
            if sequence.workload_phase == phase
        }
        _require(
            powers == {46, 61},
            f"Phase-gain training lost the paired 46/61 W {phase} excitation.",
        )
    measured_power = np.concatenate([sequence.measured_power_w for sequence in sequences])
    temperature = np.concatenate([sequence.temperature_c for sequence in sequences])
    power_span = float(np.ptp(measured_power))
    _require(power_span >= 1.0, "Phase training measured power lacks excitation.")

    lower = np.asarray(
        [
            0.01,
            math.log(1.0e-4),
            math.log(0.05),
            math.log(_MINIMUM_PREFILL_POWER_GAIN),
        ]
    )
    upper = np.asarray(
        [
            100.0,
            math.log(10.0),
            math.log(1.0e5),
            math.log(_MAXIMUM_PREFILL_POWER_GAIN),
        ]
    )
    resistance_guess = float(
        np.clip(np.ptp(temperature) / max(power_span, 1.0), 0.02, 1.0)
    )
    ambient_guess = float(
        np.clip(
            np.median(temperature) - resistance_guess * np.median(measured_power),
            5.0,
            90.0,
        )
    )
    cool_ambient = max(5.0, float(np.min(temperature)) - 15.0)
    physical_starts = (
        (ambient_guess, resistance_guess, 20.0, 1.0),
        (ambient_guess, resistance_guess, 50.0, 1.2),
        (ambient_guess, resistance_guess, 10.0, 0.8),
        (cool_ambient, resistance_guess, 200.0, 1.0),
        (ambient_guess, max(0.01, resistance_guess / 2.0), 20.0, 1.3),
        (cool_ambient, max(0.01, resistance_guess / 2.0), 100.0, 0.7),
    )
    starts = [
        np.asarray([ambient, math.log(resistance), math.log(tau), math.log(gain)])
        for ambient, resistance, tau, gain in physical_starts[: config.multistart_count]
    ]
    rng = np.random.default_rng(config.random_seed + 404 + len(sequences))
    while len(starts) < config.multistart_count:
        starts.append(
            np.asarray(
                [
                    rng.uniform(5.0, 85.0),
                    rng.uniform(math.log(0.005), math.log(2.0)),
                    rng.uniform(math.log(0.2), math.log(2_000.0)),
                    rng.uniform(
                        math.log(_MINIMUM_PREFILL_POWER_GAIN),
                        math.log(_MAXIMUM_PREFILL_POWER_GAIN),
                    ),
                ]
            )
        )

    def residual(transformed: np.ndarray) -> np.ndarray:
        parameters = _phase_gain_parameters(transformed)
        return np.concatenate(
            [
                simulate_one_state_phase_gain(parameters, sequence)[1:]
                - sequence.temperature_c[1:]
                for sequence in sequences
            ]
        )

    best, candidates = _run_multistart(
        residual,
        starts,
        lower,
        upper,
        max_nfev=config.max_nfev,
    )
    parameters = _phase_gain_parameters(best.transformed_parameters)
    diagnostics = _candidate_diagnostics(candidates, best, lower, upper)
    pole = -1.0 / parameters["thermal_time_constant_s"]
    diagnostics.update(
        {
            "continuous_poles_per_s": [pole],
            "discrete_poles_at_one_second": [math.exp(pole)],
            "asymptotically_stable": pole < 0.0,
            "positive_thermal_parameters": True,
            "positive_effective_power_gain": parameters["prefill_power_gain"] > 0.0,
            "measured_power_span_w": power_span,
            "prefill_power_gain_bounds": [
                _MINIMUM_PREFILL_POWER_GAIN,
                _MAXIMUM_PREFILL_POWER_GAIN,
            ],
            "training_phase_counts": {
                phase: sum(sequence.workload_phase == phase for sequence in sequences)
                for phase in ("decode", "prefill")
            },
        }
    )
    return ModelFit(
        model="one_state_phase_gain_rc",
        parameters=parameters,
        transformed_parameters=best.transformed_parameters,
        objective_sum_squared_c=best.cost,
        diagnostics=diagnostics,
    )


def _fit_to_json(fit: ModelFit) -> dict[str, Any]:
    return {
        "parameters": dict(fit.parameters),
        "objective_sum_squared_c": fit.objective_sum_squared_c,
        "diagnostics": dict(fit.diagnostics),
    }


def _evaluate_pulses(
    fit: ModelFit,
    pulses: Sequence[SequenceData],
    simulator: Callable[[Mapping[str, float], SequenceData], np.ndarray],
    *,
    config: FitConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    predictions: dict[str, np.ndarray] = {}
    per_pulse: dict[str, Any] = {}
    for pulse in pulses:
        predicted = simulator(fit.parameters, pulse)
        predictions[pulse.name] = predicted
        per_pulse[pulse.name] = {
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


def _pair_by_phase(
    pulses: Sequence[SequenceData],
) -> dict[str, SequenceData]:
    by_phase = {pulse.workload_phase: pulse for pulse in pulses}
    _require(
        set(by_phase) == {"decode", "prefill"} and len(pulses) == 2,
        "Validation is not exactly one decode/prefill pair.",
    )
    return by_phase


def _prediction_pair_summary(
    validation: Sequence[SequenceData],
    evaluation: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    by_phase = _pair_by_phase(validation)

    def summary(pulse: SequenceData) -> dict[str, Any]:
        predicted = predictions[pulse.name]
        return {
            "block_id": pulse.name,
            "observed_peak_temperature_c": float(np.max(pulse.temperature_c)),
            "predicted_peak_temperature_c": float(np.max(predicted)),
            "observed_peak_rise_c": float(
                np.max(pulse.temperature_c) - pulse.temperature_c[0]
            ),
            "predicted_peak_rise_c": float(np.max(predicted) - predicted[0]),
            "metrics": evaluation["per_pulse"][pulse.name]["metrics"],
        }

    decode = summary(by_phase["decode"])
    prefill = summary(by_phase["prefill"])
    return {
        "matching_rule": "55 W requested cap, 60 s duration, cold start",
        "decode": decode,
        "prefill": prefill,
        "observed_prefill_minus_decode_peak_c": (
            prefill["observed_peak_temperature_c"]
            - decode["observed_peak_temperature_c"]
        ),
        "predicted_prefill_minus_decode_peak_c": (
            prefill["predicted_peak_temperature_c"]
            - decode["predicted_peak_temperature_c"]
        ),
        "observed_prefill_minus_decode_peak_rise_c": (
            prefill["observed_peak_rise_c"] - decode["observed_peak_rise_c"]
        ),
        "predicted_prefill_minus_decode_peak_rise_c": (
            prefill["predicted_peak_rise_c"] - decode["predicted_peak_rise_c"]
        ),
    }


def _raw_pulse_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    times = np.asarray([row["elapsed_s"] for row in rows], dtype=float)
    powers = np.asarray([row["power_w"] for row in rows], dtype=float)
    temperatures = np.asarray([row["temperature_c"] for row in rows], dtype=float)
    graphics_clocks = np.asarray(
        [row["graphics_clock_mhz"] for row in rows], dtype=float
    )
    memory_clocks = np.asarray([row["memory_clock_mhz"] for row in rows], dtype=float)
    utilization = np.asarray(
        [row["utilization_percent"] for row in rows], dtype=float
    )
    duration = float(times[-1] - times[0])
    _require(duration > 0.0, "Raw pulse duration must be positive.")
    energy = float(np.trapezoid(powers, times))
    peak = float(np.max(temperatures))
    peak_indices = np.flatnonzero(temperatures == peak)
    return {
        "raw_rows": len(rows),
        "duration_s": duration,
        "time_weighted_mean_power_w": energy / duration,
        "integrated_energy_j": energy,
        "start_temperature_c": float(temperatures[0]),
        "end_temperature_c": float(temperatures[-1]),
        "peak_temperature_c": peak,
        "peak_rise_c": peak - float(temperatures[0]),
        "terminal_rise_c": float(temperatures[-1] - temperatures[0]),
        "peak_sample_count": int(len(peak_indices)),
        "first_peak_time_s": float(times[peak_indices[0]] - times[0]),
        "last_peak_time_s": float(times[peak_indices[-1]] - times[0]),
        "maximum_sampling_gap_s": float(np.max(np.diff(times))),
        "median_realized_graphics_clock_mhz": float(np.median(graphics_clocks)),
        "minimum_realized_graphics_clock_mhz": float(np.min(graphics_clocks)),
        "maximum_realized_graphics_clock_mhz": float(np.max(graphics_clocks)),
        "median_memory_clock_mhz": float(np.median(memory_clocks)),
        "mean_utilization_percent": float(np.mean(utilization)),
    }


def _binned_pulse_summary(pulse: SequenceData) -> dict[str, Any]:
    peak = float(np.max(pulse.temperature_c))
    return {
        "temperature_states": len(pulse.temperature_c),
        "duration_s": float(pulse.time_s[-1]),
        "start_temperature_c": float(pulse.temperature_c[0]),
        "end_temperature_c": float(pulse.temperature_c[-1]),
        "peak_temperature_c": peak,
        "peak_rise_c": peak - float(pulse.temperature_c[0]),
        "terminal_rise_c": float(pulse.temperature_c[-1] - pulse.temperature_c[0]),
    }


def _phase_difference(
    by_phase: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    decode = by_phase["decode"]
    prefill = by_phase["prefill"]
    return {
        "prefill_minus_decode_mean_power_w": float(
            prefill["time_weighted_mean_power_w"]
            - decode["time_weighted_mean_power_w"]
        ),
        "prefill_minus_decode_energy_j": float(
            prefill["integrated_energy_j"] - decode["integrated_energy_j"]
        ),
        "prefill_to_decode_energy_ratio": float(
            prefill["integrated_energy_j"] / decode["integrated_energy_j"]
        ),
    }


def _measured_power_energy_report(bundle: PhaseThermalBundle) -> dict[str, Any]:
    per_pulse: dict[str, dict[str, Any]] = {}
    for block_id in (block["block_id"] for block in _EXPECTED_BLOCKS):
        raw_rows = bundle.raw_pulse_rows[block_id]
        requests = [row for row in bundle.request_rows if row["block_id"] == block_id]
        left = float(raw_rows[0]["elapsed_s"])
        right = float(raw_rows[-1]["elapsed_s"])
        covered = _interval_coverage(
            [
                (float(row["start_elapsed_s"]), float(row["end_elapsed_s"]))
                for row in requests
            ],
            left,
            right,
        )
        per_pulse[block_id] = {
            "workload_phase": bundle.pulses[block_id].workload_phase,
            "requested_power_limit_w": bundle.pulses[block_id].requested_power_limit_w,
            **_raw_pulse_summary(raw_rows),
            "completed_request_rows": len(requests),
            "completed_batch_count": len({int(row["batch_index"]) for row in requests}),
            "request_coverage_fraction": covered / (right - left),
            "first_request_start_offset_s": min(
                float(row["start_elapsed_s"]) for row in requests
            )
            - left,
            "last_request_end_offset_s": max(
                float(row["end_elapsed_s"]) for row in requests
            )
            - right,
        }
    matched: dict[str, Any] = {}
    for split_name, pulse_ids in bundle.pulse_ids_by_sequence.items():
        grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
        for block_id in pulse_ids:
            item = per_pulse[block_id]
            cap = int(round(float(item["requested_power_limit_w"])))
            grouped.setdefault(cap, {})[str(item["workload_phase"])] = item
        matched[split_name] = {}
        for cap, by_phase in sorted(grouped.items()):
            _require(
                set(by_phase) == {"decode", "prefill"},
                f"The {cap} W phase pair is incomplete.",
            )
            matched[split_name][str(cap)] = {
                "decode_block_id": next(
                    block_id
                    for block_id in pulse_ids
                    if per_pulse[block_id] is by_phase["decode"]
                ),
                "prefill_block_id": next(
                    block_id
                    for block_id in pulse_ids
                    if per_pulse[block_id] is by_phase["prefill"]
                ),
                **_phase_difference(by_phase),
            }
    return {"per_pulse": per_pulse, "matched_phase_pairs": matched}


def _observed_validation_contrast(
    bundle: PhaseThermalBundle,
    validation: Sequence[SequenceData],
) -> dict[str, Any]:
    by_phase = _pair_by_phase(validation)
    raw = {
        phase: _raw_pulse_summary(bundle.raw_pulse_rows[pulse.name])
        for phase, pulse in by_phase.items()
    }
    aligned = {phase: _binned_pulse_summary(pulse) for phase, pulse in by_phase.items()}

    def contrast(summary: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        decode = summary["decode"]
        prefill = summary["prefill"]
        return {
            "decode": decode,
            "prefill": prefill,
            "prefill_minus_decode_peak_c": float(
                prefill["peak_temperature_c"] - decode["peak_temperature_c"]
            ),
            "prefill_minus_decode_peak_rise_c": float(
                prefill["peak_rise_c"] - decode["peak_rise_c"]
            ),
            "prefill_minus_decode_terminal_rise_c": float(
                prefill["terminal_rise_c"] - decode["terminal_rise_c"]
            ),
        }

    return {
        "raw_telemetry": contrast(raw),
        "one_second_aligned": {
            "alignment_rule": "separate grids anchored at each pulse's first telemetry row",
            **contrast(aligned),
        },
    }


def build_phase_fit_report(
    bundle: PhaseThermalBundle,
    *,
    config: FitConfig = FitConfig(),
) -> dict[str, Any]:
    """Fit training, then score the untouched validation pair once per model."""

    _require(
        math.isclose(config.bin_width_s, 1.0, rel_tol=0.0, abs_tol=1.0e-12),
        "The pre-registered phase report requires one-second bins.",
    )

    training_ids = bundle.pulse_ids_by_sequence[TRAINING_SEQUENCE]
    validation_ids = bundle.pulse_ids_by_sequence[VALIDATION_SEQUENCE]
    training = [bundle.pulses[block_id] for block_id in training_ids]
    validation = [bundle.pulses[block_id] for block_id in validation_ids]
    _require(
        len(training) == 4 and all(pulse.split == "training" for pulse in training),
        "Internal split error: phase training is not exactly four pulses.",
    )
    _require(
        len(validation) == 2
        and all(pulse.split == "validation" for pulse in validation),
        "Internal split error: validation is not exactly the final pair.",
    )

    fits = {
        "power_only_one_state_rc": fit_one_state_rc(training, config=config),
        "phase_gain_one_state_rc": fit_one_state_phase_gain(training, config=config),
    }
    simulators: dict[str, Callable[[Mapping[str, float], SequenceData], np.ndarray]] = {
        "power_only_one_state_rc": simulate_one_state_rc,
        "phase_gain_one_state_rc": simulate_one_state_phase_gain,
    }
    models: dict[str, Any] = {}
    for model_name in ("power_only_one_state_rc", "phase_gain_one_state_rc"):
        fit = fits[model_name]
        simulator = simulators[model_name]
        training_evaluation, _ = _evaluate_pulses(
            fit, training, simulator, config=config
        )
        # This is the only scoring pass over validation temperatures for this model.
        validation_evaluation, validation_predictions = _evaluate_pulses(
            fit, validation, simulator, config=config
        )
        validation_evaluation["evaluation_passes"] = 1
        validation_evaluation["matched_55w_decode_prefill"] = (
            _prediction_pair_summary(
                validation, validation_evaluation, validation_predictions
            )
        )
        models[model_name] = {
            "final_training_fit": _fit_to_json(fit),
            "training_evaluation": training_evaluation,
            "validation_evaluation": validation_evaluation,
        }

    power_training_rmse = models["power_only_one_state_rc"]["training_evaluation"][
        "aggregate"
    ]["rmse_c"]
    gain_training_rmse = models["phase_gain_one_state_rc"]["training_evaluation"][
        "aggregate"
    ]["rmse_c"]
    power_validation_rmse = models["power_only_one_state_rc"]["validation_evaluation"][
        "aggregate"
    ]["rmse_c"]
    gain_validation_rmse = models["phase_gain_one_state_rc"]["validation_evaluation"][
        "aggregate"
    ]["rmse_c"]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "method": (
            "pre-registered one-second level fits using measured power; exact-discrete "
            "free-run trajectories; no temperature finite differences"
        ),
        "source": {
            "manifest": bundle.manifest_path.name,
            "telemetry": bundle.telemetry_path.name,
            "requests": bundle.requests_path.name,
            "telemetry_sha256": bundle.telemetry_sha256,
            "requests_sha256": bundle.requests_sha256,
            "telemetry_row_count": bundle.manifest.get("telemetry_row_count"),
            "request_row_count": bundle.manifest.get("request_row_count"),
            "source_manifest_schema_version": bundle.manifest.get("schema_version"),
            "source_git_revision": bundle.manifest.get("git_revision"),
        },
        "pre_registration": {
            "training_sequence": TRAINING_SEQUENCE,
            "training_pulse_ids": list(training_ids),
            "training_pulse_count": 4,
            "untouched_validation_sequence": VALIDATION_SEQUENCE,
            "validation_pulse_ids": list(validation_ids),
            "validation_pulse_count": 2,
            "validation_used_for_fit_or_model_selection": False,
            "candidate_models": [
                "power_only_one_state_rc",
                "phase_gain_one_state_rc",
            ],
            "phase_gain_equation": "P_eff = P * (1 + beta * I_prefill)",
            "prefill_power_gain_bounds": [
                _MINIMUM_PREFILL_POWER_GAIN,
                _MAXIMUM_PREFILL_POWER_GAIN,
            ],
            "random_seed": config.random_seed,
            "multistart_count": config.multistart_count,
            "maximum_optimizer_evaluations": config.max_nfev,
        },
        "preprocessing": {
            "bin_width_s": 1.0,
            "power_statistic": "time-weighted mean of measured board power",
            "temperature_statistic": "nearest integer sensor reading at each bin boundary",
            "finite_differencing": False,
            "pulse_initialization": "each cold pulse starts at its first observed junction temperature",
            "pulse_samples": {
                pulse.name: {
                    "split": pulse.split,
                    "workload_phase": pulse.workload_phase,
                    "requested_power_limit_w": pulse.requested_power_limit_w,
                    "raw_rows": pulse.raw_row_count,
                    "binned_temperature_states": len(pulse.temperature_c),
                    "measured_power_intervals": len(pulse.measured_power_w),
                    "binned_duration_s": float(pulse.time_s[-1]),
                }
                for pulse in (*training, *validation)
            },
        },
        "measured_power_and_energy": _measured_power_energy_report(bundle),
        "observed_validation_contrast": _observed_validation_contrast(
            bundle, validation
        ),
        "models": models,
        "model_comparison": {
            "phase_gain_minus_power_only_training_rmse_c": (
                gain_training_rmse - power_training_rmse
            ),
            "phase_gain_minus_power_only_validation_rmse_c": (
                gain_validation_rmse - power_validation_rmse
            ),
            "validation_was_not_used_to_choose_or_refit_a_model": True,
        },
    }


def write_phase_fit_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Atomically write a deterministic finite JSON phase-fit report."""

    path = Path(output_path).resolve()
    _require(
        path.name not in {MANIFEST_FILENAME, TELEMETRY_FILENAME, REQUESTS_FILENAME},
        "Refusing to overwrite a phase-identification source artifact.",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
    temporary.replace(path)
    return path


def fit_phase_thermal_bundle(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    config: FitConfig = FitConfig(),
) -> dict[str, Any]:
    """Validate, fit both pre-registered models, score validation, and export."""

    bundle = load_phase_thermal_bundle(manifest_path, config=config)
    report = build_phase_fit_report(bundle, config=config)
    write_phase_fit_report(report, output_path)
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help=f"Path to {MANIFEST_FILENAME}")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: thermal_phase_fit_report.json beside manifest)",
    )
    parser.add_argument("--multistarts", type=int, default=FitConfig().multistart_count)
    parser.add_argument("--max-nfev", type=int, default=FitConfig().max_nfev)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    output = arguments.output or arguments.manifest.parent / "thermal_phase_fit_report.json"
    config = FitConfig(
        bin_width_s=1.0,
        multistart_count=arguments.multistarts,
        max_nfev=arguments.max_nfev,
    )
    fit_phase_thermal_bundle(arguments.manifest, output, config=config)
    print(output.resolve())


if __name__ == "__main__":
    main()
