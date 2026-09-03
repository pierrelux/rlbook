"""Hardware-free tests for the offline L4 thermal pulse fitter."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from fit_inference_thermal import (  # noqa: E402
    FitConfig,
    REQUIRED_TELEMETRY_COLUMNS,
    SequenceData,
    ThermalFitError,
    build_fit_report,
    fit_one_state_rc,
    fit_two_state_rc,
    load_thermal_bundle,
    simulate_one_state_rc,
    simulate_two_state_rc,
    write_fit_report,
)


TRAINING_SCHEDULE = {
    "training_a": ((40, 120), (52, 90), (64, 60), (46, 105), (58, 75)),
    "training_b": ((58, 60), (46, 90), (40, 105), (64, 45), (52, 75)),
}
VALIDATION_SCHEDULE = (
    (43, 90, "decode", "intermediate_cap"),
    (55, 60, "decode", "intermediate_cap"),
    (49, 75, "decode", "intermediate_cap"),
    (61, 45, "decode", "intermediate_cap"),
    (55, 60, "prefill", "workload_transfer"),
    (55, 60, "decode", "workload_transfer"),
)


def _measured_power(requested_power_w: float, length: int, index: int) -> np.ndarray:
    """Return a deterministic measured trace distinct from the requested cap."""

    time = np.arange(length, dtype=float)
    baseline = 0.82 * requested_power_w + 9.0
    return baseline + 0.35 * np.sin(0.17 * time + index)


def _synthetic_pulse(
    name: str,
    repeat: str,
    split: str,
    role: str,
    workload_phase: str,
    requested_power_w: float,
    duration_s: int,
    simulator,
    parameters: dict[str, float],
    *,
    initial_temperature_c: float,
    pulse_index: int,
) -> SequenceData:
    power = _measured_power(requested_power_w, duration_s, pulse_index)
    placeholder = SequenceData(
        name=name,
        split=split,
        time_s=np.arange(duration_s + 1, dtype=float),
        temperature_c=np.full(duration_s + 1, initial_temperature_c),
        measured_power_w=power,
        raw_row_count=duration_s + 1,
        repeat=repeat,
        role=role,
        workload_phase=workload_phase,
        requested_power_limit_w=requested_power_w,
        scheduled_duration_s=float(duration_s),
    )
    temperature = np.round(simulator(parameters, placeholder))
    return SequenceData(
        name=name,
        split=split,
        time_s=placeholder.time_s,
        temperature_c=temperature,
        measured_power_w=power,
        raw_row_count=duration_s + 1,
        repeat=repeat,
        role=role,
        workload_phase=workload_phase,
        requested_power_limit_w=requested_power_w,
        scheduled_duration_s=float(duration_s),
    )


def _training_pulses(simulator, parameters: dict[str, float]) -> list[SequenceData]:
    pulses = []
    pulse_index = 0
    for repeat, specifications in TRAINING_SCHEDULE.items():
        for index, (power, duration) in enumerate(specifications):
            pulses.append(
                _synthetic_pulse(
                    f"{repeat}_pulse_{index:02d}",
                    repeat,
                    "training",
                    "training_pulse",
                    "decode",
                    power,
                    duration,
                    simulator,
                    parameters,
                    initial_temperature_c=35.0 + (pulse_index % 3),
                    pulse_index=pulse_index,
                )
            )
            pulse_index += 1
    return pulses


def _schedule() -> tuple[list[dict], list[dict]]:
    schedule = []
    flat_blocks = []
    for repeat, specifications in TRAINING_SCHEDULE.items():
        blocks = []
        for index, (power, duration) in enumerate(specifications):
            block = {
                "block_id": f"{repeat}_pulse_{index:02d}",
                "split": "training",
                "sequence": repeat,
                "role": "training_pulse",
                "requested_power_limit_w": float(power),
                "duration_s": float(duration),
                "condition": {
                    "phase": "decode",
                    "prompt_tokens": 128,
                    "output_tokens": 32,
                    "concurrency": 8,
                },
            }
            blocks.append(block)
            flat_blocks.append(block)
        schedule.append(
            {
                "sequence": repeat,
                "split": "training",
                "requires_cooldown_before_every_pulse": True,
                "blocks": blocks,
            }
        )
    validation_blocks = []
    for index, (power, duration, phase, role) in enumerate(VALIDATION_SCHEDULE):
        block = {
            "block_id": f"validation_pulse_{index:02d}",
            "split": "validation",
            "sequence": "validation",
            "role": role,
            "requested_power_limit_w": float(power),
            "duration_s": float(duration),
            "condition": {
                "phase": phase,
                "prompt_tokens": 4096 if phase == "prefill" else 128,
                "output_tokens": 1 if phase == "prefill" else 32,
                "concurrency": 8,
            },
        }
        validation_blocks.append(block)
        flat_blocks.append(block)
    schedule.append(
        {
            "sequence": "validation",
            "split": "validation",
            "requires_cooldown_before_every_pulse": True,
            "blocks": validation_blocks,
        }
    )
    return schedule, flat_blocks


def _telemetry_row(
    elapsed_s: float,
    temperature_c: float,
    power_w: float,
    *,
    phase: str,
    split: str,
    sequence: str,
    block_id: str,
    block_role: str,
    requested_power_w: float | str,
    workload_phase: str,
) -> dict[str, object]:
    return {
        "elapsed_s": elapsed_s,
        "utc": "2026-09-03T00:00:00+00:00",
        "phase": phase,
        "graphics_clock_mhz": 210 if split == "conditioning" else 2040,
        "memory_clock_mhz": 6251,
        "power_w": power_w,
        "temperature_c": temperature_c,
        "utilization_percent": 0 if split == "conditioning" else 100,
        "memory_used_mib": 18000,
        "split": split,
        "sequence": sequence,
        "block_id": block_id,
        "block_role": block_role,
        "requested_power_limit_w": requested_power_w,
        "requested_clock_mhz": 210 if split == "conditioning" else 2040,
        "workload_phase": workload_phase,
    }


def _write_bundle(root: Path, *, validation_offset_c: float = 0.0) -> Path:
    truth = {
        "ambient_temperature_c": 25.0,
        "junction_capacitance_j_per_c": 15.0,
        "sink_capacitance_j_per_c": 150.0,
        "junction_sink_resistance_c_per_w": 0.08,
        "sink_ambient_resistance_c_per_w": 0.30,
    }
    schedule, flat_blocks = _schedule()
    rows = [
        _telemetry_row(
            0.0,
            45.0,
            30.0,
            phase="initialization",
            split="",
            sequence="",
            block_id="",
            block_role="",
            requested_power_w="",
            workload_phase="",
        )
    ]
    cooldown_events = []
    elapsed = 0.0
    for pulse_index, block in enumerate(flat_blocks):
        block_id = block["block_id"]
        repeat = block["sequence"]
        split = block["split"]
        duration = int(block["duration_s"])
        initial_temperature = 45.0 + (pulse_index % 3)
        elapsed += 2.0
        rows.append(
            _telemetry_row(
                elapsed,
                initial_temperature,
                30.0,
                phase=f"thermal_identification_cooldown_before_{block_id}",
                split="conditioning",
                sequence=repeat,
                block_id=f"cooldown_before_{block_id}",
                block_role="cooldown",
                requested_power_w=40,
                workload_phase="idle",
            )
        )
        cooldown_events.append(
            {
                "before_block_id": block_id,
                "sequence": repeat,
                "status": "complete",
            }
        )
        elapsed += 2.0
        pulse = _synthetic_pulse(
            block_id,
            repeat,
            split,
            block["role"],
            block["condition"]["phase"],
            block["requested_power_limit_w"],
            duration,
            simulate_two_state_rc,
            truth,
            initial_temperature_c=initial_temperature,
            pulse_index=pulse_index,
        )
        temperature = pulse.temperature_c.copy()
        if split == "validation":
            temperature += validation_offset_c
        for index, local_time in enumerate(pulse.time_s):
            power_index = min(index, len(pulse.measured_power_w) - 1)
            rows.append(
                _telemetry_row(
                    elapsed + float(local_time),
                    temperature[index],
                    pulse.measured_power_w[power_index],
                    phase=f"thermal_identification_{block_id}",
                    split=split,
                    sequence=repeat,
                    block_id=block_id,
                    block_role=block["role"],
                    requested_power_w=block["requested_power_limit_w"],
                    workload_phase=block["condition"]["phase"],
                )
            )
        elapsed += duration + 1.0
        rows.append(
            _telemetry_row(
                elapsed,
                temperature[-1],
                30.0,
                phase=f"thermal_identification_transition_after_{block_id}",
                split="conditioning",
                sequence=repeat,
                block_id=f"transition_after_{block_id}",
                block_role="workload_stop",
                requested_power_w="",
                workload_phase="transition_idle",
            )
        )

    telemetry_path = root / "l4_thermal_telemetry.csv"
    with telemetry_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_TELEMETRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 2,
        "mode": "thermal-identification",
        "protocol": "cold-start-pulses-v2",
        "status": "complete",
        "git_revision": "synthetic-test",
        "telemetry_period_s": 1.0,
        "telemetry_columns": list(REQUIRED_TELEMETRY_COLUMNS),
        "telemetry_row_count": len(rows),
        "schedule": schedule,
        "completed_block_ids": [block["block_id"] for block in flat_blocks],
        "training_power_limits_w": [40, 46, 52, 58, 64],
        "validation_power_limits_w": [43, 49, 55, 61],
        "cooldown_protocol": {"before_every_pulse": True},
        "cooldown_events": cooldown_events,
        "fit_protocol": {
            "acquisition_only": True,
            "training_sequences": ["training_a", "training_b"],
            "untouched_validation_sequence": "validation",
            "fit_input": "measured power rather than requested cap",
            "temperature_fit": (
                "continuous-time trajectory fit; do not finite-difference the "
                "integer-valued temperature samples"
            ),
        },
    }
    manifest["sha256"] = {
        "l4_thermal_telemetry.csv": sha256(telemetry_path.read_bytes()).hexdigest()
    }
    manifest_path = root / "thermal_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class InferenceThermalFitTests(unittest.TestCase):
    def test_loader_separates_cold_start_pulses_across_cooldown_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = load_thermal_bundle(_write_bundle(Path(directory)))
        self.assertEqual(len(bundle.pulses), 16)
        self.assertEqual(len(bundle.pulse_ids_by_sequence["training_a"]), 5)
        self.assertEqual(len(bundle.pulse_ids_by_sequence["training_b"]), 5)
        self.assertEqual(len(bundle.pulse_ids_by_sequence["validation"]), 6)
        for pulse in bundle.pulses.values():
            self.assertEqual(pulse.time_s[0], 0.0)
            self.assertAlmostEqual(pulse.time_s[-1], pulse.scheduled_duration_s)

    def test_one_state_parameter_recovery_from_quantized_cold_pulses(self) -> None:
        truth = {
            "ambient_temperature_c": 25.0,
            "thermal_resistance_c_per_w": 0.45,
            "thermal_time_constant_s": 15.0,
        }
        fit = fit_one_state_rc(
            _training_pulses(simulate_one_state_rc, truth),
            config=FitConfig(multistart_count=6, max_nfev=1_200),
        )
        self.assertAlmostEqual(
            fit.parameters["ambient_temperature_c"], truth["ambient_temperature_c"], delta=1.6
        )
        self.assertAlmostEqual(
            fit.parameters["thermal_resistance_c_per_w"],
            truth["thermal_resistance_c_per_w"],
            delta=0.04,
        )
        self.assertAlmostEqual(
            fit.parameters["thermal_time_constant_s"],
            truth["thermal_time_constant_s"],
            delta=1.5,
        )
        self.assertTrue(fit.diagnostics["asymptotically_stable"])

    def test_two_state_parameter_recovery_from_quantized_cold_pulses(self) -> None:
        truth = {
            "ambient_temperature_c": 25.0,
            "junction_capacitance_j_per_c": 15.0,
            "sink_capacitance_j_per_c": 150.0,
            "junction_sink_resistance_c_per_w": 0.08,
            "sink_ambient_resistance_c_per_w": 0.30,
        }
        fit = fit_two_state_rc(
            _training_pulses(simulate_two_state_rc, truth),
            config=FitConfig(multistart_count=8, max_nfev=1_500),
        )
        for name, tolerance in (
            ("ambient_temperature_c", 2.0),
            ("junction_capacitance_j_per_c", 3.0),
            ("sink_capacitance_j_per_c", 20.0),
            ("junction_sink_resistance_c_per_w", 0.02),
            ("sink_ambient_resistance_c_per_w", 0.03),
        ):
            self.assertAlmostEqual(fit.parameters[name], truth[name], delta=tolerance, msg=name)
        self.assertTrue(fit.diagnostics["asymptotically_stable"])
        self.assertTrue(fit.diagnostics["positive_thermal_parameters"])
        self.assertTrue(fit.diagnostics["locally_identifiable"])

    def test_validation_never_changes_fits_and_matched_pair_is_reported(self) -> None:
        config = FitConfig(multistart_count=2, max_nfev=1_000)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_bundle = load_thermal_bundle(_write_bundle(Path(first)), config=config)
            second_bundle = load_thermal_bundle(
                _write_bundle(Path(second), validation_offset_c=20.0), config=config
            )
            first_report = build_fit_report(first_bundle, config=config)
            second_report = build_fit_report(second_bundle, config=config)
            output_path = write_fit_report(first_report, Path(first) / "thermal_fit.json")
            exported = json.loads(output_path.read_text(encoding="utf-8"))
        for model_name in ("one_state_rc", "two_state_junction_sink_rc"):
            first_model = first_report["models"][model_name]
            second_model = second_report["models"][model_name]
            self.assertEqual(
                first_model["final_training_fit"], second_model["final_training_fit"]
            )
            self.assertEqual(
                first_model["leave_one_training_repeat_out"],
                second_model["leave_one_training_repeat_out"],
            )
            self.assertNotEqual(
                first_model["validation_evaluation"]["aggregate"]["rmse_c"],
                second_model["validation_evaluation"]["aggregate"]["rmse_c"],
            )
            validation = first_model["validation_evaluation"]
            self.assertEqual(validation["evaluation_passes"], 1)
            self.assertEqual(len(validation["per_pulse"]), 6)
            pair = validation["matched_55w_prefill_decode"]
            self.assertEqual(pair["prefill"]["block_id"], "validation_pulse_04")
            self.assertEqual(pair["decode"]["block_id"], "validation_pulse_05")
        metrics = exported["models"]["one_state_rc"]["validation_evaluation"][
            "aggregate"
        ]
        self.assertTrue(
            {
                "rmse_c",
                "mae_c",
                "maximum_absolute_error_c",
                "peak_temperature_error_c",
                "residual_autocorrelation",
            }.issubset(metrics)
        )

    def test_loader_fails_closed_on_schema_split_and_checksum(self) -> None:
        config = FitConfig(multistart_count=1, max_nfev=100)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_bundle(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ThermalFitError, "schema"):
                load_thermal_bundle(manifest_path, config=config)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_bundle(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schedule"][2]["split"] = "training"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ThermalFitError, "split"):
                load_thermal_bundle(manifest_path, config=config)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_bundle(root)
            telemetry_path = root / "l4_thermal_telemetry.csv"
            telemetry_path.write_text(
                telemetry_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ThermalFitError, "checksum"):
                load_thermal_bundle(manifest_path, config=config)


if __name__ == "__main__":
    unittest.main()
