"""Hardware-free tests for the pre-registered L4 phase-confirmation fitter."""

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

from fit_inference_thermal import FitConfig, SequenceData  # noqa: E402
from fit_inference_thermal_phase import (  # noqa: E402
    MANIFEST_FILENAME,
    REQUIRED_REQUEST_COLUMNS,
    REQUIRED_TELEMETRY_COLUMNS,
    REQUESTS_FILENAME,
    TELEMETRY_FILENAME,
    ThermalFitError,
    _EXPECTED_BLOCKS,
    build_phase_fit_report,
    load_phase_thermal_bundle,
    simulate_one_state_phase_gain,
    write_phase_fit_report,
)


def _telemetry_row(
    elapsed_s: float,
    temperature_c: float,
    power_w: float,
    *,
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
        "phase": f"thermal_phase_{block_id}" if block_id else "initialization",
        "graphics_clock_mhz": 210 if split == "conditioning" else 2040,
        "memory_clock_mhz": 6251,
        "power_w": power_w,
        "temperature_c": temperature_c,
        "utilization_percent": 0 if split in ("", "conditioning") else 100,
        "memory_used_mib": 18000,
        "split": split,
        "sequence": sequence,
        "block_id": block_id,
        "block_role": block_role,
        "requested_power_limit_w": requested_power_w,
        "requested_clock_mhz": 210 if split == "conditioning" else 2040,
        "workload_phase": workload_phase,
    }


def _request_rows(
    block: dict, start_elapsed_s: float, end_elapsed_s: float
) -> list[dict[str, object]]:
    condition = block["condition"]
    return [
        {
            "split": block["split"],
            "sequence": block["sequence"],
            "block_id": block["block_id"],
            "workload_phase": condition["phase"],
            "prompt_tokens": condition["prompt_tokens"],
            "output_tokens": condition["output_tokens"],
            "concurrency": condition["concurrency"],
            "requested_power_limit_w": block["requested_power_limit_w"],
            "requested_clock_mhz": 2040,
            "batch_index": 0,
            "request_index": request_index,
            "prompt_tokens_observed": condition["prompt_tokens"],
            "completion_tokens": condition["output_tokens"],
            "ttft_s": 0.1,
            "tpot_s": 0.05 if condition["output_tokens"] > 1 else 0.0,
            "total_s": end_elapsed_s - start_elapsed_s,
            "start_elapsed_s": start_elapsed_s,
            "end_elapsed_s": end_elapsed_s,
        }
        for request_index in range(condition["concurrency"])
    ]


def _schedule() -> list[dict]:
    schedule = []
    for sequence, split in (("phase_training", "training"), ("phase_validation", "validation")):
        blocks = [
            json.loads(json.dumps(block))
            for block in _EXPECTED_BLOCKS
            if block["sequence"] == sequence
        ]
        schedule.append(
            {
                "sequence": sequence,
                "split": split,
                "requires_cooldown_before_every_pulse": True,
                "blocks": blocks,
            }
        )
    return schedule


def _write_bundle(root: Path, *, validation_offset_c: float = 0.0) -> Path:
    truth = {
        "ambient_temperature_c": 27.0,
        "thermal_resistance_c_per_w": 0.48,
        "thermal_time_constant_s": 24.0,
        "beta": 0.20,
    }
    rows = [
        _telemetry_row(
            0.0,
            45.0,
            25.0,
            split="",
            sequence="",
            block_id="",
            block_role="",
            requested_power_w="",
            workload_phase="",
        )
    ]
    requests = []
    cooldown_events = []
    checkpoints = []
    elapsed = 0.0
    for pulse_index, fixed in enumerate(_EXPECTED_BLOCKS):
        block = json.loads(json.dumps(fixed))
        block_id = block["block_id"]
        duration = int(block["duration_s"])
        initial_temperature = 45.0 + (pulse_index % 2)
        elapsed += 1.0
        cooldown_started = elapsed
        for cooldown_second in range(61):
            rows.append(
                _telemetry_row(
                    cooldown_started + cooldown_second,
                    initial_temperature,
                    25.0,
                    split="conditioning",
                    sequence=block["sequence"],
                    block_id=f"cooldown_before_{block_id}",
                    block_role="cooldown",
                    requested_power_w=40.0,
                    workload_phase="idle",
                )
            )
        cooldown_completed = cooldown_started + 60.0
        cooldown_events.append(
            {
                "before_block_id": block_id,
                "sequence": block["sequence"],
                "started_elapsed_s": cooldown_started,
                "completed_elapsed_s": cooldown_completed,
                "duration_s": 60.0,
                "target_temperature_c": 52.0,
                "stability_band_c": 1.0,
                "stability_window_s": 60.0,
                "timeout_s": 600.0,
                "final_temperature_c": initial_temperature,
                "window_min_temperature_c": initial_temperature,
                "window_max_temperature_c": initial_temperature,
                "status": "complete",
            }
        )
        elapsed = cooldown_completed + 1.0
        pulse_started = elapsed
        local_time = np.arange(duration + 1, dtype=float)
        measured_power = (
            0.90 * float(block["requested_power_limit_w"])
            + 5.0
            + 0.3 * np.sin(0.19 * np.arange(duration, dtype=float) + pulse_index)
        )
        placeholder = SequenceData(
            name=block_id,
            split=block["split"],
            time_s=local_time,
            temperature_c=np.full(duration + 1, initial_temperature),
            measured_power_w=measured_power,
            raw_row_count=duration + 1,
            repeat=block["sequence"],
            role=block["role"],
            workload_phase=block["condition"]["phase"],
            requested_power_limit_w=float(block["requested_power_limit_w"]),
            scheduled_duration_s=float(duration),
        )
        temperature = np.round(simulate_one_state_phase_gain(truth, placeholder))
        if block["split"] == "validation":
            temperature += validation_offset_c
        for index, time_s in enumerate(local_time):
            rows.append(
                _telemetry_row(
                    elapsed + float(time_s),
                    float(temperature[index]),
                    float(measured_power[min(index, duration - 1)]),
                    split=block["split"],
                    sequence=block["sequence"],
                    block_id=block_id,
                    block_role=block["role"],
                    requested_power_w=float(block["requested_power_limit_w"]),
                    workload_phase=block["condition"]["phase"],
                )
            )
        requests.extend(
            _request_rows(
                block,
                pulse_started + 0.01,
                pulse_started + duration - 0.01,
            )
        )
        checkpoints.append(
            {
                "block": block,
                "status": "complete",
                "started_elapsed_s": pulse_started,
                "ended_elapsed_s": pulse_started + duration,
                "actual_duration_s": float(duration),
                "block_telemetry_rows": duration + 1,
                "maximum_temperature_c": float(np.max(temperature)),
            }
        )
        elapsed = pulse_started + duration + 1.0

    telemetry_path = root / TELEMETRY_FILENAME
    with telemetry_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_TELEMETRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    requests_path = root / REQUESTS_FILENAME
    with requests_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_REQUEST_COLUMNS)
        writer.writeheader()
        writer.writerows(requests)
    manifest = {
        "schema_version": 2,
        "mode": "thermal-phase-identification",
        "protocol": "cold-start-phase-pairs-v1",
        "status": "complete",
        "git_revision": "synthetic-phase-test",
        "cloud": {
            "project": "potent-arcade-491015-g7",
            "zone": "us-central1-a",
            "provisioning_model": "STANDARD",
            "machine_type": "g2-standard-8",
        },
        "gpu": {"name": "NVIDIA L4"},
        "telemetry_period_s": 1.0,
        "requested_graphics_clock_mhz": 2040,
        "selected_memory_clock_mhz": 6251,
        "cooldown_graphics_clock_mhz": 210,
        "telemetry_columns": list(REQUIRED_TELEMETRY_COLUMNS),
        "request_columns": list(REQUIRED_REQUEST_COLUMNS),
        "telemetry_row_count": len(rows),
        "request_row_count": len(requests),
        "schedule": _schedule(),
        "completed_block_ids": [block["block_id"] for block in _EXPECTED_BLOCKS],
        "cooldown_protocol": {
            "before_every_pulse": True,
            "power_limit_w": 40.0,
            "graphics_clock_mhz": 210,
            "workload": "idle",
            "target_temperature_c": 52.0,
            "stability_band_c": 1.0,
            "stability_window_s": 60.0,
            "timeout_s": 600.0,
        },
        "safety_protocol": {
            "independent_of_request_loop": True,
            "safe_down_temperature_c": 77.0,
            "abort_temperature_c": 79.0,
            "safe_power_limit_w": 40.0,
        },
        "fit_protocol": {
            "acquisition_only": True,
            "training_sequences": ["phase_training"],
            "untouched_validation_sequence": "phase_validation",
            "validation_evaluation_passes": 1,
            "fit_input": "measured power rather than requested cap",
            "prespecified_phase_model": "P_eff = P * (1 + beta * I_prefill)",
        },
        "cooldown_events": cooldown_events,
        "block_checkpoints": checkpoints,
        "sha256": {
            TELEMETRY_FILENAME: sha256(telemetry_path.read_bytes()).hexdigest(),
            REQUESTS_FILENAME: sha256(requests_path.read_bytes()).hexdigest(),
        },
    }
    manifest_path = root / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class InferenceThermalPhaseFitTests(unittest.TestCase):
    def test_fit_is_training_only_and_reports_phase_pair_diagnostics(self) -> None:
        config = FitConfig(multistart_count=6, max_nfev=1_200)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path = _write_bundle(Path(first))
            second_path = _write_bundle(Path(second), validation_offset_c=8.0)
            first_report = build_phase_fit_report(
                load_phase_thermal_bundle(first_path, config=config), config=config
            )
            second_report = build_phase_fit_report(
                load_phase_thermal_bundle(second_path, config=config), config=config
            )
            output = write_phase_fit_report(
                first_report, Path(first) / "thermal_phase_fit_report.json"
            )
            exported = json.loads(output.read_text(encoding="utf-8"))

        for model_name in ("power_only_one_state_rc", "phase_gain_one_state_rc"):
            self.assertEqual(
                first_report["models"][model_name]["final_training_fit"],
                second_report["models"][model_name]["final_training_fit"],
            )
            self.assertEqual(
                first_report["models"][model_name]["validation_evaluation"][
                    "evaluation_passes"
                ],
                1,
            )
            self.assertNotEqual(
                first_report["models"][model_name]["validation_evaluation"][
                    "aggregate"
                ]["rmse_c"],
                second_report["models"][model_name]["validation_evaluation"][
                    "aggregate"
                ]["rmse_c"],
            )
        phase_fit = exported["models"]["phase_gain_one_state_rc"][
            "final_training_fit"
        ]
        self.assertAlmostEqual(phase_fit["parameters"]["beta"], 0.20, delta=0.04)
        self.assertEqual(phase_fit["diagnostics"]["jacobian_numerical_rank"], 4)
        self.assertEqual(phase_fit["diagnostics"]["parameter_count"], 4)
        self.assertEqual(phase_fit["diagnostics"]["successful_multistarts"], 6)
        self.assertLess(
            exported["models"]["phase_gain_one_state_rc"]["validation_evaluation"][
                "aggregate"
            ]["rmse_c"],
            exported["models"]["power_only_one_state_rc"]["validation_evaluation"][
                "aggregate"
            ]["rmse_c"],
        )
        contrast = exported["observed_validation_contrast"]
        self.assertIn("raw_telemetry", contrast)
        self.assertIn("one_second_aligned", contrast)
        balance = exported["measured_power_and_energy"]["matched_phase_pairs"]
        self.assertEqual(set(balance["phase_training"]), {"46", "61"})
        self.assertEqual(set(balance["phase_validation"]), {"55"})

    def test_loader_fails_closed_on_schedule_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_bundle(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schedule"][0]["blocks"][0]["condition"]["phase"] = "prefill"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ThermalFitError, "condition phase"):
                load_phase_thermal_bundle(manifest_path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_bundle(root)
            telemetry_path = root / TELEMETRY_FILENAME
            telemetry_path.write_text(
                telemetry_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ThermalFitError, "checksum"):
                load_phase_thermal_bundle(manifest_path)

    def test_loader_rejects_weak_cooldown_duration_and_request_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_bundle(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cooldown_events"][0]["window_max_temperature_c"] = 53.0
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ThermalFitError, "stable cold-start evidence"):
                load_phase_thermal_bundle(manifest_path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_bundle(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["block_checkpoints"][0]["actual_duration_s"] = 70.0
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ThermalFitError, "Checkpoint duration"):
                load_phase_thermal_bundle(manifest_path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _write_bundle(root)
            requests_path = root / REQUESTS_FILENAME
            with requests_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows[:8]:
                row["end_elapsed_s"] = str(float(row["start_elapsed_s"]) + 1.0)
                row["total_s"] = "1.0"
            with requests_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=REQUIRED_REQUEST_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sha256"][REQUESTS_FILENAME] = sha256(
                requests_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ThermalFitError, "cover too little"):
                load_phase_thermal_bundle(manifest_path)

    def test_report_rejects_nonregistered_bins_and_source_output_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = load_phase_thermal_bundle(_write_bundle(root))
            with self.assertRaisesRegex(ThermalFitError, "one-second bins"):
                build_phase_fit_report(bundle, config=FitConfig(bin_width_s=0.5))
            with self.assertRaisesRegex(ThermalFitError, "source artifact"):
                write_phase_fit_report({}, root / MANIFEST_FILENAME)


if __name__ == "__main__":
    unittest.main()
