"""Hardware-free tests for the L4 profiling protocol."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv
import json
import math
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from profile_inference_gpu import (  # noqa: E402
    BenchmarkCondition,
    GpuControls,
    GpuMetadata,
    PROFILE_COLUMNS,
    PROFILE_SCHEMA_VERSION,
    RAW_PROFILE_COLUMNS,
    ProfilingError,
    THERMAL_ABORT_C,
    THERMAL_COOLDOWN_STABILITY_S,
    THERMAL_COOLDOWN_TARGET_C,
    THERMAL_COOLDOWN_TIMEOUT_S,
    THERMAL_SAFE_DOWN_C,
    ThermalSafetyWatchdog,
    VLLM_PREFIX_CACHING_ENABLED,
    _telemetry_summary,
    apply_verified_thermal_power_limit,
    aggregate_profile,
    batch_balanced_realized_clock_medians,
    checkpoint_condition_progress,
    completed_profile_metadata,
    fit_thermal_rc,
    managed_gpu_controls,
    main,
    parse_arguments,
    parse_supported_clocks,
    parse_temperature_thresholds,
    select_thermal_limit,
    profile_conditions,
    run_thermal_identification_sequence,
    select_clock_levels,
    select_usable_clock_levels,
    thermal_identification_sequences,
    thermal_phase_identification_sequences,
    wait_for_thermal_cooldown,
)


class RecordingRunner:
    def __init__(self, *, fail_memory: bool = False) -> None:
        self.commands: list[list[str]] = []
        self.fail_memory = fail_memory

    def __call__(self, arguments, **kwargs):
        command = list(arguments)
        self.commands.append(command)
        if self.fail_memory and "-lmc" in command:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, "", "")


class ScriptedCooldownSampler:
    def __init__(self, temperatures: list[float], *, initial: float | None = None) -> None:
        self.rows = (
            [{"phase": "previous", "temperature_c": initial}]
            if initial is not None
            else []
        )
        self.temperatures = iter(temperatures)
        self.phase = "initialization"
        self.health_checks = 0

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def ensure_healthy(self) -> None:
        self.health_checks += 1

    def emit_next(self) -> None:
        try:
            temperature = next(self.temperatures)
        except StopIteration:
            return
        self.rows.append({"phase": self.phase, "temperature_c": temperature})


class ScriptedClock:
    def __init__(self, sampler: ScriptedCooldownSampler) -> None:
        self.elapsed_s = 0.0
        self.sampler = sampler

    def monotonic(self) -> float:
        return self.elapsed_s

    def sleep(self, duration_s: float) -> None:
        self.elapsed_s += duration_s
        self.sampler.emit_next()


class WatchdogSampler:
    def __init__(self, rows: list[dict[str, float]] | None = None) -> None:
        self.rows = rows or []
        self.error: Exception | None = None

    def snapshot(self) -> list[dict[str, float]]:
        return [dict(row) for row in self.rows]


class RecordingSafeDownControls:
    def __init__(self, errors: tuple[str, ...] = ()) -> None:
        self.calls: list[tuple[float, int]] = []
        self.errors = errors

    def emergency_safe_down(
        self, *, power_limit_w: float, graphics_clock_mhz: int
    ) -> tuple[str, ...]:
        self.calls.append((power_limit_w, graphics_clock_mhz))
        return self.errors


def metadata() -> GpuMetadata:
    return GpuMetadata(
        name="NVIDIA L4",
        uuid="GPU-test",
        driver_version="580.00",
        cuda_version="12.9",
        default_power_limit_w=72.0,
        minimum_power_limit_w=60.0,
        maximum_power_limit_w=72.0,
        slowdown_temperature_c=85.0,
        shutdown_temperature_c=92.0,
    )


class InferenceProfilerTests(unittest.TestCase):
    def test_clock_parser_and_even_selection(self) -> None:
        output = "5001, 420\n5001, 690\n5001, 960\n5001, 1230\n5001, 1500\n"
        supported = parse_supported_clocks(output)
        memory, clocks = select_clock_levels(supported)
        self.assertEqual(memory, 5001)
        self.assertEqual(clocks, [420, 690, 960, 1230, 1500])

    def test_clock_selection_aborts_below_minimum(self) -> None:
        with self.assertRaisesRegex(ProfilingError, "at least 4"):
            select_clock_levels({5001: [600, 900, 1200]})

    def test_modeled_clock_selection_excludes_a_power_capped_inversion(self) -> None:
        requested = [210, 660, 1125, 1575, 2040]
        realized = [210.0, 660.0, 1123.0, 1297.5, 1233.75]
        raw = []
        for clock, realized_clock in zip(requested, realized):
            for request_index in range(2):
                raw.append(
                    {
                        "phase": "decode",
                        "prompt_tokens": 1024,
                        "output_tokens": 128,
                        "concurrency": 2,
                        "repeat": 0,
                        "requested_clock_mhz": clock,
                        "realized_clock_mhz": realized_clock,
                        "request_index": request_index,
                    }
                )
        aggregate = [
            {
                "clock_mhz": clock,
                "prefill_tokens_per_s": prefill,
                "decode_tokens_per_s": decode,
            }
            for clock, prefill, decode in zip(
                requested,
                [10.0, 20.0, 30.0, 40.0, 41.0],
                [10.0, 20.0, 30.0, 40.0, 39.9],
            )
        ]

        medians = batch_balanced_realized_clock_medians(raw, requested)
        modeled, metadata = select_usable_clock_levels(raw, aggregate, requested)

        self.assertEqual(medians, dict(zip(requested, realized)))
        self.assertEqual(modeled, [210, 660, 1125, 1575])
        self.assertEqual(metadata["excluded_requested_clocks_mhz"], [2040])

    def test_modeled_clock_selection_fails_below_four_jointly_usable_levels(self) -> None:
        requested = [210, 660, 1125, 1575, 2040]
        raw = [
            {
                "phase": "decode",
                "prompt_tokens": 1024,
                "output_tokens": 128,
                "concurrency": 1,
                "repeat": 0,
                "requested_clock_mhz": clock,
                "realized_clock_mhz": realized,
            }
            for clock, realized in zip(requested, [210, 660, 1125, 600, 590])
        ]
        aggregate = [
            {
                "clock_mhz": clock,
                "prefill_tokens_per_s": float(index + 1),
                "decode_tokens_per_s": float(index + 1),
            }
            for index, clock in enumerate(requested)
        ]
        with self.assertRaisesRegex(ProfilingError, "Fewer than four"):
            select_usable_clock_levels(raw, aggregate, requested)

    def test_temperature_thresholds_are_read_from_xml(self) -> None:
        slowdown, shutdown = parse_temperature_thresholds(
            "<nvidia_smi_log><gpu><temperature>"
            "<gpu_temp_slow_threshold>85 C</gpu_temp_slow_threshold>"
            "<gpu_temp_max_threshold>92 C</gpu_temp_max_threshold>"
            "</temperature></gpu></nvidia_smi_log>"
        )
        self.assertEqual(slowdown, 85.0)
        self.assertEqual(shutdown, 92.0)

    def test_missing_temperature_thresholds_remain_explicit(self) -> None:
        slowdown, shutdown = parse_temperature_thresholds(
            "<nvidia_smi_log><gpu><temperature>"
            "<gpu_temp_slow_threshold>N/A</gpu_temp_slow_threshold>"
            "<gpu_temp_max_threshold>N/A</gpu_temp_max_threshold>"
            "</temperature></gpu></nvidia_smi_log>"
        )
        self.assertIsNone(slowdown)
        self.assertIsNone(shutdown)

    def test_absent_temperature_threshold_tags_remain_explicit(self) -> None:
        slowdown, shutdown = parse_temperature_thresholds(
            "<nvidia_smi_log><gpu><temperature>"
            "<gpu_temp>41 C</gpu_temp>"
            "</temperature></gpu></nvidia_smi_log>"
        )
        self.assertIsNone(slowdown)
        self.assertIsNone(shutdown)

    def test_thermal_limit_uses_absolute_ceiling_when_threshold_missing(self) -> None:
        limit, source = select_thermal_limit(None)
        self.assertEqual(limit, 80.0)
        self.assertEqual(
            source, "protocol_absolute_80_c_no_reported_slowdown_threshold"
        )

    def test_thermal_limit_records_reported_threshold_provenance(self) -> None:
        _, source = select_thermal_limit(78.0)
        self.assertEqual(source, "min_80_c_and_5_c_below_reported_slowdown")

    def test_thermal_limit_clamps_exact_85_c_boundary_to_80_c(self) -> None:
        self.assertEqual(select_thermal_limit(85.0)[0], 80.0)

    def test_thermal_limit_stays_five_degrees_below_reported_slowdown(self) -> None:
        self.assertEqual(select_thermal_limit(78.0)[0], 73.0)
        self.assertEqual(select_thermal_limit(90.0)[0], 80.0)

    def test_sweep_cooldown_requires_consecutive_fresh_stable_samples(self) -> None:
        sampler = ScriptedCooldownSampler(
            [66.0, 64.0, 63.0, 65.0, 64.0, 63.0, 64.0], initial=62.0
        )
        clock = ScriptedClock(sampler)
        events: list[dict[str, object]] = []

        event = wait_for_thermal_cooldown(
            sampler,  # type: ignore[arg-type]
            requested_clock_mhz=660,
            thermal_limit_c=80.0,
            cooldown_clock_mhz=210,
            events=events,  # type: ignore[arg-type]
            stable_samples_required=3,
            timeout_s=10.0,
            poll_s=0.1,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

        self.assertIs(event, events[0])
        self.assertEqual(event["phase"], "cooldown_before_f660")
        self.assertEqual(event["status"], "complete")
        self.assertEqual(event["target_temperature_c"], 64.0)
        self.assertEqual(event["observed_samples"], 7)
        self.assertEqual(event["stable_samples_observed"], 3)
        self.assertEqual(event["final_temperature_c"], 64.0)
        self.assertEqual(event["initial_temperature_c"], 62.0)

    def test_condition_cooldown_phase_and_manifest_metadata_are_unique(self) -> None:
        events: list[dict[str, object]] = []
        phases: list[str] = []
        conditions = [
            BenchmarkCondition("prefill", 512, 1, 4),
            BenchmarkCondition("decode", 1024, 128, 4),
        ]
        for condition_index, condition in enumerate(conditions):
            sampler = ScriptedCooldownSampler([64.0, 64.0], initial=67.0)
            clock = ScriptedClock(sampler)
            event = wait_for_thermal_cooldown(
                sampler,  # type: ignore[arg-type]
                requested_clock_mhz=660,
                thermal_limit_c=80.0,
                cooldown_clock_mhz=210,
                condition=condition,
                condition_index=condition_index,
                events=events,  # type: ignore[arg-type]
                stable_samples_required=2,
                timeout_s=2.0,
                poll_s=0.1,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
            phases.append(str(event["phase"]))

        self.assertEqual(len(set(phases)), 2)
        self.assertEqual(
            phases[0], "cooldown_before_condition_00_prefill_p512_o1_c4_f660"
        )
        self.assertEqual(events[0]["scope"], "condition")
        self.assertEqual(events[0]["condition_index"], 0)
        self.assertEqual(
            events[0]["condition"],
            {
                "phase": "prefill",
                "prompt_tokens": 512,
                "output_tokens": 1,
                "concurrency": 4,
            },
        )

    def test_sweep_cooldown_fails_closed_above_thermal_limit(self) -> None:
        sampler = ScriptedCooldownSampler([79.0, 81.0], initial=77.0)
        clock = ScriptedClock(sampler)
        events: list[dict[str, object]] = []

        with self.assertRaisesRegex(ProfilingError, "above the protocol limit"):
            wait_for_thermal_cooldown(
                sampler,  # type: ignore[arg-type]
                requested_clock_mhz=1125,
                thermal_limit_c=80.0,
                cooldown_clock_mhz=210,
                events=events,  # type: ignore[arg-type]
                stable_samples_required=3,
                timeout_s=10.0,
                poll_s=0.1,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )

        self.assertEqual(events[0]["status"], "failed_over_limit")
        self.assertEqual(events[0]["final_temperature_c"], 81.0)
        self.assertEqual(events[0]["observed_samples"], 2)

    def test_sweep_cooldown_times_out_without_stable_target(self) -> None:
        sampler = ScriptedCooldownSampler([70.0, 69.0, 68.0], initial=72.0)
        clock = ScriptedClock(sampler)
        events: list[dict[str, object]] = []

        with self.assertRaisesRegex(ProfilingError, "timed out"):
            wait_for_thermal_cooldown(
                sampler,  # type: ignore[arg-type]
                requested_clock_mhz=1575,
                thermal_limit_c=80.0,
                cooldown_clock_mhz=210,
                events=events,  # type: ignore[arg-type]
                stable_samples_required=3,
                timeout_s=0.3,
                poll_s=0.1,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )

        self.assertEqual(events[0]["status"], "failed_timeout")
        self.assertEqual(events[0]["stable_samples_observed"], 0)
        self.assertEqual(events[0]["final_temperature_c"], 68.0)

    def test_condition_checkpoint_keeps_only_complete_clock_aggregates(self) -> None:
        condition = BenchmarkCondition("prefill", 128, 1, 1)
        rows = [{"phase": "prefill", "requested_clock_mhz": 900}]
        telemetry = [{"phase": "idle_f900", "temperature_c": 65.0}]
        manifest: dict[str, object] = {"status": "profiling"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "raw_profile_path": root / "l4_profile_raw.csv",
                "telemetry_path": root / "l4_telemetry.csv",
                "profile_path": root / "l4_profile.csv",
                "manifest_path": root / "profile_manifest.json",
            }
            with patch(
                "profile_inference_gpu.aggregate_profile",
                side_effect=AssertionError("incomplete clock must not be aggregated"),
            ):
                aggregate = checkpoint_condition_progress(
                    **paths,
                    rows=rows,
                    telemetry_rows=telemetry,
                    completed_clocks_mhz=[],
                    manifest=manifest,  # type: ignore[arg-type]
                    requested_clock_mhz=900,
                    clock_index=0,
                    condition=condition,
                    condition_index=0,
                )

            self.assertEqual(aggregate, [])
            with paths["profile_path"].open(newline="", encoding="utf-8") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            saved_manifest = json.loads(
                paths["manifest_path"].read_text(encoding="utf-8")
            )
            checkpoint = saved_manifest["latest_condition_checkpoint"]
            self.assertEqual(checkpoint["condition"], asdict(condition))
            self.assertEqual(checkpoint["aggregate_complete_clocks_mhz"], [])
            self.assertFalse(any(root.glob("*.tmp")))

            measured_aggregate = {
                "clock_mhz": 900.0,
                "prefill_tokens_per_s": 10.0,
                "decode_tokens_per_s": 5.0,
                "idle_power_w": 20.0,
                "prefill_power_w": 30.0,
                "decode_power_w": 35.0,
            }
            with patch(
                "profile_inference_gpu.aggregate_profile",
                return_value=[measured_aggregate],
            ) as aggregate_mock:
                checkpoint_condition_progress(
                    **paths,
                    rows=rows,
                    telemetry_rows=telemetry,
                    completed_clocks_mhz=[900],
                    manifest=manifest,  # type: ignore[arg-type]
                    requested_clock_mhz=900,
                    clock_index=0,
                    condition=condition,
                    condition_index=17,
                )
            aggregate_mock.assert_called_once_with(rows, telemetry, [900])
            with paths["profile_path"].open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_condition_cooldown_and_checkpoint_are_integrated_in_order(self) -> None:
        profiler = (SCRIPTS_DIRECTORY / "profile_inference_gpu.py").read_text(
            encoding="utf-8"
        )
        loop = profiler.index(
            "for condition_index, condition in enumerate(conditions):"
        )
        cooldown = profiler.index("wait_for_thermal_cooldown(", loop)
        relock = profiler.index("controls.lock_graphics(requested_clock)", cooldown)
        warmup = profiler.index("run_concurrent_batch(", relock)
        checkpoint = profiler.index("checkpoint_condition_progress(", warmup)
        self.assertLess(cooldown, relock)
        self.assertLess(relock, warmup)
        self.assertLess(warmup, checkpoint)

    def test_gpu_controls_reset_after_measurement_failure(self) -> None:
        runner = RecordingRunner()
        with patch("profile_inference_gpu.os.geteuid", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "measurement failed"):
                with managed_gpu_controls(metadata(), 5001, runner=runner) as (controls, _):
                    controls.lock_graphics(1200)
                    raise RuntimeError("measurement failed")
        joined = [" ".join(command) for command in runner.commands]
        self.assertTrue(any(" -rgc" in command for command in joined))
        self.assertTrue(any(" -rmc" in command for command in joined))
        self.assertTrue(any(" -pl 72.0" in command for command in joined))

    def test_memory_lock_failure_is_recorded_without_unlocking_graphics(self) -> None:
        runner = RecordingRunner(fail_memory=True)
        with patch("profile_inference_gpu.os.geteuid", return_value=0):
            with managed_gpu_controls(metadata(), 5001, runner=runner) as (controls, applied):
                self.assertFalse(applied["memory_clock_locked"])
                self.assertIsNotNone(applied["memory_clock_error"])
                controls.lock_graphics(900)
        self.assertTrue(any("-lgc" in command for command in runner.commands))

    def test_profile_matrix_and_csv_contract_are_fixed(self) -> None:
        self.assertEqual(PROFILE_SCHEMA_VERSION, 2)
        self.assertFalse(VLLM_PREFIX_CACHING_ENABLED)
        conditions = profile_conditions()
        self.assertEqual(len(conditions), 18)
        self.assertEqual(
            RAW_PROFILE_COLUMNS[:14],
            [
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
            ],
        )
        self.assertEqual({condition.concurrency for condition in conditions}, {1, 4, 8})
        self.assertEqual(
            {condition.prompt_tokens for condition in conditions if condition.phase == "prefill"},
            {128, 512, 2048, 4096},
        )
        self.assertEqual(
            PROFILE_COLUMNS,
            [
                "clock_mhz",
                "prefill_tokens_per_s",
                "decode_tokens_per_s",
                "idle_power_w",
                "prefill_power_w",
                "decode_power_w",
            ],
        )

    def test_book_profile_is_deterministically_aggregated_by_batch(self) -> None:
        raw = []
        for phase, work, total, ttft, tpot, power in (
            ("prefill", 128, 2.0, 1.0, 0.0, 60.0),
            ("decode", 16, 3.0, 1.0, 0.1, 70.0),
        ):
            for request_index in range(2):
                raw.append(
                    {
                        "phase": phase,
                        "requested_clock_mhz": 900,
                        "prompt_tokens": 128,
                        "output_tokens": 16 if phase == "decode" else 1,
                        "concurrency": 2,
                        "repeat": 0,
                        "prompt_tokens_observed": work,
                        "completion_tokens": work,
                        "total_s": total,
                        "ttft_s": ttft,
                        "tpot_s": tpot,
                        "mean_power_w": power,
                        "request_index": request_index,
                    }
                )
        telemetry = [
            {"phase": "idle_f900", "power_w": 30.0},
            {"phase": "idle_f900", "power_w": 32.0},
        ]
        aggregate = aggregate_profile(raw, telemetry, [900])
        self.assertEqual(len(aggregate), 1)
        self.assertAlmostEqual(aggregate[0]["prefill_tokens_per_s"], 128.0)
        self.assertAlmostEqual(aggregate[0]["decode_tokens_per_s"], 32.0 / 2.1)
        self.assertEqual(aggregate[0]["idle_power_w"], 31.0)

    def test_energy_is_integrated_and_shared_across_concurrent_requests(self) -> None:
        rows = [
            {"elapsed_s": 1.0, "power_w": 60.0, "graphics_clock_mhz": 900, "temperature_c": 40},
            {"elapsed_s": 2.0, "power_w": 80.0, "graphics_clock_mhz": 1000, "temperature_c": 45},
        ]
        summary = _telemetry_summary(
            rows, start_elapsed_s=1.0, end_elapsed_s=2.0, concurrency=2
        )
        self.assertAlmostEqual(summary["energy_j"], 35.0)
        self.assertAlmostEqual(summary["realized_clock_mhz"], 950.0)
        self.assertEqual(summary["peak_temp_c"], 45.0)

    def test_completed_metadata_fits_thermal_model_but_loader_requires_full_bundle(self) -> None:
        thermal = []
        temperature = 35.0
        elapsed = 0.0
        time_constant = 30.0
        resistance = 0.5
        ambient = 25.0
        for phase, power in (
            ("thermal_load_1", 65.0),
            ("thermal_cool_1", 25.0),
            ("thermal_load_2", 65.0),
            ("thermal_cool_2", 25.0),
        ):
            for _ in range(300):
                thermal.append(
                    {
                        "phase": phase,
                        "elapsed_s": elapsed,
                        "power_w": power,
                        "temperature_c": temperature,
                    }
                )
                equilibrium = ambient + resistance * power
                temperature = equilibrium + (temperature - equilibrium) * math.exp(
                    -0.1 / time_constant
                )
                elapsed += 0.1
        fit = fit_thermal_rc(thermal)
        self.assertAlmostEqual(fit["thermal_time_constant_s"], time_constant, delta=2.0)
        self.assertAlmostEqual(
            fit["thermal_resistance_c_per_w"], resistance, delta=0.06
        )

        baseline_rows = [
            {
                "phase": "decode",
                "requested_clock_mhz": 1200,
                "prompt_tokens": 1024,
                "concurrency": 1,
                "ttft_s": value,
                "tpot_s": 0.006 + index * 0.0001,
            }
            for index, value in enumerate((0.20, 0.18, 0.19, 0.17, 0.21))
        ]
        completed = completed_profile_metadata(
            baseline_rows,
            thermal,
            maximum_clock_mhz=1200,
            metadata=metadata(),
            experiment_power_limit_w=64.8,
            measured_on="2026-09-01T00:00:00+00:00",
        )
        self.assertEqual(completed["profile_status"], "measured_l4")
        self.assertAlmostEqual(completed["baseline_ttft_s"], 0.19)
        self.assertAlmostEqual(completed["baseline_tpot_s"], 0.0062)

        code_directory = Path(__file__).resolve().parents[1] / "code"
        if str(code_directory) not in sys.path:
            sys.path.insert(0, str(code_directory))
        from inference_serving import load_profile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "l4_profile.csv"
            with profile_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=PROFILE_COLUMNS)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "clock_mhz": clock,
                            "prefill_tokens_per_s": 1000 + clock,
                            "decode_tokens_per_s": 100 + clock / 10,
                            "idle_power_w": 25,
                            "prefill_power_w": 55,
                            "decode_power_w": 60,
                        }
                        for clock in (600, 1200)
                    ]
                )
            (root / "profile_manifest.json").write_text(
                json.dumps(completed), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "status must be 'complete'"):
                load_profile(profile_path)

    def test_thermal_identification_schedule_preserves_split_and_excitation(self) -> None:
        sequences = thermal_identification_sequences()
        self.assertEqual(
            [name for name, _ in sequences],
            ["training_a", "training_b", "validation"],
        )
        identifiers = [
            block.block_id for _, blocks in sequences for block in blocks
        ]
        self.assertEqual(len(identifiers), len(set(identifiers)))

        duration_ceilings = {
            40: 120.0,
            46: 105.0,
            52: 90.0,
            58: 75.0,
            64: 60.0,
        }
        training_orders = []
        training_durations = []
        for _, blocks in sequences[:2]:
            self.assertEqual(len(blocks), 5)
            self.assertTrue(all(block.split == "training" for block in blocks))
            self.assertTrue(all(block.role == "training_pulse" for block in blocks))
            self.assertTrue(
                all(
                    block.condition.prompt_tokens == 128
                    and block.condition.output_tokens == 32
                    and block.condition.concurrency == 8
                    for block in blocks
                )
            )
            caps = [int(block.requested_power_limit_w) for block in blocks]
            self.assertEqual(set(caps), set(duration_ceilings))
            self.assertTrue(
                all(
                    block.duration_s <= duration_ceilings[
                        int(block.requested_power_limit_w)
                    ]
                    for block in blocks
                )
            )
            training_orders.append(caps)
            training_durations.append(
                {
                    int(block.requested_power_limit_w): block.duration_s
                    for block in blocks
                }
            )
        self.assertNotEqual(training_orders[0], training_orders[1])
        self.assertTrue(
            all(
                training_durations[0][cap] != training_durations[1][cap]
                for cap in duration_ceilings
            )
        )
        self.assertFalse(
            any(block.role == "plateau" for _, blocks in sequences for block in blocks)
        )

        validation = sequences[2][1]
        self.assertTrue(all(block.split == "validation" for block in validation))
        intermediate = [
            block for block in validation if block.role == "intermediate_cap"
        ]
        self.assertEqual(len(intermediate), 4)
        self.assertEqual(
            {int(block.requested_power_limit_w) for block in intermediate},
            {43, 49, 55, 61},
        )
        matched = [
            block
            for block in validation
            if block.role == "workload_transfer"
        ]
        self.assertEqual(len(matched), 2)
        self.assertTrue(
            all(
                block.requested_power_limit_w == 55.0
                and block.duration_s == 60.0
                for block in matched
            )
        )
        self.assertEqual(
            {block.condition.phase for block in matched}, {"prefill", "decode"}
        )
        self.assertTrue(all(block.duration_s <= 90.0 for block in validation))
        self.assertEqual(sum(block.duration_s for block in validation), 390.0)
        self.assertEqual(THERMAL_COOLDOWN_TARGET_C, 52.0)
        self.assertEqual(THERMAL_COOLDOWN_STABILITY_S, 60.0)
        self.assertEqual(THERMAL_COOLDOWN_TIMEOUT_S, 600.0)

    def test_thermal_phase_schedule_is_fixed_counterbalanced_and_held_out(self) -> None:
        sequences = thermal_phase_identification_sequences()
        self.assertEqual(
            [name for name, _ in sequences],
            ["phase_training", "phase_validation"],
        )
        observed = [
            (
                block.block_id,
                block.split,
                int(block.requested_power_limit_w),
                block.duration_s,
                block.condition.phase,
                block.condition.prompt_tokens,
                block.condition.output_tokens,
                block.condition.concurrency,
            )
            for _, blocks in sequences
            for block in blocks
        ]
        self.assertEqual(
            observed,
            [
                ("phase_training_pulse_00", "training", 46, 75.0, "decode", 128, 32, 8),
                ("phase_training_pulse_01", "training", 61, 45.0, "prefill", 4096, 1, 8),
                ("phase_training_pulse_02", "training", 46, 75.0, "prefill", 4096, 1, 8),
                ("phase_training_pulse_03", "training", 61, 45.0, "decode", 128, 32, 8),
                ("phase_validation_pulse_00", "validation", 55, 60.0, "decode", 128, 32, 8),
                ("phase_validation_pulse_01", "validation", 55, 60.0, "prefill", 4096, 1, 8),
            ],
        )
        self.assertEqual(sum(item[3] for item in observed), 360.0)
        self.assertEqual(len(sequences[0][1]), 4)
        self.assertEqual(len(sequences[1][1]), 2)
        self.assertFalse(
            any(block.split == "validation" for block in sequences[0][1])
        )

    def test_thermal_mode_bypasses_the_full_profile_and_its_repeat_constraint(self) -> None:
        arguments = parse_arguments(
            [
                "--mode",
                "thermal-identification",
                "--repeats",
                "1",
                "--thermal-load-seconds",
                "-1",
            ]
        )
        self.assertEqual(arguments.mode, "thermal-identification")
        with patch(
            "profile_inference_gpu.run_thermal_identification",
            return_value={"request_row_count": 8, "elapsed_s": 60.0},
        ) as thermal_run, patch(
            "profile_inference_gpu.run_profile",
            side_effect=AssertionError("full sweep must be bypassed"),
        ) as full_run:
            self.assertEqual(
                main(["--mode", "thermal-identification", "--repeats", "1"]),
                0,
            )
        thermal_run.assert_called_once()
        full_run.assert_not_called()

    def test_thermal_phase_failure_uses_its_own_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "profile_inference_gpu.run_thermal_identification",
            side_effect=ProfilingError("deliberate phase failure"),
        ):
            root = Path(directory)
            self.assertEqual(
                main(
                    [
                        "--mode",
                        "thermal-phase-identification",
                        "--output-directory",
                        str(root),
                    ]
                ),
                2,
            )
            self.assertTrue((root / "thermal_phase.failed").is_file())
            self.assertFalse((root / "thermal.failed").exists())
            self.assertFalse((root / "profile.failed").exists())

    def test_thermal_phase_mode_bypasses_the_full_profile(self) -> None:
        arguments = parse_arguments(
            ["--mode", "thermal-phase-identification", "--repeats", "1"]
        )
        self.assertEqual(arguments.mode, "thermal-phase-identification")
        with patch(
            "profile_inference_gpu.run_thermal_identification",
            return_value={"request_row_count": 8, "elapsed_s": 60.0},
        ) as thermal_run, patch(
            "profile_inference_gpu.run_profile",
            side_effect=AssertionError("full sweep must be bypassed"),
        ) as full_run:
            self.assertEqual(
                main(
                    [
                        "--mode",
                        "thermal-phase-identification",
                        "--repeats",
                        "1",
                    ]
                ),
                0,
            )
        thermal_run.assert_called_once()
        full_run.assert_not_called()

    def test_thermal_watchdog_safe_down_and_hard_abort_are_distinct(self) -> None:
        sampler = WatchdogSampler(
            [{"elapsed_s": 5.0, "temperature_c": THERMAL_SAFE_DOWN_C}]
        )
        controls = RecordingSafeDownControls()
        watchdog = ThermalSafetyWatchdog(
            sampler,  # type: ignore[arg-type]
            controls,  # type: ignore[arg-type]
            profile_start=0.0,
            safe_clock_mhz=210,
            monotonic=lambda: 5.0,
        )
        watchdog.started_monotonic = 0.0
        watchdog.inspect_once()
        self.assertTrue(watchdog.safe_down.is_set())
        self.assertFalse(watchdog.abort.is_set())
        self.assertEqual(controls.calls, [(40.0, 210)])

        sampler.rows[-1]["temperature_c"] = THERMAL_ABORT_C
        watchdog.inspect_once()
        self.assertTrue(watchdog.abort.is_set())
        self.assertEqual(controls.calls, [(40.0, 210)])
        with self.assertRaisesRegex(ProfilingError, "abort threshold"):
            watchdog.raise_if_stopped()

    def test_thermal_watchdog_aborts_on_stale_or_failed_telemetry(self) -> None:
        controls = RecordingSafeDownControls()
        stale = WatchdogSampler([{"elapsed_s": 1.0, "temperature_c": 60.0}])
        watchdog = ThermalSafetyWatchdog(
            stale,  # type: ignore[arg-type]
            controls,  # type: ignore[arg-type]
            profile_start=0.0,
            safe_clock_mhz=210,
            monotonic=lambda: 2.1,
        )
        watchdog.inspect_once()
        self.assertTrue(watchdog.abort.is_set())
        self.assertIn("stale", watchdog.reason or "")

        failed = WatchdogSampler([{"elapsed_s": 2.1, "temperature_c": 60.0}])
        failed.error = RuntimeError("NVML unavailable")
        failed_watchdog = ThermalSafetyWatchdog(
            failed,  # type: ignore[arg-type]
            RecordingSafeDownControls(),  # type: ignore[arg-type]
            profile_start=0.0,
            safe_clock_mhz=210,
            monotonic=lambda: 2.1,
        )
        failed_watchdog.inspect_once()
        self.assertTrue(failed_watchdog.abort.is_set())
        self.assertIn("telemetry failure", failed_watchdog.reason or "")

    def test_power_limit_readback_failure_forces_safe_down(self) -> None:
        class MismatchedPowerRunner(RecordingRunner):
            def __call__(self, arguments, **kwargs):
                result = super().__call__(arguments, **kwargs)
                if "--query-gpu=power.limit" in arguments:
                    return subprocess.CompletedProcess(arguments, 0, "41.0\n", "")
                return result

        runner = MismatchedPowerRunner()
        controls = GpuControls(
            0,
            runner,
            minimum_power_limit_w=40.0,
            maximum_power_limit_w=72.0,
        )
        sampler = WatchdogSampler([{"elapsed_s": 1.0, "temperature_c": 60.0}])
        watchdog = ThermalSafetyWatchdog(
            sampler,  # type: ignore[arg-type]
            controls,
            profile_start=0.0,
            safe_clock_mhz=210,
            monotonic=lambda: 1.0,
        )
        with self.assertRaisesRegex(ProfilingError, "application failed"):
            apply_verified_thermal_power_limit(controls, watchdog, 40.0)
        self.assertTrue(watchdog.abort.is_set())
        commands = [" ".join(command) for command in runner.commands]
        self.assertTrue(any(" -pl 40.0" in command for command in commands))
        self.assertTrue(any(" -lgc 210,210" in command for command in commands))

    def test_every_pulse_cools_after_the_previous_workload_stops(self) -> None:
        lifecycle: list[str] = []

        class Clock:
            elapsed_s = 0.0

            def monotonic(self) -> float:
                return self.elapsed_s

            def sleep(self, duration_s: float) -> None:
                self.elapsed_s += duration_s

        class Controls:
            def lock_graphics(self, frequency_mhz: int) -> None:
                lifecycle.append(f"clock:{frequency_mhz}")

            def set_power_limit(self, power_limit_w: float, *, verify: bool) -> float:
                lifecycle.append(f"cap:{power_limit_w:.0f}")
                return power_limit_w

        class Sampler:
            period_s = 0.1

            def snapshot(self):
                return []

            def ensure_healthy(self) -> None:
                return None

            def set_context(self, phase: str, **context) -> None:
                lifecycle.append(f"context:{phase}")

        class Watchdog:
            def raise_if_stopped(self) -> None:
                return None

            def abort_for_control_failure(self, reason: str) -> None:
                raise AssertionError(reason)

        class Load:
            def __init__(self, condition, block, **kwargs) -> None:
                self.block = block

            def start(self) -> None:
                lifecycle.append(f"start:{self.block.block_id}")

            def ensure_healthy(self) -> None:
                return None

            def stop(self) -> None:
                lifecycle.append(f"stop:{self.block.block_id}")

        def cooldown(*args, block, events: list[dict], **kwargs) -> None:
            events.append(
                {
                    "before_block_id": block.block_id,
                    "status": "complete",
                }
            )
            # Keep the lifecycle trace separate from the manifest event list.
            lifecycle.append(f"cooldown:{block.block_id}")

        blocks = thermal_phase_identification_sequences()[0][1][:2]
        clock = Clock()
        manifest = {"block_checkpoints": [], "completed_block_ids": []}
        cooldown_events: list[dict] = []
        with tempfile.TemporaryDirectory() as directory, patch(
            "profile_inference_gpu.ContinuousThermalLoad", Load
        ), patch(
            "profile_inference_gpu.wait_for_thermal_identification_cooldown",
            cooldown,
        ):
            root = Path(directory)
            run_thermal_identification_sequence(
                blocks,
                controls=Controls(),  # type: ignore[arg-type]
                sampler=Sampler(),  # type: ignore[arg-type]
                watchdog=Watchdog(),  # type: ignore[arg-type]
                server_url="http://example.invalid",
                served_model="test",
                profile_start=0.0,
                timeout_s=1.0,
                output_directory=root,
                telemetry_path=root / "telemetry.csv",
                requests_path=root / "requests.csv",
                manifest_path=root / "manifest.json",
                request_rows=[],
                request_rows_lock=threading.Lock(),
                manifest=manifest,
                cooldown_clock_mhz=210,
                cooldown_events=cooldown_events,
                marker_prefix="thermal-phase-block",
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
            self.assertTrue(
                (root / f"thermal-phase-block-{blocks[0].block_id}.done").is_file()
            )
            self.assertTrue(
                (root / f"thermal-phase-block-{blocks[1].block_id}.done").is_file()
            )

        first, second = blocks
        self.assertLess(
            lifecycle.index(f"cooldown:{first.block_id}"),
            lifecycle.index(f"start:{first.block_id}"),
        )
        self.assertLess(
            lifecycle.index(f"stop:{first.block_id}"),
            lifecycle.index(f"cooldown:{second.block_id}"),
        )
        self.assertLess(
            lifecycle.index(f"cooldown:{second.block_id}"),
            lifecycle.index(f"start:{second.block_id}"),
        )
        self.assertEqual(
            [event["before_block_id"] for event in cooldown_events],
            [first.block_id, second.block_id],
        )

    def test_gcp_launcher_preserves_security_and_lifecycle_flags(self) -> None:
        launcher = (SCRIPTS_DIRECTORY / "run_inference_profile_gcp.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "--no-service-account",
            "--no-scopes",
            '--provisioning-model "${provisioning_model}"',
            '--instance-termination-action DELETE',
            '--max-run-duration "${MAX_RUNTIME}"',
            "--no-restart-on-failure",
            "--reservation-affinity=none",
            "--boot-disk-auto-delete",
            "--no-deletion-protection",
            "--delete-disks=all",
            "enable-oslogin=TRUE,install-nvidia-driver=True",
            "rlbook-inference-profile-${VM_NAME}-${RUN_STAMP}",
            "sweep-*-mhz.done",
            "READINESS_ATTEMPTS=90",
            "nvidia-smi -L",
            "docker version",
            "bootstrap.log",
            'MAX_EXPOSURE_USD="6.75"',
            'PRIOR_PROFILE_COMPUTE_USD="4.35"',
            'ON_DEMAND_USD_PER_HOUR="0.853624312"',
            'create_profile_vm "${zone}" "SPOT"',
            'create_profile_vm "${STANDARD_ZONE}" "STANDARD"',
            "Non-capacity creation failure",
            "The profiling disk still exists",
            "apt-get install -y docker.io",
            "nvidia-ctk runtime configure --runtime=docker",
            "nvidia-container-cli info",
            "--source-worktree-state",
            "l4_profile_all_requested.csv",
            "Could not read the profiling VM status; retrying without cleanup.",
            "Could not query completed sweeps; retrying without cleanup.",
            "leaving the VM running and retrying.",
        ):
            self.assertIn(required, launcher)
        profiler = (SCRIPTS_DIRECTORY / "profile_inference_gpu.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--no-enable-prefix-caching"', profiler)
        self.assertNotIn('STAGING_OUTPUT="${LOCAL_OUTPUT}', launcher)
        promotion = launcher.index(
            "for completed_file in l4_profile.csv l4_profile_all_requested.csv"
        )
        self.assertIn("profile_manifest.json", launcher[promotion:])
        self.assertGreater(
            launcher[promotion:].index("profile_manifest.json"),
            launcher[promotion:].index("bootstrap.log"),
        )

    def test_thermal_launcher_is_bounded_and_keeps_outputs_separate(self) -> None:
        launcher_path = SCRIPTS_DIRECTORY / "run_inference_thermal_gcp.sh"
        launcher = launcher_path.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(launcher_path)], check=True)
        for required in (
            'PROJECT="potent-arcade-491015-g7"',
            'ACCOUNT="pierreluc@carbonforge.ai"',
            'MACHINE_TYPE="g2-standard-8"',
            'IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"',
            'MAX_RUNTIME="4h"',
            'PRIOR_FAILED_TRIAL_USD="0.40"',
            'MAX_EXPOSURE_USD="4.80"',
            "Maximum guarded cumulative exposure",
            '--provisioning-model STANDARD',
            '--instance-termination-action DELETE',
            '--boot-disk-auto-delete',
            '--no-service-account',
            '--no-scopes',
            '--delete-disks=all',
            '--mode thermal-identification',
            "RLBOOK_RUN_THERMAL_IDENTIFICATION=YES",
            "l4_thermal_telemetry.csv",
            "l4_thermal_requests.csv",
            "thermal_manifest.json",
            "thermal-block-*.done",
            "thermal.complete",
            "thermal.failed",
            "Creation failed ambiguously but a disk exists; deleting it.",
            'checksums = manifest.get("sha256")',
            "thermal manifest has no artifact checksums",
            "thermal manifest is not the safe cold-start pulse protocol",
            'checkpoint.get("block_telemetry_rows", 0)',
            "thermal block reached the abort temperature",
            "scheduled thermal blocks lack labeled telemetry",
            "thermal telemetry reached the abort temperature",
            "thermal cooldown events do not match every scheduled pulse",
            "a thermal pulse lacks a completed cold-start cooldown",
        ):
            self.assertIn(required, launcher)
        self.assertNotIn("l4_profile.csv", launcher)
        self.assertNotIn("profile_manifest.json", launcher)

    def test_thermal_phase_launcher_is_bounded_exact_and_isolated(self) -> None:
        launcher_path = SCRIPTS_DIRECTORY / "run_inference_thermal_phase_gcp.sh"
        launcher = launcher_path.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(launcher_path)], check=True)
        for required in (
            'PROJECT="potent-arcade-491015-g7"',
            'ACCOUNT="pierreluc@carbonforge.ai"',
            'VM_NAME="pierreluc-l4-rlbook-thermal-phase"',
            'MACHINE_TYPE="g2-standard-8"',
            'IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"',
            'MAX_RUNTIME="2h"',
            'PRIOR_FAILED_TRIAL_USD="0.40"',
            'PRIOR_COMPLETED_RUN_USD="1.60"',
            'MAX_NEW_COMPUTE_USD="1.707248624"',
            'EXPECTED_MAX_CUMULATIVE_USD="4.207248624"',
            'MAX_EXPOSURE_USD="4.80"',
            "RLBOOK_RUN_THERMAL_PHASE_IDENTIFICATION=YES",
            "--mode thermal-phase-identification",
            '--instance-termination-action DELETE',
            '--max-run-duration "${MAX_RUNTIME}"',
            "--boot-disk-auto-delete",
            "--no-service-account",
            "--no-scopes",
            "--delete-disks=all",
            "Creation failed ambiguously but a disk exists; deleting it.",
            "l4_thermal_phase_telemetry.csv",
            "l4_thermal_phase_requests.csv",
            "thermal_phase_manifest.json",
            "thermal_phase_vllm.log",
            "thermal-phase-block-*.done",
            "thermal_phase.complete",
            "thermal_phase.failed",
            'manifest.get("protocol") != "cold-start-phase-pairs-v1"',
            "phase_training_pulse_00",
            "phase_training_pulse_03",
            "phase_validation_pulse_00",
            "phase_validation_pulse_01",
            'sequence.get("requires_cooldown_before_every_pulse") is True',
            'checkpoint.get("block_telemetry_rows", 0)',
            'checksums = manifest.get("sha256")',
            "thermal telemetry reached the abort temperature",
            "The thermal VM still exists",
            "The thermal disk still exists",
        ):
            self.assertIn(required, launcher)
        self.assertNotIn("l4_profile.csv", launcher)
        self.assertNotIn("profile_manifest.json", launcher)
        self.assertNotIn("l4_thermal_telemetry.csv", launcher)
        self.assertNotIn("thermal_manifest.json", launcher)


if __name__ == "__main__":
    unittest.main()
