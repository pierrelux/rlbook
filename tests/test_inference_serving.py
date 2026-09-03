from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE_DIRECTORY = ROOT / "code"
sys.path.insert(0, str(CODE_DIRECTORY))

from inference_serving import (  # noqa: E402
    Request,
    ServingObservation,
    ServingPlant,
    chunked_prefill_scheduler,
    compute_metrics,
    fixed_clock_controller,
    load_profile,
    load_workload,
    maximum_clock_controller,
    normalize_offered_load,
    reactive_clock_controller,
    sample_and_hold_clock_controller,
    simulate,
    static_batch_scheduler,
)


DATA = ROOT / "data" / "inference_serving"

AGGREGATE_COLUMNS = [
    "clock_mhz",
    "prefill_tokens_per_s",
    "decode_tokens_per_s",
    "idle_power_w",
    "prefill_power_w",
    "decode_power_w",
]
RAW_COLUMNS = [
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
]
TELEMETRY_COLUMNS = [
    "elapsed_s",
    "utc",
    "phase",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "power_w",
    "temperature_c",
    "utilization_percent",
    "memory_used_mib",
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_valid_measured_bundle(
    root: Path,
    *,
    declare_modeled_clocks: bool = True,
) -> Path:
    measured_clocks = [600, 900, 1200, 1500, 1755]
    modeled_clocks = [600, 900, 1500, 1755]
    aggregate_rows = [
        {
            "clock_mhz": clock,
            "prefill_tokens_per_s": 1000 + 100 * index,
            "decode_tokens_per_s": 100 + 10 * index,
            "idle_power_w": 20 + index,
            "prefill_power_w": 40 + index,
            "decode_power_w": 35 + index,
        }
        for index, clock in enumerate(measured_clocks)
    ]
    profile_path = root / "l4_profile.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        writer.writerows(
            row for row in aggregate_rows if row["clock_mhz"] in modeled_clocks
        )
    full_profile_path = root / "l4_profile_all_requested.csv"
    with full_profile_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    conditions = [
        {"phase": "prefill", "prompt_tokens": prompt, "output_tokens": 1, "concurrency": concurrency}
        for prompt in (128, 512, 2048, 4096)
        for concurrency in (1, 4, 8)
    ] + [
        {"phase": "decode", "prompt_tokens": prompt, "output_tokens": 128, "concurrency": concurrency}
        for prompt in (128, 1024)
        for concurrency in (1, 4, 8)
    ]
    raw_rows = []
    for clock in measured_clocks:
        for condition in conditions:
            for repeat in range(5):
                for request_index in range(condition["concurrency"]):
                    raw_rows.append(
                        {
                            **condition,
                            "requested_clock_mhz": clock,
                            "realized_clock_mhz": clock,
                            "repeat": repeat,
                            "ttft_s": 0.2,
                            "tpot_s": 0.01,
                            "total_s": 0.4,
                            "energy_j": 12,
                            "mean_power_w": 40,
                            "peak_power_w": 45,
                            "peak_temp_c": 55,
                            "telemetry_sample_count": 1,
                            "telemetry_fallback_used": False,
                            "request_index": request_index,
                            "completion_tokens": condition["output_tokens"],
                            "prompt_tokens_observed": condition["prompt_tokens"],
                        }
                    )
    raw_path = root / "l4_profile_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(raw_rows)
    telemetry_path = root / "l4_telemetry.csv"
    with telemetry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TELEMETRY_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "elapsed_s": 0.0,
                "utc": "2026-09-01T00:00:00+00:00",
                "phase": "idle_f600",
                "graphics_clock_mhz": 600,
                "memory_clock_mhz": 5001,
                "power_w": 20,
                "temperature_c": 30,
                "utilization_percent": 0,
                "memory_used_mib": 100,
            }
        )
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "profile_status": "measured_l4",
        "source_label": "Measured test profile",
        "gpu": {"name": "NVIDIA L4"},
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "acbd96531cda22292a3ceaa67e984955d3965282",
        "vllm_image": "vllm/vllm-openai:v0.28.0",
        "vllm_image_digest": "sha256:" + "a" * 64,
        "vllm_prefix_caching_enabled": False,
        "vllm_server_arguments": ["--no-enable-prefix-caching"],
        "selected_graphics_clocks_mhz": measured_clocks,
        "clock_profile_selection": {
            "realized_clock_median_mhz_by_requested": {
                str(clock): float(clock) for clock in measured_clocks
            }
        },
        "measured_repeats_per_condition": 5,
        "warmup_batches_per_condition": 1,
        "conditions": conditions,
        "profile_columns": AGGREGATE_COLUMNS,
        "raw_profile_columns": RAW_COLUMNS,
        "row_count": len(raw_rows),
        "telemetry_row_count": 1,
        "telemetry_fallback_batch_count": 0,
        "measured_on": "2026-09-01T00:00:00+00:00",
        "completed_utc": "2026-09-01T00:00:00+00:00",
        "baseline_ttft_s": 0.2,
        "baseline_tpot_s": 0.01,
        "sha256": {
            "l4_profile.csv": _digest(profile_path),
            "l4_profile_all_requested.csv": _digest(full_profile_path),
            "l4_profile_raw.csv": _digest(raw_path),
            "l4_telemetry.csv": _digest(telemetry_path),
        },
    }
    if declare_modeled_clocks:
        manifest["modeled_graphics_clocks_mhz"] = modeled_clocks
    (root / "profile_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return profile_path


def _rewrite_bundle_csv(
    root: Path,
    name: str,
    columns: list[str],
    rows: list[dict[str, str]],
    *,
    manifest_updates: dict[str, object] | None = None,
) -> None:
    path = root / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = root / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"][name] = _digest(path)
    if name == "l4_profile_raw.csv":
        manifest["row_count"] = len(rows)
    if manifest_updates:
        manifest.update(manifest_updates)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture(scope="module")
def profile():
    return load_profile(DATA / "l4_profile.csv")


def test_official_azure_subsets_preserve_published_rows() -> None:
    animation = load_workload(DATA / "azure_code_animation.csv")
    evaluation = load_workload(DATA / "azure_code_evaluation.csv")

    assert len(animation) == 20
    assert len(evaluation) == 781
    assert animation[0].original_timestamp == "2023-11-16 18:17:03.9799600"
    assert (animation[0].prompt_tokens, animation[0].output_tokens) == (4808, 10)
    assert evaluation[-1].arrival_time_s <= 300.0
    assert all(
        right.arrival_time_s >= left.arrival_time_s
        for left, right in zip(evaluation, evaluation[1:])
    )


def test_proxy_profile_is_explicit_and_load_normalization_is_fixed(profile) -> None:
    workload = load_workload(DATA / "azure_code_evaluation.csv")
    normalized, dilation = normalize_offered_load(workload, profile)

    if profile.profile_status == "engineering_proxy_not_measured":
        assert not profile.is_measured
        assert not profile.measurement_validated
        assert "not hardware measurements" in str(profile.manifest["warning"])
    elif profile.profile_status == "measured_l4":
        assert profile.is_measured
        assert profile.measurement_validated
        assert profile.manifest["status"] == "complete"
        assert profile.manifest["profile_status"] == "measured_l4"
        assert profile.manifest_sha256 == _digest(DATA / "profile_manifest.json")
        assert len(profile.clock_mhz) >= 4
    else:
        pytest.fail(f"unexpected inference profile status: {profile.profile_status!r}")

    span = workload[-1].arrival_time_s - workload[0].arrival_time_s
    maximum_clock = profile.maximum_clock_mhz
    isolated_service_s = sum(
        request.prompt_tokens / profile.rate("prefill", maximum_clock)
        + request.output_tokens / profile.rate("decode", maximum_clock)
        for request in workload
    )
    expected_dilation = max(1.0, isolated_service_s / span / 0.8)
    assert dilation == pytest.approx(expected_dilation)
    assert normalized[0].arrival_time_s == 0.0
    assert [request.arrival_time_s for request in normalized] == pytest.approx(
        [
            (request.arrival_time_s - workload[0].arrival_time_s) * dilation
            for request in workload
        ]
    )
    assert [request.prompt_tokens for request in normalized] == [
        request.prompt_tokens for request in workload
    ]
    assert [request.output_tokens for request in normalized] == [
        request.output_tokens for request in workload
    ]


def test_measured_clock_reports_hardware_response_but_uses_requested_row(
    profile,
) -> None:
    assert profile.is_measured
    requested = profile.maximum_clock_mhz
    reported = profile.realized_clock_for_level(requested)
    assert requested == pytest.approx(2040.0)
    assert reported == pytest.approx(969.375)

    observations: list[ServingObservation] = []

    def observing_scheduler(observation: ServingObservation):
        observations.append(observation)
        return chunked_prefill_scheduler(observation)

    plant = ServingPlant(
        profile,
        time_step_s=0.1,
        maximum_simulation_time_s=0.4,
    )
    result = simulate(
        (Request(0, 0.0, 1, 100),),
        plant,
        observing_scheduler,
        fixed_clock_controller(requested),
    )

    assert np.all(result.requested_clock_mhz == requested)
    assert np.all(result.realized_clock_mhz == pytest.approx(reported))
    assert observations[0].previous_clock_mhz == pytest.approx(
        profile.minimum_realized_clock_mhz
    )
    assert observations[1].previous_clock_mhz == pytest.approx(reported)
    assert observations[1].realized_clock_levels_mhz == pytest.approx(
        profile.realized_clock_levels_mhz
    )

    decode_increment = np.diff(
        np.concatenate([[0.0], result.cumulative_decode_tokens])
    )
    decode_steps = np.flatnonzero(np.asarray(result.phase) == "decode")
    assert decode_steps.size
    first_decode = int(decode_steps[0])
    assert decode_increment[first_decode] == pytest.approx(
        profile.rate("decode", requested) * plant.time_step_s
    )
    assert result.power_w[first_decode] == pytest.approx(
        profile.power("decode", requested)
    )
    assert result.metrics.power_violation_w == pytest.approx(
        max(0.0, profile.power("decode", requested) - plant.power_limit_w)
    )


def test_proxy_clock_response_is_identity(tmp_path: Path) -> None:
    profile_path = tmp_path / "l4_profile.csv"
    profile_path.write_text(
        "clock_mhz,prefill_tokens_per_s,decode_tokens_per_s,idle_power_w,"
        "prefill_power_w,decode_power_w\n"
        "600,1000,100,20,40,35\n"
        "1200,1500,140,25,50,45\n",
        encoding="utf-8",
    )
    (tmp_path / "profile_manifest.json").write_text(
        json.dumps(
            {
                "profile_status": "engineering_proxy_not_measured",
                "source_label": "test proxy",
                "baseline_ttft_s": 0.2,
                "baseline_tpot_s": 0.01,
            }
        ),
        encoding="utf-8",
    )
    proxy = load_profile(profile_path)
    plant = ServingPlant(
        proxy,
        power_limit_w=1_000.0,
        maximum_simulation_time_s=0.2,
    )
    result = simulate(
        (Request(0, 0.0, 1, 10),),
        plant,
        chunked_prefill_scheduler,
        fixed_clock_controller(1200.0),
    )

    assert proxy.realized_clock_levels_mhz.tolist() == [600.0, 1200.0]
    assert np.all(result.requested_clock_mhz == 1200.0)
    assert np.all(result.realized_clock_mhz == 1200.0)


def test_observation_hides_realized_output_lengths_and_simulation_conserves_tokens(
    profile,
) -> None:
    assert "output_tokens" not in ServingObservation.__dataclass_fields__
    workload = (
        Request(0, 0.0, 64, 20),
        Request(1, 0.2, 96, 12),
        Request(2, 0.5, 48, 8),
    )
    plant = ServingPlant(profile, time_step_s=0.1, maximum_simulation_time_s=30.0)
    seen_observations: list[ServingObservation] = []

    def observing_scheduler(observation: ServingObservation):
        seen_observations.append(observation)
        return chunked_prefill_scheduler(observation)

    result = simulate(workload, plant, observing_scheduler, maximum_clock_controller)

    assert result.metrics.completed_fraction == 1.0
    assert result.cumulative_prefill_tokens[-1] == pytest.approx(
        sum(request.prompt_tokens for request in workload)
    )
    assert result.cumulative_decode_tokens[-1] == pytest.approx(
        sum(request.output_tokens for request in workload)
    )
    assert result.completed_requests[-1] == len(workload)
    assert result.kv_tokens[-1] == pytest.approx(0.0)
    assert result.metrics.kv_violation_tokens == 0.0
    if profile.is_measured:
        assert result.metrics.power_violation_w > 0.0
    else:
        assert result.metrics.power_violation_w == 0.0
    assert result.metrics.thermal_violation_c == 0.0
    assert result.metrics.unfinished_requests_at_end == 0.0
    assert all(
        record.first_token_time_s >= record.arrival_time_s
        and record.completion_time_s >= record.first_token_time_s
        for record in result.request_records
    )
    assert seen_observations
    assert all(not hasattr(observation, "output_tokens") for observation in seen_observations)
    recomputed = compute_metrics(result, plant)
    assert recomputed.energy_j == pytest.approx(result.metrics.energy_j)


def test_chunked_prefill_reduces_decode_stall_on_a_crafted_trace(profile) -> None:
    chunk_tokens = 512.0
    maximum_prefill_rate = profile.rate("prefill", profile.maximum_clock_mhz)
    time_step_s = 1.1 * chunk_tokens / maximum_prefill_rate
    workload = (
        Request(0, 0.0, 64, 100),
        Request(1, 2.0 * time_step_s, 4096, 2),
    )
    plant = ServingPlant(
        profile,
        time_step_s=time_step_s,
        maximum_simulation_time_s=30.0,
    )
    static = simulate(workload, plant, static_batch_scheduler, maximum_clock_controller)
    chunked = simulate(workload, plant, chunked_prefill_scheduler, maximum_clock_controller)

    static_first = static.request_records[0]
    chunked_first = chunked.request_records[0]
    assert chunked_first.completion_time_s < static_first.completion_time_s
    assert chunked.metrics.decode_stall_fraction < static.metrics.decode_stall_fraction
    assert chunked.request_records[1].completion_time_s > static.request_records[1].completion_time_s


def test_raw_profiler_schema_is_reduced_to_clock_medians(tmp_path: Path) -> None:
    raw = tmp_path / "l4_profile.csv"
    raw.write_text(
        "phase,prompt_tokens,output_tokens,concurrency,requested_clock_mhz,"
        "realized_clock_mhz,repeat,ttft_s,tpot_s,total_s,energy_j,mean_power_w,"
        "peak_power_w,peak_temp_c\n"
        "prefill,100,1,1,600,600,0,0.2,0.01,0.2,8,40,42,50\n"
        "decode,100,20,1,600,600,0,0.2,0.02,0.4,14,35,38,48\n"
        "prefill,100,1,1,1200,1198,0,0.1,0.005,0.1,6,60,62,55\n"
        "decode,100,20,1,1200,1198,0,0.1,0.01,0.2,10,50,54,53\n",
        encoding="utf-8",
    )
    parsed = load_profile(raw)

    assert parsed.clock_mhz.tolist() == [600.0, 1198.0]
    assert parsed.prefill_tokens_per_s.tolist() == [500.0, 1000.0]
    assert parsed.decode_tokens_per_s.tolist() == [50.0, 100.0]
    assert np.all(np.diff(parsed.prefill_power_w) > 0.0)


def test_measured_profile_bundle_is_verified_and_fails_closed_when_stale(
    tmp_path: Path,
) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    measured = load_profile(profile_path)

    assert measured.is_measured
    assert measured.measurement_validated
    assert measured.profile_csv_sha256 == _digest(profile_path)
    assert measured.clock_mhz.tolist() == [600.0, 900.0, 1500.0, 1755.0]
    assert measured.manifest["selected_graphics_clocks_mhz"] == [
        600,
        900,
        1200,
        1500,
        1755,
    ]

    profile_path.write_text(
        profile_path.read_text(encoding="utf-8").replace("1000", "1001", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_profile(profile_path)


def test_measured_profile_allows_an_undeclared_modeled_subset(
    tmp_path: Path,
) -> None:
    profile_path = _write_valid_measured_bundle(
        tmp_path,
        declare_modeled_clocks=False,
    )

    measured = load_profile(profile_path)

    assert measured.clock_mhz.tolist() == [600.0, 900.0, 1500.0, 1755.0]
    assert "modeled_graphics_clocks_mhz" not in measured.manifest


def test_measured_profile_rejects_a_mismatched_modeled_clock_declaration(
    tmp_path: Path,
) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    manifest_path = tmp_path / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["modeled_graphics_clocks_mhz"] = [600, 900, 1200, 1755]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="modeled_graphics_clocks_mhz"):
        load_profile(profile_path)


def test_measured_profile_rejects_an_aggregate_clock_outside_the_sweep(
    tmp_path: Path,
) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    rows = list(csv.DictReader(profile_path.open(encoding="utf-8")))
    rows[1]["clock_mhz"] = "1000"
    _rewrite_bundle_csv(
        tmp_path,
        "l4_profile.csv",
        AGGREGATE_COLUMNS,
        rows,
        manifest_updates={
            "modeled_graphics_clocks_mhz": [600, 1000, 1500, 1755]
        },
    )

    with pytest.raises(ValueError, match="ordered subset"):
        load_profile(profile_path)


def test_measured_profile_rejects_fewer_than_four_modeled_clocks(
    tmp_path: Path,
) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    rows = list(csv.DictReader(profile_path.open(encoding="utf-8")))[:3]
    _rewrite_bundle_csv(
        tmp_path,
        "l4_profile.csv",
        AGGREGATE_COLUMNS,
        rows,
        manifest_updates={"modeled_graphics_clocks_mhz": [600, 900, 1500]},
    )

    with pytest.raises(ValueError, match="at least four ordered clock levels"):
        load_profile(profile_path)


def test_measured_profile_requires_the_full_raw_sweep_matrix(
    tmp_path: Path,
) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    raw_path = tmp_path / "l4_profile_raw.csv"
    rows = list(csv.DictReader(raw_path.open(encoding="utf-8")))
    rows = [
        row
        for row in rows
        if not (
            row["requested_clock_mhz"] == "1200"
            and row["phase"] == "prefill"
            and row["prompt_tokens"] == "128"
            and row["concurrency"] == "1"
            and row["repeat"] == "0"
        )
    ]
    _rewrite_bundle_csv(
        tmp_path,
        "l4_profile_raw.csv",
        RAW_COLUMNS,
        rows,
    )

    with pytest.raises(ValueError, match="raw profile repetition matrix is incomplete"):
        load_profile(profile_path)


def test_measured_profile_validates_telemetry_fallback_batch_count(
    tmp_path: Path,
) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    raw_path = tmp_path / "l4_profile_raw.csv"
    rows = list(csv.DictReader(raw_path.open(encoding="utf-8")))
    for row in rows:
        if (
            row["requested_clock_mhz"] == "600"
            and row["phase"] == "prefill"
            and row["prompt_tokens"] == "128"
            and row["concurrency"] == "4"
            and row["repeat"] == "0"
        ):
            row["telemetry_fallback_used"] = "True"
    _rewrite_bundle_csv(
        tmp_path,
        "l4_profile_raw.csv",
        RAW_COLUMNS,
        rows,
        manifest_updates={"telemetry_fallback_batch_count": 1},
    )

    assert load_profile(profile_path).is_measured

    manifest_path = tmp_path / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["telemetry_fallback_batch_count"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="telemetry_fallback_batch_count"):
        load_profile(profile_path)


def test_realized_clock_validation_is_batch_balanced_and_modeled_only(
    tmp_path: Path,
) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    raw_path = tmp_path / "l4_profile_raw.csv"
    rows = list(csv.DictReader(raw_path.open(encoding="utf-8")))
    for row in rows:
        if row["requested_clock_mhz"] == "1200":
            # This excluded sweep point is deliberately nonmonotone.
            row["realized_clock_mhz"] = "700"
        elif (
            row["requested_clock_mhz"] == "900"
            and row["concurrency"] == "8"
        ):
            # Per-request weighting would make 1,600 MHz the median at this
            # level. One observation per condition/repeat batch keeps it 900.
            row["realized_clock_mhz"] = "1600"
    _rewrite_bundle_csv(
        tmp_path,
        "l4_profile_raw.csv",
        RAW_COLUMNS,
        rows,
    )

    measured = load_profile(profile_path)

    assert measured.is_measured
    assert measured.clock_mhz.tolist() == [600.0, 900.0, 1500.0, 1755.0]


def test_measured_profile_rejects_nonmonotone_modeled_realized_clocks(
    tmp_path: Path,
) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    raw_path = tmp_path / "l4_profile_raw.csv"
    rows = list(csv.DictReader(raw_path.open(encoding="utf-8")))
    for row in rows:
        if row["requested_clock_mhz"] == "1500":
            row["realized_clock_mhz"] = "800"
    _rewrite_bundle_csv(
        tmp_path,
        "l4_profile_raw.csv",
        RAW_COLUMNS,
        rows,
    )

    with pytest.raises(ValueError, match="batch-balanced aggregation"):
        load_profile(profile_path)


def test_measured_profile_rejects_incomplete_or_unpinned_manifest(
    tmp_path: Path,
) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    manifest_path = tmp_path / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "profiling"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="status must be 'complete'"):
        load_profile(profile_path)

    manifest["status"] = "complete"
    manifest["vllm_image_digest"] = "external-server-not-inspected"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved vLLM sha256"):
        load_profile(profile_path)


@pytest.mark.parametrize(
    ("manifest_update", "message"),
    [
        ({"schema_version": 1}, "unsupported measured profile schema_version"),
        (
            {"vllm_prefix_caching_enabled": 0},
            "explicitly disable vLLM prefix caching",
        ),
        (
            {"vllm_server_arguments": []},
            "--no-enable-prefix-caching",
        ),
    ],
)
def test_measured_profile_requires_schema_two_and_disabled_prefix_caching(
    tmp_path: Path,
    manifest_update: dict[str, object],
    message: str,
) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    manifest_path = tmp_path / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(manifest_update)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_profile(profile_path)


def test_measured_profile_requires_the_full_aggregate_artifact(
    tmp_path: Path,
) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    manifest_path = tmp_path / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["sha256"]["l4_profile_all_requested.csv"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="l4_profile_all_requested.csv"):
        load_profile(profile_path)


def test_measured_profile_checks_every_safe_manifest_checksum(
    tmp_path: Path,
) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    log_path = tmp_path / "vllm.log"
    log_path.write_text("complete\n", encoding="utf-8")
    manifest_path = tmp_path / "profile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"]["vllm.log"] = _digest(log_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert load_profile(profile_path).is_measured

    log_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch for vllm.log"):
        load_profile(profile_path)


def test_modeled_rows_must_match_the_full_aggregate_artifact(
    tmp_path: Path,
) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    full_profile_path = tmp_path / "l4_profile_all_requested.csv"
    rows = list(csv.DictReader(full_profile_path.open(encoding="utf-8")))
    rows[3]["decode_tokens_per_s"] = "999"
    _rewrite_bundle_csv(
        tmp_path,
        "l4_profile_all_requested.csv",
        AGGREGATE_COLUMNS,
        rows,
    )

    with pytest.raises(ValueError, match="modeled aggregate rows"):
        load_profile(profile_path)


def test_profile_rejects_nonmonotone_measured_service_curves(tmp_path: Path) -> None:
    profile_path = _write_valid_measured_bundle(tmp_path)
    rows = list(csv.DictReader(profile_path.open(encoding="utf-8")))
    rows[2]["decode_tokens_per_s"] = rows[1]["decode_tokens_per_s"]
    full_profile_path = tmp_path / "l4_profile_all_requested.csv"
    full_rows = list(csv.DictReader(full_profile_path.open(encoding="utf-8")))
    full_rows[3]["decode_tokens_per_s"] = rows[2]["decode_tokens_per_s"]
    _rewrite_bundle_csv(
        tmp_path,
        "l4_profile_all_requested.csv",
        AGGREGATE_COLUMNS,
        full_rows,
    )
    _rewrite_bundle_csv(
        tmp_path,
        "l4_profile.csv",
        AGGREGATE_COLUMNS,
        rows,
    )

    with pytest.raises(ValueError, match="decode service rate must increase"):
        load_profile(profile_path)


def test_result_dictionary_uses_the_replay_contract(profile) -> None:
    plant = ServingPlant(profile, time_step_s=0.1, maximum_simulation_time_s=10.0)
    result = simulate(
        (Request(0, 0.0, 32, 4),),
        plant,
        chunked_prefill_scheduler,
        maximum_clock_controller,
    )
    payload = result.as_dict()

    required = {
        "time_s",
        "prefill_queue",
        "decode_active",
        "completed_requests",
        "kv_tokens",
        "temperature_c",
        "power_w",
        "requested_clock_mhz",
        "realized_clock_mhz",
        "energy_j",
        "requests",
    }
    assert required.issubset(payload)
    assert np.all(np.diff(payload["time_s"]) > 0.0)


def test_sampled_reactive_governor_only_changes_at_one_second_boundaries(profile) -> None:
    workload = (
        Request(0, 0.0, 2048, 80),
        Request(1, 0.1, 2048, 80),
        Request(2, 1.5, 128, 4),
    )
    plant = ServingPlant(profile, time_step_s=0.1, maximum_simulation_time_s=20.0)
    sampled = sample_and_hold_clock_controller(
        reactive_clock_controller,
        period_s=1.0,
    )
    result = simulate(
        workload,
        plant,
        chunked_prefill_scheduler,
        sampled,
    )
    changes = np.flatnonzero(np.diff(result.requested_clock_mhz) != 0.0) + 1
    observation_times = result.time_s[changes] - plant.time_step_s

    assert sampled.update_times_s
    assert all(time == pytest.approx(round(time)) for time in sampled.update_times_s)
    assert all(time == pytest.approx(round(time)) for time in observation_times)


def test_observation_power_is_the_previous_completed_period(profile) -> None:
    observations: list[ServingObservation] = []

    def scheduler(observation: ServingObservation):
        observations.append(observation)
        return chunked_prefill_scheduler(observation)

    plant = ServingPlant(profile, time_step_s=0.1, maximum_simulation_time_s=5.0)
    result = simulate(
        (Request(0, 0.0, 256, 8),),
        plant,
        scheduler,
        maximum_clock_controller,
    )

    assert observations[0].previous_power_w == pytest.approx(
        profile.power("idle", profile.minimum_clock_mhz)
    )
    for index, observation in enumerate(observations[1:]):
        assert observation.previous_power_w == pytest.approx(result.power_w[index])


def test_prefill_cap_binds_and_unused_interval_serves_decode(profile) -> None:
    chunk_tokens = 512.0
    maximum_prefill_rate = profile.rate("prefill", profile.maximum_clock_mhz)
    time_step_s = 1.1 * chunk_tokens / maximum_prefill_rate
    workload = (
        Request(0, 0.0, 64, 100),
        Request(1, time_step_s, 5000, 2),
    )
    plant = ServingPlant(
        profile,
        time_step_s=time_step_s,
        maximum_simulation_time_s=20.0,
    )
    result = simulate(
        workload,
        plant,
        chunked_prefill_scheduler,
        maximum_clock_controller,
    )
    prefill_increment = np.diff(
        np.concatenate([[0.0], result.cumulative_prefill_tokens])
    )
    decode_increment = np.diff(
        np.concatenate([[0.0], result.cumulative_decode_tokens])
    )
    binding = np.flatnonzero(np.isclose(prefill_increment, chunk_tokens))

    assert maximum_prefill_rate * time_step_s > chunk_tokens
    assert binding.size > 0
    assert np.max(prefill_increment) <= chunk_tokens + 1e-9
    assert np.any(decode_increment[binding] > 0.0)
    assert any(result.phase[index] == "interleaved" for index in binding)
