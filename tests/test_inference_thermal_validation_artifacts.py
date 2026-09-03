"""Checks for the fixed measured L4 thermal-validation artifacts."""

from __future__ import annotations

import ast
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fit_inference_thermal as base_fit  # noqa: E402
import fit_inference_thermal_phase as phase_fit  # noqa: E402
from build_inference_thermal_validation_artifacts import (  # noqa: E402
    FIGURE_DESCRIPTION,
    MODEL_ORDER,
    SOURCE_DIRECTORY,
    build_artifacts,
    load_validation_evidence,
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def evidence():
    return load_validation_evidence()


def test_builder_uses_fixed_training_only_parameters_without_refitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("artifact generation must not fit a model")

    monkeypatch.setattr(base_fit, "fit_one_state_rc", fail)
    monkeypatch.setattr(phase_fit, "fit_one_state_rc", fail)
    monkeypatch.setattr(phase_fit, "fit_one_state_phase_gain", fail)
    loaded = load_validation_evidence()

    assert loaded.report["pre_registration"][
        "validation_used_for_fit_or_model_selection"
    ] is False
    assert tuple(loaded.fixed_parameters) == MODEL_ORDER
    assert {item.workload_phase for item in loaded.series} == {"decode", "prefill"}
    assert all(item.time_s.size == 60 for item in loaded.series)


def test_reconstructed_held_out_metrics_match_the_committed_report(evidence) -> None:
    report = evidence.report
    expected = {
        "power_only_one_state_rc": 1.4272989201154345,
        "phase_gain_one_state_rc": 0.8696553321716621,
    }
    for model_name in MODEL_ORDER:
        residuals = np.concatenate(
            [
                item.observed_temperature_c[1:]
                - item.predictions_c[model_name][1:]
                for item in evidence.series
            ]
        )
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        assert rmse == pytest.approx(expected[model_name], abs=1e-12)
        assert rmse == pytest.approx(
            report["models"][model_name]["validation_evaluation"]["aggregate"][
                "rmse_c"
            ],
            abs=1e-12,
        )

    assert evidence.acceptance_metrics_c == pytest.approx(
        {
            "held_out_rmse_c": 0.8696553321716621,
            "worst_error_c": 2.7310039794374816,
            "phase_contrast_error_c": 1.431289590989124,
            "threshold_c": 1.0,
        },
        abs=1e-12,
    )
    assert report["observed_validation_contrast"]["one_second_aligned"][
        "prefill_minus_decode_peak_rise_c"
    ] == pytest.approx(2.0)
    assert report["models"]["phase_gain_one_state_rc"]["validation_evaluation"][
        "matched_55w_decode_prefill"
    ]["predicted_prefill_minus_decode_peak_rise_c"] == pytest.approx(
        3.431289590989124
    )


def test_committed_csv_contains_only_the_untouched_pair_and_both_models(
    evidence,
) -> None:
    path = ROOT / "artifacts" / "inference_serving" / "thermal_phase_validation.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 120
    assert {row["split"] for row in rows} == {"validation"}
    assert {row["workload_phase"] for row in rows} == {"decode", "prefill"}
    assert {row["block_id"] for row in rows} == {
        "phase_validation_pulse_00",
        "phase_validation_pulse_01",
    }
    assert all(float(row["requested_power_limit_w"]) == 55.0 for row in rows)
    for item in evidence.series:
        selected = [row for row in rows if row["block_id"] == item.block_id]
        np.testing.assert_allclose(
            [float(row["observed_temperature_c"]) for row in selected],
            item.observed_temperature_c,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            [float(row["power_only_predicted_temperature_c"]) for row in selected],
            item.predictions_c["power_only_one_state_rc"],
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            [float(row["phase_gain_predicted_temperature_c"]) for row in selected],
            item.predictions_c["phase_gain_one_state_rc"],
            rtol=0.0,
            atol=1e-14,
        )
        assert selected[0]["preceding_interval_mean_power_w"] == ""
        np.testing.assert_allclose(
            [float(row["preceding_interval_mean_power_w"]) for row in selected[1:]],
            item.measured_power_w,
            rtol=0.0,
            atol=1e-14,
        )


def test_summary_states_the_improvement_rejection_balance_and_scope() -> None:
    result = (
        ROOT / "artifacts" / "inference_serving" / "thermal_phase_result.md"
    ).read_text(encoding="utf-8")

    assert "power-only model has an RMSE of 1.43 degrees C" in result
    assert "reduces this error to 0.87 degrees C" in result
    assert "worst trajectory error of 2.73 degrees C" in result
    assert "phase-contrast error of 1.43 degrees C" in result
    assert "below 1 degree C" in result
    assert "exceeds the decode rise by 2.00 degrees C" in result
    assert "predicts a difference of 3.43 degrees C" in result
    assert "0.132 W less" in result
    assert "6.66 J less" in result
    assert "one NVIDIA L4" in result
    assert "Qwen2.5-7B-Instruct" in result
    assert "vLLM 0.28.0" in result
    assert "do not establish a hardware-safety model" in result


def test_static_vectors_have_accessibility_and_caption_metadata() -> None:
    static = ROOT / "_static" / "inference_serving"
    svg = (static / "thermal-phase-validation.svg").read_text(encoding="utf-8")
    pdf = (static / "thermal-phase-validation.pdf").read_bytes()

    assert 'role="img"' in svg
    assert 'aria-labelledby="thermal-phase-validation-title thermal-phase-validation-desc"' in svg
    assert '<title id="thermal-phase-validation-title">' in svg
    assert '<desc id="thermal-phase-validation-desc">' in svg
    for phrase in (
        "lowers held-out RMSE from 1.43 to 0.87 degrees C",
        "worst error is 2.73 degrees C",
        "phase-contrast error is 1.43 degrees C",
        "observed aligned prefill-minus-decode peak-rise contrast is 2.00 degrees C",
        "model predicts 3.43 degrees C",
        "-0.132 W",
        "-6.66 J",
        "not a hardware-safety model",
    ):
        assert phrase in FIGURE_DESCRIPTION
        assert phrase.replace("&", "&amp;") in svg
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 20_000


def test_manifest_hashes_resolve_and_record_the_rejection() -> None:
    path = (
        ROOT
        / "artifacts"
        / "inference_serving"
        / "thermal_phase_validation_manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["source_experiment"] == SOURCE_DIRECTORY.name
    assert manifest["source_scope"]["validation_used_for_fit_or_model_selection"] is False
    assert manifest["acceptance"]["verdict"] == "rejected"
    assert manifest["acceptance"]["held_out_rmse_c"] < 1.0
    assert manifest["acceptance"]["worst_error_c"] > 1.0
    assert manifest["acceptance"]["phase_contrast_error_c"] > 1.0
    assert manifest["matched_electrical_input"][
        "prefill_minus_decode_mean_power_w"
    ] == pytest.approx(-0.13225922291921677)
    for relative, expected in manifest["inputs"].items():
        assert _sha(ROOT / relative) == expected
    for relative, expected in manifest["outputs"].items():
        assert _sha(ROOT / relative) == expected


def test_builder_outputs_are_byte_deterministic(tmp_path: Path) -> None:
    first_artifacts = tmp_path / "first" / "artifacts"
    first_static = tmp_path / "first" / "static"
    second_artifacts = tmp_path / "second" / "artifacts"
    second_static = tmp_path / "second" / "static"
    first = build_artifacts(
        artifact_directory=first_artifacts,
        static_directory=first_static,
    )
    second = build_artifacts(
        artifact_directory=second_artifacts,
        static_directory=second_static,
    )

    assert [path.name for path in first] == [path.name for path in second]
    assert [_sha(path) for path in first] == [_sha(path) for path in second]


def test_builder_source_has_no_fit_calls_or_network_clients() -> None:
    source_path = ROOT / "scripts" / "build_inference_thermal_validation_artifacts.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert not imported_roots.intersection({"requests", "urllib", "httpx", "socket"})
    assert "fit_one_state_rc" not in called_names
    assert "fit_one_state_phase_gain" not in called_names
    for forbidden in ("http://", "https://", "urlopen(", "requests.get(", "fetch("):
        assert forbidden not in source
