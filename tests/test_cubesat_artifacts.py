"""Cross-format checks for the committed CubeSat teaching evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from cubesat_replay import render_cubesat_replay  # noqa: E402


ARTIFACT_DIRECTORY = ROOT / "artifacts" / "cubesat"
STATIC_DIRECTORY = ROOT / "_static" / "cubesat"
RUN_NAMES = ("nominal_linear", "nonlinear_replay")


def _load_artifact() -> dict:
    return json.loads(
        (ARTIFACT_DIRECTORY / "textbook_results.json").read_text(
            encoding="utf-8"
        )
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_artifact_passes_the_real_browser_renderer() -> None:
    artifact_path = ARTIFACT_DIRECTORY / "textbook_results.json"
    rendered = render_cubesat_replay(artifact_path, replay_id="committed-cubesat")

    assert '<section id="committed-cubesat-' in rendered
    assert rendered.count('data-orbit="nominal_linear"') == 1
    assert rendered.count('data-orbit="nonlinear_replay"') == 1
    assert "complete open-loop drag plan is visible from the start" in rendered
    assert "fetch(" not in rendered


def test_json_csv_npz_and_generated_prose_share_one_metric_fingerprint() -> None:
    artifact = _load_artifact()
    fingerprint = artifact["metrics_sha256"]
    assert fingerprint == _canonical_sha256(artifact["metrics"])

    with (ARTIFACT_DIRECTORY / "metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert metric_rows
    assert {row["metrics_sha256"] for row in metric_rows} == {fingerprint}

    expected_leaves: dict[tuple[str, str], object] = {}

    def collect(path: tuple[str, ...], value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                collect((*path, key), item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                collect((*path, str(index)), item)
        elif path[-1].isdigit():
            expected_leaves[(".".join(path[:-1]), path[-1])] = value
        else:
            expected_leaves[(".".join(path), "")] = value

    collect((), artifact["metrics"])
    csv_leaves = {
        (row["metric"], row["component"]): row["value"] for row in metric_rows
    }
    assert csv_leaves.keys() == expected_leaves.keys()
    for key, expected in expected_leaves.items():
        recorded = csv_leaves[key]
        if isinstance(expected, bool):
            assert recorded == ("true" if expected else "false")
        elif isinstance(expected, (int, float)):
            assert float(recorded) == pytest.approx(expected, rel=0.0, abs=0.0)
        else:
            assert recorded == expected

    with np.load(ARTIFACT_DIRECTORY / "trajectories.npz") as arrays:
        assert arrays["metrics_sha256"].item() == fingerprint
        assert json.loads(arrays["metrics_json"].item()) == artifact["metrics"]

    results = (ARTIFACT_DIRECTORY / "results.md").read_text(encoding="utf-8")
    assert f"<!-- metrics-sha256: {fingerprint} -->" in results
    replay_error = artifact["metrics"]["nonlinear_replay"][
        "max_abs_final_cyclic_gap_error_deg"
    ]
    assert f"${replay_error:.6f}^\\circ$" in results

    for path in (
        STATIC_DIRECTORY / "differential-drag.svg",
        STATIC_DIRECTORY / "differential-drag.pdf",
    ):
        assert fingerprint.encode("ascii") in path.read_bytes()


def test_plan_and_trajectory_arrays_match_every_serialized_view() -> None:
    artifact = _load_artifact()
    plan = artifact["plan"]
    command = np.asarray(plan["U"], dtype=float)
    assert command.shape == (3, 180)
    np.testing.assert_allclose(
        command.sum(axis=1),
        plan["equivalent_high_drag_days"],
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        command.mean(axis=1), plan["duty_fraction"], rtol=0.0, atol=2e-14
    )
    np.testing.assert_allclose(
        plan["equivalent_high_drag_days"],
        artifact["metrics"]["plan"]["equivalent_high_drag_days"],
        rtol=0.0,
        atol=0.0,
    )

    with (ARTIFACT_DIRECTORY / "open_loop_plan.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        plan_rows = list(csv.DictReader(handle))
    assert len(plan_rows) == 180
    csv_command = np.asarray(
        [
            [
                float(row["leader_high_drag_fraction"]),
                float(row["follower_1_high_drag_fraction"]),
                float(row["follower_2_high_drag_fraction"]),
            ]
            for row in plan_rows
        ]
    ).T
    np.testing.assert_allclose(csv_command, command, rtol=0.0, atol=0.0)
    assert {
        row["plan_identity_sha256"] for row in plan_rows
    } == {plan["identity_sha256"]}

    vector_fields = (
        "phase_deg",
        "cyclic_gap_deg",
        "cyclic_gap_error_deg",
        "relative_rate_deg_per_day",
        "altitude_km",
        "extra_altitude_loss_km",
    )
    with np.load(ARTIFACT_DIRECTORY / "trajectories.npz") as arrays:
        np.testing.assert_allclose(arrays["plan_U"], command, rtol=0.0, atol=0.0)
        for run_name in RUN_NAMES:
            frames = artifact["runs"][run_name]["frames"]
            assert [frame["day"] for frame in frames] == list(range(181))
            np.testing.assert_array_equal(
                arrays[f"{run_name}_day"], np.arange(181)
            )
            for field in vector_fields:
                expected = np.asarray([frame[field] for frame in frames])
                np.testing.assert_allclose(
                    arrays[f"{run_name}_{field}"],
                    expected,
                    rtol=0.0,
                    atol=0.0,
                )

        assert arrays["nonlinear_replay_hourly_time_days"].shape == (4321,)
        assert arrays["nonlinear_replay_hourly_phase_deg"].shape == (4321, 3)
        density_min = min(
            arrays["nonlinear_replay_hourly_density_kg_m3"].min(),
            arrays["nonlinear_replay_hourly_reference_density_kg_m3"].min(),
        )
        density_max = max(
            arrays["nonlinear_replay_hourly_density_kg_m3"].max(),
            arrays["nonlinear_replay_hourly_reference_density_kg_m3"].max(),
        )
    replay_metrics = artifact["metrics"]["nonlinear_replay"]
    assert density_min == pytest.approx(replay_metrics["density_min_kg_m3"])
    assert density_max == pytest.approx(replay_metrics["density_max_kg_m3"])

    for run_name in RUN_NAMES:
        final = artifact["runs"][run_name]["frames"][-1]
        metrics = artifact["metrics"][run_name]
        np.testing.assert_allclose(
            final["cyclic_gap_deg"], metrics["final_cyclic_gap_deg"]
        )
        np.testing.assert_allclose(
            final["cyclic_gap_error_deg"], metrics["final_cyclic_gap_error_deg"]
        )
        np.testing.assert_allclose(
            final["altitude_km"], metrics["final_altitude_km"]
        )


def test_manifest_hashes_resolve_and_static_companion_is_registered() -> None:
    manifest = json.loads(
        (ARTIFACT_DIRECTORY / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "complete"
    assert all(manifest["protocol"]["validation"].values())
    for relative, expected in manifest["inputs"].items():
        assert _sha256(ROOT / relative) == expected
    for relative, expected in manifest["outputs"].items():
        assert _sha256(ROOT / relative) == expected

    plugin = (ROOT / "plugins" / "pdf-static-parity.mjs").read_text(
        encoding="utf-8"
    )
    assert re.search(
        r"\^_static\\/cubesat\\/differential-drag\\\.svg\$", plugin
    )
