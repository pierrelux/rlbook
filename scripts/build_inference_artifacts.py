#!/usr/bin/env python3
"""Ingest the official trace and build deterministic textbook artifacts.

This script is an explicit maintenance command.  MyST pages only read its
committed outputs and never invoke it during a normal book build.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from inference_control import build_textbook_results  # noqa: E402
from inference_replay import render_static_figure  # noqa: E402
from inference_serving import (  # noqa: E402
    PerformanceProfile,
    Request,
    ServingPlant,
    load_profile,
    load_workload,
    normalize_offered_load,
)


DATA = ROOT / "data" / "inference_serving"
ARTIFACTS = ROOT / "artifacts" / "inference_serving"
OFFICIAL_TRACE_SHA256 = "54e9a6d2a4bd06ba1e060304b900abbc74cbea53de96506e60fe5bb4f2277fb6"


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.strip())


def ingest_azure_trace(source: Path) -> None:
    """Write only the approved row subsets after checking the official file."""

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != OFFICIAL_TRACE_SHA256:
        raise ValueError(
            "the Azure code trace checksum does not match the documented official download"
        )
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    expected = ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]
    if fieldnames != expected:
        raise ValueError(f"unexpected Azure trace columns: {fieldnames}")
    start = _timestamp(rows[0]["TIMESTAMP"])
    evaluation = [
        row
        for row in rows
        if (_timestamp(row["TIMESTAMP"]) - start).total_seconds() <= 300.0
    ]
    subsets = {
        "azure_code_animation.csv": rows[:20],
        "azure_code_evaluation.csv": evaluation,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    for name, subset in subsets.items():
        destination = DATA / name
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=expected, lineterminator="\n")
            writer.writeheader()
            writer.writerows(subset)


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _metric_rows(view: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    controllers = view.get("controllers", {})
    assert isinstance(controllers, dict)
    for key, payload in controllers.items():
        assert isinstance(payload, dict)
        metrics = payload.get("metrics", {})
        assert isinstance(metrics, dict)
        diagnostics = payload.get("mpc_diagnostics", {})
        assert isinstance(diagnostics, dict)
        rows.append(
            {
                "controller": key,
                "profile_status": payload.get("profile_status", ""),
                "workload_checksum": payload.get("workload_checksum", ""),
                **metrics,
                **diagnostics,
            }
        )
    return rows


def plant_from_profile_manifest(
    profile: PerformanceProfile,
    *,
    time_step_s: float = 0.1,
) -> ServingPlant:
    """Construct the plant, requiring thermal fits for a measured profile."""

    manifest = profile.manifest
    thermal_fields = (
        "thermal_time_constant_s",
        "thermal_resistance_c_per_w",
        "fitted_ambient_temperature_c",
    )
    missing = [field for field in thermal_fields if field not in manifest]
    is_proxy = profile.profile_status == "engineering_proxy_not_measured"
    if profile.profile_status == "measured_l4" and not profile.is_measured:
        raise ValueError(
            "measured_l4 plant construction requires a completely validated profile bundle"
        )
    if missing and not is_proxy:
        raise ValueError(
            "a measured profile manifest must contain fitted thermal parameters: "
            + ", ".join(missing)
        )
    thermal_time_constant = float(manifest.get("thermal_time_constant_s", 35.0))
    thermal_resistance = float(manifest.get("thermal_resistance_c_per_w", 0.55))
    ambient_temperature = float(manifest.get("fitted_ambient_temperature_c", 25.0))
    plant = ServingPlant(
        profile=profile,
        time_step_s=time_step_s,
        ambient_temperature_c=ambient_temperature,
        thermal_time_constant_s=thermal_time_constant,
        thermal_resistance_c_per_w=thermal_resistance,
        power_limit_w=float(manifest.get("experiment_power_limit_w", 64.8)),
        thermal_limit_c=float(manifest.get("thermal_limit_c", 75.0)),
    )
    plant.validate()
    return plant


def build_artifacts(*, quick: bool = False) -> dict[str, object]:
    profile = load_profile(DATA / "l4_profile.csv")
    animation_raw = load_workload(DATA / "azure_code_animation.csv")
    evaluation_raw = load_workload(DATA / "azure_code_evaluation.csv")
    evaluation, dilation = normalize_offered_load(evaluation_raw, profile)
    animation = tuple(
        replace(request, arrival_time_s=request.arrival_time_s * dilation)
        for request in animation_raw
    )
    plant = plant_from_profile_manifest(
        profile,
        # At maximum clock the 512-token cap binds. Any interval left after a
        # chunk is returned to decode rather than discarded.
        time_step_s=0.1,
    )
    protocol = (
        {
            "fqi_transitions": 2_000,
            "fqi_sweeps": 5,
            "fqi_trees": 25,
            "evaluation_episodes": 100,
            "evaluation_horizon_steps": 100,
        }
        if quick
        else {
            "fqi_transitions": 50_000,
            "fqi_sweeps": 50,
            "fqi_trees": 200,
            "evaluation_episodes": 2_000,
            "evaluation_horizon_steps": 300,
        }
    )
    results = build_textbook_results(
        animation,
        evaluation,
        plant,
        load_dilation=dilation,
        **protocol,
    )
    results["metadata"].update(
        {
            "azure_full_trace_sha256": OFFICIAL_TRACE_SHA256,
            "azure_animation_rows": len(animation_raw),
            "azure_evaluation_rows": len(evaluation_raw),
            "artifact_protocol": "quick" if quick else "full",
            "profile_csv_path": "data/inference_serving/l4_profile.csv",
            "profile_manifest_path": "data/inference_serving/profile_manifest.json",
            "profile_csv_sha256": profile.profile_csv_sha256,
        }
    )
    clean = _finite_json(results)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "textbook_results.json").write_text(
        json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    open_loop = clean["open_loop"]
    mpc = clean["mpc"]
    scheduling = clean["scheduling"]
    fqi = clean["fqi"]
    assert isinstance(open_loop, dict)
    assert isinstance(mpc, dict)
    assert isinstance(scheduling, dict)
    assert isinstance(fqi, dict)
    _write_metrics(ARTIFACTS / "metrics_open_loop.csv", _metric_rows(open_loop))
    _write_metrics(ARTIFACTS / "metrics_mpc.csv", _metric_rows(mpc))
    _write_metrics(
        ARTIFACTS / "metrics_dp.csv",
        [
            {
                "bellman_residual": scheduling["bellman_residual"],
                "iterations": scheduling["iterations"],
                "profile_status": profile.profile_status,
                "arrival_probability": scheduling["calibration"][
                    "arrival_probability"
                ],
                "prefill_completion_probability": scheduling["calibration"][
                    "prefill_completion_probability"
                ],
                "decode_completion_probability": scheduling["calibration"][
                    "decode_completion_probability"
                ],
                "prefill_energy_j": scheduling["calibration"]["action_energy_j"][0],
                "decode_energy_j": scheduling["calibration"]["action_energy_j"][1],
                "idle_energy_j": scheduling["calibration"]["action_energy_j"][2],
            }
        ],
    )
    fqi_metrics = fqi["metrics"]
    assert isinstance(fqi_metrics, dict)
    _write_metrics(
        ARTIFACTS / "metrics_fqi.csv",
        [
            {"controller": name, **values}
            for name, values in fqi_metrics.items()
            if isinstance(values, dict)
        ],
    )
    static_directory = ROOT / "_static" / "inference_serving"
    static_directory.mkdir(parents=True, exist_ok=True)
    for view, filename in (
        ("modeling", "modeling.svg"),
        ("open_loop", "open-loop.svg"),
        ("mpc", "mpc.svg"),
        ("scheduling", "scheduling.svg"),
        ("fqi", "fqi.svg"),
    ):
        figure = render_static_figure(
            ARTIFACTS / "textbook_results.json",
            static_directory / filename,
            view=view,
        )
        plt.close(figure)
    return clean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--azure-source",
        type=Path,
        help="official Azure code trace CSV to verify and subset before building",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="create the two attributed trace subsets without running experiments",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="build a small smoke artifact that is not suitable for publication",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.azure_source is not None:
        ingest_azure_trace(args.azure_source)
    if args.ingest_only:
        if args.azure_source is None:
            raise SystemExit("--ingest-only requires --azure-source")
        return
    required = [
        DATA / "azure_code_animation.csv",
        DATA / "azure_code_evaluation.csv",
    ]
    if any(not path.exists() for path in required):
        raise SystemExit(
            "trace subsets are missing; pass --azure-source with the official download"
        )
    build_artifacts(quick=args.quick)


if __name__ == "__main__":
    main()
