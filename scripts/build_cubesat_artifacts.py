#!/usr/bin/env python3
"""Build deterministic evidence for the CubeSat differential-drag example.

This is a maintenance command, not part of an ordinary MyST build.  The book
and its browser replay read the committed files produced here; neither solves
the planning problem nor advances the nonlinear orbital model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import matplotlib


matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from cubesat_differential_drag import (  # noqa: E402
    SATELLITE_NAMES,
    STATE_COMPONENTS,
    assert_valid_scenario,
    metrics_as_dict,
    run_scenario,
    validation_checks,
)


SCHEMA_VERSION = 1
RUN_NAMES = ("nominal_linear", "nonlinear_replay")
SPACECRAFT = ("Leader", "Follower 1", "Follower 2")
GAP_LABELS = (
    "Leader → Follower 1",
    "Follower 1 → Follower 2",
    "Follower 2 → Leader",
)

ARTIFACT_DIRECTORY = ROOT / "artifacts" / "cubesat"
STATIC_DIRECTORY = ROOT / "_static" / "cubesat"
FONT_DIRECTORY = ROOT / "_static" / "battery" / "fonts"
FONT_FILES = (
    FONT_DIRECTORY / "IBMPlexSans-Regular.ttf",
    FONT_DIRECTORY / "IBMPlexSans-SemiBold.ttf",
    FONT_DIRECTORY / "IBMPlexMono-Regular.ttf",
    FONT_DIRECTORY / "Newsreader.ttf",
)
INPUT_FILES = (
    ROOT / "code" / "cubesat_differential_drag.py",
    ROOT / "code" / "cubesat_replay.py",
    ROOT / "scripts" / "build_cubesat_artifacts.py",
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
    *FONT_FILES,
    FONT_DIRECTORY / "IBM-Plex-LICENSE.txt",
    FONT_DIRECTORY / "Newsreader-OFL.txt",
    FONT_DIRECTORY / "README.md",
)

# Match the deterministic battery figure's paper, typography, and line work.
PAPER = "#F6F7F4"
INK = "#1B2430"
TEAL = "#2F6F8F"
MUTED = "#5C6874"
RULE = "#D2D9D7"
SUCCESS = "#2E7D5B"
CAUTION = "#B8860B"
WITHDRAWN = "#A83A32"
SATELLITE_COLORS = (INK, "#0072B2", "#D55E00")
GAP_COLORS = ("#0072B2", "#E69F00", "#009E73")

FIGURE_STYLE = {
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "font.family": "IBM Plex Sans",
    "font.sans-serif": ["IBM Plex Sans"],
    "font.monospace": ["IBM Plex Mono"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.65,
    "lines.linewidth": 1.35,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "svg.fonttype": "path",
    "svg.hashsalt": "cubesat-differential-drag-v1",
}


def _jsonable(value: Any) -> Any:
    """Convert dataclass and NumPy values to strict JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return path


def _register_fonts() -> None:
    missing = [path for path in FONT_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing vendored figure fonts: "
            + ", ".join(str(path) for path in missing)
        )
    for path in FONT_FILES:
        font_manager.fontManager.addfont(path)


def _daily_indices(time_days: np.ndarray, horizon_days: int) -> np.ndarray:
    requested = np.arange(horizon_days + 1, dtype=float)
    indices = np.searchsorted(time_days, requested)
    if indices[-1] >= time_days.size or not np.allclose(
        time_days[indices], requested, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("trajectory does not contain every integer-day sample")
    return indices


def _run_payload(
    rollout: object,
    target_gap_deg: np.ndarray,
    daily_indices: np.ndarray,
    *,
    status: str,
) -> dict[str, object]:
    state = np.asarray(rollout.state)[daily_indices]
    gaps = np.asarray(rollout.cyclic_gaps_deg)[daily_indices]
    cyclic_rates = np.asarray(rollout.cyclic_relative_rates_deg_per_day)[
        daily_indices
    ]
    altitude = np.asarray(rollout.altitude_km)[daily_indices]
    days = np.asarray(rollout.time_days)[daily_indices]
    frames = []
    for index, day in enumerate(days):
        frames.append(
            {
                "day": int(round(float(day))),
                "phase_deg": state[index, :, 0].tolist(),
                "cyclic_gap_deg": gaps[index].tolist(),
                "cyclic_gap_error_deg": (gaps[index] - target_gap_deg).tolist(),
                "relative_rate_deg_per_day": state[index, :, 1].tolist(),
                "altitude_km": altitude[index].tolist(),
                "extra_altitude_loss_km": state[index, :, 2].tolist(),
            }
        )
    return {
        "status": status,
        "plan_identity_sha256": rollout.plan_sha256,
        "sample_period_days": 1.0,
        "frames": frames,
    }


def _scenario_payload(scenario: object) -> dict[str, object]:
    params = scenario.parameters
    target_gap = np.asarray(params.target_cyclic_gaps_deg, dtype=float)
    target_slot = np.array(
        [0.0, target_gap[0], target_gap[0] + target_gap[1]], dtype=float
    )
    return {
        "satellite_count": params.satellite_count,
        "horizon_days": params.horizon_days,
        "control_interval_days": params.interval_days,
        "leader_index": 0,
        "initial_altitude_km": params.initial_altitude_km,
        "initial_phase_deg": list(params.initial_phase_deg),
        "initial_relative_rate_deg_per_day": [0.0] * params.satellite_count,
        "target_slot_deg": target_slot.tolist(),
        "target_gap_deg": target_gap.tolist(),
        "gap_tolerance_deg": params.gap_tolerance_deg,
        "relative_rate_tolerance_deg_per_day": (
            params.cyclic_rate_tolerance_deg_per_day
        ),
        "command_bounds": [0.0, 1.0],
        "earth_radius_km": params.earth_radius_km,
        "gravitational_parameter_m3_s2": params.gravitational_parameter_m3_s2,
        "reference_density_kg_m3": params.reference_density_kg_m3,
        "low_drag_ballistic_coefficient_kg_m2": (
            params.low_drag_ballistic_coefficient_kg_m2
        ),
        "high_drag_ballistic_coefficient_kg_m2": (
            params.high_drag_ballistic_coefficient_kg_m2
        ),
        "density_mean_factor": params.density_mean_factor,
        "density_amplitude": params.density_amplitude,
        "density_period_days": params.density_period_days,
        "density_phase_rad": params.density_phase_rad,
        "density_scale_height_km": params.density_scale_height_km,
        "primary_lock_tolerance_km": params.primary_lock_tolerance_km,
        "alpha_deg_per_day2": params.alpha_deg_per_day2,
        "d_km_per_day": params.d_km_per_day,
        "state_components": list(STATE_COMPONENTS),
        "linear_A": params.linear_A.tolist(),
        "linear_B": params.linear_B.tolist(),
        "gap_matrix": params.gap_matrix.tolist(),
    }


def _plan_payload(scenario: object) -> dict[str, object]:
    params = scenario.parameters
    plan = scenario.plan
    metrics = scenario.metrics
    action = np.asarray(plan.daily_high_drag_fraction)
    equivalent_days = np.asarray(metrics.equivalent_high_drag_days)
    return {
        "day": list(range(params.horizon_days)),
        "day_semantics": "U[:, day] applies on [day, day + 1)",
        "U": action.T.tolist(),
        "duty_fraction": (equivalent_days / params.horizon_days).tolist(),
        "equivalent_high_drag_days": equivalent_days.tolist(),
        "final_extra_altitude_loss_km": (
            np.asarray(metrics.nominal_final_extra_loss_km).tolist()
        ),
        "identity_sha256": plan.plan_sha256,
        "primary_max_final_extra_loss_km": (
            plan.primary_max_final_extra_loss_km
        ),
        "refined_max_final_extra_loss_km": (
            plan.refined_max_final_extra_loss_km
        ),
        "primary_total_variation": plan.primary_total_variation,
        "refined_total_variation": plan.refined_total_variation,
        "solver": {
            "primary_status": plan.primary_solver_status,
            "primary_message": plan.primary_solver_message,
            "refinement_status": plan.refinement_solver_status,
            "refinement_message": plan.refinement_solver_message,
        },
    }


def _metrics_payload(scenario: object) -> dict[str, object]:
    source = metrics_as_dict(scenario.metrics)
    nominal_final_altitude = np.asarray(scenario.nominal.altitude_km[-1]).tolist()
    checks = validation_checks(scenario)
    return {
        "plan": {
            "identity_sha256": source["plan_sha256"],
            "alpha_deg_per_day2": source["alpha_deg_per_day2"],
            "d_km_per_day": source["d_km_per_day"],
            "primary_max_final_extra_loss_km": source[
                "primary_max_final_extra_loss_km"
            ],
            "refined_max_final_extra_loss_km": source[
                "refined_max_final_extra_loss_km"
            ],
            "primary_total_variation": source["primary_total_variation"],
            "refined_total_variation": source["refined_total_variation"],
            "command_min": source["action_min"],
            "command_max": source["action_max"],
            "equivalent_high_drag_days": source["equivalent_high_drag_days"],
        },
        "nominal_linear": {
            "final_extra_altitude_loss_km": source[
                "nominal_final_extra_loss_km"
            ],
            "final_altitude_km": nominal_final_altitude,
            "final_cyclic_gap_deg": source["nominal_terminal_cyclic_gaps_deg"],
            "final_cyclic_gap_error_deg": source[
                "nominal_terminal_gap_error_deg"
            ],
            "max_abs_final_cyclic_gap_error_deg": source[
                "nominal_max_gap_error_deg"
            ],
            "final_cyclic_relative_rate_deg_per_day": source[
                "nominal_terminal_cyclic_rates_deg_per_day"
            ],
            "max_abs_final_cyclic_relative_rate_deg_per_day": source[
                "nominal_max_cyclic_rate_deg_per_day"
            ],
            "max_dynamics_residual": source["max_nominal_dynamics_residual"],
        },
        "nonlinear_replay": {
            "final_extra_altitude_loss_km": source[
                "nonlinear_final_extra_loss_km"
            ],
            "final_altitude_km": source["nonlinear_final_altitude_km"],
            "final_cyclic_gap_deg": source[
                "nonlinear_terminal_cyclic_gaps_deg"
            ],
            "final_cyclic_gap_error_deg": source[
                "nonlinear_terminal_gap_error_deg"
            ],
            "max_abs_final_cyclic_gap_error_deg": source[
                "nonlinear_max_gap_error_deg"
            ],
            "final_cyclic_relative_rate_deg_per_day": source[
                "nonlinear_terminal_cyclic_rates_deg_per_day"
            ],
            "max_abs_final_cyclic_relative_rate_deg_per_day": source[
                "nonlinear_max_cyclic_rate_deg_per_day"
            ],
            "minimum_altitude_km": source["nonlinear_min_altitude_km"],
            "reference_final_altitude_km": source[
                "nonlinear_reference_final_altitude_km"
            ],
            "density_min_kg_m3": source["density_min_kg_m3"],
            "density_max_kg_m3": source["density_max_kg_m3"],
        },
        "replay_refinement": _jsonable(asdict(scenario.resolution_check)),
        "validation": checks,
    }


def _build_artifact(scenario: object) -> dict[str, object]:
    params = scenario.parameters
    target_gap = np.asarray(params.target_cyclic_gaps_deg, dtype=float)
    nominal_indices = _daily_indices(scenario.nominal.time_days, params.horizon_days)
    nonlinear_indices = _daily_indices(
        scenario.nonlinear.time_days, params.horizon_days
    )
    metrics = _metrics_payload(scenario)
    metrics_sha256 = _canonical_sha256(metrics)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "generated_by": "scripts/build_cubesat_artifacts.py",
        "description": (
            "A fixed open-loop differential-drag plan, its exact linear rollout, "
            "and an independent variable-density nonlinear replay."
        ),
        "spacecraft": list(SPACECRAFT),
        "spacecraft_ids": list(SATELLITE_NAMES),
        "units": {
            "time": "day",
            "angle": "deg",
            "angular_rate": "deg/day",
            "altitude": "km",
            "density": "kg/m^3",
            "command_fraction": "unitless",
        },
        "scenario": _scenario_payload(scenario),
        "plan": _plan_payload(scenario),
        "runs": {
            "nominal_linear": _run_payload(
                scenario.nominal,
                target_gap,
                nominal_indices,
                status="complete",
            ),
            "nonlinear_replay": _run_payload(
                scenario.nonlinear,
                target_gap,
                nonlinear_indices,
                status="complete",
            ),
        },
        "metrics": metrics,
        "metrics_sha256": metrics_sha256,
        "conclusion": "nominal endpoint constraints do not transfer to the nonlinear replay",
        "limitations": [
            "The commands are an open-loop teaching plan, not flight software.",
            "The nonlinear replay adds deterministic density and altitude dependence but is not a high-fidelity orbit propagator.",
            "All phase coordinates are unwrapped relative to an all-low-drag reference orbit.",
            "Formation diagrams encode angular position only; marker radius never encodes altitude.",
        ],
    }


def _metric_unit(metric: str) -> str:
    if metric.endswith("_sha256"):
        return "sha256"
    if metric.endswith("_kg_m3"):
        return "kg/m^3"
    if metric.endswith("_m3_s2"):
        return "m^3/s^2"
    if metric.endswith("_deg_per_day2"):
        return "degree/day^2"
    if metric.endswith("_deg_per_day"):
        return "degree/day"
    if metric.endswith("_deg"):
        return "degree"
    if metric.endswith("_km_per_day"):
        return "km/day"
    if metric.endswith("_km"):
        return "km"
    if metric.endswith("_hours"):
        return "hour"
    if metric.endswith("_days"):
        return "day"
    if "variation" in metric or metric.startswith("command_"):
        return "dimensionless"
    return "boolean" if metric.startswith("validation.") else "dimensionless"


def _metric_rows(
    metrics: Mapping[str, object], metrics_sha256: str
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def visit(path: tuple[str, ...], value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit((*path, str(key)), item)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                visit((*path, str(index)), item)
            return
        leaf_path = ".".join(path)
        component = ""
        spacecraft = ""
        metric_path = leaf_path
        if path and path[-1].isdigit():
            component_index = int(path[-1])
            component = str(component_index)
            if 0 <= component_index < len(SPACECRAFT):
                spacecraft = SPACECRAFT[component_index]
            metric_path = ".".join(path[:-1])
        if isinstance(value, bool):
            formatted = "true" if value else "false"
        elif isinstance(value, float):
            formatted = format(value, ".17g")
        else:
            formatted = str(value)
        rows.append(
            {
                "metric": metric_path,
                "component": component,
                "spacecraft": spacecraft,
                "value": formatted,
                "unit": _metric_unit(metric_path),
                "metrics_sha256": metrics_sha256,
            }
        )

    visit((), metrics)
    return rows


def _write_metrics_csv(artifact: Mapping[str, object]) -> Path:
    path = ARTIFACT_DIRECTORY / "metrics.csv"
    rows = _metric_rows(artifact["metrics"], str(artifact["metrics_sha256"]))
    fields = ("metric", "component", "spacecraft", "value", "unit", "metrics_sha256")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_plan_csv(artifact: Mapping[str, object]) -> Path:
    path = ARTIFACT_DIRECTORY / "open_loop_plan.csv"
    plan = artifact["plan"]
    action = np.asarray(plan["U"], dtype=float)
    fields = (
        "day",
        "interval_start_day",
        "interval_end_day",
        "leader_high_drag_fraction",
        "follower_1_high_drag_fraction",
        "follower_2_high_drag_fraction",
        "plan_identity_sha256",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for day in plan["day"]:
            writer.writerow(
                {
                    "day": day,
                    "interval_start_day": day,
                    "interval_end_day": day + 1,
                    "leader_high_drag_fraction": format(action[0, day], ".17g"),
                    "follower_1_high_drag_fraction": format(action[1, day], ".17g"),
                    "follower_2_high_drag_fraction": format(action[2, day], ".17g"),
                    "plan_identity_sha256": plan["identity_sha256"],
                }
            )
    return path


def _frames_to_arrays(run: Mapping[str, object]) -> dict[str, np.ndarray]:
    frames = run["frames"]
    fields = (
        "phase_deg",
        "cyclic_gap_deg",
        "cyclic_gap_error_deg",
        "relative_rate_deg_per_day",
        "altitude_km",
        "extra_altitude_loss_km",
    )
    result = {"day": np.asarray([frame["day"] for frame in frames], dtype=int)}
    result.update(
        {
            field: np.asarray([frame[field] for frame in frames], dtype=float)
            for field in fields
        }
    )
    return result


def _write_trajectories_npz(
    artifact: Mapping[str, object], scenario: object
) -> Path:
    path = ARTIFACT_DIRECTORY / "trajectories.npz"
    arrays: dict[str, np.ndarray] = {
        "spacecraft": np.asarray(artifact["spacecraft"], dtype="U"),
        "target_slot_deg": np.asarray(
            artifact["scenario"]["target_slot_deg"], dtype=float
        ),
        "target_gap_deg": np.asarray(
            artifact["scenario"]["target_gap_deg"], dtype=float
        ),
        "plan_day": np.asarray(artifact["plan"]["day"], dtype=int),
        "plan_U": np.asarray(artifact["plan"]["U"], dtype=float),
        "plan_duty_fraction": np.asarray(
            artifact["plan"]["duty_fraction"], dtype=float
        ),
        "plan_equivalent_high_drag_days": np.asarray(
            artifact["plan"]["equivalent_high_drag_days"], dtype=float
        ),
        "plan_identity_sha256": np.asarray(
            artifact["plan"]["identity_sha256"], dtype="U"
        ),
        "metrics_json": np.asarray(
            json.dumps(
                artifact["metrics"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            dtype="U",
        ),
        "metrics_sha256": np.asarray(artifact["metrics_sha256"], dtype="U"),
    }
    for run_name in RUN_NAMES:
        for field, values in _frames_to_arrays(artifact["runs"][run_name]).items():
            arrays[f"{run_name}_{field}"] = values

    nonlinear = scenario.nonlinear
    arrays.update(
        {
            "nonlinear_replay_hourly_time_days": nonlinear.time_days,
            "nonlinear_replay_hourly_phase_deg": nonlinear.state[:, :, 0],
            "nonlinear_replay_hourly_relative_rate_deg_per_day": nonlinear.state[
                :, :, 1
            ],
            "nonlinear_replay_hourly_extra_altitude_loss_km": nonlinear.state[
                :, :, 2
            ],
            "nonlinear_replay_hourly_cyclic_gap_deg": nonlinear.cyclic_gaps_deg,
            "nonlinear_replay_hourly_cyclic_gap_error_deg": (
                nonlinear.cyclic_gaps_deg
                - np.asarray(scenario.parameters.target_cyclic_gaps_deg)
            ),
            "nonlinear_replay_hourly_cyclic_relative_rate_deg_per_day": (
                nonlinear.cyclic_relative_rates_deg_per_day
            ),
            "nonlinear_replay_hourly_altitude_km": nonlinear.altitude_km,
            "nonlinear_replay_hourly_reference_altitude_km": (
                nonlinear.reference_altitude_km
            ),
            "nonlinear_replay_hourly_density_kg_m3": nonlinear.density_kg_m3,
            "nonlinear_replay_hourly_reference_density_kg_m3": (
                nonlinear.reference_density_kg_m3
            ),
        }
    )
    np.savez_compressed(path, **arrays)
    return path


def _write_results_markdown(artifact: Mapping[str, object]) -> Path:
    path = ARTIFACT_DIRECTORY / "results.md"
    metrics = artifact["metrics"]
    plan = metrics["plan"]
    nominal = metrics["nominal_linear"]
    nonlinear = metrics["nonlinear_replay"]
    refinement = metrics["replay_refinement"]
    checks = metrics["validation"]
    fingerprint = artifact["metrics_sha256"]

    lines = [
        "<!-- Generated by scripts/build_cubesat_artifacts.py; do not edit by hand. -->",
        f"<!-- metrics-sha256: {fingerprint} -->",
        "",
        (
            "The lexicographic solve limits the largest nominal extra altitude "
            f"loss to {plan['refined_max_final_extra_loss_km']:.6f} km, then "
            f"reduces total variation from {plan['primary_total_variation']:.6f} "
            f"to {plan['refined_total_variation']:.6f} without changing that "
            "primary answer beyond the declared lock tolerance."
        ),
        "",
        (
            "At day 180, the linear planning model reaches a maximum cyclic-gap "
            f"error of ${nominal['max_abs_final_cyclic_gap_error_deg']:.6f}^\\circ$ and a "
            "maximum cyclic relative rate of "
            f"${nominal['max_abs_final_cyclic_relative_rate_deg_per_day']:.6f}^\\circ/\\mathrm{{day}}$. "
            "Replaying the identical command matrix through the variable-density "
            "model raises those values to "
            f"${nonlinear['max_abs_final_cyclic_gap_error_deg']:.6f}^\\circ$ and "
            f"${nonlinear['max_abs_final_cyclic_relative_rate_deg_per_day']:.6f}^\\circ/\\mathrm{{day}}$. "
            "The optimization result is therefore a statement about the planning "
            "model, not a guarantee for the replay model."
        ),
        "",
        "| spacecraft | high-drag equivalent (day) | duty | nominal final loss (km) | nonlinear final loss (km) | nonlinear altitude (km) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    duties = artifact["plan"]["duty_fraction"]
    equivalent = plan["equivalent_high_drag_days"]
    nominal_loss = nominal["final_extra_altitude_loss_km"]
    nonlinear_loss = nonlinear["final_extra_altitude_loss_km"]
    nonlinear_altitude = nonlinear["final_altitude_km"]
    for index, label in enumerate(SPACECRAFT):
        lines.append(
            f"| {label} | {equivalent[index]:.6f} | {100.0 * duties[index]:.3f}% "
            f"| {nominal_loss[index]:.6f} | {nonlinear_loss[index]:.6f} "
            f"| {nonlinear_altitude[index]:.6f} |"
        )

    lines.extend(
        [
            "",
            "| directed cyclic gap | target (degrees) | nominal final (degrees) | nominal error (degrees) | nonlinear final (degrees) | nonlinear error (degrees) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    targets = artifact["scenario"]["target_gap_deg"]
    for index, label in enumerate(GAP_LABELS):
        table_label = label.replace(" → ", " to ")
        lines.append(
            f"| {table_label} | {targets[index]:.3f} "
            f"| {nominal['final_cyclic_gap_deg'][index]:.6f} "
            f"| {nominal['final_cyclic_gap_error_deg'][index]:.6f} "
            f"| {nonlinear['final_cyclic_gap_deg'][index]:.6f} "
            f"| {nonlinear['final_cyclic_gap_error_deg'][index]:.6f} |"
        )

    lines.extend(
        [
            "",
            (
                "The hourly RK4 replay agrees with a 30-minute reference to within "
                f"${refinement['max_phase_delta_deg']:.3e}^\\circ$ in phase, "
                f"${refinement['max_relative_rate_delta_deg_per_day']:.3e}^\\circ/\\mathrm{{day}}$ "
                "in relative rate, and "
                f"{refinement['max_altitude_delta_km']:.3e} km in altitude. "
                f"Its sampled density spans {nonlinear['density_min_kg_m3']:.3e} "
                f"to {nonlinear['density_max_kg_m3']:.3e} kg/m³."
            ),
            "",
            (
                "All "
                f"{sum(bool(value) for value in checks.values())} deterministic "
                "acceptance checks pass. This is a transparent stress test of one "
                "open-loop teaching model, not a flight-operations prescription."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _style_axis(axis: plt.Axes, *, xgrid: bool = False) -> None:
    axis.spines["left"].set_color(RULE)
    axis.spines["bottom"].set_color(RULE)
    axis.tick_params(colors=MUTED)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)
    axis.grid(
        axis="x" if xgrid else "y",
        color=RULE,
        linewidth=0.45,
        alpha=0.6,
    )


def _draw_orbit_view(
    axis: plt.Axes,
    phase_deg: np.ndarray,
    target_slot_deg: np.ndarray,
    *,
    title: str,
    subtitle: str,
    maximum_gap_error_deg: float,
) -> None:
    relative_phase = np.mod(phase_deg - phase_deg[0], 360.0)
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    axis.fill_between(theta, 0.0, 0.34, color=TEAL, alpha=0.16, linewidth=0.0)
    axis.plot(theta, np.ones_like(theta), color=RULE, linewidth=1.0)
    for slot, color in zip(target_slot_deg, SATELLITE_COLORS):
        angle = np.deg2rad(slot)
        axis.scatter(
            [angle],
            [1.0],
            s=92,
            marker="o",
            facecolors=PAPER,
            edgecolors=color,
            linewidths=1.0,
            zorder=3,
        )
    for index, (phase, color) in enumerate(zip(relative_phase, SATELLITE_COLORS)):
        angle = np.deg2rad(phase)
        axis.scatter(
            [angle],
            [1.0],
            s=30,
            marker="o",
            facecolors=color,
            edgecolors=PAPER,
            linewidths=0.6,
            zorder=4,
        )
        label_radius = 1.17 if index != 0 else 1.20
        axis.text(
            angle,
            label_radius,
            ("L", "F1", "F2")[index],
            color=color,
            fontsize=7.0,
            ha="center",
            va="center",
            fontweight="semibold",
        )
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_ylim(0.0, 1.27)
    axis.set_xticks(np.deg2rad(target_slot_deg))
    axis.set_xticklabels(("0°", "120°", "240°"), color=MUTED)
    axis.set_yticks([])
    axis.grid(color=RULE, linewidth=0.45, alpha=0.65)
    axis.spines["polar"].set_color(RULE)
    axis.set_title(title, color=INK, pad=13, fontsize=9.2)
    axis.text(
        0.5,
        0.015,
        (
            f"fixed radius; altitude below  ·  {subtitle}\n"
            f"max final gap error {maximum_gap_error_deg:.3f}°"
        ),
        transform=axis.transAxes,
        color=MUTED,
        fontsize=6.8,
        ha="center",
        va="bottom",
        linespacing=1.15,
        bbox={"facecolor": PAPER, "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )


def _make_figure(scenario: object, artifact: Mapping[str, object]) -> plt.Figure:
    _register_fonts()
    params = scenario.parameters
    metrics = artifact["metrics"]
    target_slot = np.asarray(artifact["scenario"]["target_slot_deg"], dtype=float)
    target_gap = np.asarray(artifact["scenario"]["target_gap_deg"], dtype=float)
    action = np.asarray(artifact["plan"]["U"], dtype=float)

    with mpl.rc_context(FIGURE_STYLE):
        # Explicit normalized positions keep the dense, multi-panel fallback
        # stable across Matplotlib versions and both vector backends.
        figure = plt.figure(figsize=(7.2, 8.4))
        header_axis = figure.add_axes((0.06, 0.955, 0.88, 0.04))
        header_axis.axis("off")
        header_axis.text(
            0.5,
            0.5,
            "A nominal endpoint is not a nonlinear guarantee",
            color=INK,
            fontsize=14.5,
            fontfamily="Newsreader",
            fontweight="normal",
            ha="center",
            va="center",
        )
        nominal_orbit = figure.add_axes(
            (0.11, 0.65, 0.32, 0.225), projection="polar"
        )
        nonlinear_orbit = figure.add_axes(
            (0.57, 0.65, 0.32, 0.225), projection="polar"
        )
        _draw_orbit_view(
            nominal_orbit,
            scenario.nominal.state[-1, :, 0],
            target_slot,
            title="Linear planning endpoint",
            subtitle="constraints satisfied",
            maximum_gap_error_deg=metrics["nominal_linear"][
                "max_abs_final_cyclic_gap_error_deg"
            ],
        )
        _draw_orbit_view(
            nonlinear_orbit,
            scenario.nonlinear.state[-1, :, 0],
            target_slot,
            title="Variable-density replay endpoint",
            subtitle="same open-loop commands",
            maximum_gap_error_deg=metrics["nonlinear_replay"][
                "max_abs_final_cyclic_gap_error_deg"
            ],
        )

        duty_axis = figure.add_axes((0.10, 0.525, 0.80, 0.085))
        duty = np.asarray(artifact["plan"]["duty_fraction"], dtype=float)
        y = np.arange(3)
        duty_axis.barh(y, np.ones(3), height=0.48, color=RULE, alpha=0.45)
        duty_axis.barh(y, duty, height=0.48, color=SATELLITE_COLORS)
        equivalent_days = np.asarray(
            artifact["plan"]["equivalent_high_drag_days"], dtype=float
        )
        for index in range(3):
            duty_axis.text(
                min(duty[index] + 0.018, 0.88),
                index,
                f"{100.0 * duty[index]:.2f}%  ({equivalent_days[index]:.2f} d)",
                color=INK,
                va="center",
                fontsize=7.3,
                fontfamily="monospace",
            )
        duty_axis.set(
            xlim=(0.0, 1.0),
            yticks=y,
            yticklabels=SPACECRAFT,
            xlabel="fraction of 180-day horizon",
            title="High-drag duty: comparable burden across the formation",
        )
        duty_axis.invert_yaxis()
        duty_axis.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        _style_axis(duty_axis, xgrid=True)

        altitude_axis = figure.add_axes((0.09, 0.285, 0.39, 0.175))
        error_axis = figure.add_axes((0.57, 0.285, 0.35, 0.175))
        for index, (label, color) in enumerate(zip(SPACECRAFT, SATELLITE_COLORS)):
            altitude_axis.plot(
                scenario.nominal.time_days,
                scenario.nominal.altitude_km[:, index],
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.78,
            )
            altitude_axis.plot(
                scenario.nonlinear.time_days,
                scenario.nonlinear.altitude_km[:, index],
                color=color,
                label=label,
            )

        nonlinear_error = scenario.nonlinear.cyclic_gaps_deg - target_gap
        nominal_error = scenario.nominal.cyclic_gaps_deg - target_gap
        for index, (label, color) in enumerate(zip(GAP_LABELS, GAP_COLORS)):
            error_axis.plot(
                scenario.nominal.time_days,
                nominal_error[:, index],
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.78,
            )
            error_axis.plot(
                scenario.nonlinear.time_days,
                nonlinear_error[:, index],
                color=color,
                label=label,
            )
        error_axis.axhspan(
            -params.gap_tolerance_deg,
            params.gap_tolerance_deg,
            color=SUCCESS,
            alpha=0.10,
            linewidth=0.0,
        )
        error_axis.axhline(0.0, color=RULE, linewidth=0.65)

        altitude_axis.set(
            xlim=(0.0, params.horizon_days),
            xlabel="day",
            ylabel="altitude (km)",
        )
        altitude_axis.set_title(
            "Altitude conventions\n"
            "nominal proxy omits shared decay (dashed); absolute replay (solid)",
            fontsize=7.7,
            linespacing=1.15,
        )
        error_axis.set(
            xlim=(0.0, params.horizon_days),
            xlabel="day",
            ylabel="cyclic-gap error (°)",
            title="Phase errors expose the model mismatch",
        )
        _style_axis(altitude_axis)
        _style_axis(error_axis)
        altitude_axis.legend(
            loc="lower left",
            ncols=1,
            frameon=False,
            fontsize=6.8,
            handlelength=1.5,
            labelspacing=0.25,
        )
        error_axis.legend(
            loc="upper left",
            ncols=1,
            frameon=False,
            fontsize=6.5,
            handlelength=1.5,
            labelspacing=0.20,
        )
        error_axis.text(
            0.98,
            0.04,
            "solid replay  ·  dashed linear",
            transform=error_axis.transAxes,
            ha="right",
            va="bottom",
            color=MUTED,
            fontsize=6.4,
        )

        command_axis = figure.add_axes((0.10, 0.095, 0.79, 0.105))
        command_map = LinearSegmentedColormap.from_list(
            "low_to_high_drag", (PAPER, TEAL)
        )
        image = command_axis.imshow(
            action,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=(0.0, params.horizon_days, 2.5, -0.5),
            vmin=0.0,
            vmax=1.0,
            cmap=command_map,
        )
        command_axis.set(
            xlim=(0.0, params.horizon_days),
            yticks=(0, 1, 2),
            yticklabels=SPACECRAFT,
            xlabel="day; command applies on [day, day + 1)",
            title="Committed open-loop command matrix",
        )
        command_axis.set_xticks(np.arange(0, params.horizon_days + 1, 30))
        command_axis.tick_params(colors=MUTED)
        for spine in command_axis.spines.values():
            spine.set_color(RULE)
        colorbar_axis = figure.add_axes((0.91, 0.095, 0.012, 0.105))
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_ticks((0.0, 1.0))
        colorbar.set_ticklabels(("low drag", "high drag"))
        colorbar.ax.tick_params(labelsize=6.8, colors=MUTED)
        colorbar.outline.set_edgecolor(RULE)

        plan_metrics = metrics["plan"]
        replay_metrics = metrics["nonlinear_replay"]
        footer_axis = figure.add_axes((0.06, 0.008, 0.88, 0.025))
        footer_axis.axis("off")
        footer_axis.text(
            0.0,
            0.5,
            (
                f"plan {artifact['plan']['identity_sha256'][:12]}  ·  "
                f"primary max loss {plan_metrics['primary_max_final_extra_loss_km']:.6f} km  ·  "
                f"replay max final gap error {replay_metrics['max_abs_final_cyclic_gap_error_deg']:.6f}°  ·  "
                f"metrics {artifact['metrics_sha256'][:12]}"
            ),
            color=MUTED,
            fontsize=6.2,
            fontfamily="monospace",
            ha="left",
            va="center",
        )
        return figure


def _figure_metric_summary(artifact: Mapping[str, object]) -> dict[str, object]:
    metrics = artifact["metrics"]
    return {
        "metrics_sha256": artifact["metrics_sha256"],
        "plan_identity_sha256": artifact["plan"]["identity_sha256"],
        "primary_max_final_extra_loss_km": metrics["plan"][
            "primary_max_final_extra_loss_km"
        ],
        "refined_max_final_extra_loss_km": metrics["plan"][
            "refined_max_final_extra_loss_km"
        ],
        "nominal_max_abs_final_cyclic_gap_error_deg": metrics[
            "nominal_linear"
        ]["max_abs_final_cyclic_gap_error_deg"],
        "nonlinear_max_abs_final_cyclic_gap_error_deg": metrics[
            "nonlinear_replay"
        ]["max_abs_final_cyclic_gap_error_deg"],
    }


def _write_figure(
    scenario: object,
    artifact: Mapping[str, object],
    *,
    include_pdf: bool,
) -> list[Path]:
    figure = _make_figure(scenario, artifact)
    summary = json.dumps(
        _figure_metric_summary(artifact),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    svg_path = STATIC_DIRECTORY / "differential-drag.svg"
    with mpl.rc_context(FIGURE_STYLE):
        figure.savefig(
            svg_path,
            metadata={
                "Date": None,
                "Title": "CubeSat differential-drag plan and nonlinear replay",
                "Description": summary,
                "Creator": "scripts/build_cubesat_artifacts.py",
            },
        )
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    outputs = [svg_path]
    if include_pdf:
        pdf_path = STATIC_DIRECTORY / "differential-drag.pdf"
        with mpl.rc_context(FIGURE_STYLE):
            figure.savefig(
                pdf_path,
                metadata={
                    "CreationDate": None,
                    "ModDate": None,
                    "Title": "CubeSat differential-drag plan and nonlinear replay",
                    "Subject": summary,
                    "Creator": "scripts/build_cubesat_artifacts.py",
                },
            )
        outputs.append(pdf_path)
    plt.close(figure)
    return outputs


def build(*, include_pdf: bool = True) -> dict[str, object]:
    """Solve the fixed scenario and write every committed evidence artifact."""

    build_started = time.perf_counter()
    scenario = run_scenario()
    assert_valid_scenario(scenario)
    artifact = _build_artifact(scenario)

    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    STATIC_DIRECTORY.mkdir(parents=True, exist_ok=True)
    artifact_path = _write_json(
        ARTIFACT_DIRECTORY / "textbook_results.json", artifact
    )
    metrics_path = _write_metrics_csv(artifact)
    plan_path = _write_plan_csv(artifact)
    trajectories_path = _write_trajectories_npz(artifact, scenario)
    results_path = _write_results_markdown(artifact)
    figure_paths = _write_figure(
        scenario,
        artifact,
        include_pdf=include_pdf,
    )

    outputs = [
        artifact_path,
        metrics_path,
        plan_path,
        trajectories_path,
        results_path,
        *figure_paths,
    ]
    missing_inputs = [path for path in INPUT_FILES if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            "manifest inputs are missing: "
            + ", ".join(str(path) for path in missing_inputs)
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "build_elapsed_s": round(time.perf_counter() - build_started, 3),
        "command": (
            "uv run python scripts/build_cubesat_artifacts.py"
            + ("" if include_pdf else " --skip-pdf")
        ),
        "inputs": {
            str(path.relative_to(ROOT)): _sha256(path) for path in INPUT_FILES
        },
        "outputs": {
            str(path.relative_to(ROOT)): _sha256(path) for path in outputs
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "protocol": {
            "plan_identity_sha256": scenario.plan.plan_sha256,
            "metrics_sha256": artifact["metrics_sha256"],
            "planning_model": {
                "steps": scenario.parameters.horizon_days,
                "step_days": scenario.parameters.interval_days,
                "primary_objective": "minimize maximum final extra altitude loss",
                "secondary_objective": "minimize total variation",
                "primary_lock_tolerance_km": (
                    scenario.parameters.primary_lock_tolerance_km
                ),
            },
            "nonlinear_replay": {
                "method": "fixed-step classical RK4",
                "step_hours": scenario.nonlinear.step_hours,
                "reference_step_hours": scenario.resolution_check.fine_step_hours,
                "immutable_plan": True,
            },
            "json_sampling": "integer days 0 through 180 inclusive",
            "npz_sampling": "daily views plus complete hourly nonlinear replay",
            "validation": validation_checks(scenario),
        },
    }
    _write_json(ARTIFACT_DIRECTORY / "manifest.json", manifest)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="write all artifacts except the PDF companion",
    )
    args = parser.parse_args()
    build(include_pdf=not args.skip_pdf)


if __name__ == "__main__":
    main()
