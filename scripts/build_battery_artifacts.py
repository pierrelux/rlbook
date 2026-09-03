#!/usr/bin/env python3
"""Build committed PyBaMM fast-charging evidence for the modeling chapter.

Normal MyST builds read these files and never import or solve PyBaMM.  Run with
``--skip-pdf`` while iterating; the complete maintenance command writes both
vector formats after the repository PDF operation marker has been recorded.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import platform
import sys
import time

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from battery_control import (  # noqa: E402
    BatteryScenario,
    FIGURE_STYLE,
    RUN_ORDER,
    audit_to_artifact,
    make_summary_figure,
    run_battery_audit,
    save_artifact,
    sha256,
)


ARTIFACT_DIRECTORY = ROOT / "artifacts" / "battery"
STATIC_DIRECTORY = ROOT / "_static" / "battery"
INPUT_FILES = (
    ROOT / "code" / "battery_control.py",
    ROOT / "code" / "battery_replay.py",
    ROOT / "scripts" / "build_battery_artifacts.py",
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
    STATIC_DIRECTORY / "fonts" / "IBMPlexSans-Regular.ttf",
    STATIC_DIRECTORY / "fonts" / "IBMPlexSans-SemiBold.ttf",
    STATIC_DIRECTORY / "fonts" / "IBMPlexMono-Regular.ttf",
    STATIC_DIRECTORY / "fonts" / "Newsreader.ttf",
    STATIC_DIRECTORY / "fonts" / "IBM-Plex-LICENSE.txt",
    STATIC_DIRECTORY / "fonts" / "Newsreader-OFL.txt",
    STATIC_DIRECTORY / "fonts" / "README.md",
)


def _write_metrics(results: dict[str, object]) -> Path:
    path = ARTIFACT_DIRECTORY / "metrics.csv"
    fields = [
        "run",
        "label",
        "verdict",
        "plant_resistance_scale",
        "model_resistance_scale",
        *asdict(results[RUN_ORDER[0]].metrics),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name in RUN_ORDER:
            trace = results[name]
            writer.writerow(
                {
                    "run": name,
                    "label": trace.label,
                    "verdict": trace.verdict,
                    "plant_resistance_scale": trace.plant_resistance_scale,
                    "model_resistance_scale": trace.model_resistance_scale,
                    **asdict(trace.metrics),
                }
            )
    return path


def _write_trajectories(results: dict[str, object]) -> tuple[Path, Path]:
    csv_path = ARTIFACT_DIRECTORY / "trajectories.csv"
    fields = [
        "run",
        "time_s",
        "soc",
        "current_a",
        "terminal_voltage_v",
        "rc_overpotential_v",
        "cell_temperature_c",
        "jig_temperature_c",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name in RUN_ORDER:
            trace = results[name]
            for index in range(trace.time_s.size):
                writer.writerow(
                    {
                        "run": name,
                        "time_s": trace.time_s[index],
                        "soc": trace.soc[index],
                        "current_a": trace.current_a[index],
                        "terminal_voltage_v": trace.terminal_voltage_v[index],
                        "rc_overpotential_v": trace.rc_overpotential_v[index],
                        "cell_temperature_c": trace.cell_temperature_c[index],
                        "jig_temperature_c": trace.jig_temperature_c[index],
                    }
                )

    npz_path = ARTIFACT_DIRECTORY / "trajectories.npz"
    arrays: dict[str, np.ndarray] = {}
    for name in RUN_ORDER:
        trace = results[name]
        arrays.update(
            {
                f"{name}_time_s": trace.time_s,
                f"{name}_soc": trace.soc,
                f"{name}_current_a": trace.current_a,
                f"{name}_terminal_voltage_v": trace.terminal_voltage_v,
                f"{name}_rc_overpotential_v": trace.rc_overpotential_v,
                f"{name}_cell_temperature_c": trace.cell_temperature_c,
                f"{name}_jig_temperature_c": trace.jig_temperature_c,
            }
        )
    np.savez_compressed(npz_path, **arrays)
    return csv_path, npz_path


def _write_diagnostic(diagnostic: object) -> Path:
    path = ARTIFACT_DIRECTORY / "diagnostic_pulse.csv"
    fields = [
        "time_s",
        "charge_current_a",
        "soc",
        "clean_voltage_v",
        "measured_voltage_v",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index in range(diagnostic.time_s.size):
            writer.writerow(
                {
                    "time_s": diagnostic.time_s[index],
                    "charge_current_a": diagnostic.current_a[index],
                    "soc": diagnostic.soc[index],
                    "clean_voltage_v": diagnostic.clean_voltage_v[index],
                    "measured_voltage_v": diagnostic.measured_voltage_v[index],
                }
            )
    return path


def _write_results(results: dict[str, object], fitted: object) -> Path:
    path = ARTIFACT_DIRECTORY / "results.md"
    lines = [
        "<!-- Generated by scripts/build_battery_artifacts.py; do not edit by hand. -->",
        "",
    ]
    fresh = results["fresh_nominal"]
    stale = results["high_resistance_stale"]
    calibrated = results["high_resistance_calibrated"]
    lines.extend(
        [
            (
                f"The diagnostic pulse estimates a resistance multiplier of "
                f"{fitted.resistance_scale:.4f}, with a voltage RMSE of "
                f"{1000.0 * fitted.voltage_rmse_v:.2f} mV. The fresh plant reaches "
                f"the target in {fresh.metrics.target_time_s / 60:.2f} minutes. "
                f"The stale model reaches "
                f"the charge target in {stale.metrics.target_time_s / 60:.2f} minutes, "
                f"but its voltage exceeds 4.20 V for "
                f"{stale.metrics.voltage_violation_duration_s:.1f} seconds and peaks "
                f"at {stale.metrics.max_voltage_v:.3f} V. The fitted model takes "
                f"{calibrated.metrics.target_time_s / 60:.2f} minutes and remains "
                "inside both plant bounds."
            ),
            "",
            (
                "Time above the voltage bound uses linear interpolation at the "
                "threshold crossings. The 35 degree C plant bound is never reached, "
                "but the conservative 34.5 degree C local thermal-headroom envelope "
                "does limit requested current. "
                "These are results for a declared teaching cell and a controlled "
                "high-resistance counterfactual, not charging guidance for a product."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build(*, include_pdf: bool = True) -> dict[str, object]:
    """Run the fixed audit and write all committed replay/evidence artifacts."""

    import pybamm

    build_started = time.perf_counter()
    scenario = BatteryScenario()
    results, diagnostic, fitted = run_battery_audit(scenario)
    artifact = audit_to_artifact(
        results, diagnostic, fitted, scenario, frame_stride=5
    )

    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    STATIC_DIRECTORY.mkdir(parents=True, exist_ok=True)
    artifact_path = save_artifact(
        artifact, ARTIFACT_DIRECTORY / "textbook_results.json"
    )
    metrics_path = _write_metrics(results)
    trajectory_paths = _write_trajectories(results)
    diagnostic_path = _write_diagnostic(diagnostic)
    results_path = _write_results(results, fitted)

    figure = make_summary_figure(results, scenario)
    svg_path = STATIC_DIRECTORY / "fast-charging.svg"
    with matplotlib.rc_context(FIGURE_STYLE):
        figure.savefig(svg_path, metadata={"Date": None})
    figure_paths = [svg_path]
    if include_pdf:
        pdf_path = STATIC_DIRECTORY / "fast-charging.pdf"
        with matplotlib.rc_context(FIGURE_STYLE):
            figure.savefig(
                pdf_path,
                metadata={"CreationDate": None, "ModDate": None},
            )
        figure_paths.append(pdf_path)
    plt.close(figure)

    outputs = [
        artifact_path,
        metrics_path,
        *trajectory_paths,
        diagnostic_path,
        results_path,
        *figure_paths,
    ]
    manifest = {
        "schema_version": 1,
        "build_elapsed_s": round(time.perf_counter() - build_started, 3),
        "command": (
            "uv run --group artifacts python scripts/build_battery_artifacts.py"
            + ("" if include_pdf else " --skip-pdf")
        ),
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path) for path in INPUT_FILES
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path) for path in outputs
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "pybamm": pybamm.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "protocol": {
            "model": artifact["model"],
            "scenario": artifact["scenario"],
            "diagnostic": artifact["diagnostic"],
            "resistance_scaling": (
                "R0 and R1 multiply by alpha; C1 divides by alpha; "
                "R1*C1 remains 24 s"
            ),
            "controller": (
                "Continuous PyBaMM CustomStepExplicit current governor; the minimum "
                "of the 10 A action limit, the immediate-voltage envelope, and a "
                "34.5 degree C local thermal-headroom ceiling derived from the "
                "declared steady heat balance; results sampled every 1 s. The "
                "35 degree C plant bound is never reached, but the thermal-headroom "
                "ceiling does limit requested current."
            ),
            "threshold_duration": (
                "piecewise-linear threshold crossing interpolation on sampled output"
            ),
        },
    }
    manifest_path = ARTIFACT_DIRECTORY / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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
