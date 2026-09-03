#!/usr/bin/env python3
"""Build deterministic camera-gimbal replay and static textbook assets."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
import sys

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from gimbal_control import (  # noqa: E402
    ESTIMATOR_ORDER,
    GimbalParameters,
    GimbalScenario,
    comparison_to_artifact,
    make_summary_figure,
    run_comparison,
    save_artifact,
)


ARTIFACT_DIRECTORY = ROOT / "artifacts" / "gimbal"
STATIC_DIRECTORY = ROOT / "_static" / "gimbal"


def build_artifacts() -> dict[str, object]:
    """Run the validated experiment and write every derived artifact."""

    parameters = GimbalParameters()
    scenario = GimbalScenario()
    results = run_comparison(parameters, scenario)
    artifact = comparison_to_artifact(results, parameters, scenario, frame_stride=4)

    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    STATIC_DIRECTORY.mkdir(parents=True, exist_ok=True)
    save_artifact(artifact, ARTIFACT_DIRECTORY / "textbook_results.json")

    with (ARTIFACT_DIRECTORY / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ["estimator", "label", *asdict(results[ESTIMATOR_ORDER[0]].metrics)]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for estimator in ESTIMATOR_ORDER:
            rollout = results[estimator]
            writer.writerow(
                {
                    "estimator": estimator,
                    "label": rollout.label,
                    **asdict(rollout.metrics),
                }
            )

    metric_lines = [
        "| state estimate supplied to the controller | RMS angle | peak during translation | final absolute error |",
        "|---|---:|---:|---:|",
    ]
    for estimator in ESTIMATOR_ORDER:
        rollout = results[estimator]
        metric_lines.append(
            f"| {rollout.label} | "
            f"{rollout.metrics.rms_angle_deg:.2f} degrees | "
            f"{rollout.metrics.peak_acceleration_window_deg:.2f} degrees | "
            f"{rollout.metrics.final_abs_angle_deg:.2f} degrees |"
        )
    (ARTIFACT_DIRECTORY / "results.md").write_text(
        "\n".join(metric_lines) + "\n", encoding="utf-8"
    )

    figure = make_summary_figure(results)
    for suffix in ("svg", "pdf"):
        figure.savefig(
            STATIC_DIRECTORY / f"partial-observability.{suffix}",
        )
    plt.close(figure)
    return artifact


if __name__ == "__main__":
    build_artifacts()
