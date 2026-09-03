#!/usr/bin/env python3
"""Build the measured L4 thermal-validation evidence used by the textbook.

The acquisition and parameter fitting have already happened.  This builder
loads the completed phase-confirmation bundle and the committed fit report,
then free-runs the two fixed training-only models on the untouched validation
pair.  It never fits a model and never contacts a network service.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import html
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import matplotlib


matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fit_inference_thermal import simulate_one_state_rc  # noqa: E402
from fit_inference_thermal_phase import (  # noqa: E402
    VALIDATION_SEQUENCE,
    load_phase_thermal_bundle,
    simulate_one_state_phase_gain,
)


SOURCE_DIRECTORY = (
    ROOT / "data" / "inference_serving" / "thermal-phase-identification-20260903T131518Z"
)
ARTIFACT_DIRECTORY = ROOT / "artifacts" / "inference_serving"
STATIC_DIRECTORY = ROOT / "_static" / "inference_serving"

MODEL_ORDER = ("power_only_one_state_rc", "phase_gain_one_state_rc")
MODEL_LABELS = {
    "power_only_one_state_rc": "power-only RC",
    "phase_gain_one_state_rc": "phase-gain RC",
}
MODEL_SIMULATORS: Mapping[
    str, Callable[[Mapping[str, float], Any], np.ndarray]
] = {
    "power_only_one_state_rc": simulate_one_state_rc,
    "phase_gain_one_state_rc": simulate_one_state_phase_gain,
}

PAPER = "#F6F7F4"
INK = "#1B2430"
TEAL = "#2F6F8F"
MUTED = "#5C6874"
RULE = "#D2D9D7"
STANDS = "#2E7D5B"
CAVEAT = "#B8860B"
WITHDRAWN = "#A83A32"

FIGURE_TITLE = "Measured thermal validation on one NVIDIA L4"
FIGURE_DESCRIPTION = (
    "Two untouched 55 W validation pulses compare measured decode and prefill "
    "temperature rises with fixed power-only and phase-gain one-state RC models. "
    "The phase gain lowers held-out RMSE from 1.43 to 0.87 degrees C, but the "
    "model is rejected because its worst error is 2.73 degrees C and its phase-"
    "contrast error is 1.43 degrees C, both above the strict 1 degree C boundary. "
    "The observed aligned prefill-minus-decode peak-rise contrast is 2.00 degrees C "
    "and the model predicts 3.43 degrees C. Prefill-minus-decode measured mean "
    "power is -0.132 W and integrated energy is -6.66 J. This one-L4 Qwen and "
    "vLLM experiment is not a hardware-safety model."
)

FONT_DIRECTORY = ROOT / "_static" / "battery" / "fonts"
FONT_FILES = {
    "body": FONT_DIRECTORY / "IBMPlexSans-Regular.ttf",
    "body_semibold": FONT_DIRECTORY / "IBMPlexSans-SemiBold.ttf",
    "mono": FONT_DIRECTORY / "IBMPlexMono-Regular.ttf",
    "heading": FONT_DIRECTORY / "Newsreader.ttf",
}

FIGURE_STYLE = {
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
    "font.monospace": ["IBM Plex Mono", "DejaVu Sans Mono"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.65,
    "lines.linewidth": 1.45,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "svg.fonttype": "path",
    "svg.hashsalt": "l4-thermal-phase-validation-v1",
}


class ThermalValidationArtifactError(ValueError):
    """Raised when the committed acquisition and fit evidence disagree."""


@dataclass(frozen=True)
class ValidationSeries:
    """One one-second held-out pulse and both fixed-model predictions."""

    block_id: str
    workload_phase: str
    requested_power_limit_w: float
    time_s: np.ndarray
    observed_temperature_c: np.ndarray
    measured_power_w: np.ndarray
    predictions_c: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ValidationEvidence:
    """Validated source metadata, fixed parameters, and held-out trajectories."""

    source_directory: Path
    manifest: Mapping[str, Any]
    report: Mapping[str, Any]
    fixed_parameters: Mapping[str, Mapping[str, float]]
    series: tuple[ValidationSeries, ...]
    acceptance_metrics_c: Mapping[str, float]
    power_difference_w: float
    energy_difference_j: float
    energy_ratio: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ThermalValidationArtifactError(message)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ThermalValidationArtifactError(f"Could not read {path}: {error}") from error
    _require(isinstance(value, Mapping), f"{path} must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left: float, right: float, *, atol: float = 1.0e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-10, abs_tol=atol)


def _trajectory_metrics(
    observed: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    residual = np.asarray(observed[1:] - predicted[1:], dtype=float)
    return {
        "rmse_c": float(np.sqrt(np.mean(np.square(residual)))),
        "maximum_absolute_error_c": float(np.max(np.abs(residual))),
        "predicted_peak_rise_c": float(np.max(predicted) - predicted[0]),
        "observed_peak_rise_c": float(np.max(observed) - observed[0]),
    }


def load_validation_evidence(
    source_directory: str | Path = SOURCE_DIRECTORY,
) -> ValidationEvidence:
    """Load fixed fits and reproduce their untouched one-second predictions."""

    source = Path(source_directory).resolve()
    manifest_path = source / "thermal_phase_manifest.json"
    report_path = source / "thermal_phase_fit_report.json"
    manifest = _read_json(manifest_path)
    report = _read_json(report_path)

    _require(manifest.get("status") == "complete", "thermal acquisition is incomplete")
    _require(
        manifest.get("protocol") == "cold-start-phase-pairs-v3",
        "unexpected thermal phase protocol",
    )
    _require(
        manifest.get("gpu", {}).get("name") == "NVIDIA L4",
        "source hardware is not an NVIDIA L4",
    )
    _require(report.get("schema_version") == 1, "unexpected thermal fit schema")
    pre_registration = report.get("pre_registration")
    _require(isinstance(pre_registration, Mapping), "fit report lacks pre-registration")
    _require(
        pre_registration.get("validation_used_for_fit_or_model_selection") is False,
        "validation data were used to fit or select a model",
    )
    _require(
        report.get("model_comparison", {}).get(
            "validation_was_not_used_to_choose_or_refit_a_model"
        )
        is True,
        "fit report does not preserve the untouched validation claim",
    )

    bundle = load_phase_thermal_bundle(manifest_path)
    _require(
        report.get("source", {}).get("telemetry_sha256") == bundle.telemetry_sha256,
        "fit report and telemetry bundle disagree",
    )
    _require(
        report.get("source", {}).get("requests_sha256") == bundle.requests_sha256,
        "fit report and request bundle disagree",
    )
    validation_ids = tuple(bundle.pulse_ids_by_sequence[VALIDATION_SEQUENCE])
    _require(
        validation_ids == tuple(pre_registration.get("validation_pulse_ids", ())),
        "fit report and bundle disagree on the held-out pulse ids",
    )
    _require(len(validation_ids) == 2, "expected exactly two validation pulses")

    report_models = report.get("models")
    _require(isinstance(report_models, Mapping), "fit report lacks model results")
    fixed_parameters: dict[str, Mapping[str, float]] = {}
    for model_name in MODEL_ORDER:
        model_report = report_models.get(model_name)
        _require(isinstance(model_report, Mapping), f"missing model {model_name}")
        training_fit = model_report.get("final_training_fit")
        _require(isinstance(training_fit, Mapping), f"{model_name} lacks a fixed fit")
        parameters = training_fit.get("parameters")
        _require(isinstance(parameters, Mapping), f"{model_name} lacks parameters")
        parsed = {str(name): float(value) for name, value in parameters.items()}
        _require(
            parsed and all(math.isfinite(value) for value in parsed.values()),
            f"{model_name} contains invalid parameters",
        )
        fixed_parameters[model_name] = parsed

    series: list[ValidationSeries] = []
    for block_id in validation_ids:
        pulse = bundle.pulses[block_id]
        predictions: dict[str, np.ndarray] = {}
        for model_name in MODEL_ORDER:
            predictions[model_name] = MODEL_SIMULATORS[model_name](
                fixed_parameters[model_name], pulse
            )
        series.append(
            ValidationSeries(
                block_id=block_id,
                workload_phase=pulse.workload_phase,
                requested_power_limit_w=float(pulse.requested_power_limit_w),
                time_s=np.asarray(pulse.time_s, dtype=float),
                observed_temperature_c=np.asarray(pulse.temperature_c, dtype=float),
                measured_power_w=np.asarray(pulse.measured_power_w, dtype=float),
                predictions_c=predictions,
            )
        )

    _require(
        {item.workload_phase for item in series} == {"decode", "prefill"},
        "validation pair must contain one decode and one prefill pulse",
    )
    for item in series:
        _require(
            item.requested_power_limit_w == 55.0,
            f"{item.block_id} is not the prespecified 55 W validation pulse",
        )
        _require(
            item.time_s.shape == item.observed_temperature_c.shape,
            f"{item.block_id} has inconsistent one-second state arrays",
        )
        _require(
            item.measured_power_w.size + 1 == item.time_s.size,
            f"{item.block_id} has inconsistent power intervals",
        )
        _require(
            np.array_equal(item.time_s, np.arange(item.time_s.size, dtype=float)),
            f"{item.block_id} is not on the one-second validation grid",
        )
        for model_name in MODEL_ORDER:
            _require(
                item.predictions_c[model_name].shape == item.time_s.shape
                and np.all(np.isfinite(item.predictions_c[model_name])),
                f"{model_name} produced an invalid trajectory for {item.block_id}",
            )
            computed = _trajectory_metrics(
                item.observed_temperature_c, item.predictions_c[model_name]
            )
            stored = report_models[model_name]["validation_evaluation"]["per_pulse"][
                item.block_id
            ]["metrics"]
            _require(
                _close(computed["rmse_c"], stored["rmse_c"])
                and _close(
                    computed["maximum_absolute_error_c"],
                    stored["maximum_absolute_error_c"],
                ),
                f"reconstructed {model_name} metrics disagree for {item.block_id}",
            )

    for model_name in MODEL_ORDER:
        observed = np.concatenate([item.observed_temperature_c[1:] for item in series])
        predicted = np.concatenate(
            [item.predictions_c[model_name][1:] for item in series]
        )
        aggregate_rmse = float(np.sqrt(np.mean(np.square(observed - predicted))))
        stored_rmse = report_models[model_name]["validation_evaluation"]["aggregate"][
            "rmse_c"
        ]
        _require(
            _close(aggregate_rmse, stored_rmse),
            f"reconstructed aggregate RMSE disagrees for {model_name}",
        )

    phase_gain_acceptance = report.get("acceptance")
    _require(isinstance(phase_gain_acceptance, Mapping), "fit report lacks acceptance")
    _require(
        phase_gain_acceptance.get("target_model") == "phase_gain_one_state_rc"
        and phase_gain_acceptance.get("verdict") == "rejected"
        and phase_gain_acceptance.get("accepted_for_mixed_serving_thermal_constraints")
        is False,
        "the committed acceptance verdict is not the expected rejection",
    )
    criteria = phase_gain_acceptance.get("criteria")
    _require(isinstance(criteria, Mapping), "fit report lacks acceptance criteria")
    acceptance_metrics = {
        "held_out_rmse_c": float(criteria["validation_rmse"]["value_c"]),
        "worst_error_c": float(
            criteria["validation_maximum_absolute_or_peak_error"]["value_c"]
        ),
        "phase_contrast_error_c": float(
            criteria["validation_phase_contrast_error"]["absolute_error_c"]
        ),
        "threshold_c": float(criteria["validation_rmse"]["threshold_c"]),
    }
    _require(
        acceptance_metrics["held_out_rmse_c"] < acceptance_metrics["threshold_c"]
        and acceptance_metrics["worst_error_c"] >= acceptance_metrics["threshold_c"]
        and acceptance_metrics["phase_contrast_error_c"]
        >= acceptance_metrics["threshold_c"],
        "stored acceptance values do not imply the reported rejection",
    )

    observed_contrast = report["observed_validation_contrast"]["one_second_aligned"]
    phase_pair = report_models["phase_gain_one_state_rc"]["validation_evaluation"][
        "matched_55w_decode_prefill"
    ]
    _require(
        _close(observed_contrast["prefill_minus_decode_peak_rise_c"], 2.0)
        and _close(phase_pair["predicted_prefill_minus_decode_peak_rise_c"], 3.431289590989124),
        "stored aligned phase contrast differs from the committed result",
    )
    matched_power = report["measured_power_and_energy"]["matched_phase_pairs"][
        "phase_validation"
    ]["55"]
    return ValidationEvidence(
        source_directory=source,
        manifest=manifest,
        report=report,
        fixed_parameters=fixed_parameters,
        series=tuple(series),
        acceptance_metrics_c=acceptance_metrics,
        power_difference_w=float(matched_power["prefill_minus_decode_mean_power_w"]),
        energy_difference_j=float(matched_power["prefill_minus_decode_energy_j"]),
        energy_ratio=float(matched_power["prefill_to_decode_energy_ratio"]),
    )


def _format_number(value: float) -> str:
    return format(float(value), ".17g")


def write_validation_csv(evidence: ValidationEvidence, destination: str | Path) -> Path:
    """Write tidy one-second observed and fixed-model validation trajectories."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "split",
        "block_id",
        "workload_phase",
        "requested_power_limit_w",
        "time_s",
        "preceding_interval_mean_power_w",
        "observed_temperature_c",
        "observed_temperature_rise_c",
        "power_only_predicted_temperature_c",
        "power_only_predicted_temperature_rise_c",
        "phase_gain_predicted_temperature_c",
        "phase_gain_predicted_temperature_rise_c",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in evidence.series:
            starts = {
                name: float(item.predictions_c[name][0]) for name in MODEL_ORDER
            }
            observed_start = float(item.observed_temperature_c[0])
            for index, time_s in enumerate(item.time_s):
                power = "" if index == 0 else _format_number(item.measured_power_w[index - 1])
                writer.writerow(
                    {
                        "split": "validation",
                        "block_id": item.block_id,
                        "workload_phase": item.workload_phase,
                        "requested_power_limit_w": _format_number(
                            item.requested_power_limit_w
                        ),
                        "time_s": _format_number(time_s),
                        "preceding_interval_mean_power_w": power,
                        "observed_temperature_c": _format_number(
                            item.observed_temperature_c[index]
                        ),
                        "observed_temperature_rise_c": _format_number(
                            item.observed_temperature_c[index] - observed_start
                        ),
                        "power_only_predicted_temperature_c": _format_number(
                            item.predictions_c["power_only_one_state_rc"][index]
                        ),
                        "power_only_predicted_temperature_rise_c": _format_number(
                            item.predictions_c["power_only_one_state_rc"][index]
                            - starts["power_only_one_state_rc"]
                        ),
                        "phase_gain_predicted_temperature_c": _format_number(
                            item.predictions_c["phase_gain_one_state_rc"][index]
                        ),
                        "phase_gain_predicted_temperature_rise_c": _format_number(
                            item.predictions_c["phase_gain_one_state_rc"][index]
                            - starts["phase_gain_one_state_rc"]
                        ),
                    }
                )
    return path


def write_result_markdown(evidence: ValidationEvidence, destination: str | Path) -> Path:
    """Write the generated textbook summary of the confirmatory result."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = evidence.report
    power_rmse = report["models"]["power_only_one_state_rc"][
        "validation_evaluation"
    ]["aggregate"]["rmse_c"]
    gain_rmse = evidence.acceptance_metrics_c["held_out_rmse_c"]
    worst = evidence.acceptance_metrics_c["worst_error_c"]
    contrast_error = evidence.acceptance_metrics_c["phase_contrast_error_c"]
    pair = report["models"]["phase_gain_one_state_rc"]["validation_evaluation"][
        "matched_55w_decode_prefill"
    ]
    text = f"""<!-- Generated by scripts/build_inference_thermal_validation_artifacts.py; do not edit by hand. -->

On the held-out pair, the power-only model has an RMSE of {power_rmse:.2f} degrees C. Adding the workload-phase gain reduces this error to {gain_rmse:.2f} degrees C. The fixed rule, however, requires all three validation errors to remain below 1 degree C. The phase-gain model reaches a worst trajectory error of {worst:.2f} degrees C and a phase-contrast error of {contrast_error:.2f} degrees C, so the model is rejected.

The measured prefill peak rise exceeds the decode rise by {pair['observed_prefill_minus_decode_peak_rise_c']:.2f} degrees C; the phase-gain model predicts a difference of {pair['predicted_prefill_minus_decode_peak_rise_c']:.2f} degrees C. Prefill draws {abs(evidence.power_difference_w):.3f} W less mean board power and consumes {abs(evidence.energy_difference_j):.2f} J less energy than decode, with an energy ratio of {evidence.energy_ratio:.5f}. Higher measured electrical input therefore cannot explain the larger prefill temperature rise in this pair.

These measurements come from one NVIDIA L4 serving Qwen2.5-7B-Instruct with vLLM 0.28.0. They do not establish a hardware-safety model or justify a mixed-serving thermal constraint.
"""
    path.write_text(text, encoding="utf-8")
    return path


def _register_fonts() -> dict[str, font_manager.FontProperties]:
    missing = [path for path in FONT_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing vendored figure fonts: " + ", ".join(str(path) for path in missing)
        )
    for path in FONT_FILES.values():
        font_manager.fontManager.addfont(path)
    return {
        name: font_manager.FontProperties(fname=path)
        for name, path in FONT_FILES.items()
    }


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["left"].set_color(RULE)
    axis.spines["bottom"].set_color(RULE)
    axis.tick_params(colors=MUTED)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)
    axis.grid(axis="y", color=RULE, linewidth=0.5, alpha=0.65)


def _direct_line_labels(axis: plt.Axes, phase: str, endpoints: Mapping[str, float]) -> None:
    label_y = {
        "decode": {"power": 15.2, "observed": 12.6, "phase_gain": 10.7},
        "prefill": {"phase_gain": 16.4, "observed": 14.7, "power": 12.9},
    }[phase]
    specifications = (
        ("observed", "measured", INK, "-"),
        ("power", "power-only RC", TEAL, "--"),
        ("phase_gain", "phase-gain RC", WITHDRAWN, "-."),
    )
    for key, label, color, linestyle in specifications:
        axis.annotate(
            label,
            xy=(59.0, endpoints[key]),
            xytext=(62.0, label_y[key]),
            color=color,
            fontsize=7.2,
            va="center",
            arrowprops={"arrowstyle": "-", "color": color, "linestyle": linestyle, "lw": 0.8},
            annotation_clip=False,
        )


def make_validation_figure(evidence: ValidationEvidence) -> plt.Figure:
    """Create the static small-multiple validation and acceptance figure."""

    fonts = _register_fonts()
    by_phase = {item.workload_phase: item for item in evidence.series}
    report = evidence.report
    power_rmse = report["models"]["power_only_one_state_rc"][
        "validation_evaluation"
    ]["aggregate"]["rmse_c"]
    gain_rmse = evidence.acceptance_metrics_c["held_out_rmse_c"]
    with mpl.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.2, 5.35))
        grid = figure.add_gridspec(2, 2, height_ratios=(1.52, 0.92))
        axes = {
            "decode": figure.add_subplot(grid[0, 0]),
            "prefill": figure.add_subplot(grid[0, 1]),
        }
        acceptance_axis = figure.add_subplot(grid[1, :])
        figure.subplots_adjust(
            left=0.105,
            right=0.965,
            bottom=0.205,
            top=0.75,
            wspace=0.24,
            hspace=0.56,
        )

        figure.text(
            0.075,
            0.955,
            FIGURE_TITLE,
            color=INK,
            fontsize=19,
            fontproperties=fonts["heading"],
            va="top",
        )
        figure.text(
            0.075,
            0.902,
            "Fixed training-only fits evaluated on one untouched 55 W decode/prefill pair",
            color=MUTED,
            fontsize=8.4,
            fontproperties=fonts["body"],
            va="top",
        )
        figure.text(
            0.075,
            0.855,
            f"held-out RMSE  {power_rmse:.2f}  ->  {gain_rmse:.2f} degrees C",
            color=TEAL,
            fontsize=8.2,
            fontproperties=fonts["mono"],
            va="top",
        )
        figure.text(
            0.925,
            0.855,
            "REJECTED",
            color=WITHDRAWN,
            fontsize=8.2,
            fontproperties=fonts["body_semibold"],
            ha="right",
            va="top",
        )

        for phase in ("decode", "prefill"):
            axis = axes[phase]
            item = by_phase[phase]
            time_s = item.time_s
            observed_rise = item.observed_temperature_c - item.observed_temperature_c[0]
            power_rise = (
                item.predictions_c["power_only_one_state_rc"]
                - item.predictions_c["power_only_one_state_rc"][0]
            )
            gain_rise = (
                item.predictions_c["phase_gain_one_state_rc"]
                - item.predictions_c["phase_gain_one_state_rc"][0]
            )
            axis.plot(
                time_s,
                observed_rise,
                color=INK,
                linestyle="-",
                marker="o",
                markevery=8,
                markersize=3.0,
                markerfacecolor=PAPER,
                markeredgewidth=0.8,
                zorder=4,
            )
            axis.plot(time_s, power_rise, color=TEAL, linestyle="--", zorder=2)
            axis.plot(time_s, gain_rise, color=WITHDRAWN, linestyle="-.", zorder=3)
            mean_power = float(np.mean(item.measured_power_w))
            axis.set_title(f"{phase.capitalize()} workload", loc="left", pad=12)
            axis.text(
                0.0,
                1.015,
                f"measured mean board power {mean_power:.2f} W",
                transform=axis.transAxes,
                color=MUTED,
                fontsize=7.0,
                va="bottom",
            )
            axis.set_xlim(0.0, 74.0)
            axis.set_ylim(-0.5, 17.5)
            axis.set_xticks((0, 20, 40, 60))
            axis.set_yticks((0, 4, 8, 12, 16))
            axis.set_xlabel("elapsed time (s)")
            _style_axis(axis)
            _direct_line_labels(
                axis,
                phase,
                {
                    "observed": float(observed_rise[-1]),
                    "power": float(power_rise[-1]),
                    "phase_gain": float(gain_rise[-1]),
                },
            )
        axes["decode"].set_ylabel("junction-temperature rise (degrees C)")
        axes["prefill"].tick_params(labelleft=False)

        metric_names = (
            "held-out RMSE",
            "worst trajectory error",
            "phase-contrast error",
        )
        metric_keys = (
            "held_out_rmse_c",
            "worst_error_c",
            "phase_contrast_error_c",
        )
        values = [evidence.acceptance_metrics_c[key] for key in metric_keys]
        y_positions = np.arange(len(values) - 1, -1, -1)
        threshold = evidence.acceptance_metrics_c["threshold_c"]
        acceptance_axis.axvline(
            threshold,
            color=CAVEAT,
            linestyle=(0, (2.0, 2.0)),
            linewidth=1.25,
            zorder=1,
        )
        acceptance_axis.text(
            threshold + 0.035,
            y_positions[0] + 0.42,
            "strict 1 degree C boundary",
            color=CAVEAT,
            fontsize=7.2,
            va="bottom",
        )
        for index, (name, value, y) in enumerate(
            zip(metric_names, values, y_positions)
        ):
            passed = value < threshold
            color = STANDS if passed else WITHDRAWN
            marker = "o" if passed else "X"
            acceptance_axis.hlines(y, 0.0, value, color=color, linewidth=1.8, alpha=0.72)
            acceptance_axis.scatter(
                [value],
                [y],
                color=color,
                marker=marker,
                s=45 if passed else 58,
                linewidth=0.9,
                edgecolor=PAPER if passed else color,
                zorder=3,
            )
            acceptance_axis.text(
                value + 0.08,
                y,
                f"{value:.2f}  {'passes' if passed else 'fails'}",
                color=color,
                fontsize=7.5,
                fontproperties=fonts["mono"],
                va="center",
            )
        acceptance_axis.set_yticks(y_positions, metric_names)
        acceptance_axis.set_xlim(0.0, 3.35)
        acceptance_axis.set_ylim(-0.55, 2.55)
        acceptance_axis.set_xlabel("validation error (degrees C); all three must be below 1")
        acceptance_axis.set_title(
            "Phase-gain model against the fixed acceptance rule",
            loc="left",
            pad=8,
        )
        _style_axis(acceptance_axis)
        acceptance_axis.grid(False)

        figure.text(
            0.075,
            0.025,
            (
                "One NVIDIA L4, Qwen2.5-7B-Instruct, vLLM 0.28.0. "
                f"Prefill minus decode: {evidence.power_difference_w:+.3f} W and "
                f"{evidence.energy_difference_j:+.2f} J. The rejected fit is not a "
                "hardware-safety model."
            ),
            color=MUTED,
            fontsize=7.1,
            fontproperties=fonts["body"],
            va="bottom",
        )
        return figure


def _add_svg_accessibility(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    start = source.find("<svg ")
    _require(start >= 0, "generated SVG lacks a root element")
    open_end = source.find(">", start)
    _require(open_end >= 0, "generated SVG root element is malformed")
    source = (
        source[: start + 5]
        + 'role="img" aria-labelledby="thermal-phase-validation-title thermal-phase-validation-desc" '
        + source[start + 5 : open_end + 1]
        + '<title id="thermal-phase-validation-title">'
        + html.escape(FIGURE_TITLE)
        + "</title>"
        + '<desc id="thermal-phase-validation-desc">'
        + html.escape(FIGURE_DESCRIPTION)
        + "</desc>"
        + source[open_end + 1 :]
    )
    path.write_text(source, encoding="utf-8")


def _write_manifest(
    evidence: ValidationEvidence,
    artifact_directory: Path,
    static_directory: Path,
    outputs: tuple[Path, ...],
) -> Path:
    source = evidence.source_directory
    input_paths = (
        source / "thermal_phase_manifest.json",
        source / "thermal_phase_fit_report.json",
        source / "l4_thermal_phase_telemetry.csv",
        source / "l4_thermal_phase_requests.csv",
        Path(__file__).resolve(),
        ROOT / "scripts" / "fit_inference_thermal.py",
        ROOT / "scripts" / "fit_inference_thermal_phase.py",
        *FONT_FILES.values(),
    )

    def logical_output(path: Path) -> str:
        if path.parent == artifact_directory:
            return f"artifacts/inference_serving/{path.name}"
        if path.parent == static_directory:
            return f"_static/inference_serving/{path.name}"
        raise ThermalValidationArtifactError(f"unexpected output location: {path}")

    manifest = {
        "schema_version": 1,
        "command": "uv run python scripts/build_inference_thermal_validation_artifacts.py",
        "source_experiment": source.name,
        "source_scope": {
            "gpu": evidence.manifest["gpu"]["name"],
            "model": evidence.manifest["model"],
            "model_revision": evidence.manifest["model_revision"],
            "vllm_version": evidence.manifest["vllm_server_version"],
            "validation_pulse_ids": evidence.report["pre_registration"][
                "validation_pulse_ids"
            ],
            "validation_used_for_fit_or_model_selection": False,
        },
        "fixed_training_only_parameters": evidence.fixed_parameters,
        "acceptance": {
            **evidence.acceptance_metrics_c,
            "verdict": "rejected",
            "accepted_for_mixed_serving_thermal_constraints": False,
        },
        "matched_electrical_input": {
            "prefill_minus_decode_mean_power_w": evidence.power_difference_w,
            "prefill_minus_decode_energy_j": evidence.energy_difference_j,
            "prefill_to_decode_energy_ratio": evidence.energy_ratio,
        },
        "inputs": {str(path.relative_to(ROOT)): _file_sha256(path) for path in input_paths},
        "outputs": {logical_output(path): _file_sha256(path) for path in outputs},
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "limitations": [
            "This is one confirmatory experiment on one NVIDIA L4.",
            "The phase-gain model failed the fixed validation rule.",
            "The fit is not a hardware-safety model.",
        ],
    }
    path = artifact_directory / "thermal_phase_validation_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def build_artifacts(
    *,
    source_directory: str | Path = SOURCE_DIRECTORY,
    artifact_directory: str | Path = ARTIFACT_DIRECTORY,
    static_directory: str | Path = STATIC_DIRECTORY,
) -> tuple[Path, ...]:
    """Write deterministic validation data, prose, and vector fallbacks."""

    artifact_dir = Path(artifact_directory).resolve()
    static_dir = Path(static_directory).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    evidence = load_validation_evidence(source_directory)

    csv_path = write_validation_csv(
        evidence, artifact_dir / "thermal_phase_validation.csv"
    )
    result_path = write_result_markdown(
        evidence, artifact_dir / "thermal_phase_result.md"
    )
    figure = make_validation_figure(evidence)
    svg_path = static_dir / "thermal-phase-validation.svg"
    pdf_path = static_dir / "thermal-phase-validation.pdf"
    with mpl.rc_context(FIGURE_STYLE):
        figure.savefig(
            svg_path,
            metadata={
                "Title": FIGURE_TITLE,
                "Description": FIGURE_DESCRIPTION,
                "Date": None,
                "Creator": "rlbook measured thermal validation artifact builder",
            },
        )
        figure.savefig(
            pdf_path,
            metadata={
                "Title": FIGURE_TITLE,
                "Subject": FIGURE_DESCRIPTION,
                "Keywords": "NVIDIA L4, thermal validation, Qwen, vLLM",
                "Creator": "rlbook measured thermal validation artifact builder",
                "CreationDate": None,
                "ModDate": None,
            },
        )
    plt.close(figure)
    _add_svg_accessibility(svg_path)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    outputs = (csv_path, result_path, svg_path, pdf_path)
    manifest_path = _write_manifest(
        evidence, artifact_dir, static_dir, outputs
    )
    return (*outputs, manifest_path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-directory", type=Path, default=SOURCE_DIRECTORY)
    parser.add_argument("--artifact-directory", type=Path, default=ARTIFACT_DIRECTORY)
    parser.add_argument("--static-directory", type=Path, default=STATIC_DIRECTORY)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    build_artifacts(
        source_directory=arguments.source_directory,
        artifact_directory=arguments.artifact_directory,
        static_directory=arguments.static_directory,
    )


if __name__ == "__main__":
    main()
