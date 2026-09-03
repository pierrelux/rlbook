"""Checks for the immutable browser-native battery replay."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import sys

import pytest


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

from battery_replay import BatteryReplayError, render_battery_replay  # noqa: E402


@pytest.fixture()
def artifact() -> dict:
    scenario = {
        "initial_soc": 0.2,
        "target_soc": 0.8,
        "current_limit_a": 10.0,
        "voltage_limit_v": 4.2,
        "temperature_limit_c": 35.0,
        "voltage_guard_v": 4.17,
        "temperature_guard_c": 34.5,
        "duration_s": 180.0,
        "control_period_s": 1.0,
    }

    def run(
        label: str,
        verdict: str,
        voltages: tuple[float, float, float],
        temperatures: tuple[float, float, float],
        violation_time: float | None,
    ) -> dict:
        frames = [
            {
                "time_s": time_s,
                "soc": soc,
                "current_a": current,
                "terminal_voltage_v": voltage,
                "cell_temperature_c": temperature,
                "jig_temperature_c": 25.0 + 0.3 * index,
                "rc_overpotential_v": 0.01 * index,
            }
            for index, (time_s, soc, current, voltage, temperature) in enumerate(
                zip(
                    (0.0, 60.0, 120.0),
                    (0.2, 0.55, 0.8),
                    (10.0, 7.0, 0.0),
                    voltages,
                    temperatures,
                )
            )
        ]
        return {
            "label": label,
            "verdict": verdict,
            "style": {"color": "ignored by the renderer"},
            "frames": frames,
            "metrics": {
                "target_time_s": 120.0,
                "max_voltage_v": max(voltages),
                "voltage_violation_duration_s": 60.0 if violation_time else 0.0,
                "max_cell_temperature_c": max(temperatures),
                "extra_metric": 123.0,
            },
            "events": {
                "first_taper_time_s": 60.0,
                "first_violation_time_s": violation_time,
                "target_time_s": 120.0,
            },
        }

    return {
        "schema_version": 1,
        "title": "Fast charging when resistance drifts",
        "description": "A recorded three-run control audit.",
        "playback_fps": 20,
        "scenario": scenario,
        "runs": {
            "fresh_nominal": run(
                "Fresh plant, nominal model",
                "stands",
                (3.55, 4.15, 4.17),
                (25.0, 27.0, 28.0),
                None,
            ),
            "high_resistance_stale": run(
                "High resistance, stale model",
                "withdrawn",
                (3.58, 4.23, 4.25),
                (25.0, 28.0, 30.0),
                60.0,
            ),
            "high_resistance_calibrated": run(
                "High resistance, fitted model",
                "stands",
                (3.58, 4.16, 4.18),
                (25.0, 27.5, 29.0),
                None,
            ),
        },
        "metadata": {"ignored_extra": True},
    }


def test_player_is_accessible_prefix_only_and_offline(artifact: dict) -> None:
    rendered = render_battery_replay(artifact, replay_id="lecture-battery")
    root_match = re.search(r'<section id="([^"]+)"', rendered)

    assert root_match is not None
    assert root_match.group(1).startswith("lecture-battery-")
    assert 'data-action="play"' in rendered
    assert 'data-action="step"' in rendered
    assert 'data-action="step-back"' in rendered
    assert 'data-action="reset"' in rendered
    assert 'type="range"' in rendered
    assert 'aria-live="polite"' in rendered
    assert '<span role="timer" data-time>0.0 min</span>' in rendered
    assert "timeOutput.textContent=" in rendered
    assert 'aria-keyshortcuts="Space"' in rendered
    assert 'data-seek="first_taper"' in rendered
    assert 'data-seek="first_violation"' in rendered
    assert 'data-seek="target"' in rendered
    assert "state of charge" in rendered
    assert "thermal-halo" in rendered
    assert "current-arrow" in rendered
    assert "4.20 V plant bound" in rendered
    assert "35 °C plant bound" in rendered
    assert "within plant bounds" in rendered
    assert "plant bound crossed" in rendered
    assert "run.frames.slice(0, index + 1)" in rendered
    assert "prefers-reduced-motion:reduce" in rendered
    assert 'data-theme="dark"' in rendered
    assert "MutationObserver(applyTheme)" in rendered
    assert 'matchMedia("(prefers-color-scheme: dark)")' in rendered
    assert "autoplay" not in rendered.lower()
    assert "fetch(" not in rendered
    assert "XMLHttpRequest" not in rendered
    assert "WebSocket" not in rendered


def test_fallback_is_hidden_only_after_successful_initial_render(artifact: dict) -> None:
    rendered = render_battery_replay(artifact)

    initialization = rendered.index("configureLines(); configureRun(); render();")
    hiding = rendered.index("fallback.hidden=true")
    assert initialization < hiding
    assert 'const fallbackId="fig-battery-fast-charging-fallback"' in rendered
    assert 'fallback.setAttribute("aria-hidden","true")' in rendered
    assert "MutationObserver" in rendered
    assert "window.parent && window.parent !== window" in rendered


def test_browser_script_contains_presentation_logic_not_a_battery_model(artifact: dict) -> None:
    rendered = render_battery_replay(artifact)
    browser_script = re.findall(r"<script>(.*?)</script>", rendered, flags=re.DOTALL)[-1]

    for forbidden in (
        "Math.exp",
        "pybamm",
        "capacity_ah",
        "thermal_mass",
        "open_circuit_voltage",
        "fit_resistance_scale",
        "rc_overpotential_v",
    ):
        assert forbidden not in browser_script
    assert "requestAnimationFrame" in browser_script
    assert "pointsPrefix" in browser_script


def test_extra_recorded_fields_are_not_embedded(artifact: dict) -> None:
    rendered = render_battery_replay(artifact)

    assert "rc_overpotential_v" not in rendered
    assert "extra_metric" not in rendered
    assert "ignored_extra" not in rendered


def test_unique_roots_and_script_termination_escaping(artifact: dict) -> None:
    first = render_battery_replay(artifact, replay_id="battery")
    second_artifact = deepcopy(artifact)
    second_artifact["runs"]["fresh_nominal"]["label"] = (
        "bad</script><script>alert(1)</script>"
    )
    second = render_battery_replay(second_artifact, replay_id="battery")

    first_id = re.search(r'<section id="([^"]+)"', first).group(1)
    second_id = re.search(r'<section id="([^"]+)"', second).group(1)
    assert first_id != second_id
    assert "</script><script>alert(1)</script>" not in second
    assert "\\u003c/script" in second


def test_path_input_and_malformed_schema_are_handled(artifact: dict, tmp_path: Path) -> None:
    path = tmp_path / "battery.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert "Recorded charging run" in render_battery_replay(path)

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(BatteryReplayError, match="invalid battery replay JSON"):
        render_battery_replay(invalid_json)

    wrong_schema = deepcopy(artifact)
    wrong_schema["schema_version"] = 2
    with pytest.raises(BatteryReplayError, match="unsupported"):
        render_battery_replay(wrong_schema)


def test_missing_nonfinite_and_nonmonotone_data_are_rejected(artifact: dict) -> None:
    missing = deepcopy(artifact)
    del missing["runs"]["high_resistance_calibrated"]
    with pytest.raises(BatteryReplayError, match="missing run"):
        render_battery_replay(missing)

    nonfinite = deepcopy(artifact)
    nonfinite["runs"]["fresh_nominal"]["frames"][1]["current_a"] = float("nan")
    with pytest.raises(BatteryReplayError, match="must be finite"):
        render_battery_replay(nonfinite)

    nonmonotone = deepcopy(artifact)
    nonmonotone["runs"]["fresh_nominal"]["frames"][1]["time_s"] = 0.0
    with pytest.raises(BatteryReplayError, match="strictly increasing"):
        render_battery_replay(nonmonotone)

    outside_action = deepcopy(artifact)
    outside_action["runs"]["fresh_nominal"]["frames"][0]["current_a"] = 10.1
    with pytest.raises(BatteryReplayError, match="action bounds"):
        render_battery_replay(outside_action)


def test_event_and_verdict_inconsistencies_are_rejected(artifact: dict) -> None:
    missing_violation = deepcopy(artifact)
    missing_violation["runs"]["high_resistance_stale"]["events"][
        "first_violation_time_s"
    ] = None
    with pytest.raises(BatteryReplayError, match="violation event disagrees"):
        render_battery_replay(missing_violation)

    unsafe_stands = deepcopy(artifact)
    unsafe_stands["runs"]["high_resistance_stale"]["verdict"] = "stands"
    with pytest.raises(BatteryReplayError, match="cannot stand"):
        render_battery_replay(unsafe_stands)

    target_disagreement = deepcopy(artifact)
    target_disagreement["runs"]["fresh_nominal"]["events"]["target_time_s"] = 119.0
    with pytest.raises(BatteryReplayError, match="target event and metric disagree"):
        render_battery_replay(target_disagreement)

    crossing_disagreement = deepcopy(artifact)
    crossing_disagreement["runs"]["high_resistance_stale"]["events"][
        "first_violation_time_s"
    ] = 119.0
    with pytest.raises(BatteryReplayError, match="does not bracket"):
        render_battery_replay(crossing_disagreement)

    unfinished = deepcopy(artifact)
    unfinished["runs"]["fresh_nominal"]["frames"][-1]["soc"] = 0.79
    with pytest.raises(BatteryReplayError, match="must end at the target"):
        render_battery_replay(unfinished)


def test_event_seek_uses_first_recorded_frame_at_or_after_event(artifact: dict) -> None:
    stale = artifact["runs"]["high_resistance_stale"]
    stale["frames"][1]["time_s"] = 59.9
    stale["frames"][1]["terminal_voltage_v"] = 4.199
    stale["events"]["first_violation_time_s"] = 59.95

    rendered = render_battery_replay(artifact)

    assert '"first_violation":2' in rendered


def test_reading_width_stacks_stage_and_voltage_keeps_crossing_precision(
    artifact: dict,
) -> None:
    rendered = render_battery_replay(artifact)

    assert "grid-template-columns:1fr; margin-top:.7rem" in rendered
    assert "grid-template-columns:minmax(14rem,19rem) minmax(0,1fr)" in rendered
    assert "frame.terminal_voltage_v.toFixed(4)" in rendered
