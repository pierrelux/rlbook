"""Checks for the immutable browser-native CubeSat replay."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE_DIRECTORY = ROOT / "code"
ARTIFACT = ROOT / "artifacts" / "cubesat" / "textbook_results.json"
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

from cubesat_replay import CubeSatReplayError, render_cubesat_replay  # noqa: E402


def test_committed_textbook_artifact_matches_the_replay_contract() -> None:
    rendered = render_cubesat_replay(ARTIFACT, replay_id="committed-cubesat")

    assert 'id="committed-cubesat-' in rendered
    assert '"nominal_linear"' in rendered
    assert '"nonlinear_replay"' in rendered
    assert len(rendered.encode("utf-8")) < 300_000


@pytest.fixture()
def artifact() -> dict:
    initial_phase = [-0.5, 0.0, 0.5]
    target_phase = [0.0, 120.0, 240.0]
    target_gap = [120.0, 120.0, -240.0]

    def frames(run: str) -> list[dict]:
        if run == "nominal_linear":
            final_phase = target_phase
        else:
            final_phase = [0.0, 113.23, 228.46]
        result = []
        for day in range(181):
            fraction = day / 180.0
            phase = [
                start + fraction * (finish - start)
                for start, finish in zip(initial_phase, final_phase)
            ]
            gaps = [
                phase[1] - phase[0],
                phase[2] - phase[1],
                phase[0] - phase[2],
            ]
            gap_error = [value - target for value, target in zip(gaps, target_gap)]
            altitude = [
                475.0 - 0.004 * day - offset * fraction
                for offset in (0.0, 0.03, 0.06)
            ]
            result.append(
                {
                    "day": float(day),
                    "phase_deg": phase,
                    "cyclic_gap_deg": gaps,
                    "cyclic_gap_error_deg": gap_error,
                    "relative_rate_deg_per_day": [
                        0.0 if day == 0 else (phase[index] - initial_phase[index]) / day
                        for index in range(3)
                    ],
                    "altitude_km": altitude,
                    "extra_altitude_loss_km": [475.0 - value for value in altitude],
                    "solver_state_that_must_not_ship": [999.0],
                }
            )
        return result

    commands = [
        [float((day + satellite) % (4 + satellite) == 0) for day in range(180)]
        for satellite in range(3)
    ]
    duty = [sum(row) / len(row) for row in commands]
    equivalent_days = [sum(row) for row in commands]
    return {
        "schema_version": 1,
        "status": "complete",
        "generated_by": "test fixture that must not be embedded",
        "units": {
            "command_fraction": "unitless",
            "angle": "deg",
            "angular_rate": "deg/day",
            "altitude": "km",
            "density": "kg/m^3",
            "time": "day",
            "unused_unit": "secret",
        },
        "spacecraft": ["Leader", "Follower 1", "Follower 2"],
        "scenario": {
            "horizon_days": 180,
            "control_interval_days": 1.0,
            "leader_index": 0,
            "initial_altitude_km": 475.0,
            "initial_phase_deg": initial_phase,
            "initial_relative_rate_deg_per_day": [0.0, 0.0, 0.0],
            "target_slot_deg": target_phase,
            "target_gap_deg": target_gap,
            "gap_tolerance_deg": 0.1,
            "relative_rate_tolerance_deg_per_day": 0.002,
            "atmosphere_model_that_must_not_ship": {"density": 123.0},
        },
        "plan": {
            "day": list(range(180)),
            "U": commands,
            "duty_fraction": duty,
            "equivalent_high_drag_days": equivalent_days,
            "final_extra_altitude_loss_km": [0.72, 0.75, 0.78],
            "identity_sha256": "a" * 64,
            "primary_max_final_extra_loss_km": 0.79,
            "refined_max_final_extra_loss_km": 0.78,
            "primary_total_variation": 12.0,
            "refined_total_variation": 10.0,
            "optimizer_diagnostics_that_must_not_ship": "secret",
        },
        "runs": {
            "nominal_linear": {"status": "complete", "frames": frames("nominal_linear")},
            "nonlinear_replay": {"status": "complete", "frames": frames("nonlinear_replay")},
        },
        "metrics": {
            "nominal_linear": {"private_metric": 1.0},
            "nonlinear_replay": {"private_metric": 2.0},
            "replay_refinement": {"private_metric": 3.0},
            "validation": {"private_metric": 4.0},
        },
        "limitations": ["fixture limitation that must not be embedded"],
    }


def test_player_is_accessible_causal_and_offline(artifact: dict) -> None:
    rendered = render_cubesat_replay(artifact, replay_id="lecture-cubesat")
    root_match = re.search(r'<section id="([^"]+)"', rendered)

    assert root_match is not None
    assert root_match.group(1).startswith("lecture-cubesat-")
    assert 'tabindex="0"' in rendered
    assert 'aria-labelledby="' in rendered
    assert 'aria-describedby="' in rendered
    assert 'data-action="play"' in rendered
    assert 'data-action="step-back"' in rendered
    assert 'data-action="step"' in rendered
    assert 'data-action="reset"' in rendered
    assert 'type="range"' in rendered
    assert 'role="timer"' in rendered
    assert 'aria-live="polite"' in rendered
    assert 'aria-keyshortcuts="Space"' in rendered
    assert 'event.key==="ArrowRight"' in rendered
    assert 'event.key==="ArrowLeft"' in rendered
    assert 'event.key==="Home"' in rendered
    assert 'event.key==="End"' in rendered
    assert "prefers-reduced-motion:reduce" in rendered
    assert '@media (max-width:720px)' in rendered
    assert '[data-theme="dark"]' in rendered
    assert 'matchMedia("(prefers-color-scheme: dark)")' in rendered
    assert "MutationObserver(applyTheme)" in rendered
    assert "autoplay" not in rendered.lower()
    assert "fetch(" not in rendered
    assert "XMLHttpRequest" not in rendered
    assert "WebSocket" not in rendered


def test_orbits_share_literal_fixed_scale_and_show_direct_gaps(artifact: dict) -> None:
    rendered = render_cubesat_replay(artifact)

    assert rendered.count('viewBox="0 0 320 320"') == 2
    assert rendered.count('class="orbit-ring" cx="160" cy="160" r="112"') == 2
    assert 'data-orbit="nominal_linear"' in rendered
    assert 'data-orbit="nonlinear_replay"' in rendered
    assert "orbitPoint(relativePhase(current,index))" in rendered
    assert "angle-replay.scenario.target_slot_deg" in rendered
    assert rendered.count("altitude not encoded radially") == 2
    assert "Leader → Follower 1" in rendered
    assert "Follower 1 → Follower 2" in rendered
    assert "Follower 2 → Leader" in rendered
    assert "current.cyclic_gap_deg[index]" in rendered
    assert "current.cyclic_gap_error_deg[index]" in rendered


def test_state_prefixes_are_censored_but_full_plan_is_visible(artifact: dict) -> None:
    rendered = render_cubesat_replay(artifact)
    browser_script = re.findall(r"<script>(.*?)</script>", rendered, flags=re.DOTALL)[-1]

    assert ".frames.slice(0,frameIndex+1)" in browser_script
    assert "replay.plan.U[satellite].forEach" in browser_script
    assert "replay.plan.U[satellite][actionDay]" in browser_script
    assert "complete open-loop drag plan is visible from the start" in rendered
    assert 'class="heatmap"' in rendered
    assert rendered.count('<meter min="0" max="1" value="0" data-duty-meter') == 3
    assert rendered.count("whole-plan duty") == 3
    assert 'aria-label="Complete 180-day high-drag command heatmap available from day zero"' in rendered

    for forbidden in (
        "solve_ivp",
        "scipy",
        "casadi",
        "optimize",
        "atmospheric_density",
        "drag_acceleration",
        "propagate_orbit",
        "fabricate",
    ):
        assert forbidden not in browser_script


def test_only_whitelisted_artifact_fields_are_embedded(artifact: dict) -> None:
    rendered = render_cubesat_replay(artifact)

    assert "solver_state_that_must_not_ship" not in rendered
    assert "atmosphere_model_that_must_not_ship" not in rendered
    assert "optimizer_diagnostics_that_must_not_ship" not in rendered
    assert "generated_by" not in rendered
    assert "private_metric" not in rendered
    assert "fixture limitation" not in rendered
    assert "unused_unit" not in rendered


def test_fallback_is_hidden_only_after_successful_initial_render(artifact: dict) -> None:
    rendered = render_cubesat_replay(artifact)

    initialization = rendered.index(
        "RUNS.forEach(makeOrbit); drawHeatmap(); configureCharts(); render();"
    )
    hiding = rendered.index("fallback.hidden=true")
    assert initialization < hiding
    assert 'const fallbackId="fig-cubesat-formation-fallback"' in rendered
    assert 'fallback.setAttribute("aria-hidden","true")' in rendered
    assert "window.parent&&window.parent!==window" in rendered
    assert "fallbackObserver.observe" in rendered


def test_unique_roots_path_input_and_script_escaping(
    artifact: dict, tmp_path: Path
) -> None:
    first = render_cubesat_replay(artifact, replay_id="cubesat")
    second_artifact = deepcopy(artifact)
    second_artifact["spacecraft"][1] = "bad</script><script>alert(1)</script>"
    path = tmp_path / "cubesat.json"
    path.write_text(json.dumps(second_artifact), encoding="utf-8")
    second = render_cubesat_replay(path, replay_id="cubesat")

    first_id = re.search(r'<section id="([^"]+)"', first).group(1)
    second_id = re.search(r'<section id="([^"]+)"', second).group(1)
    assert first_id != second_id
    assert "</script><script>alert(1)</script>" not in second
    assert "\\u003c/script" in second


def test_root_plan_and_run_shape_errors_are_rejected(artifact: dict) -> None:
    wrong_schema = deepcopy(artifact)
    wrong_schema["schema_version"] = 2
    with pytest.raises(CubeSatReplayError, match="unsupported"):
        render_cubesat_replay(wrong_schema)

    unfinished = deepcopy(artifact)
    unfinished["status"] = "running"
    with pytest.raises(CubeSatReplayError, match="status 'complete'"):
        render_cubesat_replay(unfinished)

    missing_run = deepcopy(artifact)
    del missing_run["runs"]["nonlinear_replay"]
    with pytest.raises(CubeSatReplayError, match="missing run"):
        render_cubesat_replay(missing_run)

    transposed_plan = deepcopy(artifact)
    transposed_plan["plan"]["U"] = list(map(list, zip(*transposed_plan["plan"]["U"])))
    with pytest.raises(CubeSatReplayError, match="one row per spacecraft"):
        render_cubesat_replay(transposed_plan)

    bad_frame_day = deepcopy(artifact)
    bad_frame_day["runs"]["nominal_linear"]["frames"][2]["day"] = 2.5
    with pytest.raises(CubeSatReplayError, match="frame days"):
        render_cubesat_replay(bad_frame_day)


def test_nonfinite_bounds_and_cross_field_disagreement_are_rejected(
    artifact: dict,
) -> None:
    nonfinite = deepcopy(artifact)
    nonfinite["runs"]["nonlinear_replay"]["frames"][5]["altitude_km"][1] = float("nan")
    with pytest.raises(CubeSatReplayError, match="must be finite"):
        render_cubesat_replay(nonfinite)

    bad_command = deepcopy(artifact)
    bad_command["plan"]["U"][0][0] = 1.1
    with pytest.raises(CubeSatReplayError, match=r"plan\.U values"):
        render_cubesat_replay(bad_command)

    bad_duty = deepcopy(artifact)
    bad_duty["plan"]["duty_fraction"][0] += 0.1
    with pytest.raises(CubeSatReplayError, match="duty_fraction disagrees"):
        render_cubesat_replay(bad_duty)

    bad_gap = deepcopy(artifact)
    bad_gap["runs"]["nominal_linear"]["frames"][3]["cyclic_gap_deg"][0] += 0.2
    with pytest.raises(CubeSatReplayError, match="cyclic_gap_deg disagrees"):
        render_cubesat_replay(bad_gap)

    bad_error = deepcopy(artifact)
    bad_error["runs"]["nominal_linear"]["frames"][3]["cyclic_gap_error_deg"][0] += 0.2
    with pytest.raises(CubeSatReplayError, match="cyclic_gap_error_deg disagrees"):
        render_cubesat_replay(bad_error)

    bad_digest = deepcopy(artifact)
    bad_digest["plan"]["identity_sha256"] = "not-a-digest"
    with pytest.raises(CubeSatReplayError, match="SHA-256"):
        render_cubesat_replay(bad_digest)


def test_invalid_json_and_required_containers_are_rejected(
    artifact: dict, tmp_path: Path
) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(CubeSatReplayError, match="invalid CubeSat replay JSON"):
        render_cubesat_replay(invalid_json)

    wrong_units = deepcopy(artifact)
    wrong_units["units"]["angle"] = "rad"
    with pytest.raises(CubeSatReplayError, match="units.angle"):
        render_cubesat_replay(wrong_units)

    missing_metrics = deepcopy(artifact)
    del missing_metrics["metrics"]["validation"]
    with pytest.raises(CubeSatReplayError, match="metrics must contain"):
        render_cubesat_replay(missing_metrics)
