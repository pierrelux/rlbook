from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import sys

import pytest


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIRECTORY))

from gimbal_control import (  # noqa: E402
    GimbalParameters,
    GimbalScenario,
    comparison_to_artifact,
    run_comparison,
)
from gimbal_replay import GimbalReplayError, render_gimbal_replay  # noqa: E402


@pytest.fixture(scope="module")
def artifact():
    parameters = GimbalParameters()
    scenario = GimbalScenario()
    return comparison_to_artifact(
        run_comparison(parameters, scenario), parameters, scenario
    )


def test_player_is_scoped_accessible_causal_and_offline(artifact) -> None:
    rendered = render_gimbal_replay(artifact, replay_id="lecture-gimbal")
    root_match = re.search(r'<section id="([^"]+)"', rendered)

    assert root_match is not None
    root = root_match.group(1)
    assert root.startswith("lecture-gimbal-")
    assert f"#{root} .controls" in rendered
    assert 'data-action="play"' in rendered
    assert 'data-action="step"' in rendered
    assert 'data-action="reset"' in rendered
    assert 'aria-live="polite"' in rendered
    assert 'type="range"' in rendered
    assert "prefers-reduced-motion: reduce" in rendered
    assert "run.frames.slice(0, index + 1)" in rendered
    assert "toggleAttribute" in rendered
    assert "fig-gimbal-observation-fallback" in rendered
    assert 'fallback.hidden = true' in rendered
    assert "MutationObserver" in rendered
    assert "fetch(" not in rendered
    assert "XMLHttpRequest" not in rendered
    assert "WebSocket" not in rendered
    assert "autoplay" not in rendered.lower()
    assert len(rendered.encode("utf-8")) < 1_000_000


def test_each_player_has_a_unique_root(artifact) -> None:
    first = render_gimbal_replay(artifact, replay_id="gimbal")
    second = render_gimbal_replay(artifact, replay_id="gimbal")
    first_id = re.search(r'<section id="([^"]+)"', first).group(1)
    second_id = re.search(r'<section id="([^"]+)"', second).group(1)

    assert first_id != second_id


def test_path_input_and_script_termination_escaping(artifact, tmp_path: Path) -> None:
    modified = deepcopy(artifact)
    modified["runs"]["gyro"]["label"] = "bad</script><script>alert(1)</script>"
    path = tmp_path / "gimbal.json"
    path.write_text(json.dumps(modified), encoding="utf-8")
    rendered = render_gimbal_replay(path)

    assert "</script><script>alert(1)</script>" not in rendered
    assert "\\u003c/script" in rendered


def test_invalid_or_misaligned_replay_is_rejected(artifact) -> None:
    invalid = deepcopy(artifact)
    invalid["runs"]["gyro"]["frames"][1]["time_s"] = 0.0
    with pytest.raises(GimbalReplayError, match="strictly increasing"):
        render_gimbal_replay(invalid)

    misaligned = deepcopy(artifact)
    misaligned["runs"]["gyro"]["frames"][1]["time_s"] += 0.001
    with pytest.raises(GimbalReplayError, match="same replay time grid"):
        render_gimbal_replay(misaligned)

    wrong_horizon = deepcopy(artifact)
    wrong_horizon["scenario"]["duration_s"] = 11.0
    with pytest.raises(GimbalReplayError, match="must end"):
        render_gimbal_replay(wrong_horizon)
