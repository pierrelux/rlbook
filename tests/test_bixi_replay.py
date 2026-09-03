"""Checks for the BIXI recorded player and static fallbacks."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import sys

import matplotlib.pyplot as plt
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from bixi_replay import (  # noqa: E402
    BixiReplayDataError,
    make_censoring_figure,
    make_feedback_evidence_figure,
    make_model_interface_figure,
    render_bixi_replay,
)


ARTIFACT = ROOT / "artifacts" / "bixi" / "textbook_results.json"


def test_player_uses_only_recorded_frames_and_has_accessible_controls() -> None:
    rendered = render_bixi_replay(ARTIFACT, replay_id="bixi-test")
    assert 'id="bixi-test"' in rendered
    assert 'type="range"' in rendered
    assert 'aria-live="polite"' in rendered
    assert 'role="img"' in rendered
    assert "prefers-reduced-motion" in rendered
    assert "The browser only replays committed Python trajectories" in rendered
    assert 'const fallbackId="fig-bixi-replay-fallback"' in rendered
    assert "fallback.hidden=true" in rendered
    assert "fallback.setAttribute('aria-hidden','true')" in rendered
    assert "MutationObserver" in rendered
    assert "window.parent && window.parent!==window" in rendered
    assert "fetch(" not in rendered
    assert "XMLHttpRequest" not in rendered
    assert "WebSocket" not in rendered
    assert "autoplay" not in rendered.lower()
    assert len(re.findall(r'"minute":180', rendered)) == 3


def test_path_loader_rejects_a_stale_input(tmp_path: Path) -> None:
    repository = tmp_path
    copied_artifact = repository / "artifacts" / "bixi" / "textbook_results.json"
    copied_artifact.parent.mkdir(parents=True)
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for relative in payload["metadata"]["input_files"]:
        source = ROOT / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    copied_artifact.write_text(json.dumps(payload), encoding="utf-8")
    render_bixi_replay(copied_artifact)
    (repository / "data" / "bixi" / "stations.json").write_text(
        "changed\n", encoding="utf-8"
    )
    with pytest.raises(BixiReplayDataError, match="stale"):
        render_bixi_replay(copied_artifact)


def test_all_static_figures_render() -> None:
    for factory in (
        make_model_interface_figure,
        make_feedback_evidence_figure,
        make_censoring_figure,
    ):
        figure = factory(ARTIFACT)
        assert figure.axes
        figure.canvas.draw()
        plt.close(figure)
