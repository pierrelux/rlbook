"""Consistency checks for the browser replay data and its chapter wiring."""
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from thermoacoustic_pulldown import (FridgeParameters, SCENARIOS, initial_state,
                                     make_problem, powers, stage_cost, step,
                                     terminal_cost)

PAGE = ROOT / "interactive" / "thermoacoustic-refrigerator.html"
DATA = ROOT / "interactive" / "thermoacoustic-refrigerator-data.json"
CHAPTER = ROOT / "iterative-trajectory-optimization.md"


@pytest.fixture(scope="module")
def replay():
    return json.loads(DATA.read_text())


@pytest.fixture(scope="module")
def metrics():
    return json.loads((ROOT / "artifacts" / "thermoacoustic_pulldown" / "metrics.json").read_text())


def ledger(states, controls, p):
    stage = sum(float(stage_cost(x, np.array([u]), p)) for x, u in zip(states[:-1], controls))
    return stage + float(terminal_cost(states[-1], p))


def test_replay_data_matches_metrics_and_model(replay, metrics):
    assert replay["schema_version"] == 1
    assert replay["parameters"] == asdict(FridgeParameters())
    assert set(replay["scenarios"]) == set(SCENARIOS)
    for key, cfg in SCENARIOS.items():
        scenario = replay["scenarios"][key]
        assert scenario["parameters"] == asdict(cfg["parameters"])
        assert scenario["name"] == cfg["name"]
        p = cfg["parameters"]
        for method in ("ilqr", "ddp"):
            run = scenario["runs"][method]
            recorded = next(r for r in metrics["runs"]
                            if r["scenario"] == key and r["method"] == method)
            states, controls = np.asarray(run["states"]), np.asarray(run["controls"])
            assert states.shape == (p.steps + 1, 2) and controls.shape == (p.steps,)
            np.testing.assert_allclose(states, recorded["states"], atol=1e-5, rtol=0)
            np.testing.assert_allclose(controls, np.asarray(recorded["controls"])[:, 0],
                                       atol=1e-5, rtol=0)
            assert run["cost"] == pytest.approx(recorded["cost"], rel=1e-6)
            assert run["energy_J"] == pytest.approx(recorded["energy_J"], rel=1e-6)
            assert run["status"] == recorded["status"]
            assert np.all(controls >= 0) and np.all(controls <= 1)
            # The page rolls the recorded controls through the same RK4 step.
            x, replayed = initial_state(p), [initial_state(p)]
            for u in controls:
                x = np.asarray(step(x, np.array([u]), p))
                replayed.append(x)
            np.testing.assert_allclose(np.asarray(replayed), states, atol=1e-5, rtol=0)
            # The page's stage-plus-terminal ledger must reproduce the recorded cost.
            assert ledger(states, controls, p) == pytest.approx(run["cost"], abs=1e-3)

    full = replay["full_amplitude_comparison"]
    p = SCENARIOS[full["scenario"]]["parameters"]
    problem = make_problem(p)
    ones = np.full((p.steps, 1), full["amplitude"])
    expected = problem.rollout(initial_state(p), ones)
    np.testing.assert_allclose(full["states"], expected, atol=1e-5, rtol=0)
    assert full["cost"] == pytest.approx(float(problem.cost(expected, ones)), rel=1e-6)
    assert full["cost"] == pytest.approx(metrics["full_amplitude_comparison"]["cost"], rel=1e-6)
    assert np.all(np.asarray(full["controls"]) == full["amplitude"])


def test_page_is_self_contained_and_reads_the_data_file():
    html = PAGE.read_text()
    assert "thermoacoustic-refrigerator-data.json" in html
    for token in ("(a) device", "(b) lumped model", "prefers-reduced-motion", 'role="img"',
                  "schema_version", "aria-live"):
        assert token in html
    for forbidden in ("<script src=", "<link", "http://", "https://"):
        assert forbidden not in html


def test_page_model_matches_python_powers():
    """The JS power law is transcribed by hand; pin its coefficients to the Python model."""
    html = PAGE.read_text()
    js = html[html.index("function powers"):html.index("function dynamics")]
    assert "p.pumping * u * u * eta" in js
    assert "p.work * u * u * eta + p.viscous * u * u + p.minor_loss * u * u * u" in js
    assert "1 - (x[1] - x[0]) / p.critical_span" in js
    p = FridgeParameters()
    cooling, work = powers(np.array([12., 25.]), np.array([.7]), p)
    eta = 1 - 13 / p.critical_span
    assert float(cooling) == pytest.approx(p.pumping * .49 * eta)
    assert float(work) == pytest.approx(p.work * .49 * eta + p.viscous * .49 + p.minor_loss * .343)


def test_chapter_embeds_replay_with_static_placeholder():
    text = CHAPTER.read_text()
    block = re.search(r":::\{iframe\} \.\./interactive/thermoacoustic-refrigerator\.html\n(.*?)\n:::",
                      text, re.S)
    assert block, "chapter does not embed the thermoacoustic replay"
    body = block.group(1)
    for option in (":label: fig-thermoacoustic-geometry", ":class: thermoacoustic-replay",
                   ":placeholder: _static/thermoacoustic_pulldown/geometry.svg"):
        assert option in body
    assert "{figure} _static/thermoacoustic_pulldown/geometry.svg" not in text
    assert text.count("fig-thermoacoustic-geometry`a") == 1
    assert text.count("fig-thermoacoustic-geometry`b") == 1
    for path in (PAGE, DATA, ROOT / "_static/thermoacoustic_pulldown/geometry.svg",
                 ROOT / "_static/thermoacoustic_pulldown/geometry.pdf"):
        assert path.exists(), path
    css = (ROOT / "_static" / "custom.css").read_text()
    assert ".thermoacoustic-replay > div > div" in css
    plugin = (ROOT / "plugins" / "pdf-static-parity.mjs").read_text()
    assert "promoteIframePlaceholder" in plugin and "placeholder === true" in plugin
    assert re.search(r"_static\\/thermoacoustic_pulldown", plugin)
