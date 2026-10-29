"""Checks for the radiotherapy fractionation model, its artifacts, and chapter wiring."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from radiotherapy_fractionation import (CourseParameters, dose_matrix, gaps,
                                        initial_state, make_problem, make_step)

DATA = ROOT / "interactive" / "radiotherapy-fractionation-data.json"
METRICS = ROOT / "artifacts" / "radiotherapy_fractionation" / "metrics.json"


@pytest.fixture(scope="module")
def A():
    return dose_matrix()


def test_each_beam_crosses_one_organ(A):
    assert np.allclose(A[0], 1.0)
    organs = A[1:]
    assert np.all(np.diag(organs) > 1.0)
    assert np.all(organs[~np.eye(3, dtype=bool)] < 0.01)


def test_step_matches_linear_quadratic_formula(A):
    p = CourseParameters()
    step = make_step(A, p)
    h = np.array([0.6, -0.3, -0.2, -0.4])
    u = np.array([0.5, 1.0, 2.0])
    for t in (0, 4):
        d = A @ u
        damaged = h - (np.array(p.alpha) * d + np.array(p.beta) * d ** 2) / np.array(p.effect_scale)
        g = gaps(p)[t]
        factor = np.exp(-np.array(p.repair) * g)
        factor[0] = np.exp(p.tumor_growth * g)
        assert np.allclose(step(h, u, t), damaged * factor)
    assert gaps(p).tolist().count(3.0) == 4


def test_recorded_plans_replay_and_respect_limits(A):
    data, metrics = json.loads(DATA.read_text()), json.loads(METRICS.read_text())
    assert metrics["validation_issues"] == []
    problem = make_problem(A)
    for method in ("ilqr", "ddp"):
        final = data["runs"][method]["iterates"][-1]
        controls = np.array(final["controls"])
        states = problem.rollout(initial_state(), controls)
        assert np.allclose(states, final["states"], atol=1e-4)
        assert np.all(controls >= 0) and np.all(controls <= CourseParameters().max_beam)
        assert states[-1, 0] < 0.01 and states[:, 1:].min() > -1
        costs = [item["cost"] for item in data["runs"][method]["iterates"]]
        assert np.all(np.diff(costs) < 0)
    assert np.array(data["uniform"]["states"])[:, 1].min() < -2


def test_chapter_embeds_replay_with_static_placeholder():
    chapter = (ROOT / "iterative-trajectory-optimization.md").read_text()
    assert "../interactive/radiotherapy-fractionation.html" in chapter
    assert ":placeholder: _static/radiotherapy_fractionation/plan.svg" in chapter
    for name in ("plan", "iterations"):
        for suffix in ("svg", "pdf", "png"):
            assert (ROOT / "_static" / "radiotherapy_fractionation" / f"{name}.{suffix}").exists()
