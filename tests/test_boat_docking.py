"""Physics, reproducibility, and independent acceptance checks for docking."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from boat_docking import (BoatParameters, SCENARIOS, audit, dynamics, hull_corners,
                         initial_controls, make_problem)
from trajectory_optimization import solve_trajectory


@pytest.fixture(scope="module")
def problem():
    return make_problem()


@pytest.fixture(scope="module")
def recorded():
    return json.loads((ROOT / "interactive" / "boat-docking-data.json").read_text())


def test_unforced_drag_dissipates_kinetic_energy_and_thrusters_turn():
    p = BoatParameters()
    x = np.array([0., 3., .4, .6, -.3, .2])
    dx = np.asarray(dynamics(x, np.zeros(2), p))
    energy_derivative = p.mass*x[3:5]@dx[3:5] + p.inertia*x[5]*dx[5]
    forward, side = np.array([np.cos(x[2]), np.sin(x[2])]), np.array([-np.sin(x[2]), np.cos(x[2])])
    expected = (-p.longitudinal_drag*(forward@x[3:5])**2
                -p.lateral_drag*(side@x[3:5])**2-p.angular_drag*x[5]**2)
    assert energy_derivative == pytest.approx(expected)
    assert energy_derivative < 0
    resting = np.array([0., 3., 0., 0., 0., 0.])
    both = np.asarray(dynamics(resting, np.ones(2), p))
    opposite = np.asarray(dynamics(resting, np.array([-1., 1.]), p))
    assert both[3] > 0 and both[5] == 0
    assert opposite[3] == 0 and opposite[5] > 0


def test_hull_clearance_uses_all_corners():
    p = BoatParameters()
    x = np.array([0., .8, np.pi/2, 0., 0., 0.])
    corners = np.asarray(hull_corners(x, p))
    assert x[1] > 0
    assert corners[:, 1].min() == pytest.approx(-.2)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("method", ["ilqr", "ddp"])
def test_docking_solve_and_fine_replay_match_recorded_experiment(problem, recorded, scenario, method):
    p = BoatParameters()
    result = solve_trajectory(problem, SCENARIOS[scenario]["initial"], initial_controls(p), method)
    report = audit(result.states, result.controls, p)
    saved = recorded["scenarios"][scenario]["runs"][method]
    assert result.converged and report["docking_pass"]
    assert result.projected_gradient <= 1e-5
    assert np.all(np.diff([r["cost"] for r in result.history]) < 0)
    assert np.max(np.abs(result.controls)) <= 1
    assert report["fine_replay_max_position_difference_m"] < 1e-6
    assert len(result.history) == len(saved["history"])
    np.testing.assert_allclose(result.states, saved["history"][-1]["states"], atol=2e-8, rtol=1e-8)
    np.testing.assert_allclose(result.controls, saved["history"][-1]["controls"], atol=2e-8, rtol=1e-8)
    assert result.history[-1]["cost"] == pytest.approx(saved["metrics"]["cost"], rel=1e-8)
    assert report["minimum_clearance_m"] > .3


def test_replay_histories_are_nonlinear_rollouts_and_share_initialization(problem, recorded):
    assert recorded["schema_version"] == 1
    assert recorded["parameters"] == asdict(BoatParameters())
    for scenario in recorded["scenarios"].values():
        histories = [scenario["runs"][method]["history"] for method in ("ilqr", "ddp")]
        np.testing.assert_array_equal(histories[0][0]["controls"], histories[1][0]["controls"])
        np.testing.assert_array_equal(histories[0][0]["states"], histories[1][0]["states"])
        for history in histories:
            for entry in history:
                xs, us = np.asarray(entry["states"]), np.asarray(entry["controls"])
                np.testing.assert_allclose(problem.rollout(scenario["initial"], us), xs, atol=2e-8, rtol=1e-8)
                assert problem.cost(xs, us) == pytest.approx(entry["cost"], rel=2e-8, abs=1e-6)
                assert np.all(np.isfinite(xs)) and np.max(np.abs(us)) <= 1


def test_artifacts_record_current_sources_and_pdf_companions(recorded):
    for path, digest in recorded["provenance"]["sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for figure in ("iterations", "paths", "convergence", "future-and-motion"):
        for extension in ("svg", "pdf", "png"):
            assert (ROOT / "_static" / "boat_docking" / f"{figure}.{extension}").stat().st_size > 1000
