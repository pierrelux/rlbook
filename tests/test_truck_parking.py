"""Kinematics, reproducibility, sensitivity, and wiring checks for the truck demo."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from truck_parking import (SCENARIOS, TruckParameters, audit, bay_state, body_corners,
                           configuration, dynamics, hitch_sensitivity, initial_controls,
                           lane_state, linearized_amplification, pull_out_plan, rollout,
                           shooting_objective, solve_single_shooting, steering_amplification,
                           trailer_axle)

PAGE = ROOT / "interactive" / "truck-parking.html"
DATA = ROOT / "interactive" / "truck-parking-data.json"
CHAPTER = ROOT / "numerical-trajectory-optimization.md"


@pytest.fixture(scope="module")
def recorded():
    return json.loads(DATA.read_text())


@pytest.fixture(scope="module")
def metrics():
    return json.loads((ROOT / "artifacts" / "truck_parking" / "metrics.json").read_text())


def test_straight_driving_and_steering_signs():
    p = TruckParameters()
    x = np.array([3., 4., .3, .3])
    forward = np.asarray(dynamics(x, np.array([1., 0.]), p))
    np.testing.assert_allclose(forward, [np.cos(.3), np.sin(.3), 0., 0.])
    turning = np.asarray(dynamics(x, np.array([1., .2]), p))
    assert turning[2] == pytest.approx(np.tan(.2) / p.tractor_wheelbase)
    assert turning[3] == 0.
    # With the tractor turned left of the trailer, backing swings the trailer heading down.
    bent = np.array([0., 0., .4, 0.])
    assert float(dynamics(bent, np.array([-1., 0.]), p)[3]) < 0
    assert float(dynamics(bent, np.array([1., 0.]), p)[3]) > 0


def test_hitch_angle_is_stable_forward_and_unstable_in_reverse():
    p = TruckParameters()

    def hitch_rate(beta, speed):
        x = jnp.array([0., 0., beta, 0.])
        rate = dynamics(x, jnp.array([speed, 0.]), p)
        return rate[2] - rate[3]

    forward = float(jax.grad(hitch_rate)(0., 1.))
    reverse = float(jax.grad(hitch_rate)(0., -1.))
    assert forward == pytest.approx(-1 / p.trailer_length)
    assert reverse == pytest.approx(1 / p.trailer_length)


def test_docked_rig_geometry():
    p = TruckParameters()
    docked = bay_state(p)
    axle = np.asarray(trailer_axle(jnp.asarray(docked), p))
    corners = np.asarray(body_corners(jnp.asarray(docked), p))
    assert axle[0] == pytest.approx(p.bay_x)
    assert corners[:, 1].min() == pytest.approx(p.bumper_gap)
    assert docked[2] == docked[3] == pytest.approx(np.pi / 2)
    assert lane_state(p)[2] == 0. and configuration("lane", p)[1] == p.lane_y
    with pytest.raises(ValueError):
        configuration("roof", p)


def test_seed_outside_the_box_is_rejected():
    p = TruckParameters()
    with pytest.raises(ValueError, match="bounds"):
        solve_single_shooting(lane_state(p), bay_state(p), np.tile([-2., 0.], (p.steps, 1)), p,
                              max_iterations=0)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_solve_reproduces_recorded_run_and_passes_finer_replay(recorded, scenario):
    p, cfg = TruckParameters(), SCENARIOS[scenario]
    initial, goal = configuration(cfg["initial"], p), configuration(cfg["goal"], p)
    options = recorded["provenance"]["solver_options"]
    result = solve_single_shooting(initial, goal, initial_controls(p, cfg["seed_speed"]), p,
                                   **options)
    saved = recorded["scenarios"][scenario]
    report = audit(initial, result.controls, goal, p)
    assert result.status == saved["metrics"]["status"]
    assert result.iterations == saved["metrics"]["iterations"]
    assert result.cost_trace[-1] == pytest.approx(saved["metrics"]["cost"], rel=1e-8)
    np.testing.assert_allclose(result.controls, saved["history"][-1]["controls"], atol=2e-6, rtol=0)
    np.testing.assert_allclose(result.states, saved["history"][-1]["states"], atol=2e-6, rtol=0)
    assert report["docking_pass"]
    assert report["fine_replay_max_position_difference_m"] < 1e-6
    assert np.all(np.diff(result.cost_trace) <= 0)
    lower, upper = -np.array([p.speed_limit, p.steering_limit]), np.array([p.speed_limit, p.steering_limit])
    assert np.all(result.controls >= lower) and np.all(result.controls <= upper)


def test_recorded_checkpoints_are_rollouts_and_share_the_seed_shape(recorded):
    p = TruckParameters()
    assert recorded["schema_version"] == 1
    assert recorded["parameters"] == asdict(TruckParameters())
    assert set(recorded["scenarios"]) == set(SCENARIOS)
    for key, scenario in recorded["scenarios"].items():
        initial, goal = np.asarray(scenario["initial"]), np.asarray(scenario["goal"])
        np.testing.assert_allclose(initial, configuration(SCENARIOS[key]["initial"], p))
        np.testing.assert_allclose(goal, configuration(SCENARIOS[key]["goal"], p))
        history, trace = scenario["history"], np.asarray(scenario["cost_trace"])
        assert history[0]["iteration"] == 0
        assert history[-1]["iteration"] == scenario["metrics"]["iterations"] == len(trace) - 1
        assert [h["iteration"] for h in history] == sorted({h["iteration"] for h in history})
        np.testing.assert_array_equal(history[0]["controls"],
                                      initial_controls(p, SCENARIOS[key]["seed_speed"]))
        roll = jax.jit(lambda us: rollout(initial, us, p))
        objective = jax.jit(lambda us: shooting_objective(us, initial, goal, p))
        for entry in history:
            us, xs = np.asarray(entry["controls"]), np.asarray(entry["states"])
            assert xs.shape == (p.steps + 1, 4) and us.shape == (p.steps, 2)
            # Controls are stored to six decimals; the rollout amplifies that rounding.
            np.testing.assert_allclose(np.asarray(roll(jnp.asarray(us))), xs, atol=5e-5, rtol=0)
            assert float(objective(jnp.asarray(us))) == pytest.approx(entry["cost"], rel=1e-5)
            assert entry["cost"] == pytest.approx(trace[entry["iteration"]], rel=1e-9)
            assert np.max(np.abs(us[:, 0])) <= p.speed_limit + 1e-12
            assert np.max(np.abs(us[:, 1])) <= p.steering_limit + 1e-12


def test_amplification_matches_per_interval_product_and_direction(recorded):
    p = TruckParameters()
    dock = recorded["scenarios"]["dock"]
    initial = np.asarray(dock["initial"])
    controls = np.asarray(dock["history"][-1]["controls"])
    states = np.asarray(rollout(initial, controls, p))
    autodiff = steering_amplification(initial, controls, p)
    product = linearized_amplification(states, controls, p)
    moving = ~np.isnan(autodiff)
    assert moving[0] and np.max(np.abs(autodiff[moving] / product[moving] - 1)) < 0.01
    assert autodiff[0] > 10
    assert np.all(np.diff(autodiff[moving]) <= 1e-9)
    # The same path driven forward forgets an early steering error instead.
    back_initial, back_controls = pull_out_plan(states, controls)
    forward = steering_amplification(back_initial, back_controls, p)
    first = forward[~np.isnan(forward)][0]
    assert first < 0.1 and abs(first - dock["metrics"]["reversed_plan_first_moving_amplification"]) < 1e-6
    returned = np.asarray(rollout(back_initial, back_controls, p))[-1]
    assert np.linalg.norm(returned[:2] - initial[:2]) < 1e-6
    # Reverse mode agrees with the JSON up to the six-decimal rounding of the controls.
    saved = np.asarray([np.nan if v is None else v for v in dock["amplification"]["autodiff"]])
    np.testing.assert_allclose(autodiff[moving], saved[moving], rtol=1e-4)
    assert hitch_sensitivity(initial, controls, p).shape == (p.steps,)


def test_metrics_record_current_sources_and_vector_figures(recorded, metrics):
    for path, digest in recorded["provenance"]["sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert metrics["provenance"] == recorded["provenance"]
    assert {run["scenario"] for run in metrics["runs"]} == set(SCENARIOS)
    for run in metrics["runs"]:
        assert run["docking_pass"] and run["status"] in ("converged", "small_decrease")
        assert run["cost"] == pytest.approx(recorded["scenarios"][run["scenario"]]["metrics"]["cost"])
    for name in ("iterations", "paths", "plan", "conditioning"):
        for extension in ("svg", "pdf", "png"):
            assert (ROOT / "_static" / "truck_parking" / f"{name}.{extension}").stat().st_size > 1000


def test_replay_page_and_chapter_are_wired():
    html = PAGE.read_text(encoding="utf-8")
    for token in ("truck-parking-data.json", "prefers-reduced-motion", 'role="img"',
                  "schema_version"):
        assert token in html
    for forbidden in ("<script src=", "<link", "http://", "https://"):
        assert forbidden not in html
    chapter = CHAPTER.read_text(encoding="utf-8")
    start = chapter.find(":::{iframe} ../interactive/truck-parking.html")
    assert start >= 0, "truck replay iframe missing"
    block = chapter[start:chapter.find("\n:::", start + 10)]
    for option in (":class: truck-parking-replay", ":placeholder: _static/truck_parking/paths.svg"):
        assert option in block
    assert "{figure} _static/truck_parking/paths.svg" not in chapter
    assert ":::{include} artifacts/truck_parking/results.md" in chapter
    for figure in ("iterations", "plan", "conditioning"):
        assert f"{{figure}} _static/truck_parking/{figure}.svg" in chapter
    css = (ROOT / "_static/custom.css").read_text(encoding="utf-8")
    assert ".truck-parking-replay > div > div" in css
    plugin = (ROOT / "plugins/pdf-static-parity.mjs").read_text(encoding="utf-8")
    assert "truck_parking" in plugin
