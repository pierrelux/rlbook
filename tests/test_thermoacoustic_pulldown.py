"""Physics, recorded-plan, and finer-replay checks for refrigerator pull-down."""
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
from thermoacoustic_pulldown import (FridgeParameters, SCENARIOS, audit, dynamics,
                                     initial_controls, initial_state, make_problem, powers)
from trajectory_optimization import SolverOptions, solve_trajectory


@pytest.fixture(scope="module")
def recorded():
    return json.loads((ROOT / "artifacts" / "thermoacoustic_pulldown" / "metrics.json").read_text())


def test_energy_balance_cancels_pumped_heat():
    p = FridgeParameters()
    x, u = np.array([12., 25.]), np.array([0.7])
    cold_rate, hot_rate = np.asarray(dynamics(x, u, p))
    _, work = powers(x, u, p)
    combined = p.cold_capacity * cold_rate + p.hot_capacity * hot_rate
    expected = p.load + work - p.hot_conductance * (x[1] - p.ambient)
    assert combined == pytest.approx(expected, abs=1e-12)


def test_critical_span_zero_driver_and_curvature_sign():
    p = FridgeParameters()
    u = np.array([0.6])
    assert float(powers(np.array([0., p.critical_span]), u, p)[0]) == pytest.approx(0.)
    assert float(powers(np.array([0., p.critical_span + 1.]), u, p)[0]) < 0
    assert float(powers(np.array([20., 20.]), u, p)[1]) > 0
    cooling, work = powers(np.array([20., 20.]), np.zeros(1), p)
    assert float(cooling) == 0 and float(work) == 0

    # With span = T_h - T_c, the mixed cold-rate derivative is positive.
    cold_rate = lambda span, amplitude: dynamics(
        jnp.array([10., 10. + span]), jnp.array([amplitude]), p)[0]
    mixed = jax.grad(jax.grad(cold_rate, argnums=1), argnums=0)(12., 0.6)
    assert float(mixed) == pytest.approx(
        2 * p.pumping * 0.6 / (p.cold_capacity * p.critical_span), rel=1e-12)


def test_controls_are_bounded_and_out_of_box_seed_is_rejected():
    p = FridgeParameters()
    problem = make_problem(p)
    np.testing.assert_array_equal(problem.lower, [0.])
    np.testing.assert_array_equal(problem.upper, [1.])
    with pytest.raises(ValueError, match="bounds"):
        solve_trajectory(problem, initial_state(p), np.full((p.steps, 1), 1.01),
                         options=SolverOptions(max_iterations=0))


def test_baseline_ilqr_reproduces_recorded_cost_and_cold_arrival(recorded):
    p = FridgeParameters()
    report = next(r for r in recorded["runs"]
                  if r["scenario"] == "baseline" and r["method"] == "ilqr")
    result = solve_trajectory(make_problem(p), initial_state(p), initial_controls(p),
                              method="ilqr", options=SolverOptions(max_iterations=200))
    assert result.status == "converged"
    assert result.history[-1]["cost"] == pytest.approx(report["cost"], rel=0, abs=1e-6)
    assert result.states[-1, 0] == pytest.approx(report["terminal_cold_C"], rel=0, abs=1e-6)
    assert np.all(np.diff([record["cost"] for record in result.history]) < 0)


def test_every_recorded_plan_passes_finer_replay_and_target_checks(recorded):
    assert recorded["parameters"] == asdict(FridgeParameters())
    assert len(recorded["runs"]) == 2 * len(SCENARIOS)
    for report in recorded["runs"]:
        cfg = SCENARIOS[report["scenario"]]
        p = cfg["parameters"]
        states, controls = np.asarray(report["states"]), np.asarray(report["controls"])
        checked = audit(states, controls, p, require_target=cfg["target_reachable"])
        assert checked["pulldown_pass"], (report["scenario"], report["method"], checked)
        assert checked["finite"] and checked["control_violation"] == 0
        assert checked["fine_replay_max_temperature_difference_K"] < 1e-4
        if cfg["target_reachable"]:
            assert checked["target_reached"]
            assert checked["terminal_cold_error_K"] < 0.6
        else:
            assert not checked["target_reached"]
        problem = make_problem(p)
        np.testing.assert_allclose(problem.rollout(initial_state(p), controls), states,
                                   rtol=1e-10, atol=1e-9)
        assert problem.cost(states, controls) == pytest.approx(report["cost"], abs=1e-6)
        assert report["status"] in ("converged", "iteration_limit")
        accepted_costs = np.asarray(report["accepted_costs"])
        assert len(accepted_costs) == report["accepted_iterations"] + 1
        assert np.all(np.diff(accepted_costs) < 0)


def test_no_minor_loss_exposes_different_stopping_statuses(recorded):
    runs = {r["method"]: r for r in recorded["runs"]
            if r["scenario"] == "no_minor_loss"}
    assert runs["ilqr"]["status"] == "iteration_limit"
    assert runs["ddp"]["status"] == "converged"
    assert runs["ilqr"]["accepted_iterations"] == 200
    assert runs["ilqr"]["cost"] == pytest.approx(runs["ddp"]["cost"], abs=1e-3)


def test_artifacts_track_sources_and_have_vector_and_raster_figures(recorded):
    for path, digest in recorded["provenance"]["sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for name in ("pulldown", "variants", "convergence"):
        for extension in ("svg", "pdf", "png"):
            path = ROOT / "_static" / "thermoacoustic_pulldown" / f"{name}.{extension}"
            assert path.stat().st_size > 1000
