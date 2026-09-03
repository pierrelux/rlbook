from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIRECTORY))

from cubesat_differential_drag import (  # noqa: E402
    CubeSatParameters,
    assert_valid_scenario,
    metrics_as_dict,
    nominal_dynamics_residual,
    run_scenario,
    validation_checks,
)


@pytest.fixture(scope="module")
def scenario():
    return run_scenario()


def test_physical_constants_derive_the_linear_model() -> None:
    params = CubeSatParameters()

    assert params.d_km_per_day == pytest.approx(
        0.045157234506305814,
        abs=1e-14,
    )
    assert params.alpha_deg_per_day2 == pytest.approx(
        0.05445030597065867,
        abs=1e-14,
    )
    assert params.linear_A == pytest.approx(
        np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    assert params.linear_B == pytest.approx(
        np.array(
            [
                0.5 * params.alpha_deg_per_day2,
                params.alpha_deg_per_day2,
                params.d_km_per_day,
            ]
        )
    )


def test_lexicographic_plan_is_feasible_immutable_and_smoother(scenario) -> None:
    params = scenario.parameters
    plan = scenario.plan
    action = plan.daily_high_drag_fraction

    assert action.shape == (180, 3)
    assert np.min(action) >= -1e-12
    assert np.max(action) <= 1.0 + 1e-12
    assert len(plan.plan_sha256) == 64
    assert scenario.nominal.plan_sha256 == plan.plan_sha256
    assert scenario.nonlinear.plan_sha256 == plan.plan_sha256
    with pytest.raises(ValueError):
        action[0, 0] = 0.5

    refined_loss = params.d_km_per_day * np.sum(action, axis=0)
    assert np.max(refined_loss) == pytest.approx(
        plan.refined_max_final_extra_loss_km,
        abs=1e-12,
    )
    assert plan.refined_max_final_extra_loss_km <= (
        plan.primary_max_final_extra_loss_km
        + params.primary_lock_tolerance_km
        + 5e-10
    )
    assert plan.refined_total_variation == pytest.approx(
        np.sum(np.abs(np.diff(action, axis=0))),
        abs=1e-10,
    )
    assert plan.refined_total_variation < 0.1 * plan.primary_total_variation


def test_nominal_rollout_satisfies_dynamics_and_terminal_constraints(scenario) -> None:
    params = scenario.parameters
    nominal = scenario.nominal
    residual = nominal_dynamics_residual(scenario.plan, nominal, params)
    target = np.asarray(params.target_cyclic_gaps_deg)

    assert nominal.time_days.shape == (181,)
    assert nominal.state.shape == (181, 3, 3)
    assert np.max(np.abs(residual)) < 1e-12
    assert np.max(np.abs(nominal.cyclic_gaps_deg[-1] - target)) <= (
        params.gap_tolerance_deg + 1e-9
    )
    assert np.max(np.abs(nominal.cyclic_relative_rates_deg_per_day[-1])) <= (
        params.cyclic_rate_tolerance_deg_per_day + 1e-9
    )
    assert np.max(np.abs(np.sum(nominal.cyclic_gaps_deg, axis=1))) < 1e-12
    assert nominal.state[-1, :, 2] == pytest.approx(
        params.initial_altitude_km - nominal.altitude_km[-1],
        abs=1e-12,
    )


def test_nonlinear_replay_is_converged_physical_and_nontrivial(scenario) -> None:
    replay = scenario.nonlinear
    check = scenario.resolution_check
    metrics = scenario.metrics

    assert replay.step_hours == pytest.approx(1.0)
    assert replay.time_days.shape == (4_321,)
    assert replay.state.shape == (4_321, 3, 3)
    assert replay.time_days[-1] == pytest.approx(180.0)
    assert np.all(np.isfinite(replay.state))
    assert np.min(replay.density_kg_m3) > 0.0
    assert metrics.density_max_kg_m3 > 1.1 * metrics.density_min_kg_m3
    assert metrics.nonlinear_min_altitude_km > 450.0
    assert np.all(np.diff(replay.altitude_km, axis=0) <= 1e-10)
    assert np.min(replay.state[:, :, 2]) >= -1e-9

    assert 10.0 < metrics.nonlinear_max_gap_error_deg < 15.0
    assert check.fine_step_hours == pytest.approx(0.5)
    assert check.max_phase_delta_deg < 1e-5
    assert check.max_relative_rate_delta_deg_per_day < 1e-7
    assert check.max_extra_loss_delta_km < 1e-7
    assert check.terminal_gap_delta_deg < 0.02
    assert check.max_altitude_delta_km < 1e-7
    assert check.max_altitude_delta_km < 0.01


def test_metrics_and_named_validation_are_artifact_ready(scenario) -> None:
    checks = validation_checks(scenario)
    serialized = metrics_as_dict(scenario.metrics)

    assert checks
    assert all(checks.values()), checks
    assert_valid_scenario(scenario)
    assert serialized["plan_sha256"] == scenario.plan.plan_sha256
    assert len(serialized["equivalent_high_drag_days"]) == 3
    assert len(serialized["nonlinear_terminal_cyclic_gaps_deg"]) == 3
