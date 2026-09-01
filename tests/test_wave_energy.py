from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIRECTORY))

from wave_energy import (  # noqa: E402
    WaveParameters,
    project_damping,
    run_comparison,
    shifted_sea_state,
    simulate_controller,
    solve_economic_mpc,
)


def test_damping_projection_enforces_action_torque_and_rate_limits() -> None:
    params = WaveParameters()
    previous = 900.0
    velocity = 1.5
    applied = project_damping(10_000.0, velocity, previous, params)

    assert 0.0 <= applied <= params.damping_max
    assert abs(applied - previous) <= (
        params.damping_rate_limit * params.control_period + 1e-10
    )
    assert applied * abs(velocity) <= params.torque_limit + 1e-10


def test_economic_mpc_prediction_satisfies_all_constraints() -> None:
    params = WaveParameters()
    state = np.array([0.18, 0.75])
    previous = params.constant_damping
    step = solve_economic_mpc(2.4, state, previous, params)

    velocity = step.predicted_state[:-1, 1]
    changes = np.diff(np.concatenate([[previous], step.sequence]))
    assert step.success
    assert step.max_predicted_violation <= 2e-4
    assert np.max(np.abs(step.predicted_state[1:, 0])) <= params.stroke_limit + 2e-4
    assert np.max(np.abs(step.sequence * velocity)) <= params.torque_limit + 1.0
    assert np.max(np.abs(changes)) <= (
        params.damping_rate_limit * params.control_period + 1.0
    )


@pytest.fixture(scope="module")
def short_comparison():
    return run_comparison(duration=3.6)


def test_closed_loop_mpc_is_passive_feasible_and_economic(short_comparison) -> None:
    params = WaveParameters()
    mpc = short_comparison["mpc"]
    constant = short_comparison["constant"]

    assert np.all(mpc.pto_torque * mpc.angular_velocity <= 1e-10)
    assert mpc.metrics.stroke_violation_rad <= 1e-8
    assert mpc.metrics.torque_violation_nm <= 1e-8
    assert mpc.metrics.damping_rate_violation_per_s <= 1e-8
    assert mpc.metrics.mpc_success_fraction == pytest.approx(1.0)
    assert mpc.metrics.absorbed_energy_j > constant.metrics.absorbed_energy_j
    assert np.max(mpc.damping) <= params.damping_max + 1e-10


def test_shifted_three_frequency_sea_state_remains_finite_and_feasible() -> None:
    params = shifted_sea_state()
    result = simulate_controller("mpc", "Economic MPC", params, duration=3.6)

    assert np.all(np.isfinite(result.angle))
    assert np.all(np.isfinite(result.cumulative_energy))
    assert result.metrics.absorbed_energy_j > 0.0
    assert result.metrics.stroke_violation_rad <= 1e-8
    assert result.metrics.torque_violation_nm <= 1e-8
    assert result.metrics.damping_rate_violation_per_s <= 1e-8
