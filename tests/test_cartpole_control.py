"""Focused checks for the shared cart-pole teaching example."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from cartpole_control import (  # noqa: E402
    CartPoleParameters,
    cartpole_dynamics,
    design_lqr,
    replay_open_loop_with_disturbance,
    rollout_controls,
    run_lqr_cases,
    solve_swingup_comparison,
    wrap_angle,
)


def test_upright_is_an_equilibrium_and_linear_closed_loop_is_stable() -> None:
    parameters = CartPoleParameters()
    derivative = cartpole_dynamics(np.zeros(4), 0.0, parameters)
    np.testing.assert_allclose(derivative, np.zeros(4), atol=1e-12)

    design = design_lqr(parameters)
    controllability = np.hstack(
        [
            np.linalg.matrix_power(design.state_matrix, power) @ design.input_matrix
            for power in range(4)
        ]
    )
    assert np.linalg.matrix_rank(controllability) == 4
    assert np.max(np.abs(design.closed_loop_eigenvalues)) < 1.0


def test_lqr_recovers_locally_and_exposes_the_large_angle_failure() -> None:
    parameters = CartPoleParameters()
    cases = run_lqr_cases(parameters)

    assert cases["local"].stabilized
    assert abs(cases["local"].final_angle_error_deg) < 1.0
    assert not cases["uncontrolled"].stabilized
    assert cases["large"].terminated
    assert cases["large"].termination_reason == "rail limit"
    assert np.isclose(abs(cases["large"].state[-1, 0]), parameters.rail_limit)
    assert np.isclose(
        np.max(np.abs(cases["large"].control)),
        parameters.acceleration_limit,
    )


def test_matched_swingup_formulations_reach_the_same_physical_target() -> None:
    comparison = solve_swingup_comparison()
    assert set(comparison) == {"direct", "shooting"}

    direct = comparison["direct"]
    shooting = comparison["shooting"]
    assert direct.success
    assert shooting.success
    assert direct.decision_variables > shooting.decision_variables
    assert direct.dynamics_equalities > shooting.dynamics_equalities

    for result in comparison.values():
        assert np.all(np.isfinite(result.state))
        assert np.all(np.isfinite(result.control))
        assert abs(float(wrap_angle(result.state[-1, 2]))) < np.deg2rad(1.0)
        assert np.cos(result.state[-1, 2]) > 0.99
        assert np.max(np.abs(result.state[:, 0])) <= 2.40 + 1e-4
        assert np.max(np.abs(result.state[:, 1])) <= 4.00 + 1e-4
        assert np.max(np.abs(result.state[:, 3])) <= 12.00 + 1e-4
        assert np.max(np.abs(result.control)) <= 8.00 + 1e-4
        assert result.dynamics_defect < 1e-4

    replayed = rollout_controls(shooting.control)
    np.testing.assert_allclose(replayed, shooting.state, atol=1e-10)


def test_fixed_open_loop_plan_does_not_correct_an_unmodeled_impulse() -> None:
    direct = solve_swingup_comparison()["direct"]
    replay = replay_open_loop_with_disturbance(direct)

    assert np.cos(replay.nominal_state[-1, 2]) > 0.99
    assert np.cos(replay.disturbed_state[-1, 2]) < 0.0
    assert np.count_nonzero(replay.disturbance) == 1
