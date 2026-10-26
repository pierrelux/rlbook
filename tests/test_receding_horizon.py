from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIRECTORY))

from receding_horizon import (  # noqa: E402
    StickParameters,
    discretize,
    lyapunov_decrease_margin,
    make_mpc,
    riccati_limit,
    saturated_lqr,
    simulate,
    solve_mpc,
    stage_cost,
    tap,
    terminal_level,
)

X0 = np.array([0.2, 0.6])


@pytest.fixture(scope="module")
def params() -> StickParameters:
    return StickParameters()


def test_terminal_cost_solves_lyapunov_equation_and_matches_riccati(params) -> None:
    mpc = make_mpc(params, 10)
    AK = mpc.A + mpc.B @ mpc.K
    residual = AK.T @ mpc.P @ AK - mpc.P + mpc.Q + mpc.K.T @ mpc.R @ mpc.K
    assert np.max(np.abs(residual)) < 1e-9
    assert np.max(np.abs(np.linalg.eigvals(AK))) < 1.0
    assert np.max(np.abs(riccati_limit(mpc.A, mpc.B, mpc.Q, mpc.R) - mpc.P)) < 1e-8
    assert lyapunov_decrease_margin(mpc, samples=500) < 1e-10


def test_terminal_set_respects_constraints_under_lqr(params) -> None:
    mpc = make_mpc(params, 10)
    # Boundary points of the ellipsoid, then many steps of the LQR closed loop.
    L = np.linalg.cholesky(np.linalg.inv(mpc.P))
    angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    points = np.sqrt(mpc.alpha) * (L @ np.vstack([np.cos(angles), np.sin(angles)]))
    for x in points.T:
        for _ in range(50):
            u = mpc.K @ x
            assert abs(u[0]) <= params.u_max + 1e-9
            assert abs(x[0]) <= params.theta_max + 1e-9
            x = mpc.A @ x + mpc.B @ u
            assert x @ mpc.P @ x <= mpc.alpha + 1e-9
    # The level is tight: some boundary point touches a constraint.
    tight = max(np.max(np.abs(mpc.K @ points)) / params.u_max,
                np.max(np.abs(points[0])) / params.theta_max)
    assert tight == pytest.approx(1.0, abs=1e-3)


def test_terminal_level_matches_scalar_exercise() -> None:
    P, K = np.array([[13 / 3]]), np.array([[-1.5]])
    Pinv = np.linalg.inv(P)
    a = K.ravel()
    assert 1.0 / float(a @ Pinv @ a) == pytest.approx(52 / 27)


def test_shifted_candidate_inequality_and_recursive_feasibility(params) -> None:
    mpc = make_mpc(params, 10, "cost+set")
    x = X0.copy()
    for _ in range(25):
        sol = solve_mpc(mpc, x)
        assert sol.feasible
        x_next = mpc.A @ x + mpc.B @ sol.u[0]
        nxt = solve_mpc(mpc, x_next)
        assert nxt.feasible
        assert nxt.value <= sol.value - stage_cost(mpc, x, sol.u[0]) + 1e-5
        x = x_next


def test_value_bounds_for_each_terminal_choice(params) -> None:
    v_inf = solve_mpc(make_mpc(params, 200, "cost"), X0).value
    lower = [solve_mpc(make_mpc(params, N, "none"), X0).value for N in (1, 5, 10, 20)]
    assert all(v <= v_inf + 1e-6 for v in lower)
    assert all(a <= b + 1e-6 for a, b in zip(lower, lower[1:]))
    assert solve_mpc(make_mpc(params, 1, "cost"), X0).value < v_inf - 1.0
    assert not solve_mpc(make_mpc(params, 8, "cost+set"), X0).feasible
    for N in (9, 15):
        assert solve_mpc(make_mpc(params, N, "cost+set"), X0).value == pytest.approx(v_inf, rel=1e-5)


def test_replanning_survives_model_error_that_breaks_open_loop(params) -> None:
    mpc = make_mpc(params, 40, "cost+set")
    plant = discretize(params, length=0.85)
    push = tap(0.5, 20)
    frozen = simulate(mpc, X0, 30, plant=plant, disturbance=push, replan=False)
    replanned = simulate(mpc, X0, 30, plant=plant, disturbance=push)
    assert np.max(np.abs(frozen.x[:, 0])) > params.theta_max
    assert replanned.feasible.all()
    assert np.max(np.abs(replanned.x[:, 0])) < params.theta_max
    assert np.max(np.abs(replanned.u)) <= params.u_max + 1e-6


def test_short_horizon_without_terminal_ingredients_fails(params) -> None:
    lost = simulate(make_mpc(params, 1, "none"), X0, 10)
    assert not lost.feasible.all()
    parked = simulate(make_mpc(params, 5, "none"), X0, 120)
    assert parked.feasible.all()
    assert parked.x[-1, 0] > 0.3
    assert abs(parked.x[-1, 1]) < 1e-6


def test_mpc_law_matches_saturated_lqr_where_feasible(params) -> None:
    mpc = make_mpc(params, 20, "cost+set")
    for x in ([0.1, 0.0], [-0.2, 0.8], [0.25, -1.0]):
        x = np.array(x)
        sol = solve_mpc(mpc, x)
        assert sol.feasible
        assert sol.u[0, 0] == pytest.approx(saturated_lqr(mpc, x)[0], abs=1e-3)


def test_terminal_level_includes_optional_rate_bound(params) -> None:
    mpc = make_mpc(params, 10)
    assert terminal_level(mpc.P, mpc.K, params.u_max, params.theta_max, 1.0) == pytest.approx(mpc.alpha)
    assert terminal_level(mpc.P, mpc.K, params.u_max, params.theta_max, 0.5) < mpc.alpha
