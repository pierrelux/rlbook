"""Check the state-value construction and its equivalence to integration."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from collocation_transcription import (
    cardinal_polynomials,
    differentiation_problem,
    make_differentiation_rule,
    make_rule,
    transcribed_problem,
)


def test_worked_trapezoidal_construction():
    rule = make_differentiation_rule([0, 1], [0, 0.5, 1], [0, 1])
    np.testing.assert_allclose(rule.E, [[1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(rule.D, [[-3, 4, -1], [1, -4, 3]])
    np.testing.assert_allclose(rule.state_left, [1, 0, 0])
    np.testing.assert_allclose(rule.state_right, [0, 0, 1])
    np.testing.assert_allclose(rule.b, [0.5, 0.5])
    np.testing.assert_allclose(rule.B, np.eye(2))
    basis = cardinal_polynomials(rule.state_nodes)
    np.testing.assert_allclose(basis[0].deriv().coef, [-3, 4])

    left = np.array([1.2, -0.4])
    f_left, f_right = np.array([0.3, 0.9]), np.array([-0.8, 1.1])
    width = 1.7
    right = left + width * (f_left + f_right) / 2
    middle = (left + right) / 2 + width * (f_left - f_right) / 8
    states = np.stack([left, middle, right])
    np.testing.assert_allclose(rule.D @ states, width * np.stack([f_left, f_right]))


@pytest.mark.parametrize("stage_value,expected_residual", [(3.0, 0.5), (2.0, 0.0)])
def test_stage_consistency_closes_the_ode_matching_gap(stage_value, expected_residual):
    # The chapter's x'=x example has x_left=1, h=1 and one midpoint stage.
    # Interpolating the evaluated stage slope gives p(tau)=1+stage_value*tau.
    p = np.polynomial.Polynomial([1.0, stage_value])
    integral = make_rule([0.5], [0])
    derivative = make_differentiation_rule([0.5], [0, 1], [0])
    stored_stage = np.array([[stage_value]])
    x_mesh = np.array([[p(0)], [p(1)]])
    x_support = x_mesh[None, :, :]
    curve_stage = derivative.E @ x_support[0]
    np.testing.assert_allclose(curve_stage, [[p(0.5)]])

    # Matching the sampled slope is built in even for an infeasible state.
    np.testing.assert_allclose(derivative.D @ x_support[0], stored_stage)
    np.testing.assert_allclose(p.deriv()(0.5), stage_value)
    np.testing.assert_allclose(stored_stage - curve_stage, expected_residual)
    # The actual ODE residual uses f(p(0.5))=p(0.5), not f(stored_stage).
    np.testing.assert_allclose(p.deriv()(0.5) - p(0.5), expected_residual)

    callbacks = dict(
        dynamics=lambda x, u, t: x,
        running_cost=lambda x, u, t: 0,
        terminal_cost=lambda x, t: 0,
        boundary=lambda x0, xf, tf: [],
    )
    _, _, H_int = transcribed_problem(
        integral, [0, 1], x_mesh, stored_stage[None, :, :], [[[0]]], **callbacks
    )
    _, _, H_diff = differentiation_problem(
        derivative, [0, 1], x_mesh, x_support, [[[0]]], **callbacks
    )
    np.testing.assert_allclose(H_int, [expected_residual, 0])
    np.testing.assert_allclose(H_diff, [expected_residual, 0, 0])


@pytest.mark.parametrize("nodes", [
    [0], [1], [0.5], [0, 1], [0, 0.5, 1],
    [0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6], [1/3, 1],
])
@pytest.mark.parametrize("include_endpoints", [False, True])
def test_polynomial_exactness_and_integration_identities(nodes, include_endpoints):
    s = len(nodes)
    supports = np.linspace(0, 1, s + 1) if include_endpoints else np.linspace(0.1, 0.9, s + 1)
    rule = make_differentiation_rule(nodes, supports, [0.1, 0.9])
    integral = make_rule(nodes, rule.control_nodes)
    np.testing.assert_allclose(rule.b, integral.b, atol=1e-13)
    np.testing.assert_allclose(rule.B, integral.B, atol=1e-13)
    for degree in range(s + 1):
        values = supports**degree
        derivatives = np.zeros(s) if degree == 0 else degree * rule.nodes**(degree - 1)
        np.testing.assert_allclose(rule.E @ values, rule.nodes**degree, atol=1e-12)
        np.testing.assert_allclose(rule.D @ values, derivatives, atol=1e-12)
        np.testing.assert_allclose(rule.state_left @ values, float(degree == 0), atol=1e-12)
        np.testing.assert_allclose(rule.state_right @ values, 1, atol=1e-12)
    np.testing.assert_allclose(
        rule.E - rule.state_left[None, :], integral.A @ rule.D, atol=1e-12
    )
    np.testing.assert_allclose(
        rule.state_right - rule.state_left, rule.b @ rule.D, atol=1e-12
    )


def test_both_forms_agree_on_a_nonlinear_vector_trajectory():
    # x=(t^3, 2t^3), u=t is a manufactured solution. Nonendpoint supports
    # exercise interpolation and both endpoint evaluations on nonunit steps.
    rule = make_differentiation_rule([0, 0.5, 1], [0.1, 0.3, 0.7, 0.9], [0, 1])
    mesh = np.array([1.0, 1.4, 2.3])
    widths = np.diff(mesh)
    support_times = mesh[:-1, None] + widths[:, None] * rule.state_nodes
    factors = np.array([1.0, 2.0])
    x_mesh = mesh[:, None]**3 * factors
    x_support = support_times[:, :, None]**3 * factors
    u_support = np.stack([mesh[:-1], mesh[1:]], axis=1)[:, :, None]
    callbacks = dict(
        dynamics=lambda x, u, t: 3*u[0]**2 * factors + x - t**3 * factors,
        running_cost=lambda x, u, t: u[0]**2 + x[0] - t**3,
        terminal_cost=lambda x, t: 0.25*x[0],
        boundary=lambda x0, xf, tf: np.concatenate([x0 - factors, xf - tf**3 * factors]),
        path=lambda x, u, t: np.array([x[0] - t**3 - 1, u[0] - 3]),
        continuous_control=True,
    )
    F, G, H = differentiation_problem(
        rule, mesh, x_mesh, x_support, u_support, **callbacks
    )
    np.testing.assert_allclose(H, 0, atol=2e-12)
    np.testing.assert_allclose(F, (mesh[-1]**3 - mesh[0]**3)/3 + 0.25*mesh[-1]**3)
    assert np.all(G < 0)

    integral = make_rule(rule.nodes, rule.control_nodes)
    Fi, Gi, Hi = transcribed_problem(
        integral, mesh, x_mesh, rule.E @ x_support, u_support, **callbacks
    )
    np.testing.assert_allclose(Hi, 0, atol=2e-12)
    np.testing.assert_allclose(Fi, F)
    np.testing.assert_allclose(Gi, G)


@pytest.mark.parametrize("nodes,control_nodes,continuous", [
    ([0], [0], False), ([1], [0], False),
    ([0, 1], [0, 1], True), ([0, 0.5, 1], [0, 1], True),
])
def test_differentiation_nlp_recovers_minimum_energy_trajectory(nodes, control_nodes, continuous):
    # min integral u^2 dt, x'=u, x(0)=0, x(1)=1 has x=t,u=1,cost=1.
    rule = make_differentiation_rule(nodes, np.linspace(0, 1, len(nodes) + 1), control_nodes)
    mesh = np.array([0, 0.3, 1])
    N, d, r = 2, len(rule.state_nodes), len(control_nodes)

    def unpack(z):
        return (z[:N+1, None], z[N+1:N+1+N*d].reshape(N, d, 1),
                z[N+1+N*d:].reshape(N, r, 1))

    def evaluate(z):
        return differentiation_problem(
            rule, mesh, *unpack(z),
            dynamics=lambda x, u, t: u,
            running_cost=lambda x, u, t: u[0]**2,
            terminal_cost=lambda x, t: 0,
            boundary=lambda x0, xf, tf: [x0[0], xf[0] - 1],
            path=lambda x, u, t: [-u[0], u[0] - 2],
            continuous_control=continuous,
        )

    initial = np.concatenate([mesh, np.zeros(N*d), np.full(N*r, 0.7)])
    result = minimize(
        lambda z: evaluate(z)[0], initial, method="SLSQP",
        constraints=[{"type": "eq", "fun": lambda z: evaluate(z)[2]},
                     {"type": "ineq", "fun": lambda z: -evaluate(z)[1]}],
        options={"ftol": 1e-11, "maxiter": 100},
    )
    assert result.success, result.message
    F, G, H = evaluate(result.x)
    np.testing.assert_allclose(F, 1, atol=1e-8)
    np.testing.assert_allclose(H, 0, atol=1e-8)
    assert G.max() <= 1e-8
    xm, xs, us = unpack(result.x)
    np.testing.assert_allclose(xm[:, 0], mesh, atol=1e-5)
    support_times = mesh[:-1, None] + np.diff(mesh)[:, None] * rule.state_nodes
    np.testing.assert_allclose(xs[:, :, 0], support_times, atol=1e-5)
    np.testing.assert_allclose(us, 1, atol=1e-5)


def test_control_continuity_uses_endpoint_evaluation():
    rule = make_differentiation_rule([0.5], [0.2, 0.8], [0.25, 0.75])
    args = (rule, [0, 1, 2], [[0], [1], [3]],
            [[[0.2], [0.8]], [[1.4], [2.6]]], [[[1], [1]], [[2], [2]]])
    callbacks = dict(dynamics=lambda x, u, t: u, running_cost=lambda x, u, t: 0,
                     terminal_cost=lambda x, t: 0, boundary=lambda x0, xf, tf: [])
    _, G, H = differentiation_problem(*args, **callbacks)
    assert G.size == 0
    np.testing.assert_allclose(H, 0, atol=1e-14)
    _, _, linked = differentiation_problem(*args, **callbacks, continuous_control=True)
    assert np.max(np.abs(linked)) == pytest.approx(1)


@pytest.mark.parametrize("state_nodes", [[], [0, 1], [0, 0, 1], [-0.1, 0.5, 1],
                                         [0, 0.5, 1.1], [0, np.nan, 1], [0, 0.3, 0.7, 1]])
def test_invalid_state_supports_are_rejected(state_nodes):
    with pytest.raises(ValueError):
        make_differentiation_rule([0, 1], state_nodes, [0, 1])


@pytest.mark.parametrize("field,value,match", [
    ("mesh", [0, 0], "strictly increasing"),
    ("x_mesh", [0, 1], "x_mesh"),
    ("x_support", [[[0], [1]]], "x_support"),
    ("u_support", [[0], [1]], "u_support"),
])
def test_invalid_evaluator_shapes_are_rejected(field, value, match):
    arguments = dict(
        rule=make_differentiation_rule([0, 1], [0, 0.5, 1], [0, 1]),
        mesh=[0, 1], x_mesh=[[0], [1]], x_support=[[[0], [0.5], [1]]],
        u_support=[[[1], [1]]], dynamics=lambda x, u, t: u,
        running_cost=lambda x, u, t: 0, terminal_cost=lambda x, t: 0,
        boundary=lambda x0, xf, tf: [],
    )
    arguments[field] = value
    with pytest.raises(ValueError, match=match):
        differentiation_problem(**arguments)
