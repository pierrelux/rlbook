"""Check generated schemes against known formulas and a solved control problem."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from collocation_transcription import make_rule, transcribed_problem


@pytest.mark.parametrize("nodes,A,b", [
    ([0.0], [[0.0]], [1.0]),
    ([1.0], [[1.0]], [1.0]),
    ([0.0, 1.0], [[0.0, 0.0], [0.5, 0.5]], [0.5, 0.5]),
    ([0.0, 0.5, 1.0], [[0, 0, 0], [5/24, 1/3, -1/24], [1/6, 2/3, 1/6]],
     [1/6, 2/3, 1/6]),
])
def test_generated_low_order_rules(nodes, A, b):
    rule = make_rule(nodes, [0.0, 1.0])
    np.testing.assert_allclose(rule.A, A, atol=1e-14)
    np.testing.assert_allclose(rule.b, b, atol=1e-14)
    np.testing.assert_allclose(rule.B, np.column_stack([1-np.array(nodes), nodes]))


def test_interior_nodes_integrate_slopes_and_evaluate_controls():
    nodes = np.array([0.13, 0.44, 0.92])
    rule = make_rule(nodes, [0.0, 0.3, 1.0])
    for degree in range(3):
        np.testing.assert_allclose(
            rule.A @ nodes**degree, nodes**(degree+1)/(degree+1), atol=1e-14
        )
        np.testing.assert_allclose(rule.b @ nodes**degree, 1/(degree+1), atol=1e-14)
        np.testing.assert_allclose(rule.B @ rule.control_nodes**degree, nodes**degree,
                                   atol=1e-14)


def test_control_parameterization_changes_only_control_weights():
    linear = make_rule([0, 0.5, 1], [0, 1])
    quadratic = make_rule([0, 0.5, 1], [0, 0.5, 1])
    np.testing.assert_array_equal(linear.A, quadratic.A)
    np.testing.assert_array_equal(linear.b, quadratic.b)
    np.testing.assert_allclose(quadratic.B, np.eye(3))


def test_nonlinear_dynamics_on_a_nonunit_physical_interval():
    # x=t^3, u=t solve x'=3u^2. Simpson integrates both rates exactly.
    rule = make_rule([0, 0.5, 1], [0, 1])
    mesh = np.array([1.0, 2.3])
    times = mesh[0] + np.diff(mesh)[0] * rule.nodes
    F, G, H = transcribed_problem(
        rule, mesh, mesh[:, None]**3, times[None, :, None]**3,
        mesh[None, :, None],
        dynamics=lambda x,u,t:3*u**2,
        running_cost=lambda x,u,t:u[0]**2,
        terminal_cost=lambda x,t:0,
        boundary=lambda x0,xf,tf:np.array([x0[0]-1, xf[0]-tf**3]),
        path=lambda x,u,t:np.array([u[0]-3]),
    )
    np.testing.assert_allclose(H, 0, atol=1e-13)
    np.testing.assert_allclose(F, (mesh[-1]**3-mesh[0]**3)/3, atol=1e-13)
    assert np.all(G < 0)


@pytest.mark.parametrize("nodes,control_nodes,continuous", [
    ([0.0], [0.0], False), ([1.0], [0.0], False),
    ([0.0, 1.0], [0.0, 1.0], True),
    ([0.0, 0.5, 1.0], [0.0, 1.0], True),
])
def test_assembled_nlp_recovers_minimum_energy_trajectory(nodes, control_nodes, continuous):
    # min integral u^2 dt, x'=u, x(0)=0, x(1)=1 has x=t,u=1,cost=1.
    rule = make_rule(nodes, control_nodes)
    mesh = np.array([0.0, 0.3, 1.0])
    N, s, r = 2, len(nodes), len(control_nodes)
    def unpack(z):
        return z[:N+1, None], z[N+1:N+1+N*s].reshape(N, s, 1), z[N+1+N*s:].reshape(N, r, 1)
    def evaluate(z):
        return transcribed_problem(
            rule, mesh, *unpack(z),
            dynamics=lambda x, u, t: u,
            running_cost=lambda x, u, t: u[0]**2,
            terminal_cost=lambda x, t: 0,
            boundary=lambda x0, xf, tf: np.array([x0[0], xf[0]-1]),
            path=lambda x, u, t: np.array([-u[0], u[0]-2]),
            continuous_control=continuous,
        )
    initial = np.concatenate([mesh, np.zeros(N*s), np.full(N*r, 0.7)])
    result = minimize(
        lambda z: evaluate(z)[0], initial, method="SLSQP",
        constraints=[{"type":"eq", "fun":lambda z:evaluate(z)[2]},
                     {"type":"ineq", "fun":lambda z:-evaluate(z)[1]}],
        options={"ftol":1e-11, "maxiter":100},
    )
    assert result.success, result.message
    objective, G, H = evaluate(result.x)
    np.testing.assert_allclose(objective, 1.0, atol=1e-8)
    np.testing.assert_allclose(H, 0.0, atol=1e-8)
    assert G.max() <= 1e-8
    xm, xs, us = unpack(result.x)
    np.testing.assert_allclose(xm[:, 0], mesh, atol=1e-5)
    np.testing.assert_allclose(us, 1.0, atol=1e-5)


def test_control_jumps_are_allowed_unless_continuity_is_requested():
    rule = make_rule([0.5], [0])
    args = (rule, [0, 1, 2], [[0], [1], [3]], [[[0.5]], [[2]]], [[[1]], [[2]]])
    kwargs = dict(dynamics=lambda x,u,t:u, running_cost=lambda x,u,t:0,
                  terminal_cost=lambda x,t:0, boundary=lambda x0,xf,tf:[])
    _, G, H = transcribed_problem(*args, **kwargs)
    assert G.size == 0
    np.testing.assert_allclose(H, 0)
    _, _, linked = transcribed_problem(*args, **kwargs, continuous_control=True)
    assert np.max(np.abs(linked)) == 1


@pytest.mark.parametrize("nodes", [[], [0.2, 0.2], [-0.1], [1.1], [np.nan]])
def test_invalid_nodes_are_rejected(nodes):
    with pytest.raises(ValueError):
        make_rule(nodes, [0])
