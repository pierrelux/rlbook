"""Independent numerical checks of the successive-approximation derivation."""
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from trajectory_optimization import (DifferentiableProblem, SolverOptions,
    backward_pass, box_quadratic, control_gradient, solve_trajectory)
from boat_docking import make_problem, step, stage_cost


def linear_problem():
    A = jnp.array([[1., .15], [0., .95]])
    B = jnp.array([[.02], [.2]])
    Q = jnp.diag(jnp.array([.7, .2]))
    QT = jnp.diag(jnp.array([4., 2.]))
    return DifferentiableProblem(2,
        lambda x, u, t: A @ x + B @ u + jnp.array([.03, -.01]),
        lambda x, u, t: .5*x@Q@x + .4*u@u + .03*x[0]*u[0] + .1*x[0] - .05*u[0],
        lambda x: .5*x@QT@x + .2*x[1], [-np.inf], [np.inf])


def test_backward_elimination_matches_dense_quadratic_solve_in_one_step():
    problem = linear_problem()
    x0 = np.array([1.2, -.2])
    controls = np.zeros((7, 1))
    def full_cost(flat):
        us = flat.reshape(controls.shape)
        xs = problem._rollout(jnp.asarray(x0), us)
        return problem._cost(xs, us)
    g = np.asarray(jax.grad(full_cost)(jnp.zeros(7)))
    H = np.asarray(jax.hessian(full_cost)(jnp.zeros(7)))
    exact = np.linalg.solve(H, -g).reshape(controls.shape)
    options = SolverOptions(initial_regularization=0., minimum_regularization=0.,
                            gradient_tolerance=1e-9, max_iterations=2)
    results = [solve_trajectory(problem, x0, controls, method, options)
               for method in ("ilqr", "ddp")]
    for result in results:
        assert result.converged
        assert len(result.history) == 2
        np.testing.assert_allclose(result.controls, exact, atol=1e-10)
    np.testing.assert_allclose(results[0].gains, results[1].gains, atol=1e-12)


def test_scalar_two_stage_worked_example():
    problem = DifferentiableProblem(1, lambda x, u, t: x+u,
                                   lambda x, u, t: .5*(u@u),
                                   lambda x: .5*(x[0]-1.)**2, [-np.inf], [np.inf])
    xs, us = np.zeros((3, 1)), np.zeros((2, 1))
    k, K, _, _ = backward_pass(problem.derivatives(xs, us), us, problem.lower, problem.upper)
    np.testing.assert_allclose(k[:, 0], [1/3, 1/2])
    np.testing.assert_allclose(K[:, 0, 0], [-1/3, -1/2])
    following, controls = problem.feedback_rollout(xs, us, k, K, 1.)
    np.testing.assert_allclose(controls[:, 0], [1/3, 1/3])
    np.testing.assert_allclose(following[:, 0], [0., 1/3, 2/3])
    assert problem.cost(following, controls) == pytest.approx(1/6)


def test_box_qp_matches_independent_optimizer_and_kkt_signs():
    rng = np.random.default_rng(52)
    for _ in range(15):
        matrix = rng.normal(size=(2, 2))
        H, g = matrix.T@matrix + .3*np.eye(2), rng.normal(size=2)*4
        lower, upper = np.array([-.5, -.8]), np.array([.7, .6])
        k, free = box_quadratic(H, g, lower, upper)
        oracle = minimize(lambda v: .5*v@H@v + g@v, np.zeros(2),
                          jac=lambda v: H@v+g, bounds=list(zip(lower, upper)),
                          method="L-BFGS-B", options={"ftol": 1e-14, "gtol": 1e-10})
        np.testing.assert_allclose(k, oracle.x, atol=1e-7)
        residual = k - np.clip(k - (H@k+g), lower, upper)
        np.testing.assert_allclose(residual, 0., atol=1e-10)
        assert not np.any(free & ((k == lower) | (k == upper)))
    with pytest.raises(np.linalg.LinAlgError):
        box_quadratic(-np.eye(2), np.ones(2), lower, upper)


def test_discrete_boat_derivatives_against_finite_differences():
    z = np.array([-.2, 1.4, .4, .3, -.2, .1, .3, -.2])
    direction = np.array([.2, -.3, .7, -.4, .1, -.2, .6, -.5])
    fun = lambda zz: step(zz[:6], zz[6:])
    jacobian = np.asarray(jax.jacfwd(fun)(jnp.asarray(z)))
    h = 1e-5
    finite_first = (np.asarray(fun(z+h*direction))-np.asarray(fun(z-h*direction)))/(2*h)
    np.testing.assert_allclose(jacobian@direction, finite_first, rtol=1e-7, atol=1e-9)
    multiplier = jnp.array([.2, -.4, .1, .8, -.5, .3])
    scalar = lambda zz: multiplier@fun(zz)
    hessian = np.asarray(jax.hessian(scalar)(jnp.asarray(z)))
    h = 1e-3
    finite_second = (float(scalar(z+h*direction))-2*float(scalar(z))
                     +float(scalar(z-h*direction)))/h**2
    np.testing.assert_allclose(direction@hessian@direction, finite_second, rtol=2e-5, atol=1e-8)


def test_ddp_curvature_is_the_hessian_of_the_composed_tail():
    problem = make_problem()
    x = np.array([-.2, 1.4, .4, .3, -.2, .1])
    u = np.array([.3, -.2])
    states = problem.rollout(x, u[None])
    F, c, C, fzz, _, _ = problem.derivatives(states, u[None], "ddp")
    p, P = np.arange(1., 7.)/3, np.diag(np.arange(1., 7.))
    center = jnp.asarray(states[-1])
    def composed(z):
        delta = step(z[:6], z[6:]) - center
        return stage_cost(z[:6], z[6:]) + p@delta + .5*delta@P@delta
    expected = np.asarray(jax.hessian(composed)(jnp.asarray(np.r_[x, u])))
    curvature = C[0] + F[0].T@P@F[0] + np.einsum("i,ijk->jk", p, fzz[0])
    np.testing.assert_allclose(curvature, expected, atol=1e-10)
    assert np.linalg.norm(np.einsum("i,ijk->jk", p, fzz[0])) > .01


def test_adjoint_gradient_matches_whole_rollout_differentiation():
    problem = make_problem()
    x0 = np.array([-4., 3., -.3, .2, -.1, .05])
    us = np.full((8, 2), .1)
    xs = problem.rollout(x0, us)
    exact = jax.grad(lambda controls: problem._cost(
        problem._rollout(jnp.asarray(x0), controls), controls))(jnp.asarray(us))
    np.testing.assert_allclose(control_gradient(problem.derivatives(xs, us)), exact, rtol=1e-11)


def test_rejected_trials_preserve_incumbent_and_report_failure(monkeypatch):
    problem = linear_problem()
    initial, controls = np.array([1., 0.]), np.zeros((3, 1))
    expected = problem.rollout(initial, controls)
    monkeypatch.setattr(problem, "feedback_rollout", lambda xs, us, k, K, alpha:
                        (np.full_like(xs, np.nan), us.copy()))
    result = solve_trajectory(problem, initial, controls, options=SolverOptions(
        max_iterations=2, maximum_regularization=.001, line_search_steps=2))
    assert result.status == "no_acceptable_step"
    assert not result.converged
    assert len(result.history) == 1
    np.testing.assert_array_equal(result.states, expected)
    np.testing.assert_array_equal(result.controls, controls)


def test_iteration_limit_and_invalid_inputs_are_explicit():
    problem = linear_problem()
    result = solve_trajectory(problem, [1., 0.], np.zeros((3, 1)),
                              options=SolverOptions(max_iterations=0))
    assert result.status == "iteration_limit"
    assert not result.converged
    with pytest.raises(ValueError, match="method"):
        solve_trajectory(problem, [1., 0.], np.zeros((3, 1)), "unknown")
    with pytest.raises(ValueError, match="finite"):
        solve_trajectory(problem, [np.nan, 0.], np.zeros((3, 1)))
