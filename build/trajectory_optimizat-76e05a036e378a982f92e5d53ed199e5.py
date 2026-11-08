"""Small iLQR/DDP solver: differentiate, eliminate backward, roll out forward.

The box solver enumerates active sets and is intended for teaching problems
with at most three controls. Derivatives are of the supplied discrete map,
including its numerical integrator. No global value function is represented.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from time import perf_counter
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


@lru_cache(maxsize=None)
def _active_sets(size):
    return tuple(np.array(s) for s in product((0, -1, 1), repeat=size))


def box_quadratic(H, g, lower, upper):
    """Minimize g.T k + k.T H k / 2 over a box; H must be positive definite.

    Return the minimizer and the free coordinates of its active set. Enumerate
    each face, minimize over that face, and retain the best feasible candidate.
    This is exact for the small strictly convex QP, up to floating-point error.
    """
    H, g, lower, upper = map(np.asarray, (H, g, lower, upper))
    if not 1 <= len(g) <= 3:
        raise ValueError("the teaching box solver supports one to three controls")
    if np.any(lower > upper):
        raise ValueError("inconsistent control bounds")
    np.linalg.cholesky(H)
    unconstrained = np.linalg.solve(H, -g)
    if np.all(unconstrained > lower) and np.all(unconstrained < upper):
        return unconstrained, np.ones(len(g), dtype=bool)
    best, best_value, best_free = None, np.inf, None
    for active in _active_sets(len(g)):
        free = active == 0
        k = np.where(active == -1, lower, np.where(active == 1, upper, 0.0))
        if not np.all(np.isfinite(k)):
            continue
        if np.any(free):
            k[free] = np.linalg.solve(H[np.ix_(free, free)],
                                      -g[free] - H[np.ix_(free, ~free)] @ k[~free])
        if np.any(k < lower - 1e-10) or np.any(k > upper + 1e-10):
            continue
        k = np.clip(k, lower, upper)
        value = g @ k + 0.5 * k @ H @ k
        if value < best_value:
            best, best_value, best_free = k, value, free
    if best is None:
        raise np.linalg.LinAlgError("no finite box-QP solution")
    # A coordinate lying exactly on a bound is fixed for the local feedback.
    best_free &= (best > lower + 1e-10) & (best < upper - 1e-10)
    return best, best_free


class DifferentiableProblem:
    """Discrete dynamics f(x,u,t), cost c(x,u,t), and terminal cost phi(x)."""

    def __init__(self, state_size: int, step: Callable, stage_cost: Callable,
                 terminal_cost: Callable, lower, upper):
        self.state_size = state_size
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        if (self.lower.ndim != 1 or self.lower.shape != self.upper.shape
                or np.any(np.isnan(self.lower)) or np.any(np.isnan(self.upper))
                or np.any(self.lower >= self.upper)):
            raise ValueError("control bounds must be ordered vectors of equal size")
        self.control_size = len(self.lower)
        self.step, self.stage_cost, self.terminal_cost = step, stage_cost, terminal_cost
        nx = state_size

        def fz(z, t):
            return step(z[:nx], z[nx:], t)

        def cz(z, t):
            return stage_cost(z[:nx], z[nx:], t)

        self._first = jax.jit(jax.vmap(lambda z, t: (
            jax.jacfwd(fz)(z, t), jax.grad(cz)(z, t), jax.hessian(cz)(z, t))))
        self._second = jax.jit(jax.vmap(jax.jacfwd(jax.jacrev(fz))))
        self._terminal = jax.jit(lambda x: (jax.grad(terminal_cost)(x),
                                           jax.hessian(terminal_cost)(x)))

        def rollout(x0, controls):
            def advance(x, ut):
                u, t = ut
                following = step(x, u, t)
                return following, following
            _, xs = jax.lax.scan(advance, x0, (controls, jnp.arange(len(controls))))
            return jnp.concatenate((x0[None], xs))

        self._rollout = jax.jit(rollout)
        self._cost = jax.jit(lambda xs, us: terminal_cost(xs[-1]) + jnp.sum(
            jax.vmap(stage_cost)(xs[:-1], us, jnp.arange(len(us)))))

        def feedback_rollout(xs, us, k, K, alpha):
            def advance(x, data):
                xb, ub, feedforward, gain, t = data
                u = jnp.clip(ub + alpha * feedforward + gain @ (x - xb),
                             self.lower, self.upper)
                following = step(x, u, t)
                return following, (following, u)
            _, (xx, uu) = jax.lax.scan(advance, xs[0],
                                      (xs[:-1], us, k, K, jnp.arange(len(us))))
            return jnp.concatenate((xs[:1], xx)), uu

        self._feedback = jax.jit(feedback_rollout)

    def rollout(self, initial, controls):
        return np.asarray(self._rollout(jnp.asarray(initial), jnp.asarray(controls)))

    def cost(self, states, controls):
        return float(self._cost(jnp.asarray(states), jnp.asarray(controls)))

    def derivatives(self, states, controls, method="ilqr"):
        if method not in ("ilqr", "ddp"):
            raise ValueError("method must be ilqr or ddp")
        z = jnp.asarray(np.concatenate((states[:-1], controls), axis=1))
        times = jnp.arange(len(controls))
        F, c, C = map(np.asarray, self._first(z, times))
        fzz = np.asarray(self._second(z, times)) if method == "ddp" else None
        p, P = map(np.asarray, self._terminal(jnp.asarray(states[-1])))
        return F, c, C, fzz, p, P

    def feedback_rollout(self, states, controls, k, K, alpha):
        return tuple(map(np.asarray, self._feedback(
            jnp.asarray(states), jnp.asarray(controls), jnp.asarray(k),
            jnp.asarray(K), jnp.asarray(alpha))))


def backward_pass(derivatives, controls, lower, upper, regularization=0.0):
    """Eliminate controls backward; keep original curvature in tail updates.

    Regularization affects the control solve only. Substituting the resulting
    affine correction into the original quadratic model gives the full tail
    formulas, which also apply when some controls are fixed at their bounds.
    """
    F, c, C, fzz, p, P = derivatives
    nx, nu, horizon = len(p), controls.shape[1], len(controls)
    ks, Ks = np.zeros((horizon, nu)), np.zeros((horizon, nu, nx))
    linear_change, quadratic_change = 0.0, 0.0
    for t in range(horizon - 1, -1, -1):
        q = c[t] + F[t].T @ p
        H = C[t] + F[t].T @ P @ F[t]
        if fzz is not None:
            H = H + np.einsum("i,ijk->jk", p, fzz[t])
        H = 0.5 * (H + H.T)
        qx, qu = q[:nx], q[nx:]
        Hxx, Hux, Huu = H[:nx, :nx], H[nx:, :nx], H[nx:, nx:]
        stabilized = Huu + regularization * np.eye(nu)
        k, free = box_quadratic(stabilized, qu, lower - controls[t], upper - controls[t])
        K = np.zeros((nu, nx))
        if np.any(free):
            K[free] = np.linalg.solve(stabilized[np.ix_(free, free)], -Hux[free])
        ks[t], Ks[t] = k, K
        linear_change += qu @ k
        quadratic_change += 0.5 * k @ Huu @ k
        p = qx + Hux.T @ k + K.T @ qu + K.T @ Huu @ k
        P = Hxx + Hux.T @ K + K.T @ Hux + K.T @ Huu @ K
        P = 0.5 * (P + P.T)
        if not np.all(np.isfinite(P)) or not np.all(np.isfinite(p)):
            raise np.linalg.LinAlgError("nonfinite quadratic tail")
    return ks, Ks, linear_change, quadratic_change


def control_gradient(derivatives):
    """Exact gradient of the nonlinear shooting objective by the adjoint rule."""
    F, c, _, _, p, _ = derivatives
    nx = len(p)
    gradient = np.empty((len(F), F.shape[2] - nx))
    for t in range(len(F) - 1, -1, -1):
        gradient[t] = c[t, nx:] + F[t, :, nx:].T @ p
        p = c[t, :nx] + F[t, :, :nx].T @ p
    return gradient


@dataclass(frozen=True)
class SolverOptions:
    max_iterations: int = 120
    gradient_tolerance: float = 1e-5
    initial_regularization: float = 1e-4
    minimum_regularization: float = 1e-9
    maximum_regularization: float = 1e12
    line_search_steps: int = 12


@dataclass
class TrajectoryResult:
    method: str
    states: np.ndarray
    controls: np.ndarray
    feedforward: np.ndarray
    gains: np.ndarray
    history: list[dict]
    status: str
    projected_gradient: float
    forward_evaluations: int
    backward_attempts: int
    solver_seconds: float
    gain_regularization: float

    @property
    def converged(self):
        return self.status == "converged"


def solve_trajectory(problem, initial, initial_controls, method="ilqr", options=SolverOptions()):
    """Run a local solve, retaining only finite, cost-decreasing nonlinear trials.

    History entry zero is the initial rollout. Later entries are accepted
    iterates, not trial trajectories. Compilation is excluded from solver time.
    A rejected iteration retains the last accepted plan and reports its status.
    """
    if method not in ("ilqr", "ddp"):
        raise ValueError("method must be ilqr or ddp")
    initial, controls = np.asarray(initial, float), np.asarray(initial_controls, float).copy()
    if (initial.shape != (problem.state_size,) or controls.ndim != 2
            or controls.shape[1] != problem.control_size or len(controls) == 0
            or not np.all(np.isfinite(initial)) or not np.all(np.isfinite(controls))):
        raise ValueError("initial state and controls must have finite, compatible shapes")
    if np.any(controls < problem.lower) or np.any(controls > problem.upper):
        raise ValueError("initial controls must respect their bounds")
    if options.max_iterations < 0 or options.line_search_steps < 1:
        raise ValueError("invalid iteration budget")
    states = problem.rollout(initial, controls)
    cost = problem.cost(states, controls)
    if not np.isfinite(cost) or not np.all(np.isfinite(states)):
        raise ValueError("initial rollout must have finite states and cost")
    # Compile all array kernels before timing this run.
    derivatives = problem.derivatives(states, controls, method)
    problem.feedback_rollout(states, controls, np.zeros_like(controls),
                             np.zeros((len(controls), problem.control_size, problem.state_size)), 0.)
    started = perf_counter()
    history = [dict(iteration=0, cost=cost, states=states.copy(), controls=controls.copy(),
                    alpha=0., regularization=0., projected_gradient=None)]
    mu = max(options.minimum_regularization, options.initial_regularization)
    status, forward_evaluations, backward_attempts = "iteration_limit", 0, 0

    for iteration in range(options.max_iterations + 1):
        if iteration:
            derivatives = problem.derivatives(states, controls, method)
        gradient = control_gradient(derivatives)
        residual = controls - np.clip(controls - gradient, problem.lower, problem.upper)
        norm = float(np.max(np.abs(residual)))
        history[-1]["projected_gradient"] = norm
        if not np.isfinite(norm):
            status = "nonfinite_derivatives"
            break
        if norm <= options.gradient_tolerance:
            status = "converged"
            break
        if iteration == options.max_iterations:
            break
        accepted = False
        while mu <= options.maximum_regularization:
            backward_attempts += 1
            try:
                k, K, d1, d2 = backward_pass(derivatives, controls, problem.lower, problem.upper, mu)
            except np.linalg.LinAlgError:
                mu = max(1e-8, mu * 10.)
                continue
            for exponent in range(options.line_search_steps):
                alpha = 0.5 ** exponent
                xs, us = problem.feedback_rollout(states, controls, k, K, alpha)
                candidate = problem.cost(xs, us)
                forward_evaluations += 1
                predicted = -(alpha * d1 + alpha * alpha * d2)
                if (np.all(np.isfinite(xs)) and np.isfinite(candidate)
                        and candidate < cost
                        and cost - candidate >= 1e-4 * max(predicted, 0.)):
                    states, controls, cost = xs, us, candidate
                    history.append(dict(iteration=iteration + 1, cost=cost, states=states.copy(),
                                        controls=controls.copy(), alpha=alpha, regularization=mu,
                                        projected_gradient=None))
                    mu = max(options.minimum_regularization, mu / 3.)
                    accepted = True
                    break
            if accepted:
                break
            mu = max(1e-8, mu * 10.)
        if not accepted:
            status = "no_acceptable_step"
            break

    # Return gains about the final accepted trajectory, not its predecessor.
    gain_mu = min(mu, options.maximum_regularization)
    while True:
        try:
            k, K, _, _ = backward_pass(derivatives, controls, problem.lower, problem.upper, gain_mu)
            break
        except np.linalg.LinAlgError:
            gain_mu = max(1e-8, gain_mu * 10.)
            if gain_mu > options.maximum_regularization:
                k, K = np.full_like(controls, np.nan), np.full(
                    (len(controls), problem.control_size, problem.state_size), np.nan)
                status = "backward_failure"
                break
    return TrajectoryResult(method, states, controls, k, K, history, status, norm,
                            forward_evaluations, backward_attempts, perf_counter() - started, gain_mu)
