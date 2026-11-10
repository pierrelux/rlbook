"""Linear receding-horizon control of a balanced stick.

The model is the linearized upright pendulum ``theta'' = (g / l) theta + u``
sampled exactly with a zero-order hold. The control ``u`` is an angular
acceleration command with a hard bound, and the angle has a hard bound.
Each MPC subproblem is a convex quadratic program solved with CVXPY.

The module supports three terminal choices so that the chapter can compare
them on one system:

``"none"``
    no terminal cost and no terminal constraint;
``"cost"``
    the infinite-horizon LQR cost ``x' P x`` as terminal cost;
``"cost+set"``
    the LQR terminal cost and the ellipsoidal terminal set
    ``{x : x' P x <= alpha}``, sized so that the LQR law satisfies every
    constraint inside it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Literal

import cvxpy as cp
import numpy as np
from scipy.linalg import expm, solve_discrete_are

Terminal = Literal["none", "cost", "cost+set"]


@dataclass(frozen=True)
class StickParameters:
    gravity: float = 9.81
    length: float = 1.0
    period: float = 0.05
    u_max: float = 5.0
    theta_max: float = 0.35
    rate_max: float = np.inf
    q: tuple[float, float] = (10.0, 1.0)
    r: float = 0.05


@dataclass(frozen=True)
class LinearMPC:
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    u_max: float
    theta_max: float
    rate_max: float
    horizon: int
    terminal: Terminal = "cost+set"
    P: np.ndarray = field(default=None)  # type: ignore[assignment]
    K: np.ndarray = field(default=None)  # type: ignore[assignment]
    alpha: float = np.inf


@dataclass
class MPCSolution:
    feasible: bool
    u: np.ndarray
    x: np.ndarray
    value: float


@dataclass
class ClosedLoop:
    x: np.ndarray
    u: np.ndarray
    feasible: np.ndarray
    cost: float
    values: np.ndarray


def discretize(params: StickParameters, length: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold sampling of ``theta'' = (g/l) theta + u``."""
    a = params.gravity / (params.length if length is None else length)
    Ac = np.array([[0.0, 1.0], [a, 0.0]])
    Bc = np.array([[0.0], [1.0]])
    M = np.zeros((3, 3))
    M[:2, :2], M[:2, 2:] = Ac, Bc
    E = expm(M * params.period)
    return E[:2, :2], E[:2, 2:]


def lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infinite-horizon LQR: value matrix ``P`` and gain ``K`` with ``u = K x``."""
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K


def riccati_limit(A, B, Q, R, iterations: int = 2000, tol: float = 1e-12) -> np.ndarray:
    """Iterate the finite-horizon Riccati recursion from ``P = Q`` to its limit."""
    P = Q.copy()
    for _ in range(iterations):
        G = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P_next = Q + A.T @ P @ (A - B @ G)
        if np.max(np.abs(P_next - P)) < tol:
            return P_next
        P = P_next
    return P


def terminal_level(
    P: np.ndarray, K: np.ndarray, u_max: float, theta_max: float, rate_max: float = np.inf
) -> float:
    """Largest ``alpha`` with ``{x' P x <= alpha}`` inside the constraints under ``u = K x``.

    For a half-space ``a' x <= h`` the ellipsoid ``x' P x <= alpha`` fits
    inside it exactly when ``alpha <= h^2 / (a' P^{-1} a)``.
    """
    Pinv = np.linalg.inv(P)
    rows = [(K.ravel(), u_max), (np.array([1.0, 0.0]), theta_max), (np.array([0.0, 1.0]), rate_max)]
    return min(h**2 / float(a @ Pinv @ a) for a, h in rows)


def make_mpc(params: StickParameters, horizon: int, terminal: Terminal = "cost+set") -> LinearMPC:
    A, B = discretize(params)
    Q, R = np.diag(params.q), np.array([[params.r]])
    P, K = lqr(A, B, Q, R)
    alpha = terminal_level(P, K, params.u_max, params.theta_max, params.rate_max)
    return LinearMPC(
        A, B, Q, R, params.u_max, params.theta_max, params.rate_max, horizon, terminal, P, K, alpha
    )


_CACHE: dict[tuple, tuple] = {}


def _problem(mpc: LinearMPC):
    key = (id(mpc), mpc.horizon, mpc.terminal)
    if key in _CACHE:
        return _CACHE[key]
    N, n, m = mpc.horizon, mpc.A.shape[0], mpc.B.shape[1]
    x0 = cp.Parameter(n)
    x = cp.Variable((N + 1, n))
    u = cp.Variable((N, m))
    cost = 0
    constraints = [x[0] == x0]
    for k in range(N):
        cost += cp.quad_form(x[k], mpc.Q) + cp.quad_form(u[k], mpc.R)
        constraints += [
            x[k + 1] == mpc.A @ x[k] + mpc.B @ u[k],
            cp.abs(u[k]) <= mpc.u_max,
            cp.abs(x[k + 1, 0]) <= mpc.theta_max,
        ]
        if np.isfinite(mpc.rate_max):
            constraints.append(cp.abs(x[k + 1, 1]) <= mpc.rate_max)
    if mpc.terminal in ("cost", "cost+set"):
        cost += cp.quad_form(x[N], mpc.P)
    if mpc.terminal == "cost+set":
        constraints.append(cp.quad_form(x[N], mpc.P) <= mpc.alpha)
    problem = cp.Problem(cp.Minimize(cost), constraints)
    # Keep a reference to ``mpc`` so that its id cannot be reused.
    _CACHE[key] = (problem, x0, x, u, mpc)
    return _CACHE[key]


def solve_mpc(mpc: LinearMPC, state: np.ndarray) -> MPCSolution:
    problem, x0, x, u, _ = _problem(mpc)
    x0.value = np.asarray(state, dtype=float)
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        problem.solve(solver=cp.SCS)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        N, n, m = mpc.horizon, mpc.A.shape[0], mpc.B.shape[1]
        return MPCSolution(False, np.full((N, m), np.nan), np.full((N + 1, n), np.nan), np.inf)
    return MPCSolution(True, u.value.copy(), x.value.copy(), float(problem.value))


def saturated_lqr(mpc: LinearMPC, state: np.ndarray) -> np.ndarray:
    return np.clip(mpc.K @ state, -mpc.u_max, mpc.u_max)


def stage_cost(mpc: LinearMPC, x: np.ndarray, u: np.ndarray) -> float:
    return float(x @ mpc.Q @ x + u @ mpc.R @ u)


def simulate(
    mpc: LinearMPC,
    x0: np.ndarray,
    steps: int,
    plant: tuple[np.ndarray, np.ndarray] | None = None,
    disturbance: Callable[[int], np.ndarray] | None = None,
    replan: bool = True,
) -> ClosedLoop:
    """Run the plant under MPC (``replan=True``) or under the first plan only.

    Infeasible subproblems fall back to the saturated LQR law so that the
    trajectory can still be plotted; the ``feasible`` flags record them.
    """
    Ap, Bp = (mpc.A, mpc.B) if plant is None else plant
    x = np.asarray(x0, dtype=float)
    xs, us, ok, values = [x], [], [], []
    plan = solve_mpc(mpc, x) if not replan else None
    for t in range(steps):
        if replan:
            sol = solve_mpc(mpc, x)
            ok.append(sol.feasible)
            values.append(sol.value)
            u = sol.u[0] if sol.feasible else saturated_lqr(mpc, x)
        else:
            ok.append(plan.feasible)
            values.append(np.nan)
            u = plan.u[t] if t < mpc.horizon else np.zeros(mpc.B.shape[1])
        w = np.zeros_like(x) if disturbance is None else disturbance(t)
        x = Ap @ x + (Bp @ u).ravel() + w
        xs.append(x)
        us.append(u)
    X, U = np.array(xs), np.array(us)
    cost = sum(stage_cost(mpc, X[t], U[t]) for t in range(steps))
    return ClosedLoop(X, U, np.array(ok), cost, np.array(values))


def policy_slice(mpc: LinearMPC, thetas: np.ndarray, rate: float = 0.0) -> np.ndarray:
    """First MPC action ``kappa_N(theta, rate)`` along a line of states (NaN if infeasible)."""
    out = np.full(len(thetas), np.nan)
    for i, th in enumerate(thetas):
        sol = solve_mpc(mpc, np.array([th, rate]))
        if sol.feasible:
            out[i] = sol.u[0, 0]
    return out


def lyapunov_decrease_margin(mpc: LinearMPC, samples: int = 2000, seed: int = 0) -> float:
    """Max over sampled ``x`` in ``X_f`` of ``c_N(x+) - c_N(x) + c(x, Kx)`` (should be <= 0)."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(np.linalg.inv(mpc.P))
    worst = -np.inf
    for _ in range(samples):
        z = rng.normal(size=2)
        z *= rng.uniform() ** 0.5 / np.linalg.norm(z)
        x = np.sqrt(mpc.alpha) * (L @ z)
        u = mpc.K @ x
        xn = mpc.A @ x + mpc.B @ u
        worst = max(worst, float(xn @ mpc.P @ xn - x @ mpc.P @ x + stage_cost(mpc, x, u)))
    return worst


def tap(size: float, at: int) -> Callable[[int], np.ndarray]:
    """A single impulse on the angular rate at time ``at``."""
    return lambda t: np.array([0.0, size]) if t == at else np.zeros(2)

