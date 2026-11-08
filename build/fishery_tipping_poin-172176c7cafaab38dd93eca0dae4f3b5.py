"""Harvesting a fish stock near a tipping point.

The setting is a single fish stock managed by an annual quota. The regulator
observes the stock x (as a fraction of the carrying capacity) before the
season and chooses the escapement s in [0, x], the stock left in the water;
the harvest x - s is sold at a unit price. The stock that escapes spawns and
grows to

    x' = z f(s),    f(s) = s exp(r (1 - s / K) (s - C) / K),

where z is a log-normal environmental shock with mean one. This growth law has
an Allee effect: below the threshold C the stock shrinks even when nobody
fishes it, which is the tipping point. It is the "Allen" model that
Boettiger, Mangel and Munch (2015, Proc. R. Soc. B 282: 20141631) use to study
tipping points in fisheries management. Reed (1979, J. Environ. Econ. Manage.
6: 350-363) proved that with multiplicative shocks and a linear harvest value
the optimal policy is a constant escapement S* where gamma E[z] f'(S*) = 1,
wherever the stock exceeds S*.

Teaching values: R, K, C, SIGMA, GAMMA and the unit price are chosen for the
example; no stock assessment enters the model. With these values the optimal
policy has three regions: fish the stock out below a level x_hat, close the
fishery between x_hat and S*, and harvest down to S* above it. Clark (1973,
J. Political Economy 81: 950-961) gives the economics of the first region.

The module provides the exact solution on a fine grid (the reference), two
weighted-residual approximations with n coefficients (Chebyshev polynomials
collocated at Chebyshev nodes, and piecewise-linear interpolation on uniform
nodes), both solved by function iteration, and exact evaluation of the greedy
policy of any approximation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import chebyshev
from scipy import sparse
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve

# Teaching values.
R = 2.0          # intrinsic growth rate
K = 1.0          # carrying capacity (the unit of the stock)
C = 0.3          # Allee threshold: below C the stock declines without fishing
SIGMA = 0.15     # standard deviation of log z
GAMMA = 0.95     # annual discount factor
PRICE = 1.0      # value of one unit of harvest

X_MAX = 1.6      # upper end of the state space; z f(s) above it is clipped
N_FINE = 4001    # nodes of the reference grid
N_SHOCK = 21     # Gauss-Hermite nodes for the expectation over z


def growth(s):
    """Expected stock next year after an escapement s: f(s)."""
    s = np.asarray(s, dtype=float)
    return s * np.exp(R * (1.0 - s / K) * (s - C) / K)


def growth_slope(s):
    """f'(s)."""
    s = np.asarray(s, dtype=float)
    g = R * (1.0 - s / K) * (s - C) / K
    dg = R * (1.0 + C / K - 2.0 * s / K) / K
    return np.exp(g) * (1.0 + s * dg)


def shocks(n: int = N_SHOCK, sigma: float = SIGMA):
    """Gauss-Hermite nodes and weights for z = exp(sigma e - sigma^2 / 2), e ~ N(0, 1)."""
    e, w = np.polynomial.hermite_e.hermegauss(n)
    return np.exp(sigma * e - 0.5 * sigma ** 2), w / w.sum()


def reed_escapement(gamma: float = GAMMA) -> float:
    """Reed's constant escapement: the root of gamma f'(S) = 1 on the falling side of f'."""
    peak = 0.5 * (K + C)            # f' is decreasing from about here to K
    return brentq(lambda s: gamma * growth_slope(s) - 1.0, peak - 0.2, K)


def fine_grid(n: int = N_FINE) -> np.ndarray:
    return np.linspace(0.0, X_MAX, n)


def _cumulative_argmax(values: np.ndarray) -> np.ndarray:
    """index of the maximum of values[: i + 1], for every i (first maximum on ties)."""
    running = np.maximum.accumulate(values)
    hits = np.where(values >= running, np.arange(values.size), 0)
    return np.maximum.accumulate(hits)


class Bellman:
    """The Bellman operator of the fishery, evaluated with escapements on a fine grid.

    For a value function v, (L v)(x) = x + max_{s <= x} G(s) with
    G(s) = gamma E[v(z f(s))] - s, because the harvest value x - s is linear.
    With escapements restricted to the fine grid, the maximum over s <= x is a
    running maximum, so one application costs one pass over the grid.
    """

    def __init__(self, n_fine: int = N_FINE, n_shock: int = N_SHOCK, gamma: float = GAMMA):
        self.x = fine_grid(n_fine)
        self.gamma = gamma
        self.z, self.w = shocks(n_shock)
        self.next_stock = np.clip(np.outer(growth(self.x), self.z), 0.0, X_MAX)

    def continuation(self, v) -> np.ndarray:
        """G(s) = gamma E[v(z f(s))] - PRICE s on the fine grid; v is a callable."""
        return self.gamma * (v(self.next_stock) @ self.w) - PRICE * self.x

    def apply(self, v, points=None) -> np.ndarray:
        """(L v) at the fine grid, or at arbitrary points (escapements still on the grid)."""
        running = np.maximum.accumulate(self.continuation(v))
        if points is None:
            return PRICE * self.x + running
        index = np.searchsorted(self.x, np.asarray(points), side="right") - 1
        return PRICE * np.asarray(points) + running[np.clip(index, 0, self.x.size - 1)]

    def greedy(self, v) -> np.ndarray:
        """Escapement chosen by the greedy policy of v at every fine-grid stock."""
        return self.x[_cumulative_argmax(self.continuation(v))]

    def evaluate(self, escapement: np.ndarray) -> np.ndarray:
        """Exact value of a policy given by its escapement at each fine-grid stock.

        Linear interpolation between grid nodes defines the transition matrix;
        the policy-evaluation equations are then solved directly.
        """
        x, n = self.x, self.x.size
        y = np.clip(np.outer(growth(escapement), self.z), 0.0, X_MAX)
        h = x[1] - x[0]
        j = np.minimum((y / h).astype(int), n - 2)
        t = y / h - j
        rows = np.repeat(np.arange(n), self.z.size)
        data = np.concatenate([((1 - t) * self.w).ravel(), (t * self.w).ravel()])
        cols = np.concatenate([j.ravel(), j.ravel() + 1])
        P = sparse.csr_matrix((data, (np.concatenate([rows, rows]), cols)), shape=(n, n))
        return spsolve(sparse.identity(n, format="csr") - self.gamma * P, PRICE * (x - escapement))


@dataclass
class Exact:
    x: np.ndarray
    v: np.ndarray
    escapement: np.ndarray
    iterations: int

    def __call__(self, points):
        return np.interp(points, self.x, self.v)

    @property
    def fish_out_level(self) -> float:
        return fish_out_level(self.x, self.escapement)

    @property
    def constant_escapement(self) -> float:
        return float(self.escapement[np.searchsorted(self.x, K)])


def fish_out_level(x: np.ndarray, escapement: np.ndarray) -> float:
    """Largest stock that the policy harvests down to zero (0 if none above the origin)."""
    out = x[(escapement == 0.0) & (x > 0.0)]
    return float(out.max()) if out.size else 0.0


def solve_exact(bellman: Bellman | None = None, tol: float = 1e-11, max_iter: int = 5000) -> Exact:
    """Value iteration on the fine grid with linear interpolation (the reference solution)."""
    bellman = bellman or Bellman()
    x = bellman.x
    v = np.zeros_like(x)
    for k in range(1, max_iter + 1):
        new = bellman.apply(lambda q, v=v: np.interp(q, x, v))
        step = np.abs(new - v).max()
        v = new
        if step < tol:
            break
    escapement = bellman.greedy(lambda q: np.interp(q, x, v))
    return Exact(x, v, escapement, k)


def chebyshev_nodes(n: int) -> np.ndarray:
    k = np.arange(n)
    return np.sort(0.5 * X_MAX * (1.0 + np.cos((2 * k + 1) * np.pi / (2 * n))))


class ChebyshevBasis:
    """v(x) = sum_j theta_j T_j(2 x / X_MAX - 1), collocated at the n Chebyshev nodes."""

    name = "Chebyshev polynomial"

    def __init__(self, n: int):
        self.n = n
        self.nodes = chebyshev_nodes(n)
        self.matrix = chebyshev.chebvander(self._t(self.nodes), n - 1)

    @staticmethod
    def _t(x):
        return 2.0 * np.asarray(x) / X_MAX - 1.0

    def fit(self, targets: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.matrix, targets)

    def function(self, theta: np.ndarray):
        return lambda q: chebyshev.chebval(self._t(q), theta)


class HatBasis:
    """Piecewise-linear interpolation on n uniform nodes (hat functions); theta are the nodal values."""

    name = "Piecewise linear"

    def __init__(self, n: int):
        self.n = n
        self.nodes = np.linspace(0.0, X_MAX, n)

    def fit(self, targets: np.ndarray) -> np.ndarray:
        return np.array(targets, dtype=float)

    def function(self, theta: np.ndarray):
        nodes = self.nodes
        return lambda q: np.interp(q, nodes, theta)


BASES = {"chebyshev": ChebyshevBasis, "linear": HatBasis}


@dataclass
class Collocation:
    basis: object
    theta: np.ndarray
    steps: list = field(default_factory=list)
    diverged: bool = False

    @property
    def iterations(self) -> int:
        return len(self.steps)

    @property
    def v(self):
        return self.basis.function(self.theta)


def collocate(basis, bellman: Bellman, tol: float = 1e-10, max_iter: int = 2000,
              blow_up: float = 1e6) -> Collocation:
    """Function iteration theta <- Phi^{-1} (L v_theta)(nodes): fitted value iteration with an exact fit at n nodes."""
    theta = np.zeros(basis.n)
    steps = []
    for _ in range(max_iter):
        targets = bellman.apply(basis.function(theta), basis.nodes)
        new = basis.fit(targets)
        step = float(np.abs(basis.function(new)(basis.nodes) - basis.function(theta)(basis.nodes)).max())
        theta = new
        steps.append(step)
        if not np.isfinite(step) or step > blow_up:
            return Collocation(basis, theta, steps, diverged=True)
        if step < tol:
            break
    return Collocation(basis, theta, steps)


def assess(result: Collocation, exact: Exact, bellman: Bellman) -> dict:
    """Errors of an approximation against the reference, and the loss of its greedy policy."""
    x = bellman.x
    if result.diverged:
        return {"basis": result.basis.name, "n": result.basis.n, "diverged": True,
                "iterations": result.iterations}
    vhat = result.v
    values = vhat(x)
    residual = bellman.apply(vhat) - values
    escapement = bellman.greedy(vhat)
    loss = exact.v - bellman.evaluate(escapement)
    return {
        "basis": result.basis.name, "n": result.basis.n, "diverged": False,
        "iterations": result.iterations,
        "value_error": float(np.abs(values - exact.v).max()),
        "residual": float(np.abs(residual).max()),
        "fish_out_level": fish_out_level(x, escapement),
        "constant_escapement": float(escapement[np.searchsorted(x, K)]),
        "policy_loss": float(loss.max()),
        "policy_loss_mean": float(loss.mean()),
    }

