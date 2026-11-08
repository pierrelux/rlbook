"""Setting duck-hunting regulations for midcontinent mallards.

Every summer the U.S. Fish and Wildlife Service picks one of four regulation
packages for the Mississippi and Central Flyways (closed, restrictive,
moderate, liberal) from two survey numbers: the breeding population N of
midcontinent mallards and the number P of ponds in Prairie Canada in May,
both in millions. The choice is made with a policy computed by stochastic
dynamic programming (Johnson et al. 1997, J. Wildl. Manage. 61: 202-216), and
published each year as a table of N against P.

The population models, harvest rates, model weights, population goal and
closed-season constraint below are those of the Service's report for the 2012
season (Adaptive Harvest Management: 2012 Hunting Season, Appendix C and
Sections 4-7), which also prints that year's policy table (Table 3):

- balance equation with bias corrections, proportion of males m = 0.5246,
  summer survival ratio of young females to young males 0.897;
- survival either additive in the harvest (natural survival 0.7896 males,
  0.6886 females) or compensatory (0.6467, 0.5965), with the kill rate equal
  to the harvest rate inflated for 20% crippling loss;
- recruitment weakly (0.7166 + 0.1083 P - 0.0373 N) or strongly
  (1.1390 + 0.1376 P - 0.1131 N) density dependent;
- ponds P' = 2.2127 + 0.3420 P + e with Var(e) = 1.2567;
- adult-male harvest rates 0.0088, 0.0552, 0.1011, 0.1169 and fixed ratios for
  the other cohorts; model weights 66% additive and 94% weakly density
  dependent; goal 8.5 million; season closed below 4.75 million.

Our choices, which the report does not spell out: the four models are
weighted by the products of the two marginal weights; harvest is counted from
the fall flight with the cohort structure of the balance equation; harvest is
devalued in proportion to the shortfall of the expected next breeding
population below the goal; the mean harvest rates are used without their
year-to-year variation; the population is clipped to [N_MIN, N_MAX] and
ponds to [0, P_MAX]; and future harvests are discounted with GAMMA (the
Service maximizes undiscounted long-term harvest).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

PACKAGES = ("closed", "restrictive", "moderate", "liberal")
CODES = ("C", "R", "M", "L")
ADULT_MALE_RATE = np.array([0.0088, 0.0552, 0.1011, 0.1169])
COHORT_RATIO = {"am": 1.0, "jm": 1.5407, "af": 0.7191, "jf": 1.1175}
CRIPPLING = 0.20
MALE_FRACTION = 0.5246
YOUNG_SEX_RATIO = 0.897           # summer survival of young females over young males
SURVIVAL_BIAS, RECRUIT_BIAS = 0.9407, 0.8647
ADDITIVE_S0 = {"m": 0.7896, "f": 0.6886}
COMPENSATORY_S0 = {"m": 0.6467, "f": 0.5965}
RECRUITMENT = {"weak": (0.7166, 0.1083, -0.0373), "strong": (1.1390, 0.1376, -0.1131)}
PONDS = (2.2127, 0.3420, 1.2567)  # intercept, slope, error variance
WEIGHT_ADDITIVE, WEIGHT_WEAK = 0.66, 0.94
GOAL = 8.5                        # million birds
CLOSED_BELOW = 4.75               # closed-season constraint, million birds

# Our choices.
GAMMA = 0.98
N_MIN, N_MAX = 1.0, 16.0          # breeding population range of the grid, million birds
P_MAX = 8.0
N_POND_SHOCK = 15

MODELS = [(surv, rec) for surv in ("additive", "compensatory") for rec in ("weak", "strong")]
MODEL_WEIGHTS = np.array([(WEIGHT_ADDITIVE if s == "additive" else 1 - WEIGHT_ADDITIVE)
                          * (WEIGHT_WEAK if r == "weak" else 1 - WEIGHT_WEAK) for s, r in MODELS])


def harvest_rates(package: int) -> dict:
    return {c: ADULT_MALE_RATE[package] * ratio for c, ratio in COHORT_RATIO.items()}


def survival(model: str, sex: str, harvest_rate):
    kill = np.asarray(harvest_rate) / (1.0 - CRIPPLING)
    if model == "additive":
        return ADDITIVE_S0[sex] * (1.0 - kill)
    s0 = COMPENSATORY_S0[sex]
    return np.where(kill <= 1.0 - s0, s0, 1.0 - kill)


def recruitment(model: str, N, P):
    a, b, c = RECRUITMENT[model]
    return np.maximum(a + b * np.asarray(P) + c * np.asarray(N), 0.0)


def step_population(N, P, package: int, model: tuple[str, str]):
    """Breeding population next year (millions) under one of the four models."""
    surv, rec = model
    h = harvest_rates(package)
    R = RECRUIT_BIAS * recruitment(rec, N, P)
    s = {"am": survival(surv, "m", h["am"]), "af": survival(surv, "f", h["af"]),
         "jm": survival(surv, "m", h["jm"]), "jf": survival(surv, "f", h["jf"])}
    per_bird = MALE_FRACTION * s["am"] + (1 - MALE_FRACTION) * (s["af"] + R * (s["jf"] + YOUNG_SEX_RATIO * s["jm"]))
    return SURVIVAL_BIAS * np.asarray(N) * per_bird


def harvest(N, P, package: int, model: tuple[str, str]):
    """Birds retrieved in the fall (millions), counted on the fall flight."""
    h = harvest_rates(package)
    R = RECRUIT_BIAS * recruitment(model[1], N, P)
    per_bird = MALE_FRACTION * h["am"] + (1 - MALE_FRACTION) * (h["af"] + R * (h["jf"] + YOUNG_SEX_RATIO * h["jm"]))
    return np.asarray(N) * per_bird


def allowed(N) -> np.ndarray:
    """Mask of allowed packages, shape (..., 4): the season is closed below CLOSED_BELOW and open above."""
    N = np.asarray(N)
    open_season = (N >= CLOSED_BELOW)[..., None]
    mask = np.ones(N.shape + (4,), dtype=bool)
    mask[..., :1] = ~open_season
    mask[..., 1:] = open_season
    return mask


def pond_shocks(n: int = N_POND_SHOCK):
    e, w = np.polynomial.hermite_e.hermegauss(n)
    return np.sqrt(PONDS[2]) * e, w / w.sum()


@dataclass
class Model:
    """Rewards and successor states on a set of (N, P) points, for every package.

    reward[a] has shape (M,); next_N[a] has shape (M, 4 models); next_P has
    shape (M, n_shock). The successor of (N, P) under a is (next_N[a, :, m],
    next_P[:, k]) with probability MODEL_WEIGHTS[m] * pond_weight[k].
    """
    N: np.ndarray
    P: np.ndarray
    reward: np.ndarray
    next_N: np.ndarray
    next_P: np.ndarray
    pond_weight: np.ndarray
    mask: np.ndarray


def build_model(N, P) -> Model:
    N, P = np.asarray(N, float).ravel(), np.asarray(P, float).ravel()
    eps, wp = pond_shocks()
    next_P = np.clip(PONDS[0] + PONDS[1] * P[:, None] + eps[None, :], 0.0, P_MAX)
    reward, next_N = [], []
    for a in range(4):
        nn = np.stack([step_population(N, P, a, m) for m in MODELS], axis=1)
        expected = nn @ MODEL_WEIGHTS
        h = np.stack([harvest(N, P, a, m) for m in MODELS], axis=1) @ MODEL_WEIGHTS
        reward.append(h * np.minimum(1.0, expected / GOAL))
        next_N.append(np.clip(nn, N_MIN, N_MAX))
    return Model(N, P, np.array(reward), np.array(next_N), next_P, wp, allowed(N))


def q_values(model: Model, v, gamma: float = GAMMA) -> np.ndarray:
    """Q(s, a) for every point of the model and package; v maps (k, 2) arrays of (N, P) to values."""
    M, S = model.N.size, model.pond_weight.size
    q = np.empty((4, M))
    for a in range(4):
        pts = np.empty((M, len(MODELS), S, 2))
        pts[..., 0] = model.next_N[a][:, :, None]
        pts[..., 1] = model.next_P[:, None, :]
        vals = v(pts.reshape(-1, 2)).reshape(M, len(MODELS), S)
        q[a] = model.reward[a] + gamma * np.einsum("mjk,j,k->m", vals, MODEL_WEIGHTS, model.pond_weight)
    return np.where(model.mask.T, q, -np.inf)


class Grid:
    """Bilinear interpolation on a rectangular grid of (N, P): an averager."""

    def __init__(self, dN: float, dP: float):
        self.n_axis = np.linspace(N_MIN, N_MAX, int(round((N_MAX - N_MIN) / dN)) + 1)
        self.p_axis = np.linspace(0.0, P_MAX, int(round(P_MAX / dP)) + 1)
        NN, PP = np.meshgrid(self.n_axis, self.p_axis, indexing="ij")
        self.points = np.column_stack([NN.ravel(), PP.ravel()])

    @property
    def size(self) -> int:
        return self.points.shape[0]

    def weights(self, pts: np.ndarray) -> sparse.csr_matrix:
        """Sparse matrix W with (W @ values)[i] the bilinear interpolant at pts[i]."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        n_n, n_p = self.n_axis.size, self.p_axis.size
        fn = np.clip((pts[:, 0] - N_MIN) / (self.n_axis[1] - self.n_axis[0]), 0, n_n - 1 - 1e-9)
        fp = np.clip(pts[:, 1] / (self.p_axis[1] - self.p_axis[0]), 0, n_p - 1 - 1e-9)
        i, j = fn.astype(int), fp.astype(int)
        tn, tp = fn - i, fp - j
        rows = np.repeat(np.arange(len(pts)), 4)
        cols = np.stack([i * n_p + j, (i + 1) * n_p + j, i * n_p + j + 1, (i + 1) * n_p + j + 1], 1).ravel()
        data = np.stack([(1 - tn) * (1 - tp), tn * (1 - tp), (1 - tn) * tp, tn * tp], 1).ravel()
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(pts), self.size))

    def function(self, values: np.ndarray):
        return lambda pts: self.weights(pts) @ values


@dataclass
class Solution:
    grid: Grid
    values: np.ndarray
    policy: np.ndarray
    iterations: int

    @property
    def v(self):
        return self.grid.function(self.values)


def transition_matrices(grid: Grid, model: Model) -> list:
    """For each package a, the sparse matrix P_a with (P_a v)(s) = E[v(s') | s, a] under bilinear interpolation."""
    M, S = model.N.size, model.pond_weight.size
    prob = np.einsum("j,k->jk", MODEL_WEIGHTS, model.pond_weight).ravel()
    out = []
    for a in range(4):
        pts = np.empty((M, len(MODELS), S, 2))
        pts[..., 0] = model.next_N[a][:, :, None]
        pts[..., 1] = model.next_P[:, None, :]
        W = grid.weights(pts.reshape(-1, 2))                       # (M * 4 * S, grid)
        avg = sparse.csr_matrix((np.tile(prob, M), (np.repeat(np.arange(M), prob.size),
                                                     np.arange(M * prob.size))), shape=(M, M * prob.size))
        out.append((avg @ W).tocsr())
    return out


def value_iteration(grid: Grid, gamma: float = GAMMA, tol: float = 1e-7, max_iter: int = 20000) -> Solution:
    """Value iteration on the grid with bilinear interpolation of the successor values."""
    model = build_model(grid.points[:, 0], grid.points[:, 1])
    P = transition_matrices(grid, model)
    reward = np.where(model.mask.T, model.reward, -np.inf)
    values = np.zeros(grid.size)
    for k in range(1, max_iter + 1):
        q = reward + gamma * np.stack([Pa @ values for Pa in P])
        new = q.max(axis=0)
        step = np.abs(new - values).max()
        values = new
        if step < tol * (1 - gamma):
            break
    return Solution(grid, values, q.argmax(axis=0), k)


def greedy(points, v, gamma: float = GAMMA) -> np.ndarray:
    return q_values(build_model(points[:, 0], points[:, 1]), v, gamma).argmax(axis=0)


def evaluate_policy(grid: Grid, actions: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Exact value on the grid of the policy that takes actions[i] at grid point i (bilinear successors)."""
    model = build_model(grid.points[:, 0], grid.points[:, 1])
    P = transition_matrices(grid, model)
    P_pi = _select_rows(P, actions)
    reward = model.reward[actions, np.arange(grid.size)]
    return spsolve((sparse.identity(grid.size, format="csr") - gamma * P_pi).tocsc(), reward)


def _select_rows(P: list, actions: np.ndarray) -> sparse.csr_matrix:
    out = P[0].copy().tolil()
    for a in range(1, 4):
        idx = np.flatnonzero(actions == a)
        if idx.size:
            out[idx] = P[a][idx]
    return out.tocsr()


def published_2012() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Table 3 of the 2012 report: breeding-population rows, pond columns, package indices."""
    ponds = np.arange(1.5, 6.01, 0.5)
    rows = [(4.5, "CCCCCCCCCC"), (4.75, "RRRRRRRRRR"), (5.0, "RRRRRRRRRR"), (5.25, "RRRRRRRRRR"),
            (5.5, "RRRRRRRRRR"), (5.75, "RRRRRRRRRR"), (6.0, "RRRRRRRRRR"), (6.25, "RRRRRRRRRM"),
            (6.5, "RRRRRRRMMM"), (6.75, "RRRRRMMMLL"), (7.0, "RRRMMMLLLL"), (7.25, "RRMMLLLLLL"),
            (7.5, "RMLLLLLLLL"), (7.75, "MLLLLLLLLL"), (8.0, "LLLLLLLLLL")]
    population = np.array([r[0] for r in rows])
    table = np.array([[CODES.index(c) for c in r[1]] for r in rows])
    return population, ponds, table


def bellman_targets(model: Model, v, gamma: float = GAMMA) -> np.ndarray:
    """(L v)(s) at the points of the model."""
    return q_values(model, v, gamma).max(axis=0)


class GridFit:
    """Fit by interpolation at the nodes of a grid: targets are computed at the nodes."""

    def __init__(self, step: float):
        self.grid = Grid(step, step)
        self.states = self.grid.points
        self.name = f"bilinear grid, step {step:g}"

    def __call__(self, states, targets, k):
        return self.grid.function(np.asarray(targets, float))


class TreeFit:
    """Extra-trees regression on sampled states (an averager, Ernst et al. 2005)."""

    def __init__(self, states, n_estimators: int = 30, min_samples_leaf: int = 3):
        self.states = states
        self.options = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
        self.name = "extra trees"

    def __call__(self, states, targets, k):
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(random_state=k, **self.options).fit(states, targets)
        return lambda pts: model.predict(np.asarray(pts, float).reshape(-1, 2))


class NetworkFit:
    """A two-layer network, warm-started from the previous iteration and trained for a few epochs."""

    SCALE = np.array([N_MAX, P_MAX])
    VALUE_SCALE = 80.0

    def __init__(self, states, hidden=(64, 64), epochs: int = 30, seed: int = 0):
        from sklearn.neural_network import MLPRegressor
        self.states = states
        self.model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=epochs, warm_start=True,
                                  random_state=seed, learning_rate_init=1e-3)
        self.name = "neural network"

    def __call__(self, states, targets, k):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(states / self.SCALE, np.asarray(targets) / self.VALUE_SCALE)
        coefs = [w.copy() for w in self.model.coefs_]
        intercepts = [b.copy() for b in self.model.intercepts_]

        def v(pts, coefs=coefs, intercepts=intercepts):
            h = np.asarray(pts, float).reshape(-1, 2) / self.SCALE
            for W, b in zip(coefs[:-1], intercepts[:-1]):
                h = np.maximum(h @ W + b, 0.0)
            return (h @ coefs[-1] + intercepts[-1]).ravel() * self.VALUE_SCALE
        return v


def sample_states(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(N_MIN, N_MAX, n), rng.uniform(0.0, P_MAX, n)])


def fitted_value_iteration(fit, iterations: int, monitor: np.ndarray, gamma: float = GAMMA,
                           monitor_last: int = 20):
    """v_{k+1} = fit(L v_k at fit.states). Returns the last v and, for each of the last
    monitor_last iterations, the fitting error max |v_{k+1} - L v_k| over the monitor points."""
    model = build_model(fit.states[:, 0], fit.states[:, 1])
    monitor_model = build_model(monitor[:, 0], monitor[:, 1])
    v = lambda pts: np.zeros(np.asarray(pts).reshape(-1, 2).shape[0])
    errors = []
    for k in range(iterations):
        new = fit(fit.states, bellman_targets(model, v, gamma), k)
        if k >= iterations - monitor_last:
            errors.append(float(np.abs(new(monitor) - bellman_targets(monitor_model, v, gamma)).max()))
        v = new
    return v, errors
