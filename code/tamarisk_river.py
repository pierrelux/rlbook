"""Managing invasive tamarisk in a river network as a discounted MDP.

The model follows the tamarisk benchmark of Hall et al. (2012) and Taleghan et
al. (2015, JMLR 16:3877-3903). A river is a tree of reaches; water flows from
the leaves toward the outlet reach 0. Each reach has one slot (Taleghan et
al.'s H = 1) that is empty, native or tamarisk, so there are 3**n states.

Each year, in this order:
  1. the manager acts on each reach: do nothing, eradicate (kills a tamarisk
     plant with probability CHI) or restore (eradicate, then plant a native in
     the empty slot with probability BETA);
  2. every plant dies with probability DEATH;
  3. each surviving plant releases SEEDS seeds, which spread over the reaches
     with a kernel that favours downstream moves; exogenous seeds also arrive;
  4. an empty slot stays empty if no seed arrives, otherwise one seed takes it,
     tamarisk with a competitive weight ADVANTAGE.

Values from Taleghan et al. (2015): DEATH, SEEDS, the upstream/downstream
dispersal parameters, CHI, BETA, the action costs, the damage costs, the mean
exogenous arrivals, GAMMA and the one-treatment-per-year budget. ADVANTAGE is
one of the example values in Hall et al. (2012). The geometric form of the
kernel, the Poisson seed rain and the order of events are teaching choices,
because the papers delegate them to a simulator.

Seed rain is Poisson, so given the surviving plants the slots fill
independently and the exact kernel factors as
p(. | s, a) = A_a D G, with A_a and D Kronecker products of 3x3 matrices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from itertools import product
import time

import numpy as np

EMPTY, NATIVE, TAMARISK = 0, 1, 2
CODES = "ENT"
NOTHING, ERADICATE, RESTORE = 0, 1, 2
ACTION_NAMES = ("nothing", "eradicate", "restore")

# Taleghan et al. (2015), Section 3 and Section 5.
DEATH = 0.2
SEEDS = 100
DOWN, UP = 0.5, 0.1
CHI, BETA = 0.85, 0.65
ACTION_COST = (0.0, 0.5, 0.9)
INVADED_COST, SLOT_COST = 1.0, 0.1
EXO_TAMARISK, EXO_NATIVE = 1.0, 4.0     # means of 10 Bernoulli(0.1) and 10 Bernoulli(0.4) draws
GAMMA = 0.9
# Hall et al. (2012) slides: competitive advantage examples 1.0, 1.5, 2.0.
ADVANTAGE = 1.5

# Per-reach kernels, rows = current code, columns = next code.
ACTION_KERNELS = np.array([
    np.eye(3),
    [[1, 0, 0], [0, 1, 0], [CHI, 0, 1 - CHI]],
    [[1 - BETA, BETA, 0], [0, 1, 0], [CHI * (1 - BETA), CHI * BETA, 1 - CHI]],
])
DEATH_KERNEL = np.array([[1, 0, 0], [DEATH, 1 - DEATH, 0], [DEATH, 0, 1 - DEATH]])


def parent(j: int) -> int:
    return (j - 1) // 2 if j > 0 else -1


def depth(j: int) -> int:
    d = 0
    while j > 0:
        j, d = parent(j), d + 1
    return d


def path_steps(i: int, j: int) -> tuple[int, int]:
    """(downstream steps, upstream steps) on the tree path from reach i to reach j."""
    anc_i = []
    k = i
    while k >= 0:
        anc_i.append(k)
        k = parent(k)
    k, up = j, 0
    while k not in anc_i:
        k, up = parent(k), up + 1
    return anc_i.index(k), up


def dispersal(n: int) -> np.ndarray:
    """K[i, j]: probability that a seed from reach i lands in reach j.

    Weight DOWN**(downstream steps) * UP**(upstream steps), the seed itself
    staying put with weight 1, and leaving through the outlet with weight
    DOWN**(depth + 1). Rows sum to less than one; the rest is lost to the sea.
    """
    K = np.zeros((n, n))
    for i in range(n):
        w = np.array([DOWN ** d * UP ** u for d, u in (path_steps(i, j) for j in range(n))])
        K[i] = w / (w.sum() + DOWN ** (depth(i) + 1))
    return K


def decode(n: int) -> np.ndarray:
    """codes[s, j] for every state index s (reach 0 is the most significant digit)."""
    return np.array(list(product(range(3), repeat=n)), dtype=np.int8).reshape(-1, n)


def encode(x) -> int:
    s = 0
    for c in x:
        s = 3 * s + int(c)
    return s


def establishment(survivors: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Probability that an empty slot ends empty/native/tamarisk, per reach.

    survivors: (..., n) codes after mortality. Returns (..., n, 3).
    """
    lam_t = SEEDS * (survivors == TAMARISK).astype(float) @ K + EXO_TAMARISK
    lam_n = SEEDS * (survivors == NATIVE).astype(float) @ K + EXO_NATIVE
    p_empty = np.exp(-(lam_t + lam_n))
    share_t = ADVANTAGE * lam_t / (ADVANTAGE * lam_t + lam_n)
    return np.stack([p_empty, (1 - p_empty) * (1 - share_t), (1 - p_empty) * share_t], axis=-1)


def regrowth_matrix(n: int, K: np.ndarray) -> np.ndarray:
    """G[y, s']: from the survivors y to the next state s' (occupied slots keep their plant)."""
    Y = decode(n)
    est = establishment(Y, K)                                   # (S, n, 3)
    occupied = Y != EMPTY
    rows = np.where(occupied[..., None], np.eye(3)[Y], est)    # (S, n, 3)
    return reduce(lambda acc, j: (acc[:, :, None] * rows[:, None, j, :]).reshape(len(Y), -1),
                  range(1, n), rows[:, 0, :])


def apply_axis(M: np.ndarray, w: np.ndarray, j: int, n: int) -> np.ndarray:
    """(I x ... x M x ... x I) w, with M acting on reach j; works on (S,) or (S, m)."""
    tail = w.shape[1:]
    W = w.reshape((3,) * n + tail)
    W = np.moveaxis(np.tensordot(M, W, axes=([1], [j])), 0, j)
    return W.reshape((-1,) + tail)


@dataclass
class River:
    n: int
    budget: int | None = 1
    K: np.ndarray = field(init=False)
    DG: np.ndarray = field(init=False)
    actions: list = field(init=False)
    cost: np.ndarray = field(init=False)
    build_seconds: float = field(init=False)

    def __post_init__(self):
        t0 = time.perf_counter()
        n = self.n
        self.K = dispersal(n)
        G = regrowth_matrix(n, self.K)
        for j in range(n):
            G = apply_axis(DEATH_KERNEL, G, j, n)
        self.DG = G                                             # (D G)[x, s']
        if self.budget is None:
            self.actions = list(product(range(3), repeat=n))
        else:
            self.actions = [tuple([NOTHING] * n)]
            for b in range(1, self.budget + 1):
                for reaches in _combinations(n, b):
                    for kinds in product((ERADICATE, RESTORE), repeat=b):
                        a = [NOTHING] * n
                        for r, k in zip(reaches, kinds):
                            a[r] = k
                        self.actions.append(tuple(a))
        X = decode(n)
        damage = (INVADED_COST + SLOT_COST) * (X == TAMARISK).sum(axis=1)
        act = np.array([sum(ACTION_COST[k] for k in a) for a in self.actions])
        self.cost = damage[:, None] + act[None, :]
        self.build_seconds = time.perf_counter() - t0

    @property
    def n_states(self) -> int:
        return 3 ** self.n

    def action_matrix(self, a) -> np.ndarray:
        return reduce(np.kron, [ACTION_KERNELS[k] for k in a])

    def transition(self, a) -> np.ndarray:
        """Dense p(j | s, a) for one joint action (for tests and small n)."""
        return self.action_matrix(a) @ self.DG

    def expected_next(self, v: np.ndarray) -> np.ndarray:
        """E[v(s') | s, a] for every (s, a), without forming p(. | s, a)."""
        n, w = self.n, self.DG @ v
        if self.budget is None:
            # Contract each reach with all three action kernels at once.
            # Axes go from (x_0, ..., x_{n-1}) to (a_0, s_0, ..., a_{n-1}, s_{n-1}).
            W = w.reshape((3,) * n)
            for _ in range(n):
                W = np.tensordot(W, ACTION_KERNELS, axes=([0], [2]))
            W = W.transpose(list(range(1, 2 * n, 2)) + list(range(0, 2 * n, 2)))
            return W.reshape(3 ** n, 3 ** n)
        out = np.empty((self.n_states, len(self.actions)))
        for i, a in enumerate(self.actions):
            u = w
            for j, k in enumerate(a):
                if k != NOTHING:
                    u = apply_axis(ACTION_KERNELS[k], u, j, n)
            out[:, i] = u
        return out

    def q_values(self, v: np.ndarray) -> np.ndarray:
        return self.cost + GAMMA * self.expected_next(v)

    def policy_matrix(self, policy) -> tuple[np.ndarray, np.ndarray]:
        """(P_pi, c_pi) for a policy given as one joint-action tuple per state."""
        X = decode(self.n)
        A = np.zeros((self.n_states, self.n_states))
        for s, a in enumerate(policy):
            row = reduce(np.kron, [ACTION_KERNELS[k][X[s, j]] for j, k in enumerate(a)])
            A[s] = row
        X_t = (X == TAMARISK).sum(axis=1)
        c = (INVADED_COST + SLOT_COST) * X_t + np.array([sum(ACTION_COST[k] for k in a) for a in policy])
        return A @ self.DG, c


def _combinations(n, b):
    from itertools import combinations
    return combinations(range(n), b)


@dataclass
class Solution:
    v: np.ndarray
    policy: list             # joint-action tuple per state
    iterations: int
    seconds: float


def value_iteration(river: River, eps: float = 1e-6, v0=None) -> Solution:
    """Algorithm value-iteration (cost version): stop when the sup-norm change
    falls below eps (1 - gamma) / (2 gamma)."""
    t0 = time.perf_counter()
    v = np.zeros(river.n_states) if v0 is None else v0.copy()
    it = 0
    while True:
        q = river.q_values(v)
        v_new = q.min(axis=1)
        it += 1
        delta = np.abs(v_new - v).max()
        v = v_new
        if delta < eps * (1 - GAMMA) / (2 * GAMMA):
            break
    q = river.q_values(v)
    greedy = q.argmin(axis=1)
    return Solution(v, [river.actions[i] for i in greedy], it, time.perf_counter() - t0)


def evaluate(river: River, policy) -> np.ndarray:
    P, c = river.policy_matrix(policy)
    return np.linalg.solve(np.eye(river.n_states) - GAMMA * P, c)


def policy_iteration(river: River, max_iter: int = 100) -> Solution:
    """Algorithm policy-iteration-standard, starting from 'do nothing everywhere'."""
    t0 = time.perf_counter()
    idx = np.zeros(river.n_states, dtype=int)
    for it in range(1, max_iter + 1):
        v = evaluate(river, [river.actions[i] for i in idx])
        q = river.q_values(v)
        best = q.min(axis=1)
        # Keep the current action unless another is strictly better.
        keep = q[np.arange(river.n_states), idx] <= best + 1e-10
        new = np.where(keep, idx, q.argmin(axis=1))
        if np.array_equal(new, idx):
            break
        idx = new
    return Solution(v, [river.actions[i] for i in idx], it, time.perf_counter() - t0)


# Heuristics: functions from a state's codes to a joint action.
def nothing_rule(x):
    return tuple([NOTHING] * len(x))


def _restore_one(x, key):
    invaded = [j for j, c in enumerate(x) if c == TAMARISK]
    a = [NOTHING] * len(x)
    if invaded:
        a[min(invaded, key=key)] = RESTORE
    return tuple(a)


def downstream_first(x):
    """Restore the invaded reach closest to the outlet."""
    return _restore_one(x, key=lambda j: (depth(j), j))


def upstream_first(x):
    """Restore the invaded reach farthest from the outlet (the seed source)."""
    return _restore_one(x, key=lambda j: (-depth(j), j))


def treat_all(x):
    """Restore every invaded reach (ignores the budget)."""
    return tuple(RESTORE if c == TAMARISK else NOTHING for c in x)


HEURISTICS = {"nothing": nothing_rule, "downstream": downstream_first,
              "upstream": upstream_first, "all": treat_all}


def rule_policy(n: int, rule) -> list:
    return [rule(tuple(x)) for x in decode(n)]


def simulate_step(x: np.ndarray, a, K: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Sample one year with uniforms u[j] = (action, death, establishment)."""
    n = len(x)
    y = x.copy()
    for j, k in enumerate(a):
        cdf = np.cumsum(ACTION_KERNELS[k][y[j]])
        y[j] = int(np.searchsorted(cdf, u[j, 0], side="right"))
    for j in range(n):
        if y[j] != EMPTY and u[j, 1] < DEATH:
            y[j] = EMPTY
    est = establishment(y, K)
    z = y.copy()
    for j in range(n):
        if y[j] == EMPTY:
            z[j] = int(np.searchsorted(np.cumsum(est[j]), u[j, 2], side="right"))
    return np.minimum(z, 2)


def stage_cost(x, a) -> float:
    return (INVADED_COST + SLOT_COST) * int((np.asarray(x) == TAMARISK).sum()) + sum(ACTION_COST[k] for k in a)
