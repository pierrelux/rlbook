"""Life-cycle maintenance of a bridge deck as a discounted MDP.

The state is the deck condition rating of the US National Bridge Inventory
(NBI item 58), observed once a year. Ratings 9 to 5 are kept as separate
states and ratings 4 and below are merged into one "poor" state, as in the
deck model of Saifullah et al. (arXiv:2401.12455), whose failure
probabilities and unit costs are used below.

The do-nothing transition matrix is estimated by counting year-to-year rating
changes of decks that received no work. Counts are supplied as a 7 x 7 array
over ratings 9, 8, ..., 3 (see ``count_transitions``); ratings 4 and 3 are
merged by ``do_nothing_matrix``.

The effects of the three maintenance actions are teaching assumptions in the
spirit of Andriotis and Papakonstantinou (2020): a minor repair raises the
rating by one (but cannot fix a poor deck), a major repair raises it by two up
to at most 8, replacement restores a new deck (rating 9), and the deck then
deteriorates for the rest of the year as if nothing had been done.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ratings in state order. The last state stands for "4 or below".
RATINGS = (9, 8, 7, 6, 5, 4)
RATING_LABELS = ("9", "8", "7", "6", "5", "≤4")
N_STATES = len(RATINGS)
# Ratings carried by the raw count matrix, in row/column order.
COUNT_RATINGS = (9, 8, 7, 6, 5, 4, 3)

ACTIONS = ("nothing", "minor", "major", "replace")
ACTION_LABELS = ("Do nothing", "Minor repair", "Major repair", "Replace deck")
# Ratings gained by each repair before the year's deterioration (replace -> new deck).
REPAIR_GAIN = {"nothing": 0, "minor": 1, "major": 2}
# A repair never makes a deck as good as new, and a minor repair cannot fix a poor deck.
REPAIR_CEILING = {"nothing": 9, "minor": 9, "major": 8}

# Saifullah et al. (arXiv:2401.12455), Table 7: action costs in USD per m^2 of deck.
ACTION_COST = np.array([0.0, 400.0, 1200.0, 2650.0])
# Saifullah et al., Table 6: annual failure probability for deck ratings 9, 8, 7, 6, 5, 4 (4 = "4 or below").
FAILURE_PROB = np.array([0.001, 0.001, 0.005, 0.005, 0.005, 0.01])
# Their failure cost: ten times the reconstruction cost.
FAILURE_COST = 10.0 * ACTION_COST[3]
GAMMA = 0.97  # Saifullah et al. use gamma = 0.97 (a 3% discount rate).


def collapse_counts(counts: np.ndarray) -> np.ndarray:
    """Merge ratings 4 and 3 of a 7 x 7 count matrix (ratings 9..3) into one state."""
    counts = np.asarray(counts, dtype=float)
    assert counts.shape == (len(COUNT_RATINGS), len(COUNT_RATINGS))
    merged = counts[:, :N_STATES].copy()
    merged[:, -1] += counts[:, -1]
    out = merged[:N_STATES].copy()
    out[-1] += merged[-1]
    return out


def do_nothing_matrix(counts: np.ndarray) -> np.ndarray:
    """Maximum-likelihood transition matrix from counts; the poor state is absorbing."""
    c = collapse_counts(counts)
    P = c / c.sum(axis=1, keepdims=True)
    # No deck improves without work, and nothing below "4 or below" is modeled.
    P[-1] = 0.0
    P[-1, -1] = 1.0
    return P


def improved_state(s: int, action: str) -> int:
    """State index right after the work, before the year's deterioration."""
    if action == "replace":
        return 0
    if action == "minor" and s == N_STATES - 1:
        return s
    best = RATINGS.index(REPAIR_CEILING[action]) if action != "nothing" else 0
    return max(best, s - REPAIR_GAIN[action]) if action != "nothing" else s


def transition_tensor(P0: np.ndarray) -> np.ndarray:
    """p[a, s, j]: the work moves s to improved_state(s, a), then a year of P0."""
    P = np.zeros((len(ACTIONS), N_STATES, N_STATES))
    for a, name in enumerate(ACTIONS):
        for s in range(N_STATES):
            P[a, s] = P0[improved_state(s, name)]
    return P


def cost_matrix() -> np.ndarray:
    """c[s, a] in USD per m^2: work cost plus the expected failure cost of the year.

    The failure risk is charged on the condition after the work.
    """
    c = np.zeros((N_STATES, len(ACTIONS)))
    for a, name in enumerate(ACTIONS):
        for s in range(N_STATES):
            c[s, a] = ACTION_COST[a] + FAILURE_PROB[improved_state(s, name)] * FAILURE_COST
    return c


@dataclass(frozen=True)
class DeckMDP:
    P: np.ndarray       # (action, s, j)
    c: np.ndarray       # (s, action)
    gamma: float = GAMMA


def build_mdp(counts: np.ndarray, gamma: float = GAMMA) -> DeckMDP:
    return DeckMDP(transition_tensor(do_nothing_matrix(counts)), cost_matrix(), gamma)


def q_values(mdp: DeckMDP, v: np.ndarray) -> np.ndarray:
    """q[s, a] = c(s, a) + gamma * sum_j p(j | s, a) v(j)."""
    return mdp.c + mdp.gamma * np.einsum("asj,j->sa", mdp.P, v)


def value_iteration(mdp: DeckMDP, tol: float = 1e-8, v0=None, max_iter: int = 100_000):
    """Cost-minimizing value iteration with the stopping rule of the chapter.

    Stops when ||v_{n+1} - v_n|| < tol (1 - gamma) / (2 gamma), which makes the
    greedy policy tol-optimal. Returns (v, policy, n_iter, residual history).
    """
    v = np.zeros(N_STATES) if v0 is None else np.asarray(v0, float)
    thresh = tol * (1 - mdp.gamma) / (2 * mdp.gamma)
    history = []
    for n in range(1, max_iter + 1):
        v_new = q_values(mdp, v).min(axis=1)
        delta = float(np.max(np.abs(v_new - v)))
        history.append(delta)
        v = v_new
        if delta < thresh:
            break
    return v, greedy(mdp, v), n, history


def greedy(mdp: DeckMDP, v: np.ndarray) -> np.ndarray:
    q = q_values(mdp, v)
    # Break near-ties toward the cheaper (lower-index) action.
    return np.array([int(np.flatnonzero(row <= row.min() + 1e-9)[0]) for row in q])


def evaluate(mdp: DeckMDP, policy) -> np.ndarray:
    """Solve (I - gamma P_pi) v = c_pi for a stationary deterministic policy."""
    policy = np.asarray(policy)
    idx = np.arange(N_STATES)
    P_pi = mdp.P[policy, idx]
    c_pi = mdp.c[idx, policy]
    return np.linalg.solve(np.eye(N_STATES) - mdp.gamma * P_pi, c_pi)


def policy_iteration(mdp: DeckMDP, policy0=None):
    """Policy iteration (Newton's method on the Bellman equation).

    Starts from "do nothing everywhere" unless told otherwise. Returns
    (v, policy, list of (policy, v) for each iterate).
    """
    policy = np.zeros(N_STATES, int) if policy0 is None else np.asarray(policy0, int)
    trace = []
    while True:
        v = evaluate(mdp, policy)
        trace.append((policy.copy(), v.copy()))
        q = q_values(mdp, v)
        new = policy.copy()
        for s in range(N_STATES):
            if q[s].min() < q[s, policy[s]] - 1e-9:
                new[s] = int(np.argmin(q[s]))
        if np.array_equal(new, policy):
            return v, policy, trace
        policy = new


# ---------------------------------------------------------------- baselines

def reactive_policy(trigger: int = N_STATES - 1) -> np.ndarray:
    """Do nothing until the deck is poor (rating 4 or below), then replace it."""
    pol = np.zeros(N_STATES, int)
    pol[trigger:] = ACTIONS.index("replace")
    return pol


def scheduled_value(mdp: DeckMDP, period: int, action: str = "minor") -> np.ndarray:
    """Preventive work every `period` years whatever the rating, replacement when poor.

    The policy depends on the year modulo `period`, so it is evaluated exactly on
    the chain over (phase, rating). Phase 0 is a work year, and year t has phase
    (t + 1) mod period, so the first work comes at the end of the first period.
    Returns v[phase, s]; a deck starts in phase ``1 % period``.
    """
    a_work, a_rep = ACTIONS.index(action), ACTIONS.index("replace")
    idx = np.arange(N_STATES)
    Ps, cs = [], []
    for k in range(period):
        acts = np.where(idx == N_STATES - 1, a_rep, a_work if k == 0 else 0)
        Ps.append(mdp.P[acts, idx])
        cs.append(mdp.c[idx, acts])
    # v_0 = sum_k gamma^k (P_0 ... P_{k-1}) c_k + gamma^period (P_0 ... P_{period-1}) v_0.
    M, b = np.eye(N_STATES), np.zeros(N_STATES)
    for k in range(period):
        b += M @ cs[k]
        M = mdp.gamma * M @ Ps[k]
    v = np.zeros((period, N_STATES))
    v[0] = np.linalg.solve(np.eye(N_STATES) - M, b)
    for k in range(period - 1, 0, -1):
        v[k] = cs[k] + mdp.gamma * Ps[k] @ v[(k + 1) % period]
    return v


def schedule_start(mdp: DeckMDP, period: int, action: str = "minor") -> np.ndarray:
    """Value of the schedule for a deck at the start of year 0, by rating."""
    return scheduled_value(mdp, period, action)[1 % period]


def scheduled_action(period: int, year: int, s: int, action: str = "minor") -> int:
    if s == N_STATES - 1:
        return ACTIONS.index("replace")
    return ACTIONS.index(action) if (year + 1) % period == 0 else 0


# ---------------------------------------------------------------- counting

def count_transitions(histories, work_flags=None) -> np.ndarray:
    """Count year-to-year rating changes of decks that received no work.

    `histories` is an iterable of rating sequences, one per bridge and one
    entry per consecutive year (None or a value outside 0..9 for a missing or
    'N' rating). A pair is kept only if both ratings are known and the rating
    did not go up; an increase is taken as a sign of work. `work_flags`, if
    given, has the same shape as each history minus one and marks pairs during
    which the reconstruction year (NBI item 106) changed; those pairs are
    dropped too. Ratings 0 to 3 are pooled into rating 3.
    Returns a 7 x 7 matrix over ratings 9..3.
    """
    C = np.zeros((len(COUNT_RATINGS), len(COUNT_RATINGS)), dtype=np.int64)
    index = {r: i for i, r in enumerate(COUNT_RATINGS)}
    for b, h in enumerate(histories):
        for t in range(len(h) - 1):
            x, y = h[t], h[t + 1]
            if x is None or y is None or not (0 <= x <= 9 and 0 <= y <= 9):
                continue
            if y > x:
                continue
            if work_flags is not None and work_flags[b][t]:
                continue
            C[index[max(int(x), 3)], index[max(int(y), 3)]] += 1
    return C


def simulate(mdp: DeckMDP, choose, s0: np.ndarray, uniforms: np.ndarray):
    """Run bridges with common random numbers.

    choose(year, s) -> action index; s0[b] is the initial state of bridge b;
    uniforms[t, b] drives the transition of bridge b in year t by inverse CDF.
    Returns states (T+1, B), actions (T, B), costs (T, B).
    """
    T, B = uniforms.shape
    states = np.zeros((T + 1, B), int)
    actions = np.zeros((T, B), int)
    costs = np.zeros((T, B))
    states[0] = s0
    cache = {}
    for t in range(T):
        s = states[t]
        a = np.array([cache.setdefault((t, x), choose(t, x)) for x in s])
        actions[t] = a
        costs[t] = mdp.c[s, a]
        cdf = np.cumsum(mdp.P[a, s], axis=1)
        nxt = (uniforms[t][:, None] >= cdf).sum(axis=1)
        states[t + 1] = np.minimum(nxt, N_STATES - 1)
    return states, actions, costs
