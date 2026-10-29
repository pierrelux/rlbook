"""Sailing to a windward mark in a shifting wind, as a stochastic shortest-path MDP.

The model is Vanderbei's sailing problem (https://vanderbei.princeton.edu/sail/sail.html),
also the "sailing" benchmark of Peret and Garcia (2004) and Kocsis and Szepesvari (2006).
A boat sails on an n x n grid of waypoints. Each leg goes to one of the 8
neighbouring waypoints. The time of a leg depends on the angle between the
heading and the wind (the point of sail); sailing straight into the wind is
impossible. Changing between port and starboard tack wastes a fixed delay, so
the state remembers the current tack. After every leg the wind keeps its
direction or shifts by 45 degrees according to a Markov chain.

Leg times in minutes for a straight (N, E, S, W) leg, from Vanderbei's page:
close-hauled (45 degrees off the wind) 4, beam reach (90) 3, broad reach (135) 2;
the tack delay is 3. The dead-downwind time 1 is the default of the
mdp-engine implementation of Bonet and Geffner (https://github.com/bonetblai/mdp-engine,
file sailing/sailing.h). A diagonal leg is sqrt(2) longer and takes sqrt(2)
times as long, which reproduces the cost range 1 to 8.6 reported by Kocsis and
Szepesvari. The wind chains are documented in `wind_matrix`.

Arrays are indexed [tack, wind, y, x] with x pointing east and y north.
Directions are numbered clockwise from north: N, NE, E, SE, S, SW, W, NW.
A wind direction is the direction the wind blows FROM.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIRS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
DX = np.array([0, 1, 1, 1, 0, -1, -1, -1])
DY = np.array([1, 1, 0, -1, -1, -1, 0, 1])
LENGTH = np.where((DX != 0) & (DY != 0), np.sqrt(2.0), 1.0)
PORT, STARBOARD = 0, 1
TACKS = ("port", "starboard")

# Minutes per straight leg, by the angle between wind and heading in 45-degree
# steps: 0 = into the wind (not allowed), 1 = close-hauled, 2 = beam reach,
# 3 = broad reach, 4 = dead downwind.
POINT_OF_SAIL = ("into the wind", "close-hauled", "beam reach", "broad reach", "dead downwind")
LEG_TIME = np.array([np.inf, 4.0, 3.0, 2.0, 1.0])
TACK_DELAY = 3.0

# Wind transition matrix of mdp-engine (sailing/sailing.h), rows = current
# wind, columns = next wind. In that code the index is the direction the wind
# blows towards and the chain drifts back to index 0. Read here as the
# direction the wind blows from, which rotates the whole chain by 180 degrees,
# the preferred wind is from the north.
MDP_ENGINE_WIND = np.array([
    [0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
    [0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.4, 0.3, 0.3, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.4, 0.2, 0.4, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.4, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.4],
    [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3],
])

SCENARIOS = ("steady", "oscillating", "shifty")
SCENARIO_LABELS = {
    "steady": "Steady wind from the north",
    "oscillating": "Oscillating breeze (teaching values)",
    "shifty": "Shifty breeze (mdp-engine matrix)",
}


def wind_matrix(kind: str) -> np.ndarray:
    """Wind transition matrix P[w, w'] for one leg.

    steady: the wind never changes.
    oscillating (teaching values): from the mean direction (north) the wind
        shifts 45 degrees left or right with probability 0.1 each; away from
        north it moves back towards north with probability 0.2, further away
        with probability 0.1, and otherwise persists.
    shifty: the mdp-engine default matrix (see MDP_ENGINE_WIND).
    """
    if kind == "steady":
        return np.eye(8)
    if kind == "shifty":
        return MDP_ENGINE_WIND.copy()
    if kind != "oscillating":
        raise ValueError(kind)
    P = np.zeros((8, 8))
    for w in range(8):
        off = (w + 4) % 8 - 4             # signed offset from north in -4..3
        if off == 0:
            P[w, 1], P[w, 7], P[w, 0] = 0.1, 0.1, 0.8
        elif off == -4:                    # from the south: both ways lead back
            P[w, 3], P[w, 5], P[w, 4] = 0.15, 0.15, 0.7
        else:
            toward = (w - np.sign(off)) % 8
            away = (w + np.sign(off)) % 8
            P[w, toward], P[w, away], P[w, w] = 0.2, 0.1, 0.7
    return P


def relative_angle(wind: int, heading: int) -> int:
    """Angle between the wind and the heading in 45-degree steps, 0..4."""
    d = (wind - heading) % 8
    return min(d, 8 - d)


def new_tack(wind: int, heading: int, tack: int) -> int:
    """Wind over the starboard (right) side means starboard tack; dead downwind keeps the tack."""
    d = (wind - heading) % 8
    if d in (1, 2, 3):
        return STARBOARD
    if d in (5, 6, 7):
        return PORT
    return tack


@dataclass(frozen=True)
class Lake:
    n: int = 21
    start: tuple[int, int] = (10, 2)
    mark: tuple[int, int] = (10, 18)


@dataclass
class Model:
    lake: Lake
    P: np.ndarray                 # wind transition matrix (8, 8)
    cost: np.ndarray              # cost[h, tack, wind, y, x], inf where not allowed
    next_tack: np.ndarray         # next_tack[h, tack, wind]


def build(lake: Lake, P: np.ndarray) -> Model:
    n = lake.n
    cost = np.full((8, 2, 8, n, n), np.inf)
    next_tack = np.zeros((8, 2, 8), dtype=int)
    ys, xs = np.mgrid[0:n, 0:n]
    for h in range(8):
        inside = (xs + DX[h] >= 0) & (xs + DX[h] < n) & (ys + DY[h] >= 0) & (ys + DY[h] < n)
        for w in range(8):
            leg = LEG_TIME[relative_angle(w, h)] * LENGTH[h]
            for t in range(2):
                nt = new_tack(w, h, t)
                next_tack[h, t, w] = nt
                c = leg + (TACK_DELAY if nt != t else 0.0)
                cost[h, t, w][inside] = c
    mx, my = lake.mark
    cost[:, :, :, my, mx] = 0.0          # the mark is absorbing and free
    return Model(lake, np.asarray(P, float), cost, next_tack)


def _next_values(model: Model, v: np.ndarray) -> np.ndarray:
    """nv[h, t, w, y, x] = sum_w' P[w, w'] v[t'(h,t,w), w', y+dy, x+dx] (0 where off the lake)."""
    n = model.lake.n
    ev = np.einsum("wu,tuyx->twyx", model.P, v)        # expectation over the next wind
    pad = np.zeros((2, 8, n + 2, n + 2))
    pad[:, :, 1:-1, 1:-1] = ev
    nv = np.empty((8, 2, 8, n, n))
    for h in range(8):
        shifted = pad[:, :, 1 + DY[h]:1 + DY[h] + n, 1 + DX[h]:1 + DX[h] + n]
        for t in range(2):
            nv[h, t] = shifted[model.next_tack[h, t], np.arange(8)]
    return nv


def q_values(model: Model, v: np.ndarray) -> np.ndarray:
    q = model.cost + _next_values(model, v)
    mx, my = model.lake.mark
    q[:, :, :, my, mx] = 0.0
    return q


def bellman(model: Model, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One application of the Bellman operator (min over headings) and a greedy heading."""
    q = q_values(model, v)
    return q.min(axis=0), q.argmin(axis=0)


@dataclass
class VIResult:
    v: np.ndarray
    policy: np.ndarray
    residuals: list[float]
    iterates: list[np.ndarray]    # v_n for the recorded iterations
    recorded: list[int]


def value_iteration(model: Model, tol: float = 1e-6, max_iter: int = 20000,
                    record: set[int] | None = None) -> VIResult:
    """Undiscounted value iteration v_{n+1} = T v_n from v_0 = 0, stopped when ||v_{n+1} - v_n|| < tol."""
    v = np.zeros((2, 8, model.lake.n, model.lake.n))
    residuals, iterates, recorded = [], [], []
    if record is not None and 0 in record:
        iterates.append(v.copy()); recorded.append(0)
    for k in range(1, max_iter + 1):
        nv, _ = bellman(model, v)
        r = float(np.max(np.abs(nv - v)))
        residuals.append(r)
        v = nv
        if record is not None and k in record:
            iterates.append(v.copy()); recorded.append(k)
        if r < tol:
            break
    else:
        raise RuntimeError("value iteration did not converge")
    _, pi = bellman(model, v)
    return VIResult(v, pi, residuals, iterates, recorded)


def evaluate(model: Model, policy: np.ndarray, tol: float = 1e-10, max_iter: int = 200000) -> np.ndarray:
    """Expected time to the mark under a stationary policy (iterative policy evaluation).

    Raises if the values keep growing, the sign of an improper policy.
    """
    n = model.lake.n
    c = np.take_along_axis(model.cost, policy[None], axis=0)[0]
    if not np.all(np.isfinite(c)):
        raise ValueError("policy uses a forbidden heading")
    v = np.zeros((2, 8, n, n))
    for _ in range(max_iter):
        nv = c + np.take_along_axis(_next_values(model, v), policy[None], axis=0)[0]
        mx, my = model.lake.mark
        nv[:, :, my, mx] = 0.0
        if np.max(np.abs(nv - v)) < tol:
            return nv
        if np.max(nv) > 1e6:
            break
        v = nv
    raise RuntimeError("policy evaluation did not converge (improper policy?)")


def policy_iteration(model: Model, policy0: np.ndarray, max_iter: int = 100):
    """Policy iteration from a proper policy; keeps the current heading on ties."""
    pi = policy0.copy()
    values, policies = [], [pi.copy()]
    for _ in range(max_iter):
        v = evaluate(model, pi)
        values.append(v)
        q = q_values(model, v)
        best = q.min(axis=0)
        keep = np.take_along_axis(q, pi[None], axis=0)[0] <= best + 1e-9
        new = np.where(keep, pi, q.argmin(axis=0))
        if np.array_equal(new, pi):
            return pi, v, values, policies
        pi = new
        policies.append(pi.copy())
    raise RuntimeError("policy iteration did not stop")


def naive_policy(model: Model) -> np.ndarray:
    """A layline sailor who ignores the wind forecast.

    The sailor wants to be on starboard tack until reaching the port-tack
    layline of the mean (north) wind, the diagonal through the mark on the
    left, then on port tack to the mark, and symmetrically on the right.
    Between the two laylines the sailor keeps the current tack. Each leg takes,
    among the allowed headings that bring the boat closer to the mark, the one
    on the wanted tack (or dead downwind) that ends closest to the mark, with
    ties broken by the shorter leg. If no heading on the wanted tack gets
    closer, the sailor takes the allowed heading that ends closest to the mark.
    """
    lake = model.lake
    n, (mx, my) = lake.n, lake.mark
    pi = np.zeros((2, 8, n, n), dtype=int)
    for t in range(2):
        for w in range(8):
            for y in range(n):
                for x in range(n):
                    if (x, y) == lake.mark:
                        continue
                    dy = my - y
                    if mx - x >= dy:
                        want = PORT
                    elif x - mx >= dy:
                        want = STARBOARD
                    else:
                        want = t
                    here = np.hypot(mx - x, my - y)
                    best, fallback = None, None
                    for h in range(8):
                        c = model.cost[h, t, w, y, x]
                        if not np.isfinite(c):
                            continue
                        dist = np.hypot(mx - x - DX[h], my - y - DY[h])
                        key = (round(dist, 9), c)
                        if fallback is None or key < fallback[0]:
                            fallback = (key, h)
                        if dist >= here - 1e-9:
                            continue
                        on_tack = (w - h) % 8 == 4 or new_tack(w, h, t) == want
                        if on_tack and (best is None or key < best[0]):
                            best = (key, h)
                    pi[t, w, y, x] = (best or fallback)[1]
    return pi


def certainty_equivalent_policy(lake: Lake) -> np.ndarray:
    """Greedy policy for a steady wind: plan as if the wind seen now never changes."""
    steady = build(lake, np.eye(8))
    return value_iteration(steady, tol=1e-9).policy


def start_value(v: np.ndarray, lake: Lake, wind: int = 0) -> tuple[float, int]:
    """Expected time from the start in the given wind, choosing the better starting tack."""
    x, y = lake.start
    t = int(np.argmin(v[:, wind, y, x]))
    return float(v[t, wind, y, x]), t


def simulate(model: Model, policy: np.ndarray, uniforms: np.ndarray, wind0: int, tack0: int,
             max_legs: int = 2000):
    """Sail one race. uniforms[k] draws the wind change after leg k (common random numbers).

    Returns the list of (x, y, wind, tack, elapsed minutes) before each leg and at the finish.
    """
    lake = model.lake
    (x, y), w, t, clock = lake.start, wind0, tack0, 0.0
    cdf = np.cumsum(model.P, axis=1)
    path = [(x, y, w, t, 0.0)]
    for k in range(max_legs):
        if (x, y) == lake.mark:
            return path
        h = int(policy[t, w, y, x])
        clock += float(model.cost[h, t, w, y, x])
        t = int(model.next_tack[h, t, w])
        x, y = x + int(DX[h]), y + int(DY[h])
        w = int(min(np.searchsorted(cdf[w], uniforms[k], side="right"), 7))
        path.append((x, y, w, t, clock))
    raise RuntimeError("race did not finish")
