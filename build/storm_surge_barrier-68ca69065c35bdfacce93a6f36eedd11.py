"""Closing a storm-surge barrier as a finite-horizon optimal stopping MDP.

The setting is the Maeslant barrier (Maeslantkering) on the Nieuwe Waterweg,
which protects Rotterdam. Its closing criterion is a forecast water level of
300 cm above NAP at Rotterdam, and its gates sink to the bottom within about
two hours (Rijkswaterstaat). The Port of Rotterdam states a failure
probability of 1 in 100 per closure.

Everything else is a teaching model. In the H hours before a forecast peak,
the operator sees each hour a forecast f of the peak level (cm above NAP).
The forecast is a martingale: it moves by a Gaussian increment with standard
deviation SIGMA_STEP cm per hour, so the remaining forecast error at lead h is
SIGMA_STEP * sqrt(h) and the forecast at lead 0 is the realized peak. The
operator may order the closure at any hour. The order is irreversible.
Shipping is warned and halted, the gates start to close WARN_HOURS later and
are closed ATTEMPT_HOURS after that, unless a fault occurs (probability
Q_FAIL). A fault is cleared, and the barrier closed, with probability
R_REPAIR in each following hour. A barrier that is not closed
at the peak lets the city flood when the peak exceeds THRESHOLD.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

THRESHOLD = 300            # cm above NAP at Rotterdam (Maeslant closing criterion)
F_MIN, F_MAX = 150, 420    # forecast grid, cm above NAP, 1 cm bins
GRID = np.arange(F_MIN, F_MAX + 1)
N_F = GRID.size
HORIZON = 24               # decision epochs h = 24, ..., 1 hours before the peak

# Teaching values.
SIGMA_STEP = 4.0           # cm per sqrt(hour): forecast error sd at lead h is 4 sqrt(h)
WARN_HOURS = 4             # hours from the order to the start of the first closing attempt
ATTEMPT_HOURS = 2          # hours for the gates to float out and sink
Q_FAIL = 0.1               # probability that a fault delays the closure
R_REPAIR = 0.7             # probability per hour that a fault is cleared and the gates close
REOPEN_HOURS = 6           # hours shipping stays halted after the peak once a closure is ordered
SHIP_COST = 1.0            # cost unit: one hour of halted shipping
FLOOD_COST = 1000.0        # flood damage, in hours of halted shipping
PRIOR_MEAN, PRIOR_SD = 280.0, 20.0   # forecast of the peak at lead HORIZON, across storms

# Barrier status: open, warned for k hours (k = 1..WARN_HOURS-1),
# closing in hour j (j = 1..ATTEMPT_HOURS), delayed by a fault, closed.
STATUS = (["open"] + [f"warn{k}" for k in range(1, WARN_HOURS)]
          + [f"closing{j}" for j in range(1, ATTEMPT_HOURS + 1)] + ["delayed", "closed"])
N_B = len(STATUS)
OPEN, DELAYED, CLOSED = 0, N_B - 2, N_B - 1
LAST_CLOSING = N_B - 3              # index of the final closing hour
WAIT, CLOSE = 0, 1


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def forecast_kernel(sigma: float = SIGMA_STEP) -> np.ndarray:
    """p(f' | f): Gaussian increment integrated over 1 cm bins, clipped at the grid edges."""
    offsets = np.arange(-N_F + 1, N_F)
    mass = _norm_cdf((offsets + 0.5) / sigma) - _norm_cdf((offsets - 0.5) / sigma)
    K = np.zeros((N_F, N_F))
    for i in range(N_F):
        row = mass[N_F - 1 - i: 2 * N_F - 1 - i].copy()
        row[0] += _norm_cdf(np.array([(-0.5 - i) / sigma]))[0]               # mass below the grid
        row[-1] += 1.0 - _norm_cdf(np.array([(N_F - 1 - i + 0.5) / sigma]))[0]  # mass above the grid
        K[i] = row
    return K


def status_kernel(q: float = Q_FAIL, r: float = R_REPAIR) -> np.ndarray:
    """p(b' | b, a), shape (action, b, b'). Only the open barrier has a choice."""
    P = np.zeros((2, N_B, N_B))
    for a in (WAIT, CLOSE):
        P[a, OPEN, OPEN if a == WAIT else 1] = 1.0
        for b in range(1, LAST_CLOSING):
            P[a, b, b + 1] = 1.0
        P[a, LAST_CLOSING, CLOSED], P[a, LAST_CLOSING, DELAYED] = 1.0 - q, q
        P[a, DELAYED, CLOSED], P[a, DELAYED, DELAYED] = r, 1.0 - r
        P[a, CLOSED, CLOSED] = 1.0
    return P


def p_not_closed(order_lead: int, q: float = Q_FAIL, r: float = R_REPAIR) -> float:
    """Probability that a closure ordered at lead order_lead is not complete at the peak."""
    nominal = WARN_HOURS + ATTEMPT_HOURS
    return 1.0 if order_lead < nominal else q * (1.0 - r) ** (order_lead - nominal)


def stage_cost(ship_cost: float = SHIP_COST) -> np.ndarray:
    """c(b, a): shipping is halted during every hour after the order."""
    c = np.full((2, N_B), ship_cost)
    c[WAIT, OPEN] = 0.0
    return c


def terminal_cost(flood_cost: float = FLOOD_COST, ship_cost: float = SHIP_COST) -> np.ndarray:
    """c_0(f, b) at the peak: flood if not closed and f > THRESHOLD; reopening delay if ordered."""
    v = np.zeros((N_F, N_B))
    v[:, 1:] += ship_cost * REOPEN_HOURS
    v[:, :CLOSED] += flood_cost * (GRID > THRESHOLD)[:, None]
    return v


def prior(mean: float = PRIOR_MEAN, sd: float = PRIOR_SD) -> np.ndarray:
    p = _norm_cdf((GRID + 0.5 - mean) / sd) - _norm_cdf((GRID - 0.5 - mean) / sd)
    return p / p.sum()


@dataclass(frozen=True)
class Model:
    K: np.ndarray            # (f, f')
    P: np.ndarray            # (a, b, b')
    c: np.ndarray            # (a, b)
    c0: np.ndarray           # (f, b)


def build_model(flood_cost: float = FLOOD_COST, q: float = Q_FAIL) -> Model:
    return Model(forecast_kernel(), status_kernel(q), stage_cost(), terminal_cost(flood_cost))


def _q_values(m: Model, v_next: np.ndarray) -> np.ndarray:
    """q[a, f, b] = c(b, a) + sum_{f', b'} p(f'|f) p(b'|b, a) v(f', b')."""
    ev = m.K @ v_next                                   # (f, b'): forecast moves independently
    return m.c[:, None, :] + np.einsum("fj,abj->afb", ev, m.P)


@dataclass(frozen=True)
class Solution:
    values: np.ndarray       # values[h, f, b]: optimal expected cost-to-go at lead h
    close: np.ndarray        # close[h, f]: True if an open barrier should be ordered closed at lead h


def backward_induction(m: Model) -> Solution:
    v = m.c0.copy()
    values, close = [v], [np.zeros(N_F, bool)]
    for _h in range(1, HORIZON + 1):
        q = _q_values(m, v)
        act = q[CLOSE] < q[WAIT] - 1e-12              # ties go to waiting
        v = np.where(act, q[CLOSE], q[WAIT])
        values.append(v)
        close.append(act[:, OPEN])
    return Solution(np.array(values), np.array(close))


def evaluate(m: Model, close: np.ndarray, stage: np.ndarray | None = None,
             terminal: np.ndarray | None = None) -> np.ndarray:
    """Backward policy evaluation of a Markov policy close[h, f]; returns values[h, f, b]."""
    c = m.c if stage is None else stage
    v = m.c0.copy() if terminal is None else terminal.copy()
    out = [v]
    for h in range(1, HORIZON + 1):
        ev = m.K @ v
        qa = c[:, None, :] + np.einsum("fj,abj->afb", ev, m.P)
        act = np.zeros((N_F, N_B), bool)
        act[:, OPEN] = close[h]
        v = np.where(act, qa[CLOSE], qa[WAIT])
        out.append(v)
    return np.array(out)


def threshold_rule(margin: float = 0.0, latest_lead: int = HORIZON) -> np.ndarray:
    """Order the closure once the forecast reaches THRESHOLD - margin, at a lead between
    WARN_HOURS + ATTEMPT_HOURS (the last lead at which the closure can still finish) and latest_lead."""
    close = np.zeros((HORIZON + 1, N_F), bool)
    for h in range(WARN_HOURS + ATTEMPT_HOURS, min(latest_lead, HORIZON) + 1):
        close[h] = GRID >= THRESHOLD - margin
    return close


def outcome_rates(m: Model, close: np.ndarray, start: np.ndarray) -> dict:
    """Exact expected cost, P(false closure), P(miss), P(exceed), mean hours halted under a policy."""
    zero_stage = np.zeros_like(m.c)
    exceed = (GRID > THRESHOLD)[:, None]
    ordered = np.zeros((N_F, N_B)); ordered[:, 1:] = 1.0
    not_closed = np.ones((N_F, N_B)); not_closed[:, CLOSED] = 0.0
    total = evaluate(m, close)[HORIZON][:, OPEN]
    false = evaluate(m, close, zero_stage, ordered * ~exceed)[HORIZON][:, OPEN]
    miss = evaluate(m, close, zero_stage, not_closed * exceed)[HORIZON][:, OPEN]
    hours = evaluate(m, close, stage_cost(1.0), terminal_cost(0.0, 1.0))[HORIZON][:, OPEN]
    p_exceed = float(start @ (GRID > THRESHOLD))
    return {"cost": float(start @ total), "p_false": float(start @ false), "p_miss": float(start @ miss),
            "p_exceed": p_exceed, "hours_halted": float(start @ hours),
            "p_close": float(start @ evaluate(m, close, zero_stage, ordered)[HORIZON][:, OPEN])}


def exceed_probability(h: int, K: np.ndarray | None = None) -> np.ndarray:
    """P(peak > THRESHOLD | forecast f at lead h), from the discretized kernel."""
    K = forecast_kernel() if K is None else K
    p = (GRID > THRESHOLD).astype(float)
    for _ in range(h):
        p = K @ p
    return p


def simulate(close: np.ndarray, f_path: np.ndarray, u: np.ndarray, q: float = Q_FAIL,
             r: float = R_REPAIR, flood_cost: float = FLOOD_COST) -> dict:
    """Replay one storm: forecast indices f_path[h] for h = HORIZON..0, uniforms u[h] for the barrier."""
    status, order_lead, cost = OPEN, None, 0.0
    statuses = []
    for h in range(HORIZON, 0, -1):
        statuses.append(status)
        a = CLOSE if (status == OPEN and close[h, f_path[h]]) else WAIT
        cost += SHIP_COST * (status != OPEN or a == CLOSE)
        if status == OPEN:
            if a == CLOSE:
                status, order_lead = 1, h
        elif status == LAST_CLOSING:
            status = DELAYED if u[h] < q else CLOSED
        elif status == DELAYED:
            status = CLOSED if u[h] < r else DELAYED
        elif status != CLOSED:
            status += 1
    statuses.append(status)
    peak = int(GRID[f_path[0]])
    flooded = status != CLOSED and peak > THRESHOLD
    cost += SHIP_COST * REOPEN_HOURS * (status != OPEN) + flood_cost * flooded
    return {"status": statuses, "order_lead": order_lead,
            "flooded": bool(flooded), "peak": peak, "cost": cost}


def sample_forecast_path(rng: np.random.Generator, start: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Forecast indices at leads HORIZON, ..., 0, returned indexed by lead."""
    path = np.empty(HORIZON + 1, int)
    path[HORIZON] = rng.choice(N_F, p=start)
    cum = np.cumsum(K, axis=1)
    for h in range(HORIZON, 0, -1):
        path[h - 1] = min(int(np.searchsorted(cum[path[h]], rng.random())), N_F - 1)
    return path
