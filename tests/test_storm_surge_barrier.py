"""Checks for the storm-surge-barrier stopping MDP and its artifacts."""
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from storm_surge_barrier import (ATTEMPT_HOURS, CLOSE, CLOSED, GRID, HORIZON,  # noqa: E402
                                 N_B, N_F, OPEN, REOPEN_HOURS, SHIP_COST, SIGMA_STEP,
                                 STATUS, THRESHOLD, WAIT, WARN_HOURS,
                                 backward_induction, build_model, evaluate,
                                 exceed_probability, forecast_kernel,
                                 outcome_rates, p_not_closed, prior, simulate,
                                 threshold_rule)

DATA = ROOT / "interactive" / "storm-surge-barrier-data.json"
NOMINAL = WARN_HOURS + ATTEMPT_HOURS


def test_kernels_are_stochastic_and_forecast_is_a_martingale():
    m = build_model()
    assert np.allclose(m.K.sum(axis=1), 1) and (m.K >= 0).all()
    assert np.allclose(m.P.sum(axis=2), 1) and (m.P >= 0).all()
    interior = (GRID > GRID[0] + 6 * SIGMA_STEP) & (GRID < GRID[-1] - 6 * SIGMA_STEP)
    assert np.allclose((m.K @ GRID)[interior], GRID[interior], atol=1e-6)
    var = m.K @ GRID ** 2 - (m.K @ GRID) ** 2
    assert np.allclose(var[interior], SIGMA_STEP ** 2 + 1 / 12, atol=1e-3)   # Gaussian plus bin width


def test_closing_delay_matches_status_chain():
    m = build_model()
    for h in range(0, 14):
        dist = np.zeros(N_B); dist[OPEN] = 1.0
        dist = dist @ m.P[CLOSE]
        for _ in range(h - 1):
            dist = dist @ m.P[WAIT]
        expected = 1.0 if h == 0 else p_not_closed(h)
        assert np.isclose(1 - dist[CLOSED], expected)
    assert np.isclose(p_not_closed(8), 0.009)    # about 1 in 100, the Port of Rotterdam figure


def test_backward_induction_matches_brute_force_over_stopping_times():
    """With the forecast held fixed (no noise), the optimal order time is found by enumeration."""
    m = build_model()
    det = replace(m, K=np.eye(N_F))
    sol = backward_induction(det)
    for f in (270, 290, 301, 330):
        i = int(np.nonzero(GRID == f)[0][0])
        best = (THRESHOLD < f) * 1000.0            # never close
        for h in range(1, HORIZON + 1):
            cost = SHIP_COST * (h + REOPEN_HOURS) + p_not_closed(h) * 1000.0 * (f > THRESHOLD)
            best = min(best, cost)
        assert np.isclose(sol.values[HORIZON, i, OPEN], best)


def test_optimal_policy_beats_every_threshold_rule_and_matches_its_evaluation():
    m, start = build_model(), prior()
    sol = backward_induction(m)
    v = evaluate(m, sol.close)
    assert np.allclose(v, sol.values)
    best = start @ sol.values[HORIZON][:, OPEN]
    for margin in (-10, 0, 10, 18, 25):
        for lead in (6, 9, 12, 24):
            assert best <= outcome_rates(m, threshold_rule(margin, lead), start)["cost"] + 1e-9
    assert not sol.close[:NOMINAL].any()


def test_exceed_probability_matches_normal_tail_in_the_interior():
    from math import erf, sqrt
    p = exceed_probability(9)
    for f in (285, 300, 315):
        i = int(np.nonzero(GRID == f)[0][0])
        sd = sqrt(9 * (SIGMA_STEP ** 2 + 1 / 12))
        z = (THRESHOLD + 0.5 - f) / sd
        assert abs(p[i] - 0.5 * (1 - erf(z / sqrt(2)))) < 0.01


def test_replay_data_matches_solver():
    data = json.loads(DATA.read_text())
    assert DATA.stat().st_size < 1_000_000
    assert data["status_names"] == list(STATUS)
    K = forecast_kernel()
    for C, entry in data["costs"].items():
        m = build_model(float(C))
        sol = backward_induction(m)
        for h, f in enumerate(entry["dp_threshold"]):
            idx = np.nonzero(sol.close[h])[0]
            assert (f is None) == (idx.size == 0)
            if f is not None:
                assert GRID[idx[0]] == f
        assert np.isclose(entry["metrics"]["dp"]["cost"], prior() @ sol.values[HORIZON][:, OPEN], atol=1e-5)
    for storm in data["storms"]:
        fc = storm["forecast"]
        path = np.array([int(np.nonzero(GRID == fc[HORIZON - h])[0][0]) for h in range(HORIZON + 1)])
        for i, p in enumerate(storm["p_exceed"]):
            assert abs(p - exceed_probability(HORIZON - i, K)[path[HORIZON - i]]) < 1e-4
        for C, runs in storm["runs"].items():
            dp = runs["dp"]
            # The recorded order time is the first hour the state enters the closing region.
            th = data["costs"][C]["dp_threshold"]
            first = next((h for h in range(HORIZON, 0, -1) if th[h] is not None and fc[HORIZON - h] >= th[h]), None)
            assert dp["order_lead"] == first
            for run in runs.values():
                st = run["status"]
                cost = sum(SHIP_COST for i in range(HORIZON) if st[i + 1] != OPEN)
                cost += REOPEN_HOURS * SHIP_COST * (run["order_lead"] is not None) + float(C) * run["flooded"]
                assert np.isclose(cost, run["cost"])
                assert run["flooded"] == (st[-1] != CLOSED and fc[-1] > THRESHOLD)


def test_simulation_is_consistent_with_policy():
    m = build_model(); sol = backward_induction(m)
    path = np.full(HORIZON + 1, int(np.nonzero(GRID == 320)[0][0]))
    run = simulate(sol.close, path, np.ones(HORIZON + 1))          # u = 1: no fault
    assert run["order_lead"] is not None and run["status"][-1] == CLOSED and not run["flooded"]


def test_files_exist():
    for suffix in ("svg", "pdf", "png"):
        assert (ROOT / "_static" / "storm_surge_barrier" / f"stopping_region.{suffix}").exists()
    for name in ("README.md", "metrics.json", "results.md"):
        assert (ROOT / "artifacts" / "storm_surge_barrier" / name).exists()
    assert (ROOT / "interactive" / "storm-surge-barrier.html").exists()
