"""Checks for the sailing-in-wind-shifts MDP, its solver, and its artifacts."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from sailing_wind_shifts import (DX, DY, LEG_TIME, SCENARIOS, TACK_DELAY,  # noqa: E402
                                 Lake, bellman, build, evaluate, naive_policy,
                                 new_tack, policy_iteration, relative_angle,
                                 start_value, value_iteration, wind_matrix)

DATA = ROOT / "interactive" / "sailing-wind-shifts-data.json"
METRICS = ROOT / "artifacts" / "sailing_wind_shifts" / "metrics.json"
SMALL = Lake(n=7, start=(3, 0), mark=(3, 5))


def test_wind_chains_are_stochastic_and_shift_at_most_45_degrees():
    for kind in SCENARIOS:
        P = wind_matrix(kind)
        assert np.allclose(P.sum(axis=1), 1) and np.all(P >= 0)
        for w in range(8):
            for u in np.nonzero(P[w])[0]:
                assert relative_angle(w, u) <= 1


def test_points_of_sail_and_tacks():
    # Wind from the north: north is forbidden, NE is close-hauled on port, NW on starboard.
    assert relative_angle(0, 0) == 0 and relative_angle(0, 1) == 1 and relative_angle(0, 4) == 4
    assert new_tack(0, 1, 1) == 0 and new_tack(0, 7, 0) == 1
    assert new_tack(0, 4, 0) == 0 and new_tack(0, 4, 1) == 1      # dead downwind keeps the tack
    m = build(SMALL, wind_matrix("oscillating"))
    x, y = SMALL.start
    assert np.isinf(m.cost[0, 0, 0, y, x])                        # straight into the wind
    assert np.isclose(m.cost[1, 0, 0, y, x], LEG_TIME[1] * np.sqrt(2))
    assert np.isclose(m.cost[7, 0, 0, y, x], LEG_TIME[1] * np.sqrt(2) + TACK_DELAY)
    assert np.isinf(m.cost[4, 0, 0, y, x])                        # off the lake


def brute_force_values(lake, P, sweeps=4000):
    """Independent Gauss-Seidel value iteration with explicit loops over states."""
    m = build(lake, P)
    n = lake.n
    v = np.zeros((2, 8, n, n))
    for _ in range(sweeps):
        delta = 0.0
        for t in range(2):
            for w in range(8):
                for y in range(n):
                    for x in range(n):
                        if (x, y) == lake.mark:
                            continue
                        best = np.inf
                        for h in range(8):
                            c = m.cost[h, t, w, y, x]
                            if not np.isfinite(c):
                                continue
                            nt = m.next_tack[h, t, w]
                            nx, ny = x + DX[h], y + DY[h]
                            best = min(best, c + sum(P[w, u] * v[nt, u, ny, nx] for u in range(8) if P[w, u] > 0))
                        delta = max(delta, abs(best - v[t, w, y, x]))
                        v[t, w, y, x] = best
        if delta < 1e-10:
            return v
    raise AssertionError("brute force did not converge")


def test_value_iteration_matches_brute_force_and_policy_iteration():
    P = wind_matrix("oscillating")
    m = build(SMALL, P)
    vi = value_iteration(m, tol=1e-10)
    assert np.allclose(vi.v, brute_force_values(SMALL, P), atol=1e-8)
    assert np.allclose(bellman(m, vi.v)[0], vi.v, atol=1e-9)
    _, v_pi, _, _ = policy_iteration(m, naive_policy(m))
    assert np.allclose(v_pi, vi.v, atol=1e-8)
    assert np.all(evaluate(m, naive_policy(m)) >= vi.v - 1e-9)


def test_steady_wind_is_one_tack_beat():
    lake = Lake()
    m = build(lake, wind_matrix("steady"))
    v = value_iteration(m, tol=1e-9).v
    rows = lake.mark[1] - lake.start[1]
    assert np.isclose(start_value(v, lake)[0], rows * 4 * np.sqrt(2) + 3)


def test_replay_data_matches_solver():
    data = json.loads(DATA.read_text())
    metrics = json.loads(METRICS.read_text())
    lake = Lake()
    assert data["n"] == lake.n and data["mark"] == list(lake.mark)
    for kind in SCENARIOS:
        s = data["scenarios"][kind]
        m = build(lake, wind_matrix(kind))
        vi = value_iteration(m, tol=1e-9)
        assert np.allclose(np.array(s["v"]).reshape(vi.v.shape), vi.v, atol=0.051)
        assert np.array_equal(np.array(s["policy"]).reshape(vi.v.shape), vi.policy)
        assert len(s["residuals"]) == metrics["scenarios"][kind]["value_iteration_sweeps"] == len(vi.residuals)
        assert np.isclose(s["stats"]["optimal_minutes"], start_value(vi.v, lake)[0], atol=1e-3)
        # Every recorded race follows the recorded policies and adds up its leg costs.
        for race in s["races"]:
            for key, pol in (("optimal", vi.policy), ("naive", np.array(s["naive"]).reshape(vi.v.shape))):
                path = race[key]
                assert tuple(path[-1][:2]) == lake.mark
                clock = 0.0
                for a, b in zip(path, path[1:]):
                    x, y, w, t, _ = a
                    h = int(pol[t, w, y, x])
                    assert (b[0] - x, b[1] - y) == (DX[h], DY[h]) and b[3] == m.next_tack[h, t, w]
                    assert m.P[w, b[2]] > 0
                    clock += m.cost[h, t, w, y, x]
                    assert abs(b[4] - clock) < 1e-2
    feat = data["scenarios"][data["featured"]]
    assert len(feat["vi_frames"]) == len(feat["residuals"]) + 1
    assert DATA.stat().st_size < 1_000_000


def test_files_exist():
    for suffix in ("svg", "pdf", "png"):
        assert (ROOT / "_static" / "sailing_wind_shifts" / f"sailing.{suffix}").exists()
    for name in ("README.md", "results.md", "metrics.json"):
        assert (ROOT / "artifacts" / "sailing_wind_shifts" / name).exists()
    assert (ROOT / "interactive" / "sailing-wind-shifts.html").exists()
