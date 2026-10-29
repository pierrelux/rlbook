"""Checks for the bridge-deck maintenance MDP and its artifacts."""
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
import bridge_deck_maintenance as bdm  # noqa: E402
from build_bridge_deck_maintenance_artifacts import (  # noqa: E402
    BASE_MULTIPLIER, mdp_with_risk, read_counts,
)

ART = ROOT / "artifacts" / "bridge_deck_maintenance"
DATA = ROOT / "interactive" / "bridge-deck-maintenance-data.json"


def base_mdp():
    C, _ = read_counts()
    return bdm.build_mdp(C), C


def test_kernels_are_stochastic_and_never_improve_without_work():
    mdp, C = base_mdp()
    assert C.shape == (7, 7) and C.sum() > 2_000_000
    assert np.all(np.tril(C, -1) == 0)
    assert np.allclose(mdp.P.sum(axis=2), 1) and np.all(mdp.P >= 0)
    P0 = mdp.P[0]
    assert np.all(np.tril(P0, -1) == 0)
    # Do-nothing rows are the maximum-likelihood estimates from the merged counts.
    merged = bdm.collapse_counts(C)
    assert np.allclose(P0[:-1], merged[:-1] / merged[:-1].sum(1, keepdims=True))


def test_count_transitions_drops_work_and_pools_low_ratings():
    C = bdm.count_transitions([[9, 9, 8, 9, 7, None, 2, 1]], [[False, False, False, True, False, False, False]])
    idx = {r: i for i, r in enumerate(bdm.COUNT_RATINGS)}
    expected = np.zeros((7, 7), int)
    expected[idx[9], idx[9]] = 1          # 9 -> 9
    expected[idx[9], idx[8]] = 1          # 9 -> 8; 8 -> 9 is work; 9 -> 7 flagged; 7 -> None dropped
    expected[idx[3], idx[3]] = 1          # 2 -> 1 pooled into rating 3
    assert np.array_equal(C, expected)


def test_value_and_policy_iteration_agree_with_brute_force():
    mdp, _ = base_mdp()
    v_vi, pol_vi, n, _ = bdm.value_iteration(mdp, tol=1e-6)
    v_pi, pol_pi, trace = bdm.policy_iteration(mdp)
    assert np.array_equal(pol_vi, pol_pi)
    assert np.max(np.abs(v_vi - v_pi)) < 1e-6
    assert len(trace) <= 5 < n
    # Exhaustive search over all 4**6 stationary deterministic policies.
    best = min((bdm.evaluate(mdp, np.array(p)) for p in product(range(4), repeat=bdm.N_STATES)),
               key=lambda v: v.sum())
    assert np.allclose(best, v_pi)
    assert [bdm.ACTIONS[a] for a in pol_pi] == ["nothing", "nothing", "minor", "minor", "minor", "major"]


def test_schedule_value_matches_augmented_chain():
    mdp, _ = base_mdp()
    for k in (1, 3, 9):
        # Independent construction on the (phase, rating) chain.
        n = k * bdm.N_STATES
        A, b = np.eye(n), np.zeros(n)
        for ph in range(k):
            for s in range(bdm.N_STATES):
                a = 3 if s == bdm.N_STATES - 1 else (1 if ph == 0 else 0)
                i, nxt = ph * bdm.N_STATES + s, ((ph + 1) % k) * bdm.N_STATES
                A[i, nxt:nxt + bdm.N_STATES] -= mdp.gamma * mdp.P[a, s]
                b[i] = mdp.c[s, a]
        assert np.allclose(np.linalg.solve(A, b).reshape(k, -1), bdm.scheduled_value(mdp, k))


def test_replay_follows_solver_and_costs():
    data = json.loads(DATA.read_text())
    C, _ = read_counts()
    P0 = bdm.do_nothing_matrix(C)
    disc = data["gamma"] ** np.arange(data["years"])
    for key, sc in data["scenarios"].items():
        mdp = mdp_with_risk(P0, int(key))
        _, pol, _ = bdm.policy_iteration(mdp)
        assert sc["policy"] == pol.tolist()
        for name, run in sc["runs"].items():
            S, A = np.array(run["states"]), np.array(run["actions"])
            assert np.array_equal(S[0], sc["start"])
            for t in range(data["years"]):
                for b in range(data["bridges"]):
                    s, a = S[t, b], A[t, b]
                    if name == "optimal":
                        assert a == pol[s]
                    elif name == "reactive":
                        assert a == bdm.reactive_policy()[s]
                    else:
                        assert a == bdm.scheduled_action(sc["period"], t, s)
                    assert mdp.P[a, s, S[t + 1, b]] > 0
            costs = mdp.c[S[:-1], A]
            assert np.allclose(np.cumsum(disc[:, None] * costs, axis=0).mean(axis=1), run["cum"], atol=0.01)


def test_monte_carlo_matches_exact_value():
    mdp, _ = base_mdp()
    v, pol, _ = bdm.policy_iteration(mdp)
    rng = np.random.default_rng(0)
    B, T = 4000, 400
    U = rng.random((T, B))
    _, _, costs = bdm.simulate(mdp, lambda t, s: int(pol[s]), np.zeros(B, int), U)
    est = (mdp.gamma ** np.arange(T) @ costs)
    assert abs(est.mean() - v[0]) < 4 * est.std() / np.sqrt(B) + 1.0


def test_metrics_and_files_exist():
    m = json.loads((ART / "metrics.json").read_text())
    assert m["base_failure_multiplier"] == BASE_MULTIPLIER
    assert m["transition_source"]["kind"] in ("raw NBI files", "GitHub mirror of NBI records")
    c = m["base"]["cost"]
    assert c["optimal"] < c["schedule"] and c["optimal"] < c["reactive"]
    for f in ("README.md", "results.md", "deck_transition_counts.csv", "initial_rating_counts.csv"):
        assert (ART / f).exists()
    for suffix in ("svg", "pdf", "png"):
        assert (ROOT / "_static" / "bridge_deck_maintenance" / f"policy.{suffix}").exists()
    assert (ROOT / "interactive" / "bridge-deck-maintenance.html").exists()
    assert DATA.stat().st_size < 1_000_000
