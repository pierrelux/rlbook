"""Checks for the tamarisk river-network MDP, its solvers, and its artifacts."""
import json
from itertools import product
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from tamarisk_river import (  # noqa: E402
    ACTION_KERNELS, ADVANTAGE, BETA, CHI, DEATH, EXO_NATIVE, EXO_TAMARISK,
    GAMMA, HEURISTICS, SEEDS, River, decode, dispersal, encode, evaluate,
    policy_iteration, rule_policy, simulate_step, stage_cost, value_iteration,
)

DATA = ROOT / "interactive" / "tamarisk-river-data.json"


def brute_transition(n, a):
    """Enumerate every action outcome and death pattern with plain loops."""
    K = dispersal(n)
    P = np.zeros((3 ** n, 3 ** n))
    act_out = {  # (action, code) -> {code: prob}, written out from the model's description
        (0, 0): {0: 1}, (0, 1): {1: 1}, (0, 2): {2: 1},
        (1, 0): {0: 1}, (1, 1): {1: 1}, (1, 2): {0: CHI, 2: 1 - CHI},
        (2, 0): {1: BETA, 0: 1 - BETA}, (2, 1): {1: 1},
        (2, 2): {1: CHI * BETA, 0: CHI * (1 - BETA), 2: 1 - CHI},
    }
    for s, x in enumerate(product(range(3), repeat=n)):
        for outcome in product(*[act_out[(a[j], x[j])].items() for j in range(n)]):
            y = [c for c, _ in outcome]
            p_y = math.prod(p for _, p in outcome)
            for dies in product((False, True), repeat=n):
                p_d, z = 1.0, list(y)
                for j in range(n):
                    if y[j]:
                        p_d *= DEATH if dies[j] else 1 - DEATH
                        if dies[j]:
                            z[j] = 0
                    elif dies[j]:
                        p_d = 0.0
                if p_d == 0:
                    continue
                per_reach = []
                for j in range(n):
                    if z[j]:
                        per_reach.append({z[j]: 1.0})
                        continue
                    lt = EXO_TAMARISK + SEEDS * sum(K[i, j] for i in range(n) if z[i] == 2)
                    ln = EXO_NATIVE + SEEDS * sum(K[i, j] for i in range(n) if z[i] == 1)
                    pe = math.exp(-lt - ln)
                    wt = ADVANTAGE * lt / (ADVANTAGE * lt + ln)
                    per_reach.append({0: pe, 1: (1 - pe) * (1 - wt), 2: (1 - pe) * wt})
                for nxt in product(*[r.items() for r in per_reach]):
                    P[s, encode([c for c, _ in nxt])] += p_y * p_d * math.prod(p for _, p in nxt)
    return P


def test_dispersal_prefers_downstream():
    K = dispersal(7)
    assert np.all(K >= 0) and np.all(K.sum(axis=1) < 1)
    assert K[1, 0] > K[0, 1] and K[3, 1] > K[1, 3]


def test_kernel_is_stochastic_and_matches_enumeration():
    for n in (2, 3):
        river = River(n, budget=None)
        for a in river.actions:
            P = river.transition(a)
            assert np.all(P >= -1e-15) and np.allclose(P.sum(axis=1), 1)
            assert np.allclose(P, brute_transition(n, a), atol=1e-12)


def test_factored_backup_matches_dense_backup():
    rng = np.random.default_rng(0)
    for budget in (1, None):
        river = River(4, budget=budget)
        v = rng.random(river.n_states)
        E = river.expected_next(v)
        for i, a in enumerate(river.actions[:: max(1, len(river.actions) // 12)]):
            j = river.actions.index(a)
            assert np.allclose(E[:, j], river.transition(a) @ v)


def test_value_iteration_matches_dense_solver_and_policy_iteration():
    river = River(3, budget=None)
    Ps = np.stack([river.transition(a) for a in river.actions])        # (a, s, j)
    v = np.zeros(river.n_states)
    for _ in range(2000):
        v = (river.cost + GAMMA * np.einsum("asj,j->sa", Ps, v)).min(axis=1)
    assert np.allclose(value_iteration(river, eps=1e-9).v, v, atol=1e-6)
    pi = policy_iteration(river)
    assert np.allclose(pi.v, v, atol=1e-6) and pi.iterations < 10


def test_budget_and_heuristics_are_ordered():
    river, free = River(4), River(4, budget=None)
    opt, fopt = policy_iteration(river).v, policy_iteration(free).v
    assert np.all(fopt <= opt + 1e-8)
    for key in ("upstream", "downstream", "nothing"):
        assert np.all(evaluate(river, rule_policy(4, HEURISTICS[key])) >= opt - 1e-8)


def test_sampler_matches_kernel():
    river = River(3, budget=None)
    rng = np.random.default_rng(1)
    x, a = np.array([2, 1, 0], dtype=np.int8), (2, 0, 1)
    counts = np.zeros(27)
    for _ in range(20000):
        counts[encode(simulate_step(x, a, river.K, rng.random((3, 3))))] += 1
    p = river.transition(a)[encode(x)]
    assert np.abs(counts / counts.sum() - p).max() < 0.015


def test_replay_data_follows_solver():
    data = json.loads(DATA.read_text())
    n = data["n"]
    river = River(n)
    sol = value_iteration(river)
    policies = {"optimal": sol.policy, **{k: rule_policy(n, HEURISTICS[k]) for k in ("upstream", "downstream", "all", "nothing")}}
    full = data["starts"]["full"]
    assert abs(full["expected"]["optimal"] - sol.v[encode(full["state"])]) < 1e-3
    for start in data["starts"].values():
        for ep in start["episodes"][:3]:
            for key, run in ep.items():
                for t, (x, a, c) in enumerate(zip(run["states"], run["actions"], run["costs"])):
                    assert tuple(a) == policies[key][encode(x)]
                    assert abs(c - stage_cost(x, a)) < 1e-9
                    if key == "optimal" and t < 5:
                        assert river.transition(a)[encode(x), encode(run["states"][t + 1])] > 0
    assert DATA.stat().st_size < 1_000_000


def test_files_exist():
    for suffix in ("svg", "pdf", "png"):
        for name in ("river", "scaling"):
            assert (ROOT / "_static" / "tamarisk_river" / f"{name}.{suffix}").exists()
    for name in ("README.md", "metrics.json", "results.md"):
        assert (ROOT / "artifacts" / "tamarisk_river" / name).exists()
    assert (ROOT / "interactive" / "tamarisk-river.html").exists()
    assert len(decode(2)) == 9
