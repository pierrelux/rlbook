"""Checks for the antibiotic-steering MDP, its artifacts, and chapter wiring."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from antibiotic_steering import (DRUGS, N_GENOTYPES, backward_induction,  # noqa: E402
                                 best_open_loop, label, neighbours,
                                 open_loop_probability, step_matrix,
                                 transition, uniform_start)

DATA = ROOT / "interactive" / "antibiotic-steering-data.json"


def test_kernels_are_stochastic_and_absorbed_at_peaks():
    for d in DRUGS:
        P, L = step_matrix(d), transition(d)
        assert np.allclose(P.sum(axis=1), 1) and np.allclose(L.sum(axis=1), 1)
        assert np.allclose(L @ P, L)
        for g in range(N_GENOTYPES):
            # Only one-mutation moves, and only to fitter genotypes.
            assert set(np.nonzero(P[g])[0]) <= set(neighbours(g)) | {g}


def test_reproduces_nichol_table_one_for_ampicillin():
    u = uniform_start()
    assert round(open_loop_probability((), "AMP", u), 2) == 0.62
    seq, p = best_open_loop("AMP", 1, u)
    assert seq == ("CPR",) and round(p, 2) == 0.25
    seq, p = best_open_loop("AMP", 2, u)
    assert seq == ("AMP", "CPR") and p < 1e-12


def test_backward_induction_matches_brute_force_over_markov_policies():
    # With one steering drug the Markov policy can be checked state by state.
    for final in ("AMP", "CPD", "FEP"):
        sol = backward_induction(final, 1)
        for s in range(N_GENOTYPES):
            start = np.eye(N_GENOTYPES)[s]
            best = min(open_loop_probability((d,), final, start) for d in DRUGS)
            assert np.isclose(sol.values[0, s], best)


def test_feedback_never_worse_than_fixed_sequences():
    u = uniform_start()
    for final in DRUGS:
        for k in (1, 2):
            _, p_fixed = best_open_loop(final, k, u)
            assert u @ backward_induction(final, k).values[0] <= p_fixed + 1e-12


def test_replay_data_ends_at_reported_probabilities():
    data = json.loads(DATA.read_text())
    assert data["labels"] == [label(g) for g in range(N_GENOTYPES)]
    amp = data["scenarios"]["AMP"]
    for per_k in amp["starts"].values():
        for row in per_k:
            for strategy in ("fixed", "dp"):
                run = row[strategy]
                final = run["stages"][-1]["frames"][-1]
                assert abs(sum(final) - 1) < 1e-5
                assert abs(final[amp["resistant"]] - run["p"]) < 1e-5
    assert round(amp["starts"]["uniform"][1]["dp"]["p"], 2) == 0.11


def test_chapter_embeds_replay_with_static_placeholder():
    chapter = (ROOT / "stochastic-dp.md").read_text()
    assert "../interactive/antibiotic-steering.html" in chapter
    assert ":placeholder: _static/antibiotic_steering/steering.svg" in chapter
    assert "artifacts/antibiotic_steering/results.md" in chapter
    bib = (ROOT / "references.bib").read_text()
    assert "@article{Nichol2015," in bib and "@article{Mira2015," in bib
    for suffix in ("svg", "pdf", "png"):
        assert (ROOT / "_static" / "antibiotic_steering" / f"steering.{suffix}").exists()
