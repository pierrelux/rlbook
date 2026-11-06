"""Checks for the fishery-near-a-tipping-point example and its artifacts."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from fishery_tipping_point import (C, GAMMA, K, SIGMA, Bellman, ChebyshevBasis,  # noqa: E402
                                   HatBasis, collocate, growth, growth_slope,
                                   reed_escapement, shocks, solve_exact)

ARTIFACTS = ROOT / "artifacts" / "fishery_tipping_point"
STATIC = ROOT / "_static" / "fishery_tipping_point"


def test_shocks_have_mean_one_and_the_stated_log_variance():
    z, w = shocks()
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w @ z, 1.0)
    log_z = np.log(z)
    assert np.isclose(w @ (log_z - w @ log_z) ** 2, SIGMA ** 2)


def test_growth_has_an_allee_threshold_and_a_carrying_capacity():
    assert growth(0.5 * C) < 0.5 * C          # below C the stock shrinks
    assert growth(0.5 * (C + K)) > 0.5 * (C + K)
    assert np.isclose(growth(C), C) and np.isclose(growth(K), K)
    s = np.linspace(0.05, 1.4, 200)
    h = 1e-6
    assert np.allclose(growth_slope(s), (growth(s + h) - growth(s - h)) / (2 * h), atol=1e-6)


def test_reed_escapement_solves_its_condition():
    S = reed_escapement()
    assert C < S < K
    assert np.isclose(GAMMA * growth_slope(S), 1.0)


def test_reference_solution_has_the_three_region_policy():
    bellman = Bellman(n_fine=1601)
    exact = solve_exact(bellman)
    assert 0 < exact.fish_out_level < C < exact.constant_escapement < K
    assert abs(exact.constant_escapement - reed_escapement()) < 0.01
    assert np.all(np.diff(exact.v) >= -1e-12)                       # more fish is never worse
    assert np.abs(bellman.apply(exact) - exact.v).max() < 1e-9     # fixed point of the backup
    assert np.abs(bellman.evaluate(exact.escapement) - exact.v).max() < 1e-8
    above = exact.x >= exact.constant_escapement
    assert np.allclose(exact.escapement[above], exact.escapement[above][0])


def test_hat_functions_converge_at_rate_gamma():
    bellman = Bellman(n_fine=1601)
    result = collocate(HatBasis(8), bellman)
    steps = np.array(result.steps)
    assert not result.diverged and steps[-1] < 1e-9
    ratios = steps[20:200] / steps[19:199]
    assert np.all(ratios <= GAMMA + 1e-6)


def test_chebyshev_interpolation_is_not_monotone():
    basis = ChebyshevBasis(10)
    t = np.linspace(0, 1.6, 2001)
    cardinal = basis.function(np.linalg.solve(basis.matrix, np.eye(10)[3]))(t)
    assert cardinal.min() < -0.05                 # negative weights: not an averager
    constant = basis.function(basis.fit(np.ones(10)))(t)
    assert np.allclose(constant, 1.0)             # but constants are preserved


def test_artifacts_match_the_chapter():
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    assert metrics["parameters"]["C"] == C
    sweep = {(r["key"], r["n"]): r for r in metrics["sweep"]}
    assert not any(r["diverged"] for (key, _), r in sweep.items() if key == "linear")
    assert sweep[("chebyshev", 10)]["diverged"]
    featured = metrics["featured_n"]
    cheb, hat = sweep[("chebyshev", featured)], sweep[("linear", featured)]
    assert cheb["fish_out_level"] > C > hat["fish_out_level"]
    assert cheb["policy_loss"] > 10 * hat["policy_loss"]
    assert cheb["residual"] < hat["residual"]
    for name in ("problem", "collocation", "sweep"):
        svg = (STATIC / f"{name}.svg").read_text()
        assert "prefers-color-scheme: dark" in svg
    chapter = (ROOT / "weighted-residual-methods.md").read_text()
    for label in ("fig-fishery-problem", "fig-fishery-collocation", "fig-fishery-sweep"):
        assert f":label: {label}" in chapter
    assert "artifacts/fishery_tipping_point/results.md" in chapter
