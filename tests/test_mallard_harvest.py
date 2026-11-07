"""Checks for the mallard adaptive-harvest example and its artifacts."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from mallard_harvest import (CLOSED_BELOW, MODEL_WEIGHTS, MODELS, Grid,  # noqa: E402
                             allowed, build_model, evaluate_policy, harvest,
                             published_2012, step_population, survival,
                             transition_matrices, value_iteration)

ARTIFACTS = ROOT / "artifacts" / "mallard_harvest"


def test_model_weights_and_survival_models():
    assert np.isclose(MODEL_WEIGHTS.sum(), 1.0)
    assert np.isclose(MODEL_WEIGHTS[0], 0.66 * 0.94)
    # Compensatory survival ignores light harvest and is 1 - kill rate above the threshold.
    assert np.isclose(survival("compensatory", "m", 0.05), 0.6467)
    assert np.isclose(survival("compensatory", "m", 0.4), 1 - 0.4 / 0.8)
    assert np.isclose(survival("additive", "f", 0.08), 0.6886 * (1 - 0.1))


def test_more_harvest_means_fewer_birds_next_year():
    for model in MODELS:
        nxt = [float(step_population(8.0, 3.5, a, model)) for a in range(4)]
        took = [float(harvest(8.0, 3.5, a, model)) for a in range(4)]
        assert all(np.diff(took) > 0)
        assert all(np.diff(nxt) <= 1e-12)


def test_closed_season_constraint():
    mask = allowed(np.array([CLOSED_BELOW - 0.01, CLOSED_BELOW]))
    assert mask[0].tolist() == [True, False, False, False]
    assert mask[1].tolist() == [False, True, True, True]


def test_transition_matrices_are_stochastic_and_policy_evaluation_matches():
    grid = Grid(1.0, 1.0)
    model = build_model(grid.points[:, 0], grid.points[:, 1])
    for P in transition_matrices(grid, model):
        assert np.allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0)
        assert P.min() >= 0
    solution = value_iteration(grid)
    assert np.abs(evaluate_policy(grid, solution.policy) - solution.values).max() < 1e-4


def test_published_table_shape():
    population, ponds, table = published_2012()
    assert table.shape == (population.size, ponds.size) == (15, 10)
    assert (table[population < CLOSED_BELOW] == 0).all() and (table[population >= CLOSED_BELOW] > 0).all()
    assert (np.diff(table, axis=0) >= 0).all() and (np.diff(table, axis=1) >= 0).all()


def test_artifacts_match_the_chapter():
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    published = metrics["published_2012"]
    assert published["same_package"] >= 0.8 and published["within_one_package"] == 1.0
    for row in metrics["fits"]:
        assert row["value_error"] <= row["value_bound"]
        assert 0 <= row["policy_loss"] <= row["policy_bound"]
    chapter = (ROOT / "approximate-bellman-equations.md").read_text()
    assert ":label: fig-mallard-policies" in chapter
    assert "artifacts/mallard_harvest/results.md" in chapter
