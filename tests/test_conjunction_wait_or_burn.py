"""Checks for the wait-or-burn collision-avoidance example and its artifacts."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from conjunction_wait_or_burn import (COLLISION_COST, DEADLINE, burn_cost,  # noqa: E402
                                      load_epochs, realized_costs, rule_as_soon_as_above,
                                      rule_at_deadline, rule_hindsight, rule_never, summarize)

ARTIFACTS = ROOT / "artifacts" / "conjunction_wait_or_burn"


def test_burn_cost_is_one_at_the_deadline_and_cheaper_earlier():
    assert np.isclose(burn_cost(DEADLINE), 1.0)
    t = np.linspace(DEADLINE, 7, 50)
    assert np.all(np.diff(burn_cost(t)) < 0)


def test_epochs_end_with_one_deadline_decision_per_event():
    epochs = load_epochs()
    events = np.unique(epochs.event)
    assert epochs.last.sum() == events.size
    assert np.all(epochs.t >= DEADLINE)
    assert np.all(epochs.t[epochs.last] == DEADLINE)
    # next points to the following row of the same event, and only the deadline has none
    inner = ~epochs.last
    assert np.all(epochs.event[epochs.next[inner]] == epochs.event[inner])
    assert np.all(epochs.t[epochs.next[inner]] <= epochs.t[inner])
    assert np.all(epochs.next[epochs.last] == -1)
    # the deadline decision sees the latest message before the deadline
    frame = epochs.frame
    latest = frame[~frame.at_deadline].groupby("event_id").risk.last()
    at_deadline = frame[frame.at_deadline].set_index("event_id").risk
    assert np.allclose(latest.reindex(at_deadline.index), at_deadline)


def test_hindsight_is_a_lower_bound_and_rules_are_consistent():
    epochs = load_epochs()
    totals = {name: summarize(realized_costs(epochs, rule(epochs)))
              for name, rule in [("never", rule_never), ("soon", rule_as_soon_as_above),
                                 ("deadline", rule_at_deadline), ("hindsight", rule_hindsight)]}
    assert totals["hindsight"]["total_cost"] <= min(t["total_cost"] for t in totals.values())
    assert totals["never"]["maneuvers"] == 0 and totals["never"]["missed"] == totals["never"]["needed"]
    assert totals["hindsight"]["missed"] == 0
    assert np.isclose(totals["never"]["collision_cost"], totals["never"]["total_cost"])
    # One event dominates the cost of never maneuvering.
    events = realized_costs(epochs, rule_never(epochs))
    assert events.outcome.max() > events.outcome.sum() / 2
    assert np.isclose(COLLISION_COST * 1e-5, burn_cost(DEADLINE))   # break-even at 1e-5


def test_artifacts_match_the_chapter():
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    r = metrics["results"]
    assert r["fqi_relative"]["total_cost"] < r["fqi_raw"]["total_cost"]
    assert r["fqi_relative"]["total_cost"] < r["soon"]["total_cost"]
    assert r["hindsight"]["total_cost"] <= r["deadline"]["total_cost"]
    # The raw target is driven by one event; the relative target is not.
    d = metrics["dominant_event"]
    assert d["raw_all"]["maneuvers"] > 1.5 * d["raw_without"]["maneuvers"]
    assert abs(d["relative_all"]["maneuvers"] - d["relative_without"]["maneuvers"]) < .1 * d["relative_all"]["maneuvers"]
    chapter = (ROOT / "approximate-bellman-equations.md").read_text()
    assert ":label: fig-conjunction-wait-or-burn" in chapter
    assert "artifacts/conjunction_wait_or_burn/results.md" in chapter
