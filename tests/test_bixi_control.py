"""Focused checks for the three-station BIXI teaching system."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from bixi_control import (  # noqa: E402
    load_completed_profile,
    load_event_trace,
    load_scenario,
    make_censoring_counterexample,
    make_controller,
    make_open_loop_controller,
    sample_poisson_trace,
    sha256,
    simulate,
)


DATA = ROOT / "data" / "bixi"


def _inputs():
    scenario = load_scenario(DATA / "stations.json")
    profile = load_completed_profile(DATA / "completed_event_profile.csv", scenario)
    return scenario, profile


def test_derived_data_are_checksum_pinned_and_showcase_counts_match() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["derived_files"].items():
        assert sha256(DATA / name) == expected
    assert manifest["derivation"]["training_day_count"] == 43
    assert manifest["showcase_completed_counts"] == {
        "Berri / Cherrier": {"rentals": 79, "returns": 36},
        "Prince-Arthur / St-Urbain": {"rentals": 48, "returns": 20},
        "de Maisonneuve / Aylmer (ouest)": {"rentals": 12, "returns": 84},
    }


def test_frozen_mean_flow_plan_has_the_documented_intervention() -> None:
    scenario, profile = _inputs()
    controller = make_open_loop_controller(scenario, profile)
    actions = [(action.transfer, action.destination) for action in controller.actions]
    assert actions == [
        (0, 2),
        (0, 2),
        (0, 2),
        (-3, 1),
        (3, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
    ]


def test_event_simulation_respects_bounds_and_bicycle_conservation() -> None:
    scenario, profile = _inputs()
    trace = sample_poisson_trace(scenario, profile, seed=11)
    for name in ("none", "open_loop", "feedback"):
        run = simulate(scenario, trace, make_controller(name, scenario, profile))
        assert np.all(run.history_station_inventory >= 0)
        assert np.all(run.history_station_inventory <= scenario.capacities)
        assert np.all(run.truck_inventory >= 0)
        assert np.all(run.truck_inventory <= scenario.truck_capacity)
        expected = (
            np.sum(scenario.initial_inventory)
            + run.metrics.accepted_returns
            - run.metrics.served_rentals
        )
        actual = np.sum(run.station_inventory[-1]) + run.truck_inventory[-1]
        assert np.isclose(actual, expected)


def test_paired_pulse_occurs_between_actions_and_feedback_then_diverges() -> None:
    scenario, profile = _inputs()
    nominal = sample_poisson_trace(scenario, profile, seed=0)
    pulse = sample_poisson_trace(scenario, profile, seed=0, paired_pulse=True)
    added = [event for event in pulse.events if event.source == "paired demand pulse"]
    assert len(added) == 16
    assert sum(
        event.kind == "rental"
        and event.station_index == 1
        and event.time_minutes == 47
        for event in added
    ) == 8
    assert sum(
        event.kind == "return"
        and event.station_index == 2
        and event.time_minutes == 57
        for event in added
    ) == 8
    assert 45 < min(event.time_minutes for event in added) < 60
    assert 45 < max(event.time_minutes for event in added) < 60

    nominal_run = simulate(
        scenario, nominal, make_controller("feedback", scenario, profile)
    )
    pulse_run = simulate(
        scenario, pulse, make_controller("feedback", scenario, profile)
    )
    # At 08:00 the truck unloads its same three-bike payload in both worlds;
    # the first different transfer is at Aylmer at 08:15 for seed 0.
    assert nominal_run.requested_transfer[4] == pulse_run.requested_transfer[4] == 3
    assert nominal_run.requested_transfer[5] == -6
    assert pulse_run.requested_transfer[5] == -8

    nominal_open = simulate(
        scenario, nominal, make_open_loop_controller(scenario, profile)
    )
    pulse_open = simulate(
        scenario, pulse, make_open_loop_controller(scenario, profile)
    )
    np.testing.assert_array_equal(
        nominal_open.requested_transfer, pulse_open.requested_transfer
    )
    np.testing.assert_array_equal(nominal_open.destination, pulse_open.destination)


def test_real_showcase_is_exactly_replayed_from_completed_events() -> None:
    scenario, profile = _inputs()
    trace = load_event_trace(DATA / "showcase_events.csv", scenario)
    assert len(trace.events) == 279
    feedback = simulate(
        scenario, trace, make_controller("feedback", scenario, profile)
    )
    assert feedback.metrics.service_failures == 17
    assert feedback.metrics.lost_rentals == 0
    assert feedback.metrics.rejected_returns == 17


def test_completed_trip_censoring_counterexample_is_identical() -> None:
    example = make_censoring_counterexample()
    assert example["logs_identical"]
    assert example["demand_stops"] != example["demand_continues"]
    assert example["completed"] == [2, 2, 0, 0, 0, 0, 0, 0]


def test_committed_csv_and_json_report_the_same_paired_experiment() -> None:
    artifact_path = ROOT / "artifacts" / "bixi" / "textbook_results.json"
    metrics_path = ROOT / "artifacts" / "bixi" / "controller_metrics.csv"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 512 * 3
    for scenario_name in ("nominal", "paired_pulse"):
        for controller_name in ("none", "open_loop", "feedback"):
            values = [
                float(row["service_failures"])
                for row in rows
                if row["scenario"] == scenario_name
                and row["controller"] == controller_name
            ]
            recorded = artifact["monte_carlo"]["scenarios"][scenario_name][
                "controllers"
            ][controller_name]["service_failures"]["mean"]
            assert np.mean(values) == recorded
    assert artifact["monte_carlo"]["scenarios"]["nominal"]["controllers"][
        "feedback"
    ]["service_failures"]["mean"] == 3.060546875
    paired = artifact["monte_carlo"]["scenarios"]["paired_pulse"][
        "paired_open_loop_minus_feedback_failures"
    ]
    assert paired["median"] == 11.0
    assert paired["feedback_wins_fraction"] == 0.98046875
