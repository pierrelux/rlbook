from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from battery_control import (  # noqa: E402
    BatteryScenario,
    RESISTANCE_FIT_BOUNDS,
    RUN_ORDER,
    audit_to_artifact,
    fit_resistance_scale,
    predictive_current_governor,
    pybamm_current,
    resistance_parameters,
    run_battery_audit,
    simulate_diagnostic_pulse,
    threshold_duration,
)


PYBAMM_AVAILABLE = importlib.util.find_spec("pybamm") is not None
requires_pybamm = pytest.mark.skipif(
    not PYBAMM_AVAILABLE, reason="requires optional PyBaMM artifact dependency"
)


@pytest.fixture(scope="module")
def audit():
    if not PYBAMM_AVAILABLE:
        pytest.skip("requires optional PyBaMM artifact dependency")
    scenario = BatteryScenario()
    results, diagnostic, fitted = run_battery_audit(scenario)
    return scenario, results, diagnostic, fitted


def test_default_scenario_and_resistance_scaling_preserve_rc_time_constant():
    scenario = BatteryScenario()
    scenario.validate()
    nominal = resistance_parameters(scenario, 1.0)
    high = resistance_parameters(scenario, scenario.high_resistance_scale)

    assert nominal == {"r0_ohm": 0.015, "r1_ohm": 0.010, "c1_f": 2400.0}
    assert high["r0_ohm"] == pytest.approx(1.8 * nominal["r0_ohm"])
    assert high["r1_ohm"] == pytest.approx(1.8 * nominal["r1_ohm"])
    assert high["c1_f"] == pytest.approx(nominal["c1_f"] / 1.8)
    assert high["r1_ohm"] * high["c1_f"] == pytest.approx(24.0)


def test_charge_positive_sign_is_negated_for_pybamm():
    assert pybamm_current(5.0) == -5.0
    np.testing.assert_array_equal(
        pybamm_current(np.asarray([0.0, 5.0, 10.0])),
        np.asarray([0.0, -5.0, -10.0]),
    )


@requires_pybamm
def test_diagnostic_pulse_charges_the_pybamm_plant():
    scenario = BatteryScenario()
    trace = simulate_diagnostic_pulse(scenario)
    start = int(np.flatnonzero(trace.time_s == 20.0)[0])
    stop = int(np.flatnonzero(trace.time_s == 30.0)[0])
    expected_soc_gain = 5.0 * 10.0 / (scenario.capacity_ah * 3600.0)

    assert trace.soc[start] == pytest.approx(scenario.initial_soc, abs=2e-9)
    assert trace.soc[stop] - trace.soc[start] == pytest.approx(
        expected_soc_gain, abs=2e-8
    )
    assert np.all(np.diff(trace.soc) >= -1e-10)


@requires_pybamm
def test_diagnostic_pulse_rejects_an_explicit_zero_resistance_scale():
    with pytest.raises(ValueError, match="resistance_scale must be finite and positive"):
        simulate_diagnostic_pulse(BatteryScenario(), resistance_scale=0.0)


def test_bounded_pulse_fit_is_deterministic_and_accurate(audit):
    scenario, _, diagnostic, fitted = audit
    repeated = fit_resistance_scale(diagnostic, scenario)

    assert fitted == repeated
    assert RESISTANCE_FIT_BOUNDS[0] <= fitted.resistance_scale <= RESISTANCE_FIT_BOUNDS[1]
    assert fitted.resistance_scale == pytest.approx(
        scenario.high_resistance_scale, abs=0.003
    )
    assert fitted.voltage_rmse_v < 0.002


def test_threshold_duration_does_not_integrate_a_boolean_mask_at_half_weight():
    time_s = np.asarray([0.0, 1.0, 2.0, 3.0])
    values = np.asarray([0.0, 2.0, 2.0, 0.0])

    # Linear crossings occur at 0.5 s and 2.5 s, hence two full seconds above 1.
    assert threshold_duration(time_s, values, 1.0) == pytest.approx(2.0)
    assert np.trapezoid((values > 1.0).astype(float), time_s) == pytest.approx(2.0)
    assert np.trapezoid(values > 1.0, time_s) != pytest.approx(2.0)


def test_matched_audit_has_the_intended_ordering_and_safety_verdicts(audit):
    scenario, results, _, _ = audit
    fresh = results["fresh_nominal"]
    stale = results["high_resistance_stale"]
    calibrated = results["high_resistance_calibrated"]

    assert 1080.0 < fresh.metrics.target_time_s < 1140.0
    assert 1320.0 < stale.metrics.target_time_s < 1350.0
    assert 1500.0 < calibrated.metrics.target_time_s < 1560.0
    assert fresh.metrics.target_time_s < stale.metrics.target_time_s
    assert stale.metrics.target_time_s < calibrated.metrics.target_time_s

    assert fresh.metrics.max_voltage_v <= scenario.voltage_limit_v
    assert stale.metrics.max_voltage_v == pytest.approx(4.251, abs=0.002)
    assert 240.0 < stale.metrics.voltage_violation_duration_s < 250.0
    assert calibrated.metrics.max_voltage_v <= scenario.voltage_limit_v
    assert stale.first_violation_time_s is not None
    assert fresh.first_violation_time_s is None
    assert calibrated.first_violation_time_s is None
    assert all(
        trace.metrics.max_cell_temperature_c < scenario.temperature_limit_c
        for trace in results.values()
    )


def test_charge_conservation_action_bounds_and_parameter_isolation(audit):
    scenario, results, _, fitted = audit
    for trace in results.values():
        trace.validate(scenario)
        assert trace.soc[-1] == pytest.approx(scenario.target_soc, abs=2e-7)
        assert abs(trace.metrics.charge_balance_error_ah) < 2e-6
        assert np.max(trace.current_a) <= scenario.current_limit_a + 1e-9
        assert np.min(trace.current_a) >= -1e-9

    assert results["fresh_nominal"].plant_resistance_scale == 1.0
    assert results["fresh_nominal"].model_resistance_scale == 1.0
    assert results["high_resistance_stale"].plant_resistance_scale == 1.8
    assert results["high_resistance_stale"].model_resistance_scale == 1.0
    assert results["high_resistance_calibrated"].plant_resistance_scale == 1.8
    assert results["high_resistance_calibrated"].model_resistance_scale == pytest.approx(
        fitted.resistance_scale
    )


def test_recorded_current_matches_the_shared_numeric_governor(audit):
    scenario, results, _, _ = audit
    for trace in results.values():
        sample_indices = np.unique(
            np.linspace(0, trace.time_s.size - 2, 11, dtype=int)
        )
        for index in sample_indices:
            state = type(
                "State",
                (),
                {
                    "soc": trace.soc[index],
                    "rc_overpotential_v": trace.rc_overpotential_v[index],
                    "cell_temperature_c": trace.cell_temperature_c[index],
                },
            )()
            expected = predictive_current_governor(
                state, scenario, trace.model_resistance_scale
            )
            assert trace.current_a[index] == pytest.approx(expected, abs=2e-7)


def test_replay_artifact_preserves_metrics_events_and_a_violating_frame(audit):
    scenario, results, diagnostic, fitted = audit
    artifact = audit_to_artifact(results, diagnostic, fitted, scenario)

    assert artifact["schema_version"] == 1
    assert tuple(artifact["runs"]) == RUN_ORDER
    for name in RUN_ORDER:
        run = artifact["runs"][name]
        assert run["metrics"] == pytest.approx(
            vars(results[name].metrics), rel=0.0, abs=1e-12
        )
        assert run["events"]["target_time_s"] == pytest.approx(
            run["metrics"]["target_time_s"]
        )
        assert run["frames"][-1]["time_s"] == pytest.approx(
            run["metrics"]["target_time_s"]
        )
    stale_frames = artifact["runs"]["high_resistance_stale"]["frames"]
    assert any(
        frame["terminal_voltage_v"] > scenario.voltage_limit_v
        for frame in stale_frames
    )


def test_committed_artifacts_agree_and_manifest_hashes_resolve():
    artifact_dir = ROOT / "artifacts" / "battery"
    artifact = json.loads(
        (artifact_dir / "textbook_results.json").read_text(encoding="utf-8")
    )
    with (artifact_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        metric_rows = {row["run"]: row for row in csv.DictReader(handle)}
    arrays = np.load(artifact_dir / "trajectories.npz")

    for name in RUN_ORDER:
        metrics = artifact["runs"][name]["metrics"]
        assert float(metric_rows[name]["target_time_s"]) == pytest.approx(
            metrics["target_time_s"]
        )
        assert float(metric_rows[name]["max_voltage_v"]) == pytest.approx(
            metrics["max_voltage_v"]
        )
        assert arrays[f"{name}_time_s"][-1] == pytest.approx(
            metrics["target_time_s"]
        )

    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["protocol"]["diagnostic"]["fit_bounds"] == [0.7, 2.5]
    for relative, expected in manifest["inputs"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected
    for relative, expected in manifest["outputs"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected
