from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIRECTORY))

from crane_control import (  # noqa: E402
    CraneParameters,
    ZeroVibrationShaper,
    _transcription_objective,
    run_comparison,
    transcription_defects,
)


@pytest.fixture(scope="module")
def comparison():
    return run_comparison(intervals=20, sample_period=0.04)


def test_zero_vibration_shaper_is_a_convex_delayed_pair() -> None:
    params = CraneParameters()
    shaper = ZeroVibrationShaper.from_parameters(params)
    expected_half_period = np.pi / np.sqrt(params.gravity / params.cable_length)

    assert shaper.first_weight > 0.0
    assert shaper.second_weight > 0.0
    assert shaper.first_weight + shaper.second_weight == pytest.approx(1.0)
    assert shaper.delay == pytest.approx(expected_half_period, rel=2e-4)


def test_transcription_objective_uses_matching_node_and_interval_weights() -> None:
    step = 0.25
    state = np.zeros((3, 4))
    state[:, 2] = np.array([1.0, 2.0, 3.0])
    state[:, 3] = np.array([2.0, 1.0, 4.0])
    acceleration = np.array([2.0, 5.0, 11.0])

    node_cost = (
        6.0 * state[:, 2] ** 2
        + 0.15 * state[:, 3] ** 2
        + 0.035 * acceleration**2
    )
    smooth_integral = step * (
        0.5 * node_cost[0] + node_cost[1] + 0.5 * node_cost[2]
    )
    slew_integral = step * np.sum(0.002 * (np.diff(acceleration) / step) ** 2)

    assert _transcription_objective(state, acceleration, step) == pytest.approx(
        smooth_integral + slew_integral
    )


def test_direct_collocation_satisfies_defects_bounds_and_endpoints(comparison) -> None:
    solution = comparison.collocation
    params = comparison.parameters
    defects = transcription_defects(solution, params)

    assert solution.success
    assert np.max(np.abs(defects)) < 1e-7
    assert solution.state[0] == pytest.approx(np.zeros(4), abs=1e-8)
    assert solution.state[-1] == pytest.approx(
        np.array([params.target_position, 0.0, 0.0, 0.0]),
        abs=1e-7,
    )
    assert np.max(np.abs(solution.acceleration)) <= params.acceleration_limit + 1e-8
    assert np.max(np.abs(solution.state[:, 1])) <= params.velocity_limit + 1e-8
    assert np.max(np.abs(solution.state[:, 2])) <= params.sway_limit + 1e-8


def test_all_commands_share_the_actuator_bound_and_validate_nonlinearly(comparison) -> None:
    limit = comparison.parameters.acceleration_limit
    for result in comparison.nominal.values():
        assert np.max(np.abs(result.acceleration)) <= limit + 1e-10
        assert result.metrics.terminal_position_error_m < 0.02
        assert np.all(np.isfinite(result.state))

    optimized = comparison.nominal["collocation"]
    assert optimized.metrics.residual_sway_deg < 1.0
    assert optimized.metrics.max_constraint_violation <= 1e-8


def test_cable_length_mismatch_is_replayed_without_redesign(comparison) -> None:
    nominal = comparison.nominal["zv"].metrics.residual_sway_deg
    mismatched = comparison.mismatch["zv"].metrics.residual_sway_deg

    assert comparison.mismatch_parameters.cable_length == pytest.approx(
        1.1 * comparison.parameters.cable_length
    )
    assert mismatched > 10.0 * nominal
    assert np.allclose(
        comparison.nominal["collocation"].acceleration,
        comparison.mismatch["collocation"].acceleration,
    )
