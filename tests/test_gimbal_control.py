from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIRECTORY))

from gimbal_control import (  # noqa: E402
    ESTIMATOR_ORDER,
    EstimatorMemory,
    GimbalObservation,
    GimbalParameters,
    GimbalScenario,
    comparison_to_artifact,
    continuous_dynamics,
    generate_noise,
    make_summary_figure,
    run_comparison,
    sample_observation,
    saturated_pd,
    update_estimator,
    wrap_angle,
)


@pytest.fixture(scope="module")
def comparison():
    return run_comparison()


def test_zero_equilibrium_and_zero_estimate_produce_zero_motion_and_torque() -> None:
    parameters = GimbalParameters()
    scenario = GimbalScenario(tap_start_s=2.0, tap_end_s=2.1)
    derivative = continuous_dynamics(
        0.0, np.zeros(2), 0.0, parameters, scenario
    )

    assert derivative == pytest.approx(np.zeros(2))
    assert saturated_pd(np.zeros(3), parameters) == pytest.approx(0.0)


@pytest.mark.parametrize("angle_deg", [0.0, 10.0, -14.0])
@pytest.mark.parametrize("lateral_acceleration", [0.0, 3.0])
def test_accelerometer_apparent_tilt_identity(
    angle_deg: float, lateral_acceleration: float
) -> None:
    parameters = GimbalParameters()
    angle = np.deg2rad(angle_deg)
    observation = sample_observation(
        np.array([angle, 0.0]),
        gyro_bias_rad_s=0.0,
        lateral_acceleration_mps2=lateral_acceleration,
        gyro_noise_rad_s=0.0,
        accelerometer_noise_mps2=np.zeros(2),
        parameters=parameters,
    )
    expected = wrap_angle(
        angle + np.arctan2(lateral_acceleration, parameters.gravity_mps2)
    )

    assert observation.accelerometer_angle_rad == pytest.approx(expected, abs=1e-13)


def test_complementary_observer_has_correct_bias_sign_and_converges() -> None:
    parameters = GimbalParameters()
    scenario = GimbalScenario(duration_s=30.0)
    true_bias = scenario.initial_gyro_bias_rad_s
    observation = GimbalObservation(
        gyro_rad_s=true_bias,
        accelerometer_x_mps2=0.0,
        accelerometer_y_mps2=parameters.gravity_mps2,
    )
    memory: EstimatorMemory | None = None
    initial_error = true_bias
    for _ in range(scenario.control_steps):
        memory, estimate = update_estimator(
            "complementary", memory, observation, parameters, scenario
        )

    assert memory is not None
    assert memory.bias_rad_s > 0.0
    assert abs(true_bias - memory.bias_rad_s) < 0.02 * initial_error
    assert abs(estimate[0]) < np.deg2rad(0.04)


def test_common_noise_path_reproducibility_and_shapes(comparison) -> None:
    parameters = GimbalParameters()
    scenario = GimbalScenario()
    first = generate_noise(parameters, scenario)
    second = generate_noise(parameters, scenario)

    assert np.array_equal(first.gyro_noise_rad_s, second.gyro_noise_rad_s)
    assert np.array_equal(
        first.accelerometer_noise_mps2, second.accelerometer_noise_mps2
    )
    assert np.array_equal(first.gyro_bias_rad_s, second.gyro_bias_rad_s)
    for estimator in ESTIMATOR_ORDER:
        rollout = comparison[estimator]
        assert rollout.state.shape == (scenario.control_steps + 1, 3)
        assert rollout.observation.shape == (scenario.control_steps, 3)
        assert rollout.estimate.shape == (scenario.control_steps, 3)
        assert rollout.torque_nm.shape == (scenario.control_steps,)
        assert np.all(np.isfinite(rollout.state))
        assert np.array_equal(rollout.state[:, 2], first.gyro_bias_rad_s)


def test_closed_loop_comparison_exposes_both_sensor_failures(comparison) -> None:
    parameters = GimbalParameters()
    accelerometer = comparison["accelerometer"].metrics
    gyro = comparison["gyro"].metrics
    complementary = comparison["complementary"].metrics

    for rollout in comparison.values():
        assert np.max(np.abs(rollout.torque_nm)) <= parameters.torque_limit_nm + 1e-12
    assert (
        accelerometer.peak_acceleration_window_deg
        - complementary.peak_acceleration_window_deg
        >= 8.0
    )
    assert gyro.final_abs_angle_deg - complementary.final_abs_angle_deg >= 4.0
    assert complementary.regulation_score < 0.5 * accelerometer.regulation_score
    assert complementary.regulation_score < 0.5 * gyro.regulation_score


def test_expected_metrics_are_stable(comparison) -> None:
    expected = {
        "accelerometer": (3.860841, 19.002610, 0.078816, 0.598619),
        "gyro": (4.599871, 4.314261, 7.729291, 0.845202),
        "complementary": (2.425368, 6.074239, 0.576751, 0.235927),
    }
    for estimator, values in expected.items():
        metrics = comparison[estimator].metrics
        actual = (
            metrics.rms_angle_deg,
            metrics.peak_acceleration_window_deg,
            metrics.final_abs_angle_deg,
            metrics.regulation_score,
        )
        assert actual == pytest.approx(values, abs=2e-6)


def test_artifact_is_causal_compact_and_uses_one_time_grid(comparison) -> None:
    parameters = GimbalParameters()
    scenario = GimbalScenario()
    artifact = comparison_to_artifact(comparison, parameters, scenario)
    runs = artifact["runs"]
    reference_times = [frame["time_s"] for frame in runs["accelerometer"]["frames"]]

    assert len(reference_times) == 251
    assert reference_times[0] == pytest.approx(0.0)
    assert reference_times[-1] == pytest.approx(10.0)
    for estimator in ESTIMATOR_ORDER:
        times = [frame["time_s"] for frame in runs[estimator]["frames"]]
        assert times == reference_times


def test_static_figure_uses_shared_snapshot_scales(comparison) -> None:
    figure = make_summary_figure(comparison)
    try:
        # Interactive backends may round the canvas by one device pixel.
        assert tuple(figure.get_size_inches()) == pytest.approx(
            (7.2, 4.25), abs=0.01
        )
        assert len(figure.axes) == 4
        for axis in figure.axes[:3]:
            assert axis.get_xlim() == pytest.approx((-1.05, 1.05))
            assert axis.get_ylim() == pytest.approx((-0.68, 0.86))
        assert figure.axes[-1].get_xlim() == pytest.approx((0.0, 10.0))
    finally:
        plt.close(figure)
