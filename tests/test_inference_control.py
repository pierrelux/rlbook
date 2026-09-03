from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE_DIRECTORY = ROOT / "code"
sys.path.insert(0, str(CODE_DIRECTORY))
sys.path.insert(0, str(ROOT / "scripts"))

from inference_control import (  # noqa: E402
    OpenLoopPlan,
    TransitionDataset,
    _matched_first_token_latency,
    _optional_difference,
    _policy_action_coverage_slices,
    _reduced_policy_trajectory,
    _select_frequency_experiment_window,
    build_textbook_results,
    estimate_decode_remaining_tokens,
    evaluate_scheduling_policy,
    fit_fqi,
    generate_transition_dataset,
    make_scheduling_mdp,
    mpc_latency_proxies,
    optimize_open_loop,
    open_loop_clock_controller,
    run_mpc,
    shift_largest_burst,
    solve_scheduling_mdp,
)
from inference_serving import (  # noqa: E402
    Request,
    RequestRecord,
    ServingObservation,
    ServingPlant,
    load_profile,
    load_workload,
    normalize_offered_load,
)
from build_inference_artifacts import plant_from_profile_manifest  # noqa: E402


DATA = ROOT / "data" / "inference_serving"


@pytest.fixture(scope="module")
def plant() -> ServingPlant:
    return ServingPlant(
        load_profile(DATA / "l4_profile.csv"),
        time_step_s=0.2,
        maximum_simulation_time_s=60.0,
    )


@pytest.fixture(scope="module")
def short_workload() -> tuple[Request, ...]:
    return (
        Request(0, 0.0, 256, 20),
        Request(1, 0.3, 1024, 16),
        Request(2, 2.0, 512, 24),
        Request(3, 22.0, 2048, 12),
        Request(4, 23.0, 1536, 8),
    )


def test_open_loop_plan_is_feasible_quantized_and_open_loop(
    plant: ServingPlant,
    short_workload: tuple[Request, ...],
) -> None:
    plan = optimize_open_loop(
        short_workload,
        plant,
        horizon_s=30.0,
        expected_output_tokens=24.0,
    )
    shifted = shift_largest_burst(
        short_workload,
        plant.profile,
        expected_output_tokens=24.0,
    )

    assert plan.success
    assert plan.applied_clock_mhz.size == 30
    assert set(plan.applied_clock_mhz).issubset(set(plant.profile.clock_mhz))
    assert np.all(np.isfinite(plan.predicted_backlog_s))
    assert plan.predicted_backlog_s[-1] >= 0.0
    assert sum(request.prompt_tokens for request in shifted) == sum(
        request.prompt_tokens for request in short_workload
    )
    assert [request.request_id for request in shifted] != []
    assert min(request.arrival_time_s for request in shifted) >= 0.0


def test_open_loop_forecast_never_reads_per_request_hidden_outputs(
    plant: ServingPlant,
) -> None:
    first = (
        Request(0, 0.0, 256, 1),
        Request(1, 3.0, 1024, 10_000),
    )
    changed_hidden = tuple(
        replace(request, output_tokens=99_999 - request.output_tokens)
        for request in first
    )
    first_plan = optimize_open_loop(
        first,
        plant,
        horizon_s=10.0,
        expected_output_tokens=64.0,
    )
    changed_plan = optimize_open_loop(
        changed_hidden,
        plant,
        horizon_s=10.0,
        expected_output_tokens=64.0,
    )

    assert np.array_equal(first_plan.applied_clock_mhz, changed_plan.applied_clock_mhz)
    assert np.allclose(first_plan.predicted_backlog_s, changed_plan.predicted_backlog_s)
    assert first_plan.objective == pytest.approx(changed_plan.objective)


def test_open_loop_switches_at_the_intended_floating_point_boundary() -> None:
    plan = OpenLoopPlan(
        time_s=np.array([0.0, 1.0]),
        continuous_clock_mhz=np.array([600.0, 1200.0]),
        applied_clock_mhz=np.array([600.0, 1200.0]),
        predicted_backlog_s=np.zeros(3),
        objective=0.0,
        optimization_method="test",
        success=True,
        message="test",
        control_period_s=1.0,
        workload_checksum="test",
        profile_status="engineering_proxy_not_measured",
    )
    observation = ServingObservation(
        time_s=float(np.nextafter(1.0, 0.0)),
        step_index=10,
        prefill_queue=0,
        decode_active=0,
        prefill_remaining_tokens=0.0,
        generated_decode_tokens=0.0,
        oldest_prefill_age_s=0.0,
        kv_tokens=0.0,
        kv_capacity_tokens=1.0,
        temperature_c=25.0,
        previous_clock_mhz=600.0,
        previous_power_w=0.0,
        clock_levels_mhz=(600.0, 1200.0),
        arrived_requests=0,
        completed_requests=0,
    )

    assert observation.time_s < 1.0
    assert open_loop_clock_controller(plan)(observation) == 1200.0


def test_burst_shift_uses_forecast_service_work_not_hidden_output_tokens(
    plant: ServingPlant,
) -> None:
    workload = (
        Request(0, 20.0, 4096, 1),
        Request(1, 31.0, 1, 100_000),
        Request(2, 32.0, 1, 100_000),
        Request(3, 33.0, 1, 100_000),
    )
    shifted = shift_largest_burst(
        workload,
        plant.profile,
        expected_output_tokens=1.0,
    )
    by_id = {request.request_id: request.arrival_time_s for request in shifted}

    assert by_id[0] == pytest.approx(0.0)
    assert by_id[1] == pytest.approx(31.0)


def test_frequency_window_is_rebased_and_independent_of_hidden_outputs(
    plant: ServingPlant,
) -> None:
    raw = load_workload(DATA / "azure_code_evaluation.csv")
    normalized, _ = normalize_offered_load(raw, plant.profile)
    expected_output_tokens = 32.0
    selected, metadata = _select_frequency_experiment_window(
        normalized,
        plant.profile,
        expected_output_tokens=expected_output_tokens,
    )
    changed_hidden = tuple(
        replace(request, output_tokens=100_000 - request.output_tokens)
        for request in normalized
    )
    selected_with_changed_hidden, changed_metadata = (
        _select_frequency_experiment_window(
            changed_hidden,
            plant.profile,
            expected_output_tokens=expected_output_tokens,
        )
    )

    assert selected
    assert min(request.arrival_time_s for request in selected) >= 0.0
    assert max(request.arrival_time_s for request in selected) < 60.0
    assert metadata["source_window_end_s"] == pytest.approx(
        float(metadata["source_window_start_s"]) + 60.0
    )
    assert metadata["request_count"] == len(selected)
    assert float(metadata["forecast_work_s"]) <= 60.0 + 1e-9
    assert float(metadata["shiftable_burst_start_s"]) >= 20.0
    assert int(metadata["shiftable_burst_request_count"]) > 0
    assert [request.request_id for request in selected_with_changed_hidden] == [
        request.request_id for request in selected
    ]
    assert [request.arrival_time_s for request in selected_with_changed_hidden] == [
        request.arrival_time_s for request in selected
    ]
    assert changed_metadata == metadata


def test_measured_normalized_trace_has_a_nonempty_moved_burst(
    plant: ServingPlant,
) -> None:
    raw = load_workload(DATA / "azure_code_evaluation.csv")
    normalized, _ = normalize_offered_load(raw, plant.profile)
    selected, metadata = _select_frequency_experiment_window(
        normalized,
        plant.profile,
        expected_output_tokens=32.0,
    )
    shifted = shift_largest_burst(
        selected,
        plant.profile,
        expected_output_tokens=32.0,
    )
    nominal_arrivals = {
        request.request_id: request.arrival_time_s for request in selected
    }
    moved = [
        request
        for request in shifted
        if abs(request.arrival_time_s - nominal_arrivals[request.request_id]) > 1e-9
    ]

    assert moved
    assert len(moved) == int(metadata["shiftable_burst_request_count"])
    assert all(
        request.arrival_time_s
        == pytest.approx(nominal_arrivals[request.request_id] - 20.0)
        for request in moved
    )


def test_mpc_replans_and_has_a_recorded_deadline_fallback(
    plant: ServingPlant,
    short_workload: tuple[Request, ...],
) -> None:
    nominal = run_mpc(
        short_workload[:3],
        plant,
        horizon_s=3.0,
        solve_time_limit_s=0.8,
    )
    forced_fallback = run_mpc(
        short_workload[:2],
        plant,
        horizon_s=3.0,
        solve_time_limit_s=0.0,
    )

    assert nominal.mpc_diagnostics is not None
    assert nominal.mpc_diagnostics.successful_solves > 1
    assert nominal.metrics.power_violation_w == 0.0
    assert nominal.metrics.thermal_violation_c == 0.0
    assert forced_fallback.mpc_diagnostics.fallback_count > 0
    assert np.all(np.isfinite(forced_fallback.realized_clock_mhz))
    replay_payload = nominal.as_dict(stride=11)
    plan_updates = [
        plan for plan in replay_payload["planned_clock_mhz"] if plan
    ]
    assert replay_payload["plan_dt_s"] == pytest.approx(1.0)
    assert len(plan_updates) == (
        nominal.mpc_diagnostics.successful_solves
        + nominal.mpc_diagnostics.fallback_count
    )
    update_times = np.asarray(
        [
            value
            for value in nominal.planned_clock_start_time_s
            if value is not None
        ]
    )
    assert np.allclose(np.diff(update_times), 1.0, atol=1e-12)
    assert "maximum_solve_time_s" not in replay_payload["mpc_diagnostics"]


def test_mpc_latency_proxies_keep_prefill_and_decode_delays_separate(
    plant: ServingPlant,
) -> None:
    low = plant.profile.minimum_clock_mhz
    high = plant.profile.maximum_clock_mhz
    prefill_only = mpc_latency_proxies(4096.0, 0.0, low, plant.profile)
    low_clock = mpc_latency_proxies(4096.0, 8.0, low, plant.profile)
    high_clock = mpc_latency_proxies(4096.0, 8.0, high, plant.profile)

    assert prefill_only[1] == 0.0
    assert high_clock[0] < low_clock[0]
    assert high_clock[1] < low_clock[1]


def test_mpc_decode_work_estimate_uses_observed_progress_not_true_lengths() -> None:
    initial = estimate_decode_remaining_tokens(4, 0.0, 32.0)
    progressed = estimate_decode_remaining_tokens(4, 37.0, 32.0)
    clipped = estimate_decode_remaining_tokens(1, 100.0, 32.0)

    assert initial == 128.0
    assert progressed == 91.0
    assert clipped == 0.0


def test_reduced_mdp_probabilities_masks_and_bellman_certificate() -> None:
    mdp = make_scheduling_mdp()
    solution = solve_scheduling_mdp(mdp)

    assert np.allclose(mdp.transitions.sum(axis=2)[mdp.valid_actions], 1.0)
    empty = 0
    assert not mdp.valid_actions[empty, 0]
    assert not mdp.valid_actions[empty, 1]
    assert mdp.valid_actions[empty, 2]
    full_decode_with_prefill = (1 * 7 + 6) * 5
    assert not mdp.valid_actions[full_decode_with_prefill, 0]
    assert solution.bellman_residual < 1e-8
    assert np.all(mdp.valid_actions[np.arange(245), solution.policy])
    assert "not an exact reproduction of vLLM" in mdp.description
    assert 0.0 < mdp.prefill_completion_probability < 1.0
    prefill_state = (1 * 7 + 0) * 5
    prefill_next_support = np.flatnonzero(mdp.transitions[prefill_state, 0] > 0.0)
    assert prefill_next_support.size >= 2


def test_decode_transition_respects_one_gpu_aggregate_capacity() -> None:
    mdp = make_scheduling_mdp(decode_completion_probability=0.42)

    for decode_jobs in range(1, 7):
        state = (0 * 7 + decode_jobs) * 5
        next_decode = mdp.states[:, 1]
        expected_completions = np.sum(
            (decode_jobs - next_decode) * mdp.transitions[state, 1]
        )
        assert expected_completions == pytest.approx(0.42)


def test_trace_calibration_uses_median_clock_phase_power(plant: ServingPlant) -> None:
    from inference_control import scheduling_mdp_from_trace

    workload = (
        Request(0, 0.0, 1000, 20),
        Request(1, 1.0, 500, 10),
    )
    mdp = scheduling_mdp_from_trace(workload, plant.profile)
    median_clock = plant.profile.clock_mhz[len(plant.profile.clock_mhz) // 2]
    expected = np.array(
        [
            plant.profile.power("prefill", median_clock),
            plant.profile.power("decode", median_clock),
            plant.profile.power("idle", median_clock),
        ]
    ) * mdp.decision_period_s

    assert np.allclose(mdp.action_energy, expected)
    assert 0.0 < mdp.prefill_completion_probability < 1.0


def test_reduced_replay_uses_mdp_time_energy_power_and_fixed_clock() -> None:
    mdp = make_scheduling_mdp(
        arrival_probability=0.0,
        decision_period_s=0.25,
        action_energy=(10.0, 15.0, 2.5),
    )
    idle_policy = np.full(mdp.states.shape[0], 2, dtype=int)

    trajectory = _reduced_policy_trajectory(
        mdp,
        idle_policy,
        label="Idle test policy",
        fixed_requested_clock_mhz=1125.0,
        batch_balanced_realized_clock_mhz=939.375,
        steps=4,
        random_seed=3,
    )

    assert trajectory["time_s"] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert trajectory["power_w"] == pytest.approx([10.0] * 4)
    assert trajectory["energy_j"] == pytest.approx([2.5, 5.0, 7.5, 10.0])
    assert trajectory["requested_clock_mhz"] == pytest.approx([1125.0] * 4)
    assert trajectory["realized_clock_mhz"] == pytest.approx([939.375] * 4)
    assert "temperature_c" not in trajectory


def test_fqi_is_deterministic_and_reports_the_coverage_failure_mode() -> None:
    mdp = make_scheduling_mdp()
    exact = solve_scheduling_mdp(mdp)
    broad_data = generate_transition_dataset(
        mdp, behavior="uniform", number_transitions=5_000, random_seed=0
    )
    narrow_data = generate_transition_dataset(
        mdp, behavior="decode_priority", number_transitions=5_000, random_seed=0
    )
    broad = fit_fqi(
        mdp,
        broad_data,
        sweeps=5,
        number_trees=25,
        random_seed=0,
        reference_policy=exact.policy,
    )
    repeated = fit_fqi(
        mdp,
        broad_data,
        sweeps=5,
        number_trees=25,
        random_seed=0,
        reference_policy=exact.policy,
    )

    assert broad_data.coverage_fraction > narrow_data.coverage_fraction
    assert np.array_equal(broad.policy, repeated.policy)
    assert np.allclose(broad.q_values, repeated.q_values)
    assert broad.policy_disagreement_fraction is not None
    assert 0.0 <= broad.policy_disagreement_fraction <= 1.0


def test_coverage_slices_count_only_the_displayed_state_action_pair() -> None:
    state = (2 * 7 + 3) * 5 + 4
    dataset = TransitionDataset(
        state=np.array([state, state, state]),
        action=np.array([0, 0, 1]),
        cost=np.zeros(3),
        next_state=np.array([state, state, state]),
        behavior="test",
        coverage_fraction=0.0,
        random_seed=0,
    )
    policy = np.full(245, 2, dtype=int)
    policy[state] = 1
    slices = _policy_action_coverage_slices(dataset, policy)

    assert slices["4"][2][3] == 1
    policy[state] = 0
    slices = _policy_action_coverage_slices(dataset, policy)
    assert slices["4"][2][3] == 2


def test_matched_ttft_keeps_empty_and_censored_subsets_explicit() -> None:
    records = (
        RequestRecord(7, 2.0, None, None, None, 128, 8),
        RequestRecord(8, 3.0, 3.5, 4.0, 5.0, 128, 8),
    )

    empty = _matched_first_token_latency(records, set())
    assert empty["matched_moved_burst_mean_ttft_s"] is None
    assert empty["matched_moved_burst_p95_ttft_s"] is None
    assert empty["matched_moved_burst_request_count"] == 0
    assert empty["matched_moved_burst_observed_ttft_count"] == 0
    assert empty["matched_moved_burst_censored_ttft_count"] == 0
    assert (
        empty["matched_moved_burst_ttft_status"]
        == "unavailable_no_moved_requests"
    )

    censored = _matched_first_token_latency(records, {7})
    assert censored["matched_moved_burst_mean_ttft_s"] is None
    assert censored["matched_moved_burst_p95_ttft_s"] is None
    assert censored["matched_moved_burst_request_count"] == 1
    assert censored["matched_moved_burst_observed_ttft_count"] == 0
    assert censored["matched_moved_burst_censored_ttft_count"] == 1
    assert (
        censored["matched_moved_burst_ttft_status"]
        == "unavailable_no_observed_first_tokens"
    )
    assert _optional_difference(None, 1.0) is None
    assert _optional_difference(2.5, 1.0) == pytest.approx(1.5)


def test_policy_evaluation_reports_return_queue_waiting_stalls_and_drops() -> None:
    mdp = make_scheduling_mdp()
    solution = solve_scheduling_mdp(mdp)
    evaluation = evaluate_scheduling_policy(
        mdp,
        solution.policy,
        episodes=30,
        horizon_steps=60,
        random_seed=5,
    )

    assert np.isfinite(evaluation.mean_discounted_return)
    assert evaluation.mean_queue_length >= 0.0
    assert evaluation.mean_waiting_time_s >= 0.0
    assert evaluation.mean_decode_stalls >= 0.0
    assert evaluation.mean_dropped_arrivals >= 0.0


def test_small_artifact_suite_matches_the_locked_view_schema(
    plant: ServingPlant,
    short_workload: tuple[Request, ...],
) -> None:
    results = build_textbook_results(
        short_workload[:3],
        short_workload,
        plant,
        fqi_transitions=1_000,
        fqi_sweeps=3,
        fqi_trees=10,
        evaluation_episodes=20,
        evaluation_horizon_steps=30,
    )

    assert {"modeling", "open_loop", "mpc", "scheduling", "fqi"}.issubset(results)
    metadata = results["metadata"]
    validation = metadata["measurement_validation"]
    assert metadata["profile_status"] == plant.profile.profile_status
    assert metadata["profile_source"] == plant.profile.source_label
    assert metadata["profile_csv_sha256"] == plant.profile.profile_csv_sha256
    window = metadata["frequency_experiment_window"]
    assert metadata["frequency_experiment_source_window_start_s"] == pytest.approx(
        float(window["source_window_start_s"])
    )
    assert metadata["frequency_experiment_source_window_end_s"] == pytest.approx(
        float(window["source_window_end_s"])
    )
    assert metadata["frequency_experiment_request_count"] == window["request_count"]
    assert results["open_loop"]["mismatch_metrics"]["moved_request_count"] > 0
    assert validation["validated"] is plant.profile.measurement_validated
    assert validation["manifest_status"] == plant.profile.manifest.get(
        "status", "not_measured"
    )
    assert validation["manifest_sha256"] == plant.profile.manifest_sha256
    if plant.profile.profile_status == "engineering_proxy_not_measured":
        assert not plant.profile.is_measured
        assert validation["validated"] is False
        assert "not hardware measurements" in str(metadata["warning"])
    elif plant.profile.profile_status == "measured_l4":
        assert plant.profile.is_measured
        assert validation["validated"] is True
        assert validation["manifest_status"] == "complete"
        assert len(str(validation["manifest_sha256"])) == 64
        fit_r_squared = float(plant.profile.manifest["thermal_fit_r_squared"])
        assert validation["thermal_fit_r_squared"] == pytest.approx(fit_r_squared)
        assert validation["thermal_fit_is_weak"] is (fit_r_squared < 0.1)
        if fit_r_squared < 0.1:
            assert "thermal RC fit is weak" in str(metadata["warning"])
            assert "weakly supported" in str(metadata["warning"])
        maximum_profile_power = max(
            np.max(plant.profile.idle_power_w),
            np.max(plant.profile.prefill_power_w),
            np.max(plant.profile.decode_power_w),
        )
        assert validation["maximum_profile_phase_power_w"] == pytest.approx(
            maximum_profile_power
        )
        exceeds_limit = bool(maximum_profile_power > plant.power_limit_w + 1e-9)
        assert validation["profile_phase_power_exceeds_limit"] is exceeds_limit
        if exceeds_limit:
            assert "above the configured" in str(metadata["warning"])
            assert "violations remain reported" in str(metadata["warning"])
    else:
        pytest.fail(
            f"unexpected inference profile status: {plant.profile.profile_status!r}"
        )
    assert results["scheduling"]["bellman_residual"] < 1e-8
    calibration = results["scheduling"]["calibration"]
    fixed_clock = float(
        plant.profile.clock_mhz[len(plant.profile.clock_mhz) // 2]
    )
    assert calibration["fixed_requested_clock_mhz"] == pytest.approx(fixed_clock)
    selection = plant.profile.manifest.get("clock_profile_selection", {})
    realized_by_requested = selection.get(
        "realized_clock_median_mhz_by_requested", {}
    )
    expected_realized = realized_by_requested.get(str(int(fixed_clock)))
    assert calibration["batch_balanced_realized_clock_mhz"] == expected_realized
    assert calibration["trajectory_semantics"] == {
        "source": "one_seeded_sample_path_from_reduced_mdp_not_event_simulator",
        "random_seed": 29,
        "horizon_steps": 120,
        "requested_clock_mhz": "fixed_profile_level_used_for_mdp_calibration",
        "realized_clock_mhz": (
            "batch_balanced_profile_median_repeated_when_available"
        ),
        "power_w": "action_energy_j[action] / decision_period_s",
        "energy_j": "cumulative_sum_of_action_energy_j[action]",
        "temperature_c": "omitted_not_a_reduced_mdp_state",
    }
    for view in ("scheduling", "fqi"):
        for trajectory in results[view]["controllers"].values():
            assert "temperature_c" not in trajectory
            assert set(trajectory["requested_clock_mhz"]) == {fixed_clock}
            if expected_realized is not None:
                assert set(trajectory["realized_clock_mhz"]) == {
                    float(expected_realized)
                }
    assert "mean_waiting_time_s" in results["fqi"]["metrics"]["exact_dp"]
    assert set(results["scheduling"]["policy_slices"]) == {"0", "1", "2", "3", "4"}
    for payload in results["open_loop"]["controllers"].values():
        assert payload["plan_dt_s"] == pytest.approx(1.0)
        first_plan = next(plan for plan in payload["planned_clock_mhz"] if plan)
        assert len(first_plan) == 60


def test_manifest_thermal_fit_is_used_and_required_for_measured_profiles() -> None:
    profile = load_profile(DATA / "l4_profile.csv")
    profile_plant = plant_from_profile_manifest(profile)
    if profile.profile_status == "engineering_proxy_not_measured":
        assert not profile.is_measured
        assert profile_plant.thermal_time_constant_s == pytest.approx(35.0)
        assert profile_plant.thermal_resistance_c_per_w == pytest.approx(0.55)
        assert profile_plant.ambient_temperature_c == pytest.approx(25.0)
    elif profile.profile_status == "measured_l4":
        assert profile.is_measured
        assert profile.manifest["status"] == "complete"
        assert profile_plant.thermal_time_constant_s == pytest.approx(
            float(profile.manifest["thermal_time_constant_s"])
        )
        assert profile_plant.thermal_resistance_c_per_w == pytest.approx(
            float(profile.manifest["thermal_resistance_c_per_w"])
        )
        assert profile_plant.ambient_temperature_c == pytest.approx(
            float(profile.manifest["fitted_ambient_temperature_c"])
        )
    else:
        pytest.fail(f"unexpected inference profile status: {profile.profile_status!r}")

    fitted_manifest = {
        **profile.manifest,
        "thermal_time_constant_s": 48.0,
        "thermal_resistance_c_per_w": 0.31,
        "fitted_ambient_temperature_c": 23.5,
    }
    measured = replace(
        profile,
        profile_status="measured_l4",
        manifest=fitted_manifest,
        measurement_validated=True,
    )
    measured_plant = plant_from_profile_manifest(measured)
    assert measured_plant.thermal_time_constant_s == pytest.approx(48.0)
    assert measured_plant.thermal_resistance_c_per_w == pytest.approx(0.31)
    assert measured_plant.ambient_temperature_c == pytest.approx(23.5)

    incomplete = replace(
        measured,
        manifest={
            key: value
            for key, value in fitted_manifest.items()
            if key != "thermal_time_constant_s"
        },
    )
    with pytest.raises(ValueError, match="fitted thermal parameters"):
        plant_from_profile_manifest(incomplete)

    unphysical = replace(
        measured,
        manifest={
            **fitted_manifest,
            "fitted_ambient_temperature_c": 80.0,
            "thermal_limit_c": 75.0,
        },
    )
    with pytest.raises(ValueError, match="must exceed ambient"):
        plant_from_profile_manifest(unphysical)
