from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from thermal_reactor import (ReactorConfig, physics_step, rhs, transport_rhs, heater_masks,
                             random_streams, ou_forecasts, valid_fields)
from thermal_surrogate import Surrogate, init_params
from thermal_control import PlannerConfig, controls_from_latent, shifted_latent, MPPIController


def small(**kwargs):
    return replace(ReactorConfig(), nx=8, ny=6, **kwargs)


def state(cfg, T=900., c=.4):
    return np.stack([np.full((cfg.ny, cfg.nx), T), np.full((cfg.ny, cfg.nx), c)]).astype(np.float32)


def test_finite_volume_boundary_flux_balance():
    cfg = small()
    rng = np.random.default_rng(12)
    field = rng.uniform(0, 1, (cfg.ny, cfg.nx)).astype(np.float32)
    inlet = rng.uniform(0, 1, cfg.ny).astype(np.float32)
    change = np.asarray(transport_rhs(field, inlet, cfg))
    dx, dy = cfg.length / cfg.nx, cfg.width / cfg.ny
    expected = cfg.velocity * dy * np.sum(inlet - field[:, -1])
    assert np.sum(change) * dx * dy == pytest.approx(expected, abs=2e-7)


def test_diffusion_conserves_mass_without_boundary_flux():
    cfg = small(velocity=0.)
    field = np.arange(48, dtype=np.float32).reshape(6, 8)
    assert float(transport_rhs(field, jnp.zeros(6), cfg).sum()) == pytest.approx(0., abs=1e-6)


def test_reaction_consumes_material_and_cools_by_matching_amount():
    cfg = small(velocity=0., diffusivity=0., loss=0.)
    before = state(cfg)
    after = np.asarray(physics_step(before, np.zeros(4), 0., 1., cfg))
    assert np.all(after[1] < before[1]) and np.all(after[1] > 0)
    assert np.all(after[0] < before[0])
    np.testing.assert_allclose(after[0] - cfg.reaction_cooling * after[1],
                               before[0] - cfg.reaction_cooling * before[1], atol=1e-3)


def test_uniform_inert_field_is_preserved():
    cfg = small(velocity=0., loss=0.)
    before = state(cfg, c=0.)
    np.testing.assert_allclose(physics_step(before, np.zeros(4), 0., 2., cfg), before, atol=1e-5)


def test_batched_transition_matches_individual_and_refinement():
    cfg = small()
    before = np.stack([state(cfg, 920., .7), state(cfg, 1000., .2)])
    u = np.array([[.3, .4, .5, .6], [.5, .3, .4, .6]], np.float32)
    gust = np.array([-10., 5.], np.float32)
    together = np.asarray(physics_step(before, u, gust, 2., cfg))
    for i in range(2):
        np.testing.assert_allclose(together[i], physics_step(before[i], u[i], gust[i], 2., cfg), atol=5e-4)
    fine = np.asarray(physics_step(before, u, gust, 2., replace(cfg, max_dt=.05)))
    assert np.max(abs(together[:, 0] - fine[:, 0])) < .05
    assert valid_fields(together)


def test_heating_zones_partition_the_domain():
    masks = np.asarray(heater_masks())
    assert masks.shape == (4, 32, 64)
    np.testing.assert_array_equal(masks.sum(0), 1)
    assert ReactorConfig().state_dim == 4096


def test_observation_and_future_noise_are_separate():
    physical_a, proposal_a, forecast_a = random_streams(99)
    physical_b, proposal_b, forecast_b = random_streams(99)
    ou_forecasts(10., 20, 50, forecast_a)
    proposal_a.normal(size=500)
    np.testing.assert_array_equal(physical_a.normal(size=20), physical_b.normal(size=20))
    mean = ou_forecasts(10., 20, 1, forecast_b, mean_only=True)[:, 0]
    np.testing.assert_allclose(mean, 10 * np.exp(-np.arange(20) * 2 / 20), rtol=1e-6)
    forecast = ou_forecasts(10., 20, 5, forecast_a)
    np.testing.assert_array_equal(forecast[0], 10)
    assert not np.all(forecast[1] == forecast[1, 0])


def test_bounded_mapping_is_applied_after_latent_sampling():
    p = PlannerConfig()
    tape = controls_from_latent(np.zeros((6, 4)))
    np.testing.assert_allclose(tape, .48, atol=1e-7)
    assert tape.shape == (20, 4)
    extreme = controls_from_latent(np.full((3, 6, 4), 100.))
    assert np.all((extreme >= 0) & (extreme <= 1))
    np.testing.assert_allclose(shifted_latent(tape), 0., atol=1e-6)


def dummy_model():
    cfg = small()
    return Surrogate(init_params(), {"mean": jnp.array([900., .5]), "std": jnp.array([100., .3])}, cfg)


def test_failed_batch_keeps_valid_incumbent():
    ctl = MPPIController(dummy_model(), PlannerConfig(candidates=8))
    ctl.score = lambda x, tapes, forecast: np.full(len(tapes), np.inf)
    with patch("thermal_control.check_plan", return_value=(True, None)):
        action, record, _ = ctl.plan(state(ctl.config), 0.)
    np.testing.assert_allclose(action, .48)
    assert record["rejected"] == 2 and record["accepted"] == 0
    assert not record["planning_failure"]


def test_missing_feasible_incumbent_stops_before_execution():
    ctl = MPPIController(dummy_model())
    with patch("thermal_control.check_plan", return_value=(False, None)):
        action, record, _ = ctl.plan(state(ctl.config), 0.)
    assert action is None and record["planning_failure"]


def test_invalid_proposed_mean_is_not_executed():
    ctl = MPPIController(dummy_model(), PlannerConfig(candidates=8))
    single_costs = iter([10., 0., 0.])
    ctl.score = lambda x, tapes, forecast: (np.array([next(single_costs)]) if len(tapes) == 1
                                           else np.arange(len(tapes), dtype=float))
    # Both proposals look cheaper, so the PDE check must prevent acceptance.
    # Incumbent valid; both weighted updates fail the PDE constraint check.
    with patch("thermal_control.check_plan", side_effect=[(True, None), (False, None), (False, None)]):
        action, record, _ = ctl.plan(state(ctl.config), 0.)
    np.testing.assert_allclose(action, .48)
    assert record["rejected"] == 2


def test_checkpoint_roundtrip_and_interval_contract(tmp_path):
    model = dummy_model(); x = state(model.config)
    model.save(tmp_path / "m.npz", {"seed": 10})
    restored, metadata = Surrogate.load(tmp_path / "m.npz")
    assert metadata["seed"] == 10
    np.testing.assert_array_equal(model.step(x, np.ones(4) * .48, 0.), restored.step(x, np.ones(4) * .48, 0.))
    with pytest.raises(ValueError):
        restored.step(x, np.ones(4), 0., elapsed=1.)


def test_surrogate_fraction_projection_is_explicit_and_temperature_is_unbounded():
    from thermal_surrogate import predict
    model = dummy_model()
    params = list(model.params)
    params[-1] = {**params[-1], 'b': jnp.array([2., 5.])}
    x = state(model.config)
    raw = np.asarray(predict(tuple(params), model.norm, x, np.full(4, .48), 0., model.config, False))
    bounded = np.asarray(predict(tuple(params), model.norm, x, np.full(4, .48), 0., model.config))
    assert raw[1].max() > 1
    np.testing.assert_array_equal(bounded[1], 1.)
    np.testing.assert_array_equal(raw[0], bounded[0])
    assert bounded[0].max() > model.config.temperature_limit


def test_physics_and_surrogate_share_transition_interface():
    from thermal_reactor import Physics
    model = dummy_model(); plant = Physics(model.config)
    x = np.stack([state(model.config), state(model.config, T=930.)])
    for transition in (model, plant):
        following = np.asarray(transition.step(x, np.full((2, 4), .48), np.zeros(2), 2.))
        assert following.shape == x.shape and np.isfinite(following).all()


def test_fine_diagnostics_have_correct_times_and_match_endpoint():
    from thermal_reactor import fine_rollout_diagnostics, physics_rollout, field_metrics
    cfg = small(); x = state(cfg)
    u, g = np.full((2, 4), .48), np.zeros(2)
    fine = np.asarray(fine_rollout_diagnostics(x, u, g, cfg))
    assert fine.shape == (40, 2)
    coarse = field_metrics(physics_rollout(x, u, g, cfg))
    np.testing.assert_allclose(fine[19::20, 0], coarse['peak_temperature_K'], atol=.002)
    np.testing.assert_allclose(fine[19::20, 1], coarse['outlet_conversion'], atol=1e-5)


def test_metal_backend_matches_nontrivial_jax_fields_and_costs():
    import os
    if os.environ.get('THERMAL_TEST_METAL') != '1':
        pytest.skip('Opt-in Apple GPU check: THERMAL_TEST_METAL=1')
    from thermal_metal import MetalScorer
    from thermal_control import rollout_scores
    model = dummy_model()
    # Nonzero final layer exercises all convolutions and the residual scaling.
    params = list(model.params)
    params[-1] = {'w': jnp.ones_like(params[-1]['w']) * .00001, 'b': jnp.array([-.001, -.001])}
    model.params = tuple(params)
    p = PlannerConfig(horizon=5)
    gpu = MetalScorer(model, p)
    x = state(model.config, T=1080., c=.1)  # Both thermal and quality penalties are active.
    u, g = np.full((2, 4), .48, np.float32), np.array([0, 5], np.float32)
    xx = np.repeat(x[None], 2, axis=0)
    np.testing.assert_allclose(gpu.predict(xx, u, g), model.step(xx, u, g), atol=.001, rtol=1e-6)
    tapes = controls_from_latent(np.random.default_rng(51).normal(0, .2, (3, 6, 4)), p)
    f = ou_forecasts(0, 5, 2, np.random.default_rng(10), model.config)
    cpu = np.asarray(rollout_scores(model.params, model.norm, x, tapes, f, u[0], model.config, p))
    metal = gpu.score(x, tapes, f, u[0])
    assert np.isfinite(cpu).all()
    np.testing.assert_allclose(cpu, metal, atol=.001, rtol=1e-5)


def test_peak_underprediction_compares_matching_times():
    from thermal_surrogate import peak_underprediction
    cfg = small()
    truth = np.stack([state(cfg, 1050.), state(cfg, 1000.)])
    pred = np.stack([state(cfg, 1040.), state(cfg, 1100.)])
    # An overprediction later cannot conceal an underprediction now.
    assert peak_underprediction(truth, pred) == 10.


def test_scenario_costs_are_averaged_before_exponential_weighting():
    from thermal_control import rollout_scores
    from mppi_control import normalized_weights
    cfg = small(inlet_temperature=1100.)
    model = dummy_model(); model.config = cfg
    p = PlannerConfig(horizon=3)
    x = state(cfg, T=1000., c=.02)
    tapes = np.stack([np.full((3, 4), .4), np.full((3, 4), .6)]).astype(np.float32)
    scenarios = np.tile(np.array([-100., 100.]), (3, 1)).astype(np.float32)
    def score(g):
        return np.asarray(rollout_scores(model.params, model.norm, x, tapes, g,
                                         np.full(4, .48), cfg, p, True))
    individual = np.stack([score(scenarios[:, i:i + 1]) for i in range(2)])
    ensemble = score(scenarios)
    np.testing.assert_allclose(ensemble, individual.mean(0), rtol=1e-5)
    weights, _ = normalized_weights(ensemble, 1.)
    scenario_weights = np.mean([normalized_weights(c, 1.)[0] for c in individual], axis=0)
    # Averaging independently normalized scenario weights solves another objective.
    assert np.max(abs(weights - scenario_weights)) > .01


def test_reaction_stability_bound_handles_an_excessive_requested_timestep():
    cfg = small(velocity=0., diffusivity=0., loss=0., max_dt=5.)
    before = state(cfg, T=1500., c=1.)
    after = np.asarray(physics_step(before, np.zeros(4), 0., 2., cfg))
    assert valid_fields(after) and after[1].max() < .1
    np.testing.assert_allclose(after[0] - cfg.reaction_cooling * after[1],
                               before[0] - cfg.reaction_cooling * before[1], atol=.003)


def test_uniform_transport_matches_both_boundaries():
    cfg = small()
    change = np.asarray(transport_rhs(jnp.full((cfg.ny, cfg.nx), 700.), jnp.full(cfg.ny, 700.), cfg))
    np.testing.assert_array_equal(change, 0.)
