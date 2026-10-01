from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from thermal_recovery import (P, seed_knots, interpolate_knots, decode, proposal_center,
                              tracking_error, control_cost, RecoveryController,
                              recorded_rollout, target_and_initial, validate_tape)
from thermal_reactor import ReactorConfig, physics_step
from thermal_plume import PLUME_CONFIG, plume_step
from mppi_control import gaussian_mppi_update


def small():
    return replace(PLUME_CONFIG, nx=8, ny=6)


def target(config):
    return np.stack([np.full((config.ny, config.nx), 900.), np.full((config.ny, config.nx), .4)]).astype(np.float32)


def test_tracking_units_and_conversion_sign():
    x = target(small())
    assert float(tracking_error(x, x)) == 0
    y = x.copy(); y[0] += 20; y[1] -= .1
    assert float(tracking_error(y, x)) == pytest.approx(2., abs=2e-6)
    y[1] = x[1] + .1
    assert float(tracking_error(y, x)) == pytest.approx(2., abs=2e-6)
    assert float(control_cost(jnp.full(4, .46), jnp.full(4, .46))) == 0
    assert float(control_cost(jnp.full(4, .51), jnp.full(4, .46))) == pytest.approx(.015, abs=2e-7)


def test_seed_decoder_and_bounded_latent_coordinates():
    seed = seed_knots()
    np.testing.assert_allclose(decode(np.zeros((6,4)), seed), interpolate_knots(seed), atol=1e-7)
    np.testing.assert_allclose(decode(np.full((6,4), 1e6), seed), 1)
    np.testing.assert_allclose(decode(np.full((6,4), -1e6), seed), 0)
    assert decode(np.zeros((3,6,4)), seed).shape == (3,20,4)
    tape = interpolate_knots(seed)
    assert tape[0,0] == pytest.approx(.50)
    np.testing.assert_allclose(tape[-1], .46)


def test_fixed_reference_correction_uses_latent_density_once():
    # With constant costs, a shifted proposal projects back toward the fixed
    # standard-normal reference, not its own sampling mean.
    rng = np.random.default_rng(903)
    samples = rng.normal(.3, .7, (60000,1))
    result, weights, _ = gaussian_mppi_update(samples, np.zeros(len(samples)), np.array([.3]), .49, .01)
    assert abs(float(result[0])) < .012
    assert weights.sum() == pytest.approx(1)


def test_all_invalid_samples_preserve_exact_tape_not_knot_approximation():
    config = small(); p = replace(P, candidates=8, iterations=2)
    controller = RecoveryController(target(config), config, p)
    # Deliberately outside the six-knot subspace.
    tape = controller.incumbent.copy(); tape[3,1] += .011
    controller.incumbent = tape.copy()
    assert not np.allclose(interpolate_knots(proposal_center(tape,p),p),tape)
    def score(x, tapes):
        n=len(tapes)
        return (np.full(n,np.inf),np.ones(n),np.zeros(n,dtype=bool))
    with patch.object(controller,'validate',return_value=(True,{})), patch.object(controller,'score',side_effect=score):
        action,record,_=controller.plan(target(config))
    np.testing.assert_array_equal(controller.last_plan,tape)
    np.testing.assert_array_equal(action,tape[0])
    np.testing.assert_array_equal(controller.incumbent[:-1],tape[1:])
    assert record['rejected']==2 and not record['planning_failure']


def test_invalid_incumbent_stops_before_sampling_or_execution():
    config=small();controller=RecoveryController(target(config),config)
    with patch.object(controller,'validate',return_value=(False,{'reason':'failed'})), patch.object(controller,'score') as scorer:
        action,record,_=controller.plan(target(config))
    assert action is None and record['planning_failure']
    scorer.assert_not_called()


def test_failed_fine_validation_never_replaces_feasible_incumbent():
    config=small();p=replace(P,candidates=8,iterations=1);ctl=RecoveryController(target(config),config,p)
    tape=ctl.incumbent.copy();calls=[0]
    def score(x,tapes):
        calls[0]+=1;n=len(tapes)
        cost=10. if calls[0]==1 else 0.
        return np.full(n,cost),np.full(n,cost),np.ones(n,dtype=bool)
    with patch.object(ctl,'score',side_effect=score),patch.object(ctl,'validate',side_effect=[(True,{})]+[(False,{})]*4):
        _,record,_=ctl.plan(target(config))
    np.testing.assert_array_equal(ctl.last_plan,tape)
    assert not record['iterations'][0]['accepted']


def test_held_inputs_preserve_existing_plume_and_normal_inlet_contract():
    config=small();x=target(config);u=np.array([.43,.48,.44,.49])
    got=plume_step(x,14.,2.,True,False,config,jnp.asarray(u))
    expected=physics_step(x,u,0.,2.,config)
    np.testing.assert_allclose(got,expected,atol=5e-4)
    automatic=plume_step(x,8.,.25,True,True,config)
    explicit=plume_step(x,8.,.25,True,False,config,jnp.array([.60,.48,.48,.48]))
    np.testing.assert_allclose(automatic,explicit,atol=5e-4)


def test_recorded_batch_cadence_matches_individuals():
    config=small();x=target(config)
    us=jnp.array([[[.46,.47,.45,.48],[.49,.46,.44,.48]],[[.47,.46,.45,.48],[.48,.46,.44,.48]]])
    together=recorded_rollout(jnp.stack([x,x]),us,config)
    assert together.shape==(17,2,2,6,8)
    for j in range(2):
        single=recorded_rollout(jnp.asarray(x),us[:,j],config)
        np.testing.assert_allclose(together[:,j],single,atol=5e-4)


def test_physical_initial_state_and_feasible_seed():
    desired,initial=target_and_initial()
    assert np.max(np.abs(desired-initial))>10
    ok,check=validate_tape(initial,interpolate_knots(seed_knots()))
    assert ok and check['duration_s']==80
    assert check['maximum_temperature_K']<1070
    assert check['minimum_outlet_conversion']>.95
