from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('openap')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from aircraft_mppi import (
    Aircraft, FlightConfig, MeanWind, altitude_profile, audit, conditional_gusts,
    evaluate_tapes, make_actual_gust, nominal_tapes, optimize, simulate,
    NoFeasiblePlan, validate_mean,
)
from mppi_control import normalized_weights


@pytest.fixture(scope='module')
def aircraft():
    return Aircraft(MeanWind())


def test_mean_wind_reproduces_grid_and_holds_pressure_boundaries():
    from openap import aero
    wind = MeanWind()
    with np.load(ROOT/'data/aircraft/era5_wind.npz') as d:
        i, j, k = 4, 12, 15
        # OpenAP's h_isa approximation is not the exact inverse of pressure.
        from scipy.optimize import brentq
        h = brentq(lambda z: aero.pressure(z)-d['pressure_hpa'][i]*100, 0, 20000)
        np.testing.assert_allclose(wind.at(d['lon_deg'][k], d['lat_deg'][j], h), [d['u_mps'][i,j,k], d['v_mps'][i,j,k]], atol=1e-6)
        np.testing.assert_allclose(wind.at(d['lon_deg'][k], d['lat_deg'][j], 0), [d['u_mps'][-1,j,k], d['v_mps'][-1,j,k]])


def test_numerical_model_batch_matches_scalar(aircraft):
    state = aircraft.initial.copy(); state[2] = 5000
    control = np.array([.65, 3., aircraft.heading])
    scalar = aircraft.step(state, control, 10., np.array([2., -1.]))
    batch = aircraft.step(np.broadcast_to(state,(3,2,5)), np.broadcast_to(control,(3,2,3)), np.full((3,1),10.), np.array([2.,-1.]))
    np.testing.assert_allclose(batch, np.broadcast_to(scalar,(3,2,5)))
    assert scalar[3] < state[3]
    assert scalar[2] == pytest.approx(5030.)


def test_gust_forecasts_are_conditional_and_do_not_share_actual_future():
    cfg = FlightConfig()
    observed = np.array([4.,-2.])
    forecasts = conditional_gusts(observed,8,4000,cfg,np.random.default_rng(11))
    np.testing.assert_allclose(forecasts[:,0],np.broadcast_to(observed,(4000,2)))
    rho = np.exp(-cfg.prediction_dt/cfg.gust_tau)
    np.testing.assert_allclose(forecasts[:,-1].mean(axis=0), rho**7*observed, atol=.2)
    assert not np.array_equal(forecasts[0], forecasts[1])
    before = forecasts.copy()
    make_actual_gust(cfg,999)
    np.testing.assert_array_equal(forecasts,before)


def test_scenario_averaging_precedes_exponential_weighting():
    costs = np.array([[0.,20.],[8.,8.]])
    weights, _ = normalized_weights(costs.mean(axis=1), 1.)
    assert weights[1] > weights[0]  # smaller expected cost wins.
    optimistic = np.exp(-costs).mean(axis=1)
    assert optimistic[0] > optimistic[1]  # weighting first would reverse ranking.


def test_reference_tape_flies_complete_leg_and_integrates_perturbations(aircraft):
    cfg = FlightConfig(samples=8,iterations=1,scenarios=2)
    params = np.random.default_rng(8).normal(size=(4,13))
    tape, dt, duration = nominal_tapes(aircraft,aircraft.initial,params,cfg,0,cfg.duration)
    np.testing.assert_allclose(np.sum(tape[:,:,1]*dt,axis=1),0.,atol=1e-10)
    # This is control construction, not state projection: heading/Mach changes
    # can still change terminal position and are assessed by actual rollouts.
    costs, diag = evaluate_tapes(aircraft,aircraft.initial,tape,dt,np.zeros((1,tape.shape[1],2)),cfg)
    assert np.isfinite(costs).all()
    assert np.max(diag['horizontal']) > 1.
    assert np.max(diag['vertical']) < 1e-8
    assert np.max(tape[:,:,1]) > 2.5
    assert np.min(tape[:,:,1]) < -2.5


def test_full_trip_and_frozen_have_identical_control_tapes(aircraft):
    cfg = FlightConfig(samples=8,iterations=1,scenarios=2)
    actual = make_actual_gust(cfg,cfg.seed+1)
    base = simulate(aircraft,cfg,'full_trip',actual)
    frozen = simulate(aircraft,cfg,'frozen',actual)
    np.testing.assert_array_equal(base['controls'],frozen['controls'])
    np.testing.assert_array_equal(base['dts'],frozen['dts'])
    assert np.linalg.norm(base['states'][-1,:2]-frozen['states'][-1,:2]) > 100
    report = audit(aircraft,base,cfg)
    assert report['arrival_pass']
    assert report['max_normalized_violation'] <= 1e-7
    assert report['fine_replay_position_difference_m'] < 1.
    assert report['cruise_samples'] > 0


def test_numerical_dynamics_match_original_openap_top_model():
    import casadi as ca
    from openap.top import CompleteFlight
    numerical = Aircraft()
    original = CompleteFlight('A320', 'CYUL', 'CYYZ', m0=.85)
    x, u = ca.MX.sym('x', 5), ca.MX.sym('u', 3)
    derivative = ca.Function('source_dynamics', [x, u], [original.xdot(x, u)])
    for h, mach, vs in [(1000,.4,5), (7000,.65,0), (2000,.45,-5)]:
        state = numerical.initial.copy(); state[2] = h
        control = np.array([mach, vs, numerical.heading])
        expected = np.asarray(derivative(state, control)).ravel()
        actual = numerical.velocity_and_fuel(state, control, np.zeros(2))
        # Numerical and symbolic ISA routines have small rounding differences.
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


def test_failed_updates_retain_the_validated_incumbent(aircraft, monkeypatch):
    import aircraft_mppi as module
    cfg = FlightConfig(samples=4, scenarios=2, iterations=1)
    expected, dt, duration = nominal_tapes(aircraft,aircraft.initial,np.zeros(13),cfg,0,cfg.duration,np.zeros(2))
    def fail(*args, **kwargs):
        raise ValueError('all candidate weights are invalid')
    monkeypatch.setattr(module, 'normalized_weights', fail)
    tape, actual_dt, actual_duration, diag = optimize(aircraft,aircraft.initial,np.zeros(2),cfg,0,cfg.duration,True,np.random.default_rng(1))
    np.testing.assert_array_equal(tape, expected[0])
    np.testing.assert_array_equal(actual_dt, dt[0])
    assert actual_duration == duration[0]
    assert diag['accepted'] == 0 and diag['incumbent_validated']
    assert diag['ess'] == 0


def test_no_unvalidated_initial_plan_is_returned(aircraft, monkeypatch):
    import aircraft_mppi as module
    monkeypatch.setattr(module, 'validate_mean', lambda *args: False)
    with pytest.raises(NoFeasiblePlan):
        optimize(aircraft,aircraft.initial,np.zeros(2),FlightConfig(),0,3600,False,np.random.default_rng(1))


def test_rejected_projected_mean_retains_incumbent(aircraft, monkeypatch):
    import aircraft_mppi as module
    checks = iter([True, False])
    monkeypatch.setattr(module, 'validate_mean', lambda *args: next(checks))
    cfg = FlightConfig(samples=4, iterations=1)
    tape, dt, _, diag = optimize(aircraft,aircraft.initial,np.zeros(2),cfg,0,3600,False,np.random.default_rng(1))
    expected, expected_dt, _ = nominal_tapes(aircraft,aircraft.initial,np.zeros(13),cfg,0,3600,np.zeros(2))
    np.testing.assert_array_equal(tape, expected[0])
    np.testing.assert_array_equal(dt, expected_dt[0])
    assert diag['accepted'] == 0


def test_validation_includes_boundary_from_last_executed_command(aircraft):
    cfg=FlightConfig()
    tape,dt,_=nominal_tapes(aircraft,aircraft.initial,np.zeros(13),cfg,0,3600,np.zeros(2))
    previous=tape[0,0].copy();previous[1]-=5
    assert not validate_mean(aircraft,aircraft.initial,tape[0],dt[0],np.zeros(2),cfg,previous)


def test_remaining_previous_plan_is_retained_when_new_reference_fails(aircraft, monkeypatch):
    import aircraft_mppi as module
    cfg=FlightConfig(samples=4,iterations=1)
    tape,dt,duration=nominal_tapes(aircraft,aircraft.initial,np.zeros(13),cfg,0,3600,np.zeros(2))
    checks=iter([False,True,False])
    monkeypatch.setattr(module,'validate_mean',lambda *args:next(checks))
    result,_,_,diag=optimize(aircraft,aircraft.initial,np.zeros(2),cfg,0,3600,False,np.random.default_rng(1),previous_plan=(tape[0],dt[0],duration[0]))
    np.testing.assert_array_equal(result,tape[0])
    assert diag['retained_previous_plan'] and diag['accepted']==0


def test_changing_unobserved_wind_leaves_first_decision_unchanged(aircraft, monkeypatch):
    import aircraft_mppi as module
    cfg=FlightConfig(samples=4,scenarios=2,iterations=1)
    first=make_actual_gust(cfg,10)
    second=first.copy();second[12:]+=100  # winds differ only from t=60 s onward.
    original=module.optimize
    def first_decision_only(plane,state,*args,**kwargs):
        if state[4]>=60:
            raise NoFeasiblePlan('End of causal test window')
        return original(plane,state,*args,**kwargs)
    monkeypatch.setattr(module,'optimize',first_decision_only)
    a=simulate(aircraft,cfg,'stochastic',first)
    b=simulate(aircraft,cfg,'stochastic',second)
    assert len(a['replans'])==len(b['replans'])==1
    np.testing.assert_array_equal(a['controls'],b['controls'])
    np.testing.assert_array_equal(a['states'],b['states'])


def test_recorded_fuel_matches_integrated_flow_and_mass(aircraft):
    import json
    path=ROOT/'artifacts/aircraft_mppi/full_trip.json'
    if not path.exists():
        pytest.skip('precomputed flight artifact is not present')
    run=json.loads(path.read_text())
    states=np.array(run['states']);controls=np.array(run['controls'])
    gusts=np.array(run['gusts']);dt=np.array(run['dts'])
    first=aircraft.velocity_and_fuel(states[:-1],controls,gusts)
    middle=states[:-1]+.5*dt[:,None]*first
    rates=-aircraft.velocity_and_fuel(middle,controls,gusts)[:,3]
    assert np.all(rates>0)
    np.testing.assert_allclose(rates*dt,states[:-1,3]-states[1:,3],atol=1e-10)
    assert np.sum(rates*dt)==pytest.approx(states[0,3]-states[-1,3],abs=1e-8)
