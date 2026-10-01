"""Conservation, forcing-time and artifact checks for the prescribed plume."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from thermal_reactor import (ReactorConfig, velocity_profile, transport_rhs,
                             rhs_with_inlet, physics_step, field_metrics, valid_fields)
from thermal_plume import (PLUME_CONFIG, plume_step, plume_inlet, plume_controls,
                           pulse_envelope, validate_assets)


def cfg(**kw):
    return replace(PLUME_CONFIG, nx=12, ny=8, **kw)


def state(config, T=900., c=.4):
    return jnp.stack([jnp.full((config.ny, config.nx), T), jnp.full((config.ny, config.nx), c)])


def test_shear_flux_balance_and_diffusive_boundary_conservation():
    config = cfg()
    rng = np.random.default_rng(813)
    x = rng.uniform(.1, .9, (config.ny, config.nx)).astype('float32')
    inlet = rng.uniform(.1, .9, config.ny).astype('float32')
    dx, dy = config.length / config.nx, config.width / config.ny
    change = np.asarray(transport_rhs(x, inlet, config))
    expected = dy * np.sum(np.asarray(velocity_profile(config)) * (inlet - x[:, -1]))
    assert change.sum() * dx * dy == pytest.approx(expected, abs=3e-7)
    assert np.all(np.asarray(velocity_profile(config)) > 0)


def test_explicit_inlet_uniform_field_preserved_under_shear():
    config = cfg(loss=0)
    x = state(config, c=0)
    np.testing.assert_allclose(transport_rhs(x[0], jnp.full(config.ny, 900.), config), 0, atol=1e-5)
    derivative = rhs_with_inlet(x, jnp.zeros(4), jnp.full(config.ny, 900.), config)
    np.testing.assert_allclose(derivative[0], 0, atol=1e-5)
    np.testing.assert_allclose(derivative[1, :, 1:], 0, atol=1e-5)
    assert np.all(np.asarray(derivative[1, :, 0]) > 0)  # fresh reactant enters


def test_uniform_default_has_original_velocity_and_flux_metric():
    config = ReactorConfig(nx=12, ny=8)
    np.testing.assert_allclose(velocity_profile(config), .2)
    x = np.asarray(state(config)).copy()
    assert field_metrics(x)['outlet_conversion'] == pytest.approx(field_metrics(x, config)['outlet_conversion'])
    config = replace(config, velocity_shear=.2)
    x[1] = np.linspace(0, 1, config.ny)[:, None] ** 2
    expected = 1 - np.average(x[1, :, -1], weights=np.asarray(velocity_profile(config)))
    assert field_metrics(x, config)['outlet_conversion'] == pytest.approx(expected)
    assert abs(expected - field_metrics(x)['outlet_conversion']) > .001


def test_pulse_support_and_heater_switches():
    for t in (-1., 0., 6., 14., 80.):
        assert float(pulse_envelope(t)) == 0
    assert float(pulse_envelope(10.)) == pytest.approx(1.)
    np.testing.assert_allclose(plume_controls(5.99, True), .48)
    np.testing.assert_allclose(plume_controls(6., True), [.6, .48, .48, .48])
    np.testing.assert_allclose(plume_controls(18., True), .48)
    c = cfg()
    np.testing.assert_allclose(plume_inlet(6., True, c), plume_inlet(6., False, c))
    assert np.min(plume_inlet(10., True, c) - plume_inlet(10., False, c)) < -90


def test_switch_splitting_and_stage_time_forcing():
    config = cfg()
    x = state(config)
    # A caller interval crossing either heater switch must match explicit splitting.
    for t, switch in ((5.9, 6.), (17.9, 18.)):
        a = plume_step(x, t, .2, True, True, config)
        b = plume_step(plume_step(x, t, switch - t, True, True, config), switch, t + .2 - switch, True, True, config)
        np.testing.assert_allclose(a, b, atol=3e-4)
    # The smooth inlet varies during this interval; stage-time integration
    # converges under halving and does not freeze the initial inlet.
    a = plume_step(x, 7., 1., True, False, config)
    b = plume_step(x, 7., 1., True, False, replace(config, max_dt=.05))
    np.testing.assert_allclose(a, b, atol=.02)
    assert np.max(np.abs(np.asarray(a[0] - plume_step(x, 7., 1., False, False, config)[0]))) > 1


def test_plume_batches_reaction_bounds_and_uniform_wrapper_parity():
    config = cfg()
    x = state(config)
    batched = plume_step(jnp.stack([x, x]), 9., .25, jnp.array([True, False]), jnp.array([False, True]), config)
    for i, (pulse, boost) in enumerate(((True, False), (False, True))):
        np.testing.assert_allclose(batched[i], plume_step(x, 9., .25, pulse, boost, config), atol=3e-4)
    assert valid_fields(batched)
    # No pulse/boost is the original constant-forcing physical transition.
    np.testing.assert_allclose(plume_step(x, 0., 2., False, False, config), physics_step(x, jnp.full(4, .48), 0., 2., config), atol=4e-4)


def test_exported_plume_references_motion_and_timestamps():
    assets = Path(__file__).resolve().parents[1] / 'interactive' / 'thermal-plume-data'
    report = validate_assets(assets)
    assert report['arrays_checked'] == 4 and report['frames_per_run'] == 321
