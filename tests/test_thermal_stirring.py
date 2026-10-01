from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import sys
import numpy as np
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from thermal_stirring import (CONFIG, PLAN, face_velocity, flow_basis, inlet_temperature,
    transport_rhs, transport_fluxes, stirring_step, warm_state, physical_checks,
    mixing_error, decode, filter_latent, recorded_rollout, StirringController)
from mppi_control import gaussian_mppi_update


def small():
    return replace(CONFIG,nx=16,ny=8)


def state(cfg):
    return np.stack([np.full((cfg.ny,cfg.nx),1000.), np.full((cfg.ny,cfg.nx),.01)]).astype(np.float32)


def test_discrete_divergence_boundary_flux_and_reversibility():
    cfg=CONFIG
    a=jnp.array([[1.,0,0],[0,-1.,0],[.4,.3,-.5]])
    u,v=map(np.asarray,face_velocity(a,cfg));ub,vb=flow_basis(cfg)
    div=np.diff(u,axis=-1)/(cfg.length/cfg.nx)+np.diff(v,axis=-2)/(cfg.width/cfg.ny)
    assert np.max(np.abs(div))<1e-6
    np.testing.assert_allclose(u[:,:,(0,-1)],np.broadcast_to(ub[0,:,(0,-1)].T,u[:,:,(0,-1)].shape),atol=1e-7)
    np.testing.assert_allclose(v[:,(0,-1),:],0,atol=1e-7)
    un,vn=face_velocity(-a,cfg)
    np.testing.assert_allclose((u+un)/2,np.broadcast_to(ub[0],u.shape),atol=1e-7)
    np.testing.assert_allclose((v+vn)/2,0,atol=1e-7)


def test_constant_field_and_conservative_open_boundary_balance():
    cfg=small();u,v=face_velocity(jnp.array([.7,-.4,.3]),cfg)
    f=jnp.full((cfg.ny,cfg.nx),.3)
    np.testing.assert_allclose(transport_rhs(f,jnp.full(cfg.ny,.3),u,v,cfg),0,atol=3e-7)
    f=jnp.asarray(np.random.default_rng(91).uniform(.1,.9,f.shape),jnp.float32)
    inlet=jnp.linspace(.1,.9,cfg.ny)
    fx,fy=map(np.asarray,transport_fluxes(f,inlet,u,v,cfg))
    d=np.asarray(transport_rhs(f,inlet,u,v,cfg))
    dx,dy=cfg.length/cfg.nx,cfg.width/cfg.ny
    flux=dy*np.sum(fx[:,0]-fx[:,-1])+dx*np.sum(fy[0]-fy[-1])
    assert float(d.sum()*dx*dy)==pytest.approx(flux,abs=1e-6)


def test_inlet_timing_and_fixed_flux_weighted_mean():
    for t in (0.,8.,16.,24.,32.):
        f=np.asarray(inlet_temperature(t));w=flow_basis()[0][0,:,0]
        assert np.average(f,weights=w)==pytest.approx(900.,abs=1e-4)
    y=(np.arange(CONFIG.ny)+.5)*CONFIG.width/CONFIG.ny
    np.testing.assert_allclose(inlet_temperature(8.),900+70*np.cos(np.pi*y/2),atol=1e-4)
    np.testing.assert_allclose(inlet_temperature(24.),900-50*np.cos(np.pi*y/2),atol=1e-4)


def test_filter_projection_and_latent_proposal_correction():
    rng=np.random.default_rng(92);z=rng.normal(size=(40000,3,3))
    filtered=filter_latent(z)
    assert np.corrcoef(filtered[:,0,0],filtered[:,1,0])[0,1]==pytest.approx(.6,abs=.015)
    a=decode(z,np.zeros((3,3)))
    assert np.max(np.linalg.norm(a,axis=-1))<=1+1e-6
    reference=np.tile([.4,-.3,.2],(3,1))
    np.testing.assert_allclose(decode(np.zeros((3,3)),reference),reference,atol=1e-7)
    samples=rng.normal(.15,1.,(40000,3,3))
    mu,w,_=gaussian_mppi_update(samples,np.zeros(40000),np.full((3,3),.15),1.,.02)
    assert np.max(np.abs(mu))<.02
    assert w.sum()==pytest.approx(1.)


def test_transverse_objective_preserves_longitudinal_gradient():
    cfg=small();x=state(cfg)
    x[0]=np.linspace(900,1040,cfg.nx)[None,:]
    assert float(mixing_error(x,cfg))==pytest.approx(0.,abs=1e-9)
    x[0,:cfg.ny//2,-4:]+=20;x[0,cfg.ny//2:,-4:]-=20
    assert float(mixing_error(x,cfg))==pytest.approx(1.,abs=1e-6)


def test_reaction_bounds_jacket_bound_and_endothermic_cooling():
    cfg=small();x=state(cfg);x[0]=1040.;x[1]=1.
    y=np.asarray(stirring_step(x,jnp.array([1.,0,0]),8.,4.,cfg))
    assert np.isfinite(y).all() and y[0].max()<=1050+1e-3
    assert y[1].min()>=0 and y[1].max()<=1+1e-6
    no_cooling=np.asarray(stirring_step(x,jnp.array([1.,0,0]),8.,4.,replace(cfg,reaction_cooling=0.)))
    assert y[0].mean()<no_cooling[0].mean()


def test_batch_parity_stage_time_and_recording():
    cfg=small();x=state(cfg)
    actions=jnp.array([[.3,-.4,.1],[-.2,.5,.4]])
    batch=np.asarray(stirring_step(jnp.stack([x,x]),actions,7.,2.,cfg))
    for j in range(2):
        np.testing.assert_allclose(batch[j],stirring_step(x,actions[j],7.,2.,cfg),atol=8e-4)
    one=stirring_step(x,actions[0],7.,.25,cfg)
    split=stirring_step(stirring_step(x,actions[0],7.,.125,cfg),actions[0],7.125,.125,cfg)
    np.testing.assert_allclose(one,split,atol=.012)
    tape=jnp.stack([actions,actions])
    movie=np.asarray(recorded_rollout(jnp.stack([x,x]),tape,0.,cfg))
    assert movie.shape==(17,2,2,cfg.ny,cfg.nx)
    for j in range(2):
        np.testing.assert_allclose(movie[:,j],recorded_rollout(x,tape[:,j],0.,cfg),atol=.002)


def test_failed_batches_and_failed_fine_checks_retain_exact_tape():
    cfg=small();p=replace(PLAN,candidates=8,iterations=1);ctl=StirringController(cfg,p)
    ctl.incumbent[3]=[.2,.1,-.3];tape=ctl.incumbent.copy()
    def invalid(x,tapes,t):
        n=len(tapes);return np.full(n,np.inf),np.ones(n),np.zeros(n,bool)
    with patch.object(ctl,'validate',return_value=(True,{})),patch.object(ctl,'score',side_effect=invalid):
        _,rec,_=ctl.plan(state(cfg),0.)
    np.testing.assert_array_equal(ctl.last_plan,tape)
    assert rec['rejected']==1
    ctl=StirringController(cfg,p);calls=[0]
    def improving(x,tapes,t):
        calls[0]+=1;n=len(tapes);cost=10. if calls[0]==1 else 0.
        return np.full(n,cost),np.full(n,cost),np.ones(n,bool)
    with patch.object(ctl,'validate',side_effect=[(True,{})]+[(False,{})]*4),patch.object(ctl,'score',side_effect=improving):
        _,rec,_=ctl.plan(state(cfg),0.)
    np.testing.assert_array_equal(ctl.last_plan,0)
    assert rec['rejected']==1


def test_infeasible_incumbent_stops_before_sampling():
    ctl=StirringController(small(),replace(PLAN,candidates=8))
    with patch.object(ctl,'validate',return_value=(False,{})),patch.object(ctl,'score') as score:
        action,rec,_=ctl.plan(state(small()),0.)
    assert action is None and rec['planning_failure'];score.assert_not_called()


def test_optional_metal_pde_matches_jax_over_full_horizon():
    try:
        from thermal_stirring_metal import MetalStirringScorer
    except ImportError:
        pytest.skip('Optional Metal backend unavailable in this environment')
    from thermal_stirring import score_batch
    cfg=CONFIG;x=warm_state(cfg)
    tapes=decode(np.random.default_rng(913).normal(size=(8,20,3)),np.zeros((20,3)))
    tapes[0]=0;tapes[1]=[1.,0,0];tapes[2]=[-1.,0,0]
    scorer=MetalStirringScorer(cfg,PLAN)
    _,cost,ok,fields=scorer.run(x,tapes,8.,np.zeros(3),True)
    _,expected,feasible=score_batch(x,jnp.asarray(tapes),8.,jnp.zeros(3))
    np.testing.assert_allclose(cost,expected,atol=2e-5,rtol=2e-4)
    np.testing.assert_array_equal(ok,feasible)
    exact=np.asarray(recorded_rollout(jnp.broadcast_to(x,(8,)+x.shape),jnp.asarray(tapes.swapaxes(0,1)),8.))[-1]
    np.testing.assert_allclose(fields[:,0],exact[:,0],atol=.01,rtol=0)
    np.testing.assert_allclose(fields[:,1],exact[:,1],atol=1e-5,rtol=0)
