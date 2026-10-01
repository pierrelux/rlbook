from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from brownian_mppi import (PassageProblem, scalar_control, scalar_benchmark,
    brownian_control_variance, analytic_slit,narrow_slit_control,
    solve_reference,sample_path_integral,paired_trials,terminal_desirability)


def test_brownian_scaling_preserves_physical_terminal_variance():
    nu,T=.4,2.0
    for h in [.01,.1,.25]:
        assert round(T/h)*h*h*brownian_control_variance(nu,h)==pytest.approx(nu*T)


def test_scalar_terminal_quadratic_matches_exact_projection():
    result=scalar_benchmark(samples=100000)
    assert result["corrected_mean"]==pytest.approx(.8,abs=.008)
    assert result["uncorrected_mean"]==pytest.approx(.88,abs=.008)
    assert scalar_control(0,1)==pytest.approx(.8)
    assert scalar_control(.7,0)==pytest.approx(1.2)


def test_slit_closed_form_matches_independent_quadrature_and_gradient():
    x,t,nu=.3,.4,.7
    intervals=((-1.3,-.7),(.7,1.3))
    p=PassageProblem(nu=nu)
    v,u=analytic_slit(x,t,nu=nu,intervals=intervals)
    psi=sum(quad(lambda y: norm.pdf(y,x,np.sqrt(nu*(2-t)))*terminal_desirability(y,1,p),a,b,epsabs=1e-12)[0] for a,b in intervals)
    assert v==pytest.approx(-nu*np.log(psi),abs=1e-10)
    eps=1e-5
    vp,_=analytic_slit(x+eps,t,nu=nu,intervals=intervals)
    vm,_=analytic_slit(x-eps,t,nu=nu,intervals=intervals)
    assert u==pytest.approx(-(vp-vm)/(2*eps),abs=1e-8)


def test_symmetric_slit_delays_commitment_and_has_narrow_limit():
    assert narrow_slit_control(.1,2,1)<0
    assert narrow_slit_control(.1,.5,1)>0
    _,u=analytic_slit(.3,.4,intervals=((-1.00001,-.99999),(.99999,1.00001)))
    assert u==pytest.approx(narrow_slit_control(.3,1.6,1),abs=2e-6)


def test_numerical_reference_converges_and_matches_unobstructed_scalar():
    free=PassageProblem(penalty=0)
    solution=solve_reference(free)
    for step in [0,40,100]:
        expected=scalar_control(.3,free.final_time-step*free.dt,free.r,free.kappa,0)
        assert solution.action(.3,step)==pytest.approx(expected,abs=2e-5)
    p=PassageProblem()
    coarse=solve_reference(p,dx=.01)
    fine=solve_reference(p,dx=.005)
    expanded=solve_reference(p,dx=.005,extent=8)
    assert abs(coarse.action(p.x0,0)-fine.action(p.x0,0))<.04
    assert fine.action(p.x0,0)==pytest.approx(expanded.action(p.x0,0),abs=.01)


def test_importance_sampler_estimates_reference_and_is_reproducible():
    p=PassageProblem()
    controls=[]
    for seed in range(12):
        result=sample_path_integral(p.x0,0,p,np.random.default_rng(seed),samples=16384)
        controls.append(result["action"])
        assert not result["failed"]
        assert 1<=result["ess"]<=16384
    expected=solve_reference(p).action(p.x0,0)
    assert np.mean(controls)==pytest.approx(expected,abs=.12)
    first=sample_path_integral(p.x0,0,p,np.random.default_rng(123),512)
    second=sample_path_integral(p.x0,0,p,np.random.default_rng(123),512)
    assert first==second


def test_trial_controllers_share_actual_noise_but_not_planning_noise():
    p=PassageProblem(dt=.05)
    outputs=paired_trials(p,trials=2,samples=128)
    innovations=[]
    for result in outputs.values():
        innovations.append(np.diff(result["states"],axis=1)-p.dt*result["controls"])
    np.testing.assert_allclose(innovations[0],innovations[1],atol=1e-14)
    np.testing.assert_allclose(innovations[0],innovations[2],atol=1e-14)
    other=paired_trials(p,trials=2,samples=256)
    np.testing.assert_allclose(outputs["mean_dynamics"]["states"],other["mean_dynamics"]["states"])


def test_noise_changes_narrow_versus_wide_preference():
    low=PassageProblem(nu=.015)
    high=PassageProblem(nu=.15)
    assert solve_reference(low).action(low.x0,0)>0
    assert solve_reference(high).action(high.x0,0)<0
