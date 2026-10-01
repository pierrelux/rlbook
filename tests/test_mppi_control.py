"""Numerical identities that detect missing corrections and unstable weights."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
from mppi_control import normalized_weights, gaussian_log_ratio, gaussian_mppi_update


def test_weight_offset_invariance_and_ess():
    w,ess=normalized_weights([10000,10001,10002],0.5)
    shifted,other=normalized_weights([0,1,2],0.5)
    np.testing.assert_allclose(w,shifted)
    assert ess==pytest.approx(other)
    assert 1<=ess<=3
    assert np.sum(w)==pytest.approx(1)


@pytest.mark.parametrize("temperature", [.5, 1., 2.])
def test_introductory_exponential_weight_ratios(temperature):
    costs = np.array([0., np.log(2), 4*np.log(2)])
    weights, _ = normalized_weights(costs, temperature)
    np.testing.assert_allclose(weights/weights[0], np.exp(-costs/temperature))
    if temperature == 1.:
        np.testing.assert_allclose(weights, np.array([16., 8., 1.])/25)


def test_introductory_weighting_limits():
    costs = np.array([0., 1., 3.])
    concentrated, _ = normalized_weights(costs, 1e-3)
    uniform, _ = normalized_weights(costs, 1e10)
    tied, _ = normalized_weights([0., 0., 1.], 1e-3)
    np.testing.assert_allclose(concentrated, [1., 0., 0.])
    np.testing.assert_allclose(uniform, np.full(3, 1/3))
    np.testing.assert_allclose(tied, [.5, .5, 0.])


def test_invalid_candidates_and_all_invalid_failure():
    w,ess=normalized_weights([1,np.nan,np.inf,-np.inf,2],1)
    assert np.all(w[[1,2,3]]==0)
    assert np.all(np.isfinite(w))
    with pytest.raises(ValueError,match="all candidate"):
        normalized_weights([np.nan,np.inf],1)
    for bad in [0,-1,np.inf,np.nan]:
        with pytest.raises(ValueError): normalized_weights([1,2],bad)
    with pytest.raises(ValueError): normalized_weights([1,2],1,[0])


def test_extreme_finite_cost_range_has_a_valid_winner():
    w,ess=normalized_weights([-1e308,1e308],1e-300)
    np.testing.assert_array_equal(w,[1,0])
    assert ess==1


def test_log_ratio_matches_independent_gaussian_densities():
    x=np.array([[.3,-2,1],[0,1,2]])
    mean=np.array([.4,-1,.7]);var=np.array([.5,2,1]);ref=np.array([-.2,.1,1])
    expected=np.sum(norm.logpdf(x,ref,np.sqrt(var))-norm.logpdf(x,mean,np.sqrt(var)),axis=1)
    np.testing.assert_allclose(gaussian_log_ratio(x,mean,var,ref),expected,atol=1e-14)


def test_gaussian_projection_recovers_known_tilt_from_nonzero_proposal():
    # Stratified normal quantiles suppress incidental Monte Carlo error.
    x=(.4+norm.ppf((np.arange(100000)+.5)/100000)*np.sqrt(.5))[:,None]
    cost=2*(x[:,0]-1)**2
    result,_,_=gaussian_mppi_update(x,cost,np.array([.4]),.5,.5)
    wrong,_=normalized_weights(cost,.5)
    assert result[0]==pytest.approx(.8,abs=2e-5)
    assert (wrong@x)[0]==pytest.approx(.88,abs=2e-5)


def test_invalid_samples_do_not_contaminate_update():
    samples=np.array([[0.,0.],[1.,1.],[np.nan,2.]])
    update,w,_=gaussian_mppi_update(samples,[0,0,0],[0,0],1,1)
    np.testing.assert_allclose(update,[.5,.5])
    assert w[-1]==0
