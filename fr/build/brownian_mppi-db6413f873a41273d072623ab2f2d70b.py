"""Matched-noise path-integral control and independent Brownian references.

The finite-width slit and finite-thickness passage constructions follow Kappen
(2005), sections 6.1--6.2; the finite penalties and numerical settings here are
teaching adaptations. All stochastic experiments separate planner and plant RNGs.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp, ndtr

from mppi_control import normalized_weights, gaussian_mppi_update


def brownian_control_variance(nu, dt):
    if nu <= 0 or dt <= 0:
        raise ValueError("diffusion variance and time step must be positive")
    return nu / dt


def scalar_control(x, remaining, r=1.0, kappa=4.0, target=1.0):
    return kappa * (target - np.asarray(x)) / (r + kappa * remaining)


def scalar_benchmark(seed=1907, samples=200000):
    """Ten-step terminal-quadratic benchmark, including an omitted-ratio ablation."""
    h, n, nu, r, kappa, target, mean = 0.1, 10, 0.5, 1.0, 4.0, 1.0, 0.4
    rng = np.random.default_rng(seed)
    v = mean + np.sqrt(nu / h) * rng.standard_normal((samples, n))
    cost = 0.5 * kappa * (h * v.sum(axis=1) - target)**2
    update, weights, ess = gaussian_mppi_update(v, cost, np.full(n, mean), nu/h, nu*r)
    wrong, _ = normalized_weights(cost, nu*r)
    uncorrected = wrong @ v
    return {"seed": seed, "samples": samples, "dt": h, "steps": n,
            "nu": nu, "r": r, "lambda": nu*r, "proposal_mean": mean,
            "corrected_mean": float(update.mean()), "uncorrected_mean": float(uncorrected.mean()),
            "corrected_first_control": float(update[0]), "analytic": 0.8,
            "uncorrected_analytic": 0.88, "ess": ess}


@dataclass(frozen=True)
class PassageProblem:
    nu: float = 0.15
    r: float = 1.0
    kappa: float = 1.0
    final_time: float = 3.0
    enter: float = 1.0
    leave: float = 2.0
    dt: float = 0.025
    x0: float = -0.3
    intervals: tuple = ((-0.18, 0.18), (-1.7, -0.7))
    penalty: float = 40.0

    @property
    def temperature(self):
        return self.nu * self.r

    @property
    def steps(self):
        return round(self.final_time / self.dt)

    def __post_init__(self):
        if min(self.nu, self.r, self.kappa, self.dt, self.final_time) <= 0:
            raise ValueError("positive diffusion, effort, terminal cost, time step and horizon required")
        if not (0 < self.enter < self.leave < self.final_time) or self.penalty < 0:
            raise ValueError("invalid passage times or penalty")
        for t in (self.enter, self.leave, self.final_time):
            if not np.isclose(t / self.dt, round(t / self.dt)):
                raise ValueError("passage times must lie on the time grid")
        if not self.intervals or any(a >= b for a,b in self.intervals):
            raise ValueError("passages must have positive width")


def in_passages(x, problem):
    x = np.asarray(x)
    return np.logical_or.reduce([(x >= a) & (x <= b) for a, b in problem.intervals])


def state_cost(x, step, problem):
    """Right-endpoint quadrature, active for enter < t <= leave."""
    active = round(problem.enter/problem.dt) < step <= round(problem.leave/problem.dt)
    return problem.penalty * (~in_passages(x, problem)) if active else np.zeros_like(x, dtype=float)


def terminal_desirability(x, remaining, problem):
    denom = problem.r + problem.kappa * remaining
    return np.sqrt(problem.r / denom) * np.exp(-problem.kappa * np.asarray(x)**2 / (2*problem.nu*denom))


def analytic_slit(x, time, *, nu=1.0, r=1.0, kappa=1.0, final_time=3.0,
                  slit_time=2.0, intervals=((-1.15,-0.85),(0.85,1.15)), penalty=np.inf):
    """Exact value/control for an instantaneous finite-width slit.

    ``penalty`` is a lump cost outside the openings. The infinite-penalty limit
    is Kappen's double slit. Integrating the terminal Gaussian leaves a Gaussian
    probability of lying in one of the openings at slit_time.
    """
    x = np.asarray(x, dtype=float)
    if time >= slit_time:
        raise ValueError("evaluate the analytic slit strictly before the slit")
    tau, total = slit_time-time, final_time-time
    a = r/kappa + final_time-slit_time
    factor = a/(a+tau)
    sd = np.sqrt(nu*tau*factor)
    m = factor*x
    probability, derivative = np.zeros_like(x), np.zeros_like(x)
    normal_pdf = lambda z: np.exp(-z*z/2)/np.sqrt(2*np.pi)
    for lower, upper in intervals:
        lo, hi = (lower-m)/sd, (upper-m)/sd
        # Reflect positive-tail intervals to avoid 1 - 1 cancellation.
        mass = np.where(lo > 0, ndtr(-lo)-ndtr(-hi), ndtr(hi)-ndtr(lo))
        probability += mass
        derivative += factor/sd*(normal_pdf(lo)-normal_pdf(hi))
    outside = np.exp(-penalty/(nu*r))
    gate = outside+(1-outside)*probability
    if np.any(gate <= 0):
        raise ValueError("slit probability underflow; evaluate nearer the openings")
    value = 0.5*r*x*x/(r/kappa+total) + 0.5*nu*r*np.log1p(kappa*total/r) -nu*r*np.log(gate)
    control = -x/(r/kappa+total) + nu*(1-outside)*derivative/gate
    return value, control


def narrow_slit_control(x, time_to_slit, nu, distance=1.0):
    """Infinitesimal symmetric-slit limit; u_x(0) changes sign at nu*T=a²."""
    return (distance*np.tanh(distance*np.asarray(x)/(nu*time_to_slit))-x)/time_to_slit


def sample_slit_control(x, time, rng, samples=4096, *, nu=1.0, r=1.0, kappa=1.0,
                        final_time=3.0, slit_time=2.0,
                        intervals=((-1.15,-.85),(.85,1.15)), penalty=20.0):
    """Rao--Blackwellized forward PI estimator for an instantaneous slit.

    Conditional on X_slit=y, the first Brownian increment has conditional mean
    (y-x)*dt/(slit_time-time). Integrating the intervening Brownian bridge and
    terminal Gaussian therefore gives a lower-variance exact control estimator.
    """
    tau=slit_time-time
    if tau<=0:
        return {"action":float(scalar_control(x,final_time-time,r,kappa,0)),"ess":float(samples)}
    means=np.array([x]+[(a+b)/2 for a,b in intervals])
    sd=np.sqrt(nu*tau)
    y=means[rng.integers(len(means),size=samples)]+sd*rng.standard_normal(samples)
    inside=np.logical_or.reduce([(y>=a)&(y<=b) for a,b in intervals])
    tail=final_time-slit_time
    costs=penalty*(~inside)+r*kappa*y*y/(2*(r+kappa*tail))
    logp=-0.5*((y-x)/sd)**2
    logq=logsumexp(-0.5*((y[:,None]-means)/sd)**2,axis=1)-np.log(len(means))
    weights,ess=normalized_weights(costs,nu*r,logp-logq)
    # This normalized expectation remains well defined for finite penalties.
    value=-nu*r*(logsumexp(-costs/(nu*r)+logp-logq)-np.log(samples))
    value+=0.5*nu*r*np.log1p(kappa*tail/r)
    return {"action":float(weights@(y-x)/tau),"ess":ess,"value":float(value)}


@dataclass
class ReferenceSolution:
    grid: np.ndarray
    value: np.ndarray
    control: np.ndarray
    problem: PassageProblem

    def action(self, x, step):
        return np.interp(x, self.grid, self.control[step])


def solve_reference(problem, dx=0.005, extent=6.0):
    """Backward Gaussian convolution of the discretized Feynman--Kac formula.

    The zero extension at +/-extent is a numerical boundary, checked by domain
    expansion. A derivative Gaussian kernel computes the same first-increment
    weighted estimator as the forward sampler, without Monte Carlo error.
    """
    grid = np.arange(-extent, extent+dx/2, dx)
    n = problem.steps
    psi = np.zeros((n+1, grid.size))
    control = np.zeros((n, grid.size))
    psi[-1] = terminal_desirability(grid, 0, problem)
    sigma = np.sqrt(problem.nu*problem.dt)/dx
    for j in range(n-1,-1,-1):
        next_psi = psi[j+1] * np.exp(-problem.dt*state_cost(grid, j+1, problem)/problem.temperature)
        psi[j] = gaussian_filter1d(next_psi, sigma, mode="constant", truncate=8.0)
        derivative = gaussian_filter1d(next_psi, sigma, order=1, mode="constant", truncate=8.0)/dx
        control[j] = problem.nu*derivative/np.maximum(psi[j],1e-300)
    value = -problem.temperature*np.log(np.maximum(psi,1e-300))
    return ReferenceSolution(grid,value,control,problem)


def route_means(x, step, problem):
    """Two open-loop Gaussian proposals ending at the passage exit.

    A fixed route component is chosen for an entire rollout. Its density is
    evaluated as a mixture of trajectory densities, not a mixture at each step.
    """
    enter, leave = round(problem.enter/problem.dt), round(problem.leave/problem.dt)
    times = np.arange(step+1,leave+1)*problem.dt
    centers = np.array([(a+b)/2 for a,b in problem.intervals])
    remaining = max(problem.enter-step*problem.dt, 4*problem.dt)
    fraction = np.minimum((times-step*problem.dt)/remaining,1)
    paths = x+(centers[:,None]-x)*fraction
    return np.diff(np.column_stack([np.full(centers.size,x),paths]),axis=1)


def sample_path_integral(x, step, problem, rng, samples=2048, keep_paths=False):
    """First-action estimate from forward importance-sampled Brownian futures.

    The terminal interval after the passage is integrated analytically. Sampling
    only concerns the nonlinear part; this variance reduction is exact. The
    proposal contains a zero-drift component, preserving coverage of failures.
    """
    leave = round(problem.leave/problem.dt)
    if step >= leave:
        action = float(scalar_control(x,problem.final_time-step*problem.dt,problem.r,problem.kappa,0))
        return {"action":action,"ess":float(samples),"max_weight":1/samples,"failed":False}
    if samples < 2:
        raise ValueError("at least two candidate paths required")
    means = route_means(x,step,problem)
    means = np.vstack([means, np.zeros(means.shape[1])])
    components = rng.integers(means.shape[0],size=samples)
    variance = problem.nu*problem.dt
    increments = means[components] + np.sqrt(variance)*rng.standard_normal((samples,means.shape[1]))
    paths = x+np.cumsum(increments,axis=1)
    steps = np.arange(step+1,leave+1)
    active = steps > round(problem.enter/problem.dt)
    costs = problem.penalty*problem.dt*np.sum(~in_passages(paths[:,active],problem),axis=1)
    tail = problem.final_time-problem.leave
    costs += problem.r*problem.kappa*paths[:,-1]**2/(2*(problem.r+problem.kappa*tail))
    # log(q_c/p0), with the common zero-drift density cancelled analytically.
    log_components = (increments @ means.T - 0.5*np.sum(means**2,axis=1))/variance
    log_ratio = np.log(means.shape[0])-logsumexp(log_components,axis=1)
    try:
        weights,ess = normalized_weights(costs,problem.temperature,log_ratio)
        action = float(np.dot(weights,increments[:,0])/problem.dt)
        failed = False
    except ValueError:
        action,ess,failed = 0.0,0.0,True
        weights = np.zeros(samples)
    result = {"action":action,"ess":ess,"max_weight":float(weights.max()),"failed":failed}
    if keep_paths:
        result.update(paths=np.column_stack([np.full(samples,x),paths]),weights=weights,costs=costs,log_ratio=log_ratio)
    return result


def deterministic_reference(problem, dx=0.0025, extent=4.0, control_limit=24.0):
    """Mean-dynamics finite-grid dynamic program with identical finite costs.

    Its piecewise-constant action minimizes r*h*u²/2 + h*q(x+h*u)+J_next;
    the grid does not assume that entering either passage is mandatory.
    """
    grid = np.arange(-extent,extent+dx/2,dx)
    n = problem.steps
    values = 0.5*problem.kappa*grid**2
    controls = np.zeros((n,grid.size))
    offsets = np.arange(-int(control_limit*problem.dt/dx),int(control_limit*problem.dt/dx)+1)
    us = offsets*dx/problem.dt
    for j in range(n-1,-1,-1):
        nxt = values+problem.dt*state_cost(grid,j+1,problem)
        inds = np.arange(grid.size)[:,None]+offsets
        valid = (inds>=0)&(inds<grid.size)
        costs = nxt[np.clip(inds,0,grid.size-1)]+0.5*problem.r*problem.dt*us**2
        costs[~valid]=np.inf
        best=np.argmin(costs,axis=1)
        values=costs[np.arange(grid.size),best]
        controls[j]=us[best]
    return ReferenceSolution(grid,np.array([values]),controls,problem)


def paired_trials(problem, trials=96, samples=2048, seed=8102):
    """Execute online sampling, mean-dynamics MPC and numerical feedback together."""
    reference = solve_reference(problem)
    mean = deterministic_reference(problem)
    plant_rng = np.random.default_rng(seed)
    planner_rng = np.random.default_rng(seed+1000000)
    noise= np.sqrt(problem.nu*problem.dt)*plant_rng.standard_normal((trials,problem.steps))
    result={}
    for method in ("path_integral","mean_dynamics","reference"):
        states=np.full((trials,problem.steps+1),problem.x0)
        controls=np.zeros((trials,problem.steps))
        esses=np.full_like(controls,np.nan)
        failures=np.zeros_like(controls,dtype=bool)
        for k in range(problem.steps):
            if method=="path_integral":
                for b in range(trials):
                    update=sample_path_integral(states[b,k],k,problem,planner_rng,samples)
                    controls[b,k],esses[b,k],failures[b,k]=update["action"],update["ess"],update["failed"]
            else:
                controls[:,k]=(reference if method=="reference" else mean).action(states[:,k],k)
            states[:,k+1]=states[:,k]+problem.dt*controls[:,k]+noise[:,k]
        enter,leave=round(problem.enter/problem.dt),round(problem.leave/problem.dt)
        violation=~in_passages(states[:,enter+1:leave+1],problem)
        penalty=problem.penalty*problem.dt*violation.sum(axis=1)
        effort=0.5*problem.r*problem.dt*np.sum(controls**2,axis=1)
        cost=effort+penalty+0.5*problem.kappa*states[:,-1]**2
        midpoint=states[:,round((problem.enter+problem.leave)/(2*problem.dt))]
        wide=(midpoint>=problem.intervals[1][0])&(midpoint<=problem.intervals[1][1])
        result[method]={"states":states,"controls":controls,"ess":esses,"cost":cost,"effort":effort,
                        "penalty":penalty,"failure":violation.any(axis=1),"wide":wide,"update_failures":int(failures.sum())}
    return result
