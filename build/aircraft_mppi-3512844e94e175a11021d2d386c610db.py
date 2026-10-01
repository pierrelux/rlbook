"""Aircraft trajectory sampling with a numerical OpenAP point-mass model.

The aircraft follows an open-loop tape between replanning instants. A geometric
nominal constructs initial tapes; every perturbed tape is evaluated by forward
simulation. This engineering MPPI application is not matched-noise path-integral
control. SI units are used internally; OpenAP's knots/feet interfaces are
converted only at the model boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import openap
from openap import aero
from mppi_control import normalized_weights, gaussian_log_ratio

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FlightConfig:
    prediction_dt: float = 60.0
    execution_dt: float = 10.0
    replan_dt: float = 60.0
    duration: float = 3600.0
    cruise_height: float = 7000.0
    samples: int = 32
    scenarios: int = 6
    iterations: int = 2
    temperature: float = 25.0
    gust_tau: float = 300.0
    gust_sigma_e: float = 5.0
    gust_sigma_n: float = 4.0
    horizontal_tolerance: float = 1000.0
    vertical_tolerance: float = 30.0
    seed: int = 260911


class NoFeasiblePlan(RuntimeError):
    """The current state has no validated reference in the tested tape family."""


class MeanWind:
    """Numerical frozen-time ERA5 field; a missing field is an explicit error."""
    def __init__(self, path: Path | str = ROOT / 'data/aircraft/era5_wind.npz'):
        self.path = Path(path)
        with np.load(path) as d:
            # Canonical file produced by the companion source-data preparation.
            self.latitude = d['lat_deg']
            self.longitude = d['lon_deg']
            self.pressure = d['pressure_hpa']
            u, v = d['u_mps'], d['v_mps']
        self._u = RegularGridInterpolator((self.pressure, self.latitude, self.longitude), u, bounds_error=False, fill_value=None)
        self._v = RegularGridInterpolator((self.pressure, self.latitude, self.longitude), v, bounds_error=False, fill_value=None)

    def at(self, longitude, latitude, height):
        h, lat, lon = np.broadcast_arrays(height, latitude, longitude)
        # Hold boundary values outside the pressure-level range; no extrapolated
        # polynomial winds below the lowest measured level.
        points = np.stack((np.clip(aero.pressure(h)/100, self.pressure[0], self.pressure[-1]), np.clip(lat, self.latitude[0], self.latitude[-1]), np.clip(lon, self.longitude[0], self.longitude[-1])), axis=-1)
        return np.stack((self._u(points), self._v(points)), axis=-1).reshape(h.shape + (2,))


class Aircraft:
    """OpenAP's complete-flight airborne model, without a CasADi dependency."""
    def __init__(self, wind=None):
        self.properties = openap.prop.aircraft('A320')
        self.mass0 = .85 * self.properties['mtow']
        self.h0 = 100 * aero.ft
        self.wind = wind
        a, b = openap.nav.airport('CYUL'), openap.nav.airport('CYYZ')
        self.origin = np.array([a['lon'], a['lat']])
        self.destination_lonlat = np.array([b['lon'], b['lat']])
        self.lat0 = (a['lat'] + b['lat']) / 2
        self.lon0 = (a['lon'] + b['lon']) / 2
        self.start_xy = self.project(*self.origin)
        self.end_xy = self.project(*self.destination_lonlat)
        self.distance = np.linalg.norm(self.end_xy - self.start_xy)
        self.direction = (self.end_xy - self.start_xy) / self.distance
        self.heading = np.arctan2(self.direction[0], self.direction[1])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.fuel = openap.FuelFlow('A320', wave_drag=True)
            self.drag = openap.Drag('A320', wave_drag=True)
            self.thrust = openap.Thrust('A320')
        self.initial = np.array([*self.start_xy, self.h0, self.mass0, 0.0])

    def project(self, longitude, latitude):
        # The same local azimuthal-equidistant construction as OpenAP.top.
        bearing = np.deg2rad(aero.bearing(self.lat0, self.lon0, latitude, longitude))
        distance = aero.distance(self.lat0, self.lon0, latitude, longitude)
        return np.stack((distance * np.sin(bearing), distance * np.cos(bearing)), axis=-1)

    def lonlat(self, xy):
        xy = np.asarray(xy)
        distance = np.linalg.norm(xy, axis=-1)
        bearing = np.rad2deg(np.arctan2(xy[..., 0], xy[..., 1]))
        lat, lon = aero.latlon(self.lat0, self.lon0, distance, bearing)
        return np.stack((lon, lat), axis=-1)

    def mean_wind(self, state):
        if self.wind is None:
            return np.zeros(np.shape(state)[:-1] + (2,))
        lonlat = self.lonlat(np.asarray(state)[..., :2])
        return self.wind.at(lonlat[..., 0], lonlat[..., 1], np.asarray(state)[..., 2])

    def velocity_and_fuel(self, state, control, gust):
        x, u = np.asarray(state), np.asarray(control)
        h, m = x[..., 2], x[..., 3]
        mach, vs, psi = u[..., 0], u[..., 1], u[..., 2]
        tas = aero.mach2tas(mach, h)
        gamma = np.arctan2(vs, tas)
        horizontal = tas * np.cos(gamma)
        wind = self.mean_wind(x) + gust
        ff = np.asarray(self.fuel.enroute(m.ravel(), tas.ravel() / aero.kts, h.ravel() / aero.ft, vs.ravel() / aero.fpm)).reshape(h.shape)
        return np.stack(np.broadcast_arrays(horizontal * np.sin(psi) + wind[..., 0], horizontal * np.cos(psi) + wind[..., 1], vs, -np.asarray(ff), np.ones_like(h)), axis=-1)

    def step(self, state, control, dt, gust):
        """Explicit midpoint; the gust is held over this short integration step."""
        dt = np.asarray(dt)
        first = self.velocity_and_fuel(state, control, gust)
        middle = state + .5 * dt[..., None] * first
        return state + dt[..., None] * self.velocity_and_fuel(middle, control, gust)

    def constraints(self, x, u):
        """Nonnegative violations normalized by their physical units/scales.

        Reproduces the source's approximate thrust/lift/energy checks, and uses
        OEW rather than its loose half-OEW numerical state bound.
        """
        h, mass = x[..., 2], x[..., 3]
        mach, vs, psi = u[..., 0], u[..., 1], u[..., 2]
        tas = aero.mach2tas(mach, h)
        thrust = np.asarray(self.thrust.cruise(tas.ravel() / aero.kts, h.ravel() / aero.ft)).reshape(h.shape)
        drag = np.asarray(self.drag.clean(mass.ravel(), tas.ravel() / aero.kts, h.ravel() / aero.ft)).reshape(h.shape)
        rho = aero.density(h)
        qS = .5 * rho * tas**2 * self.properties['wing']['area']
        polar = self.drag.polar['clean']
        cl = np.sqrt(np.maximum(1e-10, (.9 * thrust / (qS + 1e-10) - polar['cd0']) / polar['k']))
        heading_delta = np.arctan2(np.sin(psi-self.heading), np.cos(psi-self.heading))
        return np.stack(np.broadcast_arrays(
            np.maximum(0, .1-mach) / .1,
            np.maximum(0, mach-self.properties['mmo']) / .1,
            np.maximum(0, np.abs(vs)-2500*aero.fpm) / aero.fpm / 500,
            np.maximum(0, np.abs(heading_delta)-np.pi/2),
            np.maximum(0, self.h0-h) / 100,
            np.maximum(0, h-self.properties['limits']['ceiling']) / 100,
            np.maximum(0, self.properties['oew']-mass) / 1000,
            np.maximum(0, mass-self.mass0) / 1000,
            np.maximum(0, drag-.95*thrust) / 10000,
            np.maximum(0, mass*aero.g0-.8*cl*qS) / 100000,
            np.maximum(0, mass*aero.g0*vs-(thrust-drag)*tas) / 1e6,
            np.maximum(0, np.minimum(self.start_xy[0], self.end_xy[0])-10000-x[..., 0])/1000,
            np.maximum(0, x[..., 0]-np.maximum(self.start_xy[0], self.end_xy[0])-10000)/1000,
            np.maximum(0, np.minimum(self.start_xy[1], self.end_xy[1])-10000-x[..., 1])/1000,
            np.maximum(0, x[..., 1]-np.maximum(self.start_xy[1], self.end_xy[1])-10000)/1000,
        ), axis=-1)


def conditional_gusts(current, steps, scenarios, config, rng, dt=None):
    """OU draws conditional on the measured current gust, independent of plant.

    Return [scenario,time,component]. The same array is broadcast across all
    candidate actions (common random numbers), then costs are averaged over
    its scenario axis before applying exponential trajectory weights.
    """
    dt = config.prediction_dt if dt is None else dt
    rho = np.exp(-dt/config.gust_tau)
    sigma = np.array([config.gust_sigma_e, config.gust_sigma_n])
    noise = rng.normal(size=(scenarios, steps, 2))
    values = np.empty_like(noise)
    values[:, 0] = current
    for k in range(1, steps):
        values[:, k] = rho*values[:, k-1] + np.sqrt(1-rho*rho)*sigma*noise[:, k]
    return values


def progress(clock):
    # Smooth speed schedule: endpoint speed is 65% of average, middle 135%.
    return clock - .35*np.sin(2*np.pi*clock)/(2*np.pi)


def altitude_profile(q, aircraft, config):
    def smooth(v):
        v = np.clip(v, 0, 1)
        return v+v*v-v*v*v
    factor = np.minimum(smooth(q/.40), smooth((1-q)/.40))
    return aircraft.h0 + (config.cruise_height-aircraft.h0)*factor


def nominal_tapes(aircraft, state, parameters, config, clock_fraction, remaining, observed_gust=None):
    """Construct nominal tapes, then add smooth control-knot perturbations.

    Parameters are dimensionless: log-duration and 3 x 4 interior control knots.
    The prior N(0, 0.7^2 I) and Gaussian proposal are over these latent parameters.
    The terminal geometry initializes a proposal; simulated paths remain free
    to miss the destination and are assessed without state projection.
    """
    parameters = np.atleast_2d(parameters)
    durations = remaining*np.exp(.025*parameters[:, 0])
    durations = np.clip(durations, max(.1, .90*remaining), 1.12*remaining)
    count = max(1, int(np.ceil(np.max(durations)/config.prediction_dt)))
    times = np.minimum(np.arange(count+1)[None, :]*config.prediction_dt, durations[:, None])
    tau = times / durations[:, None]
    p = clock_fraction+(1-clock_fraction)*tau
    q = progress(p)
    q0 = progress(clock_fraction)
    fraction = (q-q0)/max(1e-9, 1-q0)
    points = state[:2]+fraction[..., None]*(aircraft.end_xy-state[:2])
    height = altitude_profile(q, aircraft, config)
    height += (state[2]-altitude_profile(q0, aircraft, config))*(1-tau)
    dts = np.diff(times, axis=1)
    safe_dt = np.maximum(dts, 1e-9)
    middle = np.zeros((len(parameters), count, 5))
    middle[..., :2] = .5*(points[:, 1:]+points[:, :-1])
    middle[..., 2] = .5*(height[:, 1:]+height[:, :-1])
    middle[..., 3] = state[3]
    middle[..., 4] = state[4] + .5*(times[:, 1:]+times[:, :-1])
    ground = np.diff(points, axis=1)/safe_dt[..., None]
    vs = np.diff(height, axis=1)/safe_dt
    air = ground-aircraft.mean_wind(middle)
    if observed_gust is not None:
        air -= np.exp(-times[:, :-1, None]/config.gust_tau)*np.asarray(observed_gust)
    vh2 = np.sum(air*air, axis=-1)
    tas = np.sqrt(.5*(vh2+np.sqrt(vh2*vh2+4*vh2*vs*vs)))
    mach = aero.tas2mach(tas, middle[..., 2])
    heading = np.arctan2(air[..., 0], air[..., 1])
    tape = np.stack((mach, vs, heading), axis=-1)
    knot_times = np.linspace(0, 1, 6)
    sample_times = .5*(tau[:, 1:]+tau[:, :-1])
    scales = [.004, .10, .0015]
    for i in range(len(parameters)):
        for j, scale in enumerate(scales):
            knots = np.r_[0, parameters[i, 1+4*j:5+4*j], 0]
            perturbation = np.interp(sample_times[i], knot_times, knots)
            envelope = np.sin(np.pi*sample_times[i])
            # Zero time integral keeps the endpoint vertical displacement of
            # the reference tape; no state is projected after simulation.
            perturbation -= envelope*np.sum(perturbation*dts[i])/max(1e-9, np.sum(envelope*dts[i]))
            tape[i, :, j] += scale*perturbation
    # Hold control on zero-duration padding; padding never advances the state.
    for i in range(len(parameters)):
        valid = np.flatnonzero(dts[i] > 0)
        tape[i, len(valid):] = tape[i, valid[-1]]
    return tape, dts, durations


def evaluate_tapes(aircraft, initial, tapes, dts, gusts, config):
    """[candidate,scenario] costs, with no access to actual future gusts."""
    nc, nt = dts.shape
    ns = gusts.shape[0]
    x = np.broadcast_to(initial, (nc, ns, 5)).copy()
    worst = np.zeros((nc, ns))
    accumulated = np.zeros((nc, ns))
    for j in range(nt):
        u = np.broadcast_to(tapes[:, j, None], (nc, ns, 3))
        dt = dts[:, j, None]
        violation = np.max(aircraft.constraints(x, u), axis=-1)
        worst = np.maximum(worst, violation)
        accumulated += violation**2 * dt/60
        x = aircraft.step(x, u, dt, gusts[None, :, j])
        worst = np.maximum(worst, np.max(aircraft.constraints(x, u), axis=-1))
    # Adjacent control-knot changes, including the final state, are checked.
    if nt > 1:
        limits = np.array([.2, 500*aero.fpm, np.deg2rad(15)])
        slew = np.max(np.maximum(0, np.abs(np.diff(tapes, axis=1))/limits-1), axis=(1,2))
        worst = np.maximum(worst, slew[:, None])
    horizontal = np.linalg.norm(x[..., :2]-aircraft.end_xy, axis=-1)
    vertical = np.abs(x[..., 2]-aircraft.h0)
    burn = initial[3]-x[..., 3]
    worst = np.maximum(worst, np.maximum(0, burn-aircraft.properties['mfc'])/1000)
    cost = burn + 80*(horizontal/config.horizontal_tolerance)**2 + 80*(vertical/config.vertical_tolerance)**2 + 1e5*(accumulated+worst**2)
    return cost, {'horizontal': horizontal, 'vertical': vertical, 'worst': worst, 'burn': burn, 'terminal': x}


def validate_mean(aircraft, state, tape, dts, observed, config, previous_control=None):
    """Replay on a 10 s grid under the conditional mean before acceptance."""
    if previous_control is not None:
        limits = np.array([.2, 500*aero.fpm, np.deg2rad(15)])
        if np.any(np.abs(tape[0]-previous_control) > limits+1e-10):
            return False
    repeats = np.maximum(1, np.ceil(dts/config.execution_dt).astype(int))
    fine_tape = np.repeat(tape, repeats, axis=0)
    fine_dt = np.repeat(dts/repeats, repeats)
    times = np.r_[0., np.cumsum(fine_dt[:-1])]
    gust = observed[None, None, :]*np.exp(-times[None, :, None]/config.gust_tau)
    _, check = evaluate_tapes(aircraft, state, fine_tape[None], fine_dt[None], gust, config)
    return bool(np.isfinite(check['terminal']).all()
        and check['horizontal'][0,0] <= config.horizontal_tolerance
        and check['vertical'][0,0] <= config.vertical_tolerance
        and check['worst'][0,0] <= 1e-7)


def optimize(aircraft, state, observed_gust, config, clock_fraction, remaining, stochastic, rng,
             previous_plan=None, previous_control=None):
    start = time.perf_counter()
    scenarios = config.scenarios if stochastic else 1
    forecast_duration = max(remaining*1.12, previous_plan[2] if previous_plan else 0)
    max_steps = int(np.ceil(forecast_duration/config.prediction_dt))+2
    scenario_rng = np.random.default_rng(rng.integers(0, 2**32))
    if stochastic:
        gusts = conditional_gusts(observed_gust, max_steps, scenarios, config, scenario_rng)
    else:
        # Mean-wind replanning uses the conditional OU mean, not a peek at future.
        decay = np.exp(-np.arange(max_steps)*config.prediction_dt/config.gust_tau)
        gusts = observed_gust[None, None, :]*decay[None, :, None]
    mean = np.zeros(13)
    nominal, ndt, nduration = nominal_tapes(aircraft, state, mean, config, clock_fraction, remaining, observed_gust)
    nominal_cost, nominal_diag = evaluate_tapes(aircraft, state, nominal, ndt, gusts, config)
    incumbent = None
    best_cost = np.inf
    retained_previous = False
    if validate_mean(aircraft, state, nominal[0], ndt[0], observed_gust, config, previous_control):
        incumbent = (nominal[0], ndt[0], float(nduration[0]))
        best_cost = float(nominal_cost.mean())
    if previous_plan is not None:
        pt, pd, duration = previous_plan
        # Previous intervals may start with a partial interval. Resample onto
        # the prediction grid so shared gust indices retain absolute meaning.
        boundaries = np.r_[0., np.cumsum(pd)]
        targets = np.minimum(np.arange(int(np.ceil(duration/config.prediction_dt))+1)*config.prediction_dt,duration)
        indices = np.minimum(np.searchsorted(boundaries,targets[:-1],side='right')-1,len(pt)-1)
        pt, pd = pt[indices], np.diff(targets)
        if validate_mean(aircraft,state,pt,pd,observed_gust,config,previous_control):
            old_cost, _ = evaluate_tapes(aircraft,state,pt[None],pd[None],gusts,config)
            if float(old_cost.mean()) < best_cost:
                incumbent = pt,pd,duration
                best_cost = float(old_cost.mean())
                retained_previous = True
    if incumbent is None:
        raise NoFeasiblePlan('Neither geometric nor remaining previous plan passes conditional-mean validation')
    accepted = 0
    ess_values = []
    for _ in range(config.iterations):
        epsilon = rng.normal(scale=.7, size=(config.samples, 13))
        candidates = mean+epsilon
        tapes, dts, durations = nominal_tapes(aircraft, state, candidates, config, clock_fraction, remaining, observed_gust)
        costs, _ = evaluate_tapes(aircraft, state, tapes, dts, gusts, config)
        expected_cost = np.mean(costs, axis=1)  # BEFORE exponentiation.
        log_ratio = gaussian_log_ratio(candidates, mean, .7**2)
        try:
            weights, ess = normalized_weights(expected_cost, config.temperature, log_ratio)
        except ValueError:
            ess_values.append(0.)
            continue
        proposal = weights@candidates
        proposed_tape, proposed_dt, proposed_duration = nominal_tapes(aircraft, state, proposal, config, clock_fraction, remaining, observed_gust)
        proposed_cost, diag = evaluate_tapes(aircraft, state, proposed_tape, proposed_dt, gusts, config)
        # Validate the mean separately: weighting feasible paths does not preserve
        # nonlinear path constraints. Feasibility concerns the mean-wind replay;
        # stochastic terminal spread is retained and reported, not erased.
        feasible = validate_mean(aircraft, state, proposed_tape[0], proposed_dt[0], observed_gust, config, previous_control)
        value = float(proposed_cost.mean())
        if feasible and value < best_cost:
            mean = proposal
            best_cost = value
            incumbent = proposed_tape[0], proposed_dt[0], float(proposed_duration[0])
            retained_previous = False
            accepted += 1
        ess_values.append(ess)
    tape, dts, duration = incumbent
    _, distribution = evaluate_tapes(aircraft, state, tape[None], dts[None], gusts, config)
    return tape, dts, duration, {
        'ess': float(np.mean(ess_values)), 'accepted': accepted,
        'rejected_updates': config.iterations-accepted, 'incumbent_validated': True,
        'retained_previous_plan': retained_previous,
        'runtime_s': time.perf_counter()-start,
        'predicted_horizontal_p10_p90': np.quantile(distribution['horizontal'], [.1, .9]).tolist(),
        'predicted_fuel_p10_p90': np.quantile(distribution['burn'], [.1, .9]).tolist(),
        'predicted_terminal_xy': distribution['terminal'][0, :, :2].tolist(),
        'nominal_cost': float(nominal_cost.mean()), 'accepted_cost': best_cost,
    }


def make_actual_gust(config, seed, duration=7200, dt=5.):
    # A distinct RNG stream: optimizers never receive this precomputed future.
    return conditional_gusts(np.zeros(2), int(duration/dt)+1, 1, config, np.random.default_rng(seed), dt=dt)[0]


def gust_at(actual, time, dt=5.):
    k = min(int(time/dt), len(actual)-1)
    return actual[k]


def simulate(aircraft, config, mode, actual_gust, seed=None):
    """Modes: full_trip (mean wind), frozen (gusts), mean, stochastic.

    full_trip establishes the optimized deterministic trajectory. frozen uses
    exactly that same solver seed and planning inputs, then executes with gusts.
    The final two modes measure state/gust and replan every 60 seconds.
    """
    rng = np.random.default_rng(config.seed if seed is None else seed)
    x = aircraft.initial.copy()
    states = [x.copy()]
    controls, gust_record, dt_record, replans = [], [], [], []
    current_tape = current_dt = None
    current_duration = config.duration
    scheduled_end = config.duration
    frozen = mode in ('full_trip', 'frozen')
    tick = 0
    failure = None
    while x[4] < scheduled_end-1e-6 and tick < 100:
        if tick == 0 or not frozen:
            previous_plan = None
            if current_tape is not None:
                pd = current_dt[tape_index:].copy()
                pd[0] -= tape_elapsed
                keep = pd > 1e-7
                if keep.any():
                    previous_plan = current_tape[tape_index:][keep],pd[keep],float(pd[keep].sum())
            observed = np.zeros(2) if mode == 'full_trip' else gust_at(actual_gust, x[4])
            # Geometric progress determines the remaining climb/cruise/descent
            # nominal; the true measured state supplies all rollout initials.
            along = np.clip(np.dot(x[:2]-aircraft.start_xy, aircraft.direction)/aircraft.distance, 0, .99999)
            grid = np.linspace(0, 1, 1001)
            phase = float(np.interp(along, progress(grid), grid))
            remaining = max(1., config.duration*(1-phase))
            if phase > .97:
                remaining = max(1., np.linalg.norm(aircraft.end_xy-x[:2])/105.)
            try:
                current_tape, current_dt, current_duration, diagnostics = optimize(aircraft, x, observed, config, phase, remaining, mode == 'stochastic', rng,
                    previous_plan=previous_plan,previous_control=controls[-1] if controls else None)
            except NoFeasiblePlan as error:
                failure = str(error)
                break
            scheduled_end = x[4]+current_duration
            diagnostics.update(time=float(x[4]), state=x.tolist())
            replans.append(diagnostics)
            tape_index = 0
            tape_elapsed = 0.
        execution = current_duration if frozen else min(config.replan_dt, scheduled_end-x[4])
        execution_end = min(scheduled_end, x[4]+execution)
        while x[4] < execution_end-1e-6:
            u = current_tape[tape_index]
            available = current_dt[tape_index]-tape_elapsed
            dt = min(config.execution_dt, execution_end-x[4], available)
            if dt <= 1e-7:
                tape_index += 1
                tape_elapsed = 0.
                continue
            gust = np.zeros(2) if mode == 'full_trip' else gust_at(actual_gust, x[4])
            controls.append(u.copy()); gust_record.append(gust.copy()); dt_record.append(dt)
            x = aircraft.step(x, u, dt, gust)
            states.append(x.copy())
            tape_elapsed += dt
        tick += 1
    if tick >= 100 and x[4] < scheduled_end-1e-6:
        failure = 'Maximum replanning count reached'
    return {'mode': mode, 'states': np.asarray(states), 'controls': np.asarray(controls), 'gusts': np.asarray(gust_record), 'dts': np.asarray(dt_record), 'replans': replans, 'failure': failure}


def audit(aircraft, run, config, fine_dt=5.):
    x = aircraft.initial.copy()
    fine_states = [x.copy()]
    max_v = 0.
    for u, gust, duration in zip(run['controls'], run['gusts'], run['dts']):
        remaining = duration
        while remaining > 1e-8:
            dt = min(fine_dt, remaining)
            max_v = max(max_v, float(np.max(aircraft.constraints(x, u))))
            x = aircraft.step(x, u, dt, gust)
            max_v = max(max_v, float(np.max(aircraft.constraints(x, u))))
            fine_states.append(x.copy())
            remaining -= dt
    terminal = run['states'][-1]
    horizontal = float(np.linalg.norm(terminal[:2]-aircraft.end_xy))
    vertical = float(abs(terminal[2]-aircraft.h0))
    controls = run['controls']
    if len(controls)>1:
        limits = np.array([.2, 500*aero.fpm, np.deg2rad(15)])
        max_v = max(max_v, float(np.max(np.maximum(0, np.abs(np.diff(controls,axis=0))/limits-1))))
    return {
        'mode': run['mode'], 'fuel_kg': float(aircraft.mass0-terminal[3]),
        'planning_failure': run.get('failure'),
        'duration_s': float(terminal[4]), 'horizontal_error_m': horizontal,
        'vertical_error_m': vertical, 'arrival_pass': horizontal <= config.horizontal_tolerance and vertical <= config.vertical_tolerance,
        'max_normalized_violation': max_v,
        'fine_replay_position_difference_m': float(np.linalg.norm(x[:3]-terminal[:3])),
        'fine_replay_fuel_difference_kg': float(abs(x[3]-terminal[3])),
        'fine_horizontal_error_m': float(np.linalg.norm(x[:2]-aircraft.end_xy)),
        'fine_vertical_error_m': float(abs(x[2]-aircraft.h0)),
        'max_altitude_m': float(run['states'][:, 2].max()),
        'min_vertical_speed_mps': float(controls[:, 1].min()) if len(controls) else 0.,
        'max_vertical_speed_mps': float(controls[:, 1].max()) if len(controls) else 0.,
        'cruise_samples': int(np.sum(np.abs(controls[:, 1]) < .5)) if len(controls) else 0,
        'planning_seconds': float(sum(p['runtime_s'] for p in run['replans'])),
        'mean_ess': float(np.mean([p['ess'] for p in run['replans']])) if run['replans'] else 0.,
        'accepted_updates': int(sum(p['accepted'] for p in run['replans'])),
        'rejected_updates': int(sum(p['rejected_updates'] for p in run['replans'])),
        'retained_previous_plans': int(sum(p['retained_previous_plan'] for p in run['replans'])),
        'replans': len(run['replans']),
    }
