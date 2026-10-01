"""Jacketed reactor with three prescribed incompressible stirring modes.

Temperature/reactant fields are (..., 2, ny, nx), in kelvin and fractions.
The velocity responds instantaneously to controls: no momentum solver is implied.
This module does not change the earlier heater-control plant or its artifacts.
"""
from dataclasses import dataclass, asdict, replace
from functools import partial, lru_cache
from pathlib import Path
import hashlib
import itertools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from thermal_reactor import ReactorConfig, reaction_rate, valid_fields
from mppi_control import gaussian_mppi_update


CONFIG = ReactorConfig(wall_temperature=1050., loss=.08, inlet_temperature=900., max_dt=.05,
                       velocity_shear=.2, heating_rate=0.)


@dataclass(frozen=True)
class StirringPlan:
    horizon: int = 20
    control_dt: float = 2.
    frame_dt: float = .25
    duration: float = 160.
    candidates: int = 256
    iterations: int = 4
    correlation: float = .6
    perturbation_scale: float = .7
    temperature: float = .02
    effort_weight: float = .01
    slew_weight: float = .005
    terminal_weight: float = 1.
    tail_steps: int = 20
    seed: int = 19744
    batch_size: int = 32
    backend: str = 'jax'


PLAN = StirringPlan()
MODE_AMPLITUDE = .15


@lru_cache(maxsize=16)
def flow_basis(config=CONFIG):
    """Vertex streamfunctions and their discrete curl on cell faces."""
    y = np.linspace(0., config.width, config.ny + 1)[:, None]
    x = np.linspace(0., config.length, config.nx + 1)[None, :]
    s, h = config.velocity_shear, config.width
    base = np.broadcast_to(config.velocity * ((1-s)*y + 3*s*y*y/h - 2*s*y**3/h**2),
                           (config.ny + 1, config.nx + 1))
    modes = np.stack([MODE_AMPLITUDE * np.maximum(1-((x-center)/2)**2, 0)**3
                      * np.sin(np.pi*y/h)**2 for center in (2., 4., 6.)])
    # Analytic boundary values avoid sin(pi) roundoff at the closed wall.
    modes[:, (0, -1), :] = 0.
    psi = np.concatenate([base[None], modes])
    u = np.diff(psi, axis=-2) / (config.width/config.ny)
    v = -np.diff(psi, axis=-1) / (config.length/config.nx)
    return u.astype(np.float32), v.astype(np.float32)


def face_velocity(controls, config=CONFIG):
    ub, vb = flow_basis(config)
    return (ub[0] + jnp.einsum('...j,jyx->...yx', controls, ub[1:]),
            vb[0] + jnp.einsum('...j,jyx->...yx', controls, vb[1:]))


def inlet_temperature(t, config=CONFIG, oscillating=True):
    y = (jnp.arange(config.ny) + .5) * config.width/config.ny
    amplitude = config.transverse_amplitude + 60*jnp.sin(2*jnp.pi*t/32)*oscillating
    return config.inlet_temperature + amplitude[..., None]*jnp.cos(jnp.pi*y/config.width)


def minmod(a, b):
    return jnp.where(a*b > 0, jnp.sign(a)*jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.)


def slopes(field, axis, order=2):
    if order == 1:
        return jnp.zeros_like(field)
    f = jnp.moveaxis(field, axis, -1)
    d = jnp.diff(f, axis=-1)
    middle = minmod(d[..., :-1], d[..., 1:])
    result = jnp.concatenate([jnp.zeros_like(f[..., :1]), middle, jnp.zeros_like(f[..., :1])], -1)
    return jnp.moveaxis(result, -1, axis)


def transport_fluxes(field, inlet, u, v, config=CONFIG, order=2):
    """Conservative MUSCL fluxes, with zero diffusive boundary fluxes.

    Supports extra field/channel axes when the caller broadcasts u and v.
    The supplied inlet is a boundary face value, not an extrapolated cell.
    """
    dx, dy = config.length/config.nx, config.width/config.ny
    inlet = jnp.broadcast_to(inlet, field.shape[:-1])
    sx, sy = slopes(field, -1, order), slopes(field, -2, order)
    left = jnp.concatenate([inlet[..., None], field + .5*sx], -1)
    right = jnp.concatenate([field - .5*sx, field[..., -1:]], -1)
    below = jnp.concatenate([field[..., :1, :], field + .5*sy], -2)
    above = jnp.concatenate([field - .5*sy, field[..., -1:, :]], -2)
    fx = u*jnp.where(u >= 0, left, right)
    fy = v*jnp.where(v >= 0, below, above)
    gx = jnp.concatenate([jnp.zeros_like(field[..., :1]), jnp.diff(field, axis=-1),
                          jnp.zeros_like(field[..., :1])], -1)
    gy = jnp.concatenate([jnp.zeros_like(field[..., :1, :]), jnp.diff(field, axis=-2),
                          jnp.zeros_like(field[..., :1, :])], -2)
    return fx-config.diffusivity*gx/dx, fy-config.diffusivity*gy/dy


def transport_rhs(field, inlet, u, v, config=CONFIG, order=2):
    fx, fy = transport_fluxes(field, inlet, u, v, config, order)
    return (-jnp.diff(fx, axis=-1)/(config.length/config.nx)
            - jnp.diff(fy, axis=-2)/(config.width/config.ny))


def field_rhs(fields, t, u, v, config=CONFIG, oscillating=True, order=2):
    tin = inlet_temperature(t, config, oscillating)
    inlet = jnp.stack([tin, jnp.ones_like(tin)], axis=-2)
    result = transport_rhs(fields, inlet, u[..., None, :, :], v[..., None, :, :], config, order)
    T, c = fields[..., 0, :, :], fields[..., 1, :, :]
    r = reaction_rate(T)*c
    return result + jnp.stack([config.loss*(config.wall_temperature-T)-config.reaction_cooling*r, -r], -3)


def stable_dt(fields, u, v, config=CONFIG):
    dx, dy = config.length/config.nx, config.width/config.ny
    outgoing = ((jnp.maximum(u[..., 1:], 0)+jnp.maximum(-u[..., :-1], 0))/dx
                + (jnp.maximum(v[..., 1:, :], 0)+jnp.maximum(-v[..., :-1, :], 0))/dy)
    # Minmod face reconstruction needs the extra factor two on advective loss.
    rate = jnp.max(reaction_rate(jnp.maximum(fields[..., 0, :, :], config.wall_temperature)))
    return jnp.minimum(config.max_dt, .9/(2*jnp.max(outgoing) + 2*config.diffusivity*(dx**-2+dy**-2)
                                          + config.loss + rate))


@partial(jax.jit, static_argnames=('config', 'oscillating', 'order'))
def stirring_step(fields, controls, start, elapsed=2., config=CONFIG, oscillating=True, order=2):
    fields, controls = jnp.asarray(fields, jnp.float32), jnp.asarray(controls, jnp.float32)
    u, v = face_velocity(controls, config)
    def advance(carry):
        local, x = carry
        h = jnp.minimum(stable_dt(x, u, v, config), elapsed-local)
        first = x+h*field_rhs(x, start+local, u, v, config, oscillating, order)
        second = .5*x + .5*(first+h*field_rhs(first, start+local+h, u, v, config, oscillating, order))
        return local+h, second
    return jax.lax.while_loop(lambda z: z[0] < elapsed-1e-6, advance, (jnp.float32(0), fields))[1]


def warm_state(config=CONFIG):
    x = jnp.stack([jnp.full((config.ny, config.nx), 950.), jnp.ones((config.ny, config.nx))])
    return np.asarray(stirring_step(x, jnp.zeros(3), 0., 400., config, False))


def mixing_components(fields, config=CONFIG):
    part = fields[..., :, :, int(round(6/config.length*config.nx)):]
    variances = jnp.mean(jnp.var(part, axis=-2), axis=-1)
    return variances[..., 0]/400, variances[..., 1]/.01


def mixing_error(fields, config=CONFIG):
    a, b = mixing_components(fields, config)
    return a+b


def physical_checks(fields, config=CONFIG):
    weights = flow_basis(config)[0][0, :, -1]
    conv = 1-jnp.sum(fields[..., 1, :, -1]*weights, -1)/jnp.sum(weights)
    peak = jnp.max(fields[..., 0, :, :], axis=(-2, -1))
    ok = (jnp.isfinite(fields).all(axis=(-3, -2, -1))
          & (jnp.min(fields[..., 0, :, :], axis=(-2, -1)) > 0)
          & (jnp.min(fields[..., 1, :, :], axis=(-2, -1)) >= -2e-5)
          & (jnp.max(fields[..., 1, :, :], axis=(-2, -1)) <= 1+2e-5)
          & (peak <= config.temperature_limit) & (conv >= config.conversion_target))
    return ok, peak, conv


def filter_latent(samples, correlation=.6):
    z = np.asarray(samples, dtype=float)
    result = np.empty_like(z); result[..., 0, :] = z[..., 0, :]
    for k in range(1, z.shape[-2]):
        result[..., k, :] = correlation*result[..., k-1, :] + np.sqrt(1-correlation**2)*z[..., k, :]
    return result


def decode(samples, reference, planner=PLAN):
    raw = reference + planner.perturbation_scale*filter_latent(samples, planner.correlation)
    return (raw/np.maximum(1., np.linalg.norm(raw, axis=-1, keepdims=True))).astype(np.float32)


@partial(jax.jit, static_argnames=('config', 'planner'))
def score_batch(initial, tapes, start, previous, config=CONFIG, planner=PLAN):
    n = len(tapes); x = jnp.broadcast_to(initial, (n,)+initial.shape)
    def advance(carry, inp):
        x, old, cost, ok = carry
        u, t = inp
        y = stirring_step(x, u, t, planner.control_dt, config)
        step_cost = (mixing_error(y, config) + planner.effort_weight*jnp.sum(u*u, -1)
                     + planner.slew_weight*jnp.sum((u-old)**2, -1))
        feasible = physical_checks(y, config)[0]
        return (y, u, cost+step_cost, ok & feasible), None
    ts = start+jnp.arange(len(tapes[0]))*planner.control_dt
    (last, _, total, ok), _ = jax.lax.scan(advance, (x, jnp.broadcast_to(previous, (n,3)),
                     jnp.zeros(n), jnp.ones(n, bool)), (tapes.swapaxes(0,1), ts))
    raw = total/len(tapes[0]) + planner.terminal_weight*mixing_error(last, config)
    ok = ok & jnp.isfinite(raw) & (jnp.max(jnp.sum(tapes*tapes, -1), -1) <= 1+1e-6)
    return jnp.where(ok, raw, jnp.inf), raw, ok


@partial(jax.jit, static_argnames=('config', 'planner'))
def recorded_rollout(initial, tape, start=0., config=CONFIG, planner=PLAN):
    controls = jnp.repeat(tape, int(round(planner.control_dt/planner.frame_dt)), axis=0)
    ts = start+jnp.arange(len(controls))*planner.frame_dt
    def advance(x, ut):
        y = stirring_step(x, ut[0], ut[1], planner.frame_dt, config)
        return y, y
    future = jax.lax.scan(advance, initial, (controls, ts))[1]
    return jnp.concatenate([initial[None], future], axis=0)


def validate_tape(initial, tape, start=0., config=CONFIG, planner=PLAN):
    tape = np.concatenate([tape, np.zeros((planner.tail_steps,3), np.float32)])
    values = np.asarray(recorded_rollout(jnp.asarray(initial), jnp.asarray(tape), start, config, planner))
    ok, peak, conv = (np.asarray(x) for x in physical_checks(values, config))
    feasible = bool(ok.all() and np.max(np.sum(tape*tape, -1)) <= 1+1e-6)
    return feasible, {'maximum_temperature_K': float(peak.max()), 'minimum_outlet_conversion': float(conv.min()),
                      'duration_s': len(tape)*planner.control_dt, 'frame_dt_s': planner.frame_dt}


class StirringController:
    def __init__(self, config=CONFIG, planner=PLAN):
        self.config, self.planner = config, planner
        self.rng = np.random.default_rng(planner.seed)
        self.incumbent = np.zeros((planner.horizon,3), np.float32)
        self.previous = np.zeros(3, np.float32)
        self.last_plan = self.incumbent.copy()
        self.scorer = None
        if planner.backend == 'metal':
            from thermal_stirring_metal import MetalStirringScorer
            self.scorer = MetalStirringScorer(config, planner)
        elif planner.backend != 'jax':
            raise ValueError('Unknown stirring backend')

    def score(self, x, tapes, start):
        if self.scorer is not None and len(tapes) >= 64:
            return self.scorer.score(x, tapes, start, self.previous)
        parts = []
        size = 1 if len(tapes) == 1 else self.planner.batch_size
        for offset in range(0, len(tapes), size):
            block = tapes[offset:offset+size]; n = len(block)
            if n < size:
                block = np.concatenate([block, np.repeat(block[-1:], size-n, axis=0)])
            parts.append(tuple(np.asarray(a)[:n] for a in score_batch(jnp.asarray(x), jnp.asarray(block),
                start, jnp.asarray(self.previous), self.config, self.planner)))
        return tuple(np.concatenate([r[i] for r in parts]) for i in range(3))

    def validate(self, x, tape, start):
        return validate_tape(x, tape, start, self.config, self.planner)

    def plan(self, x, start, capture=False):
        p = self.planner; began = time.perf_counter()
        incumbent = self.incumbent.copy()
        ok, check = self.validate(x, incumbent, start)
        record = {'time_s': float(start), 'iterations': [], 'planning_failure': not ok,
                  'initial_validation': check, 'seed': p.seed}
        if not ok:
            record['runtime_s'] = time.perf_counter()-began
            return None, record, None
        cost = float(self.score(x, incumbent[None], start)[1][0])
        record['initial_objective'] = cost
        reference, mean = incumbent.copy(), np.zeros((p.horizon,3))
        snapshot = None
        for iteration in range(p.iterations):
            z = mean+self.rng.normal(size=(p.candidates,p.horizon,3))
            tapes = decode(z, reference, p)
            masked, raw, feasible = self.score(x, tapes, start)
            entry = {'iteration': iteration+1, 'accepted': False, 'alpha': 0., 'ess': 0.,
                     'feasible_candidates': int(feasible.sum()), 'incumbent_before_objective': cost,
                     'weighted_proposal_objective': None, 'weighted_proposal_feasible': False}
            weights = np.zeros(p.candidates); proposed = incumbent.copy(); exists = False
            try:
                proposal, weights, ess = gaussian_mppi_update(z, masked, mean, 1., p.temperature)
                exists = True; entry['ess'] = ess
                proposed = decode(proposal, reference, p)
                _, pcost, pok = self.score(x, proposed[None], start)
                entry['weighted_proposal_objective'] = float(pcost[0]) if np.isfinite(pcost[0]) else None
                entry['weighted_proposal_feasible'] = bool(pok[0])
                for alpha in (1., .5, .25, .125):
                    trial_mean = mean+alpha*(proposal-mean)
                    trial = decode(trial_mean, reference, p)
                    _, value, valid = self.score(x, trial[None], start)
                    if not valid[0] or value[0] >= cost-1e-7:
                        continue
                    accepted, details = self.validate(x, trial, start)
                    if accepted:
                        incumbent, mean, cost = trial, trial_mean, float(value[0])
                        entry.update(accepted=True, alpha=alpha, accepted_validation=details)
                        break
            except ValueError as error:
                entry['failure_reason'] = str(error)
            entry['incumbent_after_objective'] = cost
            record['iterations'].append(entry)
            if capture and iteration == p.iterations-1:
                order = np.argsort(-weights)
                selected = order[[0,1,2,p.candidates//4,p.candidates//2,p.candidates-1]]
                snapshot = {'initial': np.asarray(x), 'reference': reference,
                            'candidate_controls': tapes[selected], 'candidate_costs': raw[selected],
                            'candidate_weights': weights[selected], 'candidate_feasible': feasible[selected],
                            'candidate_indices': selected, 'weighted_controls': proposed,
                            'used_controls': incumbent.copy(), 'weighted_proposal_exists': exists,
                            'iteration': entry.copy()}
        self.last_plan = incumbent.copy(); action = incumbent[0].copy()
        self.incumbent = np.concatenate([incumbent[1:], np.zeros((1,3),np.float32)])
        self.previous = action
        record.update(final_objective=cost, runtime_s=time.perf_counter()-began,
                      accepted=sum(r['accepted'] for r in record['iterations']),
                      rejected=sum(not r['accepted'] for r in record['iterations']))
        return action, record, snapshot
