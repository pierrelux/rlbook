"""Synthetic lumped pull-down model of a standing-wave thermoacoustic refrigerator."""
from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from trajectory_optimization import DifferentiableProblem


@dataclass(frozen=True)
class FridgeParameters:
    cold_capacity: float = 20.0      # J/K, cold exchanger plus payload
    hot_capacity: float = 50.0       # J/K, hot exchanger
    hot_conductance: float = 0.5     # W/K, hot exchanger to ambient
    ambient: float = 20.0            # degC
    load: float = 1.0                # W, parasitic heat leak into the cold side
    pumping: float = 5.0             # W at full amplitude and zero span
    work: float = 4.0                # W, acoustic work at full amplitude and zero span
    viscous: float = 1.0             # W at full amplitude
    minor_loss: float = 2.0          # W at full amplitude, cubic in amplitude
    critical_span: float = 40.0      # K, span at which pumping vanishes
    target: float = 5.0              # degC
    dt: float = 1.0
    steps: int = 300
    energy_weight: float = 0.05
    tracking_weight: float = 0.0
    terminal_weight: float = 10.0


_DEFAULT = FridgeParameters()
SCENARIOS = {
    "baseline": {"name": "Baseline pull-down", "parameters": _DEFAULT,
                 "target_reachable": True},
    "no_load": {"name": "No parasitic load", "parameters": replace(_DEFAULT, load=0.),
                "target_reachable": True},
    "heavy_hot": {"name": "Sluggish hot side",
                  "parameters": replace(_DEFAULT, hot_capacity=200., hot_conductance=0.2),
                  "target_reachable": True},
    "no_minor_loss": {"name": "Amplitude-independent COP",
                      "parameters": replace(_DEFAULT, minor_loss=0.),
                      "target_reachable": True},
    "unreachable": {"name": "Target out of reach in 300 s",
                    "parameters": replace(_DEFAULT, target=0.),
                    "target_reachable": False},
}


def powers(state, control, p=FridgeParameters()):
    """Return heat pumped from the cold side and acoustic work, both in watts."""
    span = state[1] - state[0]
    factor = 1.0 - span / p.critical_span
    u = control[0]
    cooling = p.pumping * u ** 2 * factor
    work = p.work * u ** 2 * factor + p.viscous * u ** 2 + p.minor_loss * u ** 3
    return cooling, work


def dynamics(state, control, p=FridgeParameters()):
    cooling, work = powers(state, control, p)
    cold = (-cooling + p.load) / p.cold_capacity
    hot = (cooling + work - p.hot_conductance * (state[1] - p.ambient)) / p.hot_capacity
    return jnp.array([cold, hot])


def step(state, control, p=FridgeParameters(), dt=None):
    h = p.dt if dt is None else dt
    a = dynamics(state, control, p)
    b = dynamics(state + h * a / 2, control, p)
    c = dynamics(state + h * b / 2, control, p)
    d = dynamics(state + h * c, control, p)
    return state + h * (a + 2 * b + 2 * c + d) / 6


def stage_cost(state, control, p=FridgeParameters()):
    _, work = powers(state, control, p)
    return p.dt * (p.energy_weight * work
                   + p.tracking_weight * (state[0] - p.target) ** 2)


def terminal_cost(state, p=FridgeParameters()):
    return p.terminal_weight * (state[0] - p.target) ** 2


def make_problem(p=FridgeParameters()):
    return DifferentiableProblem(
        2, lambda x, u, t: step(x, u, p),
        lambda x, u, t: stage_cost(x, u, p),
        lambda x: terminal_cost(x, p), [0.], [1.])


def initial_state(p=FridgeParameters()):
    return np.array([p.ambient, p.ambient])


def initial_controls(p=FridgeParameters()):
    return np.full((p.steps, 1), 0.5)


def audit(states, controls, parameters=FridgeParameters(), subdivisions=4,
          require_target=True):
    """Check controls, target, and a four-times-finer independent RK4 replay.

    The unreachable-target experiment sets ``require_target=False``; it still
    must satisfy the physical and numerical checks, and reports target_reached
    separately. The replay does not call the problem's planning-grid rollout.
    """
    p = parameters
    states, controls = np.asarray(states), np.asarray(controls)
    if (controls.shape != (p.steps, 1) or states.shape != (p.steps + 1, 2)
            or not isinstance(subdivisions, int) or subdivisions < 2):
        raise ValueError("expected one control per step and a finer replay grid")

    fine_controls = np.repeat(controls, subdivisions, axis=0)

    def fine_rollout(x0, us):
        def advance(x, u):
            nxt = step(x, u, p, p.dt / subdivisions)
            return nxt, nxt
        _, xx = jax.lax.scan(advance, x0, us)
        return jnp.concatenate((x0[None], xx))

    fine = np.asarray(jax.jit(fine_rollout)(jnp.asarray(states[0]),
                                           jnp.asarray(fine_controls)))
    end = fine[-1]
    error = float(abs(end[0] - p.target))
    violation = float(max(0., -np.min(controls), np.max(controls) - 1.))
    difference = float(np.max(np.abs(fine[::subdivisions] - states)))
    finite = bool(np.all(np.isfinite(states)) and np.all(np.isfinite(controls))
                  and np.all(np.isfinite(fine)))
    target_reached = bool(error < 0.6)
    passed = bool(finite and violation < 1e-9 and difference < 1e-4
                  and (target_reached or not require_target))
    return dict(terminal_cold_C=float(end[0]), terminal_hot_C=float(end[1]),
                terminal_cold_error_K=error, control_violation=violation,
                fine_replay_max_temperature_difference_K=difference,
                fine_dt_s=p.dt / subdivisions, finite=finite,
                target_reached=target_reached, pulldown_pass=passed)
