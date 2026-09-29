"""Synthetic planar twin-thruster boat for the successive-approximation chapter."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from trajectory_optimization import DifferentiableProblem


@dataclass(frozen=True)
class BoatParameters:
    mass: float = 50.0
    inertia: float = 20.0
    lever: float = 0.4
    thrust_limit: float = 25.0
    longitudinal_drag: float = 8.0
    lateral_drag: float = 30.0
    angular_drag: float = 10.0
    length: float = 2.0
    width: float = 0.8
    dt: float = 0.1
    steps: int = 200
    berth_x: float = 0.0
    berth_y: float = 1.0
    quay_buffer: float = 0.2
    penalty_smoothing: float = 0.08


SCENARIOS = {
    "angled": {"name": "Angled approach", "initial": (-8., 5., -.5, .2, -.1, 0.)},
    "drifting": {"name": "Sideways drift", "initial": (-8., 5., -.5, .2, -.5, 0.)},
}


def dynamics(state, control, parameters=BoatParameters()):
    """World-frame velocities; normalized controls turn into forces in newtons."""
    p = parameters
    heading, velocity, omega = state[2], state[3:5], state[5]
    forward = jnp.array([jnp.cos(heading), jnp.sin(heading)])
    side = jnp.array([-jnp.sin(heading), jnp.cos(heading)])
    drag = (p.longitudinal_drag * (forward @ velocity) * forward
            + p.lateral_drag * (side @ velocity) * side)
    thrust = p.thrust_limit * jnp.sum(control) * forward
    torque = p.lever * p.thrust_limit * (control[1] - control[0])
    acceleration = (thrust - drag) / p.mass
    return jnp.concatenate((velocity, jnp.array([omega]), acceleration,
                            jnp.array([(torque - p.angular_drag * omega) / p.inertia])))


def step(state, control, parameters=BoatParameters(), dt=None):
    h = parameters.dt if dt is None else dt
    a = dynamics(state, control, parameters)
    b = dynamics(state + h * a / 2, control, parameters)
    c = dynamics(state + h * b / 2, control, parameters)
    d = dynamics(state + h * c, control, parameters)
    return state + h * (a + 2 * b + 2 * c + d) / 6


def hull_corners(state, parameters=BoatParameters()):
    p = parameters
    body = jnp.array([[-p.length/2, -p.width/2], [p.length/2, -p.width/2],
                      [p.length/2, p.width/2], [-p.length/2, p.width/2]])
    cs, sn = jnp.cos(state[2]), jnp.sin(state[2])
    rotation = jnp.array([[cs, -sn], [sn, cs]])
    return body @ rotation.T + state[:2]


def quay_penalty(state, parameters=BoatParameters()):
    p = parameters
    depths = (p.quay_buffer - hull_corners(state, p)[:, 1]) / p.penalty_smoothing
    violation = p.penalty_smoothing * jax.nn.softplus(depths)
    return 200. * jnp.sum(violation ** 2)


def stage_cost(state, control, parameters=BoatParameters()):
    p = parameters
    position = state[:2] - jnp.array([p.berth_x, p.berth_y])
    return p.dt * (0.03 * (position @ position) + 0.1 * (1 - jnp.cos(state[2]))
                   + 0.1 * (state[3:5] @ state[3:5]) + 0.1 * state[5] ** 2
                   + 0.2 * (control @ control) + quay_penalty(state, p))


def terminal_cost(state, parameters=BoatParameters(), velocity_weight=200.):
    p = parameters
    position = state[:2] - jnp.array([p.berth_x, p.berth_y])
    return (300. * (position @ position) + 100. * (1 - jnp.cos(state[2]))
            + velocity_weight * (state[3:5] @ state[3:5]) + 100. * state[5] ** 2
            + quay_penalty(state, p))


def make_problem(parameters=BoatParameters(), velocity_weight=200.):
    return DifferentiableProblem(
        6, lambda x, u, t: step(x, u, parameters),
        lambda x, u, t: stage_cost(x, u, parameters),
        lambda x: terminal_cost(x, parameters, velocity_weight), [-1., -1.], [1., 1.])


def initial_controls(parameters=BoatParameters()):
    """An explicit coasting seed, shared by both methods and both scenarios."""
    return np.zeros((parameters.steps, 2))


def audit(states, controls, parameters=BoatParameters(), subdivisions=4):
    """Check the hull, arrival, and an independent, finer RK4 replay."""
    p = parameters
    fine_controls = np.repeat(controls, subdivisions, axis=0)
    def fine_rollout(x0, us):
        def advance(x, u):
            nxt = step(x, u, p, p.dt / subdivisions)
            return nxt, nxt
        _, xx = jax.lax.scan(advance, x0, us)
        return jnp.concatenate((x0[None], xx))
    fine = np.asarray(jax.jit(fine_rollout)(jnp.asarray(states[0]), jnp.asarray(fine_controls)))
    corners = np.asarray(jax.vmap(lambda x: hull_corners(x, p))(jnp.asarray(fine)))
    end = fine[-1]
    position_error = float(np.linalg.norm(end[:2] - [p.berth_x, p.berth_y]))
    heading_error = float(abs(np.rad2deg(np.arctan2(np.sin(end[2]), np.cos(end[2])))))
    speed = float(np.linalg.norm(end[3:5]))
    yaw_rate = float(abs(np.rad2deg(end[5])))
    clearance = float(corners[:, :, 1].min())
    thrust_violation = float(max(0., np.max(np.abs(controls)) - 1.) * p.thrust_limit)
    difference = float(np.max(np.linalg.norm(fine[::subdivisions, :2] - states[:, :2], axis=1)))
    passed = bool(np.all(np.isfinite(fine)) and position_error < .2 and heading_error < 3.
                  and speed < .05 and yaw_rate < 1. and clearance > 0. and thrust_violation < 1e-9)
    return dict(position_error_m=position_error, heading_error_deg=heading_error,
                terminal_speed_mps=speed, terminal_yaw_rate_degps=yaw_rate,
                minimum_clearance_m=clearance, thrust_violation_N=thrust_violation,
                fine_replay_max_position_difference_m=difference,
                fine_dt=p.dt / subdivisions, docking_pass=passed)
