"""Synthetic tractor-semitrailer alley dock for the single-shooting section.

The state is the tractor's rear-axle position, the tractor heading, and the
trailer heading. The fifth wheel sits over the tractor's rear axle, so the
trailer axle trails that point by a fixed distance along the trailer axis.
The two controls, tractor speed and steering angle, are piecewise constant
over each control interval and are the only decision variables.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class TruckParameters:
    tractor_wheelbase: float = 4.5          # rear axle to steered axle (m)
    trailer_length: float = 10.0            # fifth wheel to trailer axle (m)
    steering_limit: float = 0.5236          # 30 degrees
    speed_limit: float = 1.5                # m/s, either direction
    hitch_limit: float = 1.0472             # 60 degrees before the penalty starts
    tractor_rear: float = 1.0               # body behind the rear axle (m)
    tractor_front: float = 5.5              # body ahead of the rear axle (m)
    tractor_width: float = 2.5
    trailer_front: float = 1.0              # body ahead of the fifth wheel (m)
    trailer_rear: float = 12.5              # body behind the fifth wheel (m)
    trailer_width: float = 2.6
    dt: float = 0.25
    steps: int = 160
    bay_x: float = 0.0                      # bay centerline
    bumper_gap: float = 0.5                 # rear bumper to dock face at arrival (m)
    wall_buffer: float = 0.2                # every body corner stays above this line
    penalty_smoothing: float = 0.1
    lane_x: float = 26.0                    # tractor axle in the lane, straight, heading +x
    lane_y: float = 19.0


SCENARIOS = {
    "dock": {"name": "Backing into the bay", "initial": "lane", "goal": "bay",
             "seed_speed": -0.5},
    "pull_out": {"name": "Pulling out of the bay", "initial": "bay", "goal": "lane",
                 "seed_speed": 0.5},
}


def bay_state(parameters=TruckParameters()):
    """Trailer axle on the bay centerline, both units pointing away from the dock."""
    p = parameters
    trailer_axle_y = p.bumper_gap + (p.trailer_rear - p.trailer_length)
    return np.array([p.bay_x, trailer_axle_y + p.trailer_length, np.pi / 2, np.pi / 2])


def lane_state(parameters=TruckParameters()):
    return np.array([parameters.lane_x, parameters.lane_y, 0., 0.])


def configuration(name, parameters=TruckParameters()):
    if name not in ("lane", "bay"):
        raise ValueError("configuration must be lane or bay")
    return bay_state(parameters) if name == "bay" else lane_state(parameters)


def dynamics(state, control, parameters=TruckParameters()):
    p = parameters
    heading, trailer = state[2], state[3]
    speed, steering = control[0], control[1]
    return jnp.array([speed * jnp.cos(heading), speed * jnp.sin(heading),
                      speed * jnp.tan(steering) / p.tractor_wheelbase,
                      speed * jnp.sin(heading - trailer) / p.trailer_length])


def step(state, control, parameters=TruckParameters(), dt=None):
    h = parameters.dt if dt is None else dt
    a = dynamics(state, control, parameters)
    b = dynamics(state + h * a / 2, control, parameters)
    c = dynamics(state + h * b / 2, control, parameters)
    d = dynamics(state + h * c, control, parameters)
    return state + h * (a + 2 * b + 2 * c + d) / 6


def rollout(initial, controls, parameters=TruckParameters(), dt=None):
    """States x_0..x_T reached from x_0 under the control sequence."""
    def advance(x, u):
        following = step(x, u, parameters, dt)
        return following, following
    _, states = jax.lax.scan(advance, jnp.asarray(initial), jnp.asarray(controls))
    return jnp.concatenate((jnp.asarray(initial)[None], states))


def trailer_axle(state, parameters=TruckParameters()):
    p = parameters
    return state[:2] - p.trailer_length * jnp.array([jnp.cos(state[3]), jnp.sin(state[3])])


def hitch_angle(state):
    """Tractor heading minus trailer heading; zero when the rig is straight."""
    return state[2] - state[3]


def _rectangle(origin, heading, back, front, width):
    body = jnp.array([[-back, -width / 2], [front, -width / 2],
                      [front, width / 2], [-back, width / 2]])
    cs, sn = jnp.cos(heading), jnp.sin(heading)
    return body @ jnp.array([[cs, sn], [-sn, cs]]) + origin


def body_corners(state, parameters=TruckParameters()):
    """Eight corners: tractor first, then the trailer hanging from the fifth wheel."""
    p = parameters
    tractor = _rectangle(state[:2], state[2], p.tractor_rear, p.tractor_front, p.tractor_width)
    trailer = _rectangle(state[:2], state[3], p.trailer_rear, p.trailer_front, p.trailer_width)
    return jnp.concatenate((tractor, trailer))


def _smooth_violation(depth, smoothing):
    return smoothing * jax.nn.softplus(depth / smoothing)


def wall_penalty(state, parameters=TruckParameters()):
    p = parameters
    depths = p.wall_buffer - body_corners(state, p)[:, 1]
    return 50. * jnp.sum(_smooth_violation(depths, p.penalty_smoothing) ** 2)


def hitch_penalty(state, parameters=TruckParameters()):
    p = parameters
    excess = jnp.abs(hitch_angle(state)) - p.hitch_limit
    return 200. * _smooth_violation(excess, p.penalty_smoothing) ** 2


def stage_cost(state, control, goal, parameters=TruckParameters()):
    p = parameters
    offset = trailer_axle(state, p) - trailer_axle(jnp.asarray(goal), p)
    return p.dt * (0.05 * (control[0] / p.speed_limit) ** 2
                   + 0.2 * (control[1] / p.steering_limit) ** 2
                   + 0.002 * (offset @ offset)
                   + wall_penalty(state, p) + hitch_penalty(state, p))


def rate_cost(controls, parameters=TruckParameters()):
    """Charge control changes and a moving final interval; depends on the controls only."""
    p = parameters
    changes = jnp.diff(controls, axis=0) / jnp.array([p.speed_limit, p.steering_limit])
    return (jnp.sum(jnp.array([2.0, 2.0]) * changes ** 2) / p.dt
            + 5. * (controls[-1, 0] / p.speed_limit) ** 2)


def terminal_cost(state, goal, parameters=TruckParameters()):
    p = parameters
    goal = jnp.asarray(goal)
    offset = trailer_axle(state, p) - trailer_axle(goal, p)
    return (20. * (offset @ offset) + 200. * (1 - jnp.cos(state[3] - goal[3]))
            + 100. * (1 - jnp.cos(hitch_angle(state))) + wall_penalty(state, p))


def total_cost(states, controls, goal, parameters=TruckParameters()):
    running = jax.vmap(lambda x, u: stage_cost(x, u, goal, parameters))(states[:-1], controls)
    return (jnp.sum(running) + rate_cost(controls, parameters)
            + terminal_cost(states[-1], goal, parameters))


def shooting_objective(controls, initial, goal, parameters=TruckParameters()):
    """The reduced objective J(u): roll the plan out from x_0, then price it."""
    states = rollout(initial, controls, parameters)
    return total_cost(states, controls, goal, parameters)


def control_bounds(parameters=TruckParameters()):
    lower = np.tile([-parameters.speed_limit, -parameters.steering_limit], parameters.steps)
    return lower, -lower


def initial_controls(parameters=TruckParameters(), speed=-0.5):
    """Creep straight at a constant speed: a plausible seed, not a solution."""
    return np.tile([speed, 0.], (parameters.steps, 1))


def checkpoint_schedule(max_iterations, dense=10, factor=1.2):
    """Iterations whose full plans are kept: every early one, then geometrically spaced."""
    kept, value = set(range(dense + 1)), float(dense)
    while value < max_iterations:
        value *= factor
        kept.add(int(round(value)))
    kept.add(max_iterations)
    return kept


@dataclass
class ShootingResult:
    states: np.ndarray
    controls: np.ndarray
    history: list
    cost_trace: np.ndarray
    status: str
    iterations: int
    objective_evaluations: int
    projected_gradient: float
    solver_seconds: float

    @property
    def converged(self):
        return self.status == "converged"


def _projected_gradient(controls, gradient, lower, upper):
    return float(np.max(np.abs(controls - np.clip(controls - gradient, lower, upper))))


def solve_single_shooting(initial, goal, seed, parameters=TruckParameters(),
                          max_iterations=1500, gradient_tolerance=1e-5, relative_tolerance=1e-10,
                          checkpoints=None):
    """Minimize J(u) over the control sequence alone with L-BFGS-B.

    The objective and its reverse-mode gradient come from one compiled
    rollout. SciPy's bound-constrained L-BFGS-B handles the control box and
    stops on the projected gradient or on a small relative decrease. The
    cost of every iterate is kept; complete plans are kept at the checkpoint
    iterations and at the end.
    """
    p = parameters
    initial, goal, seed = (np.asarray(v, float) for v in (initial, goal, seed))
    if initial.shape != (4,) or goal.shape != (4,) or seed.shape != (p.steps, 2):
        raise ValueError("states need four entries and the seed needs steps x 2 controls")
    lower, upper = control_bounds(p)
    if np.any(seed.ravel() < lower) or np.any(seed.ravel() > upper):
        raise ValueError("initial controls must respect their bounds")
    if max_iterations < 0:
        raise ValueError("invalid iteration budget")
    checkpoints = checkpoint_schedule(max_iterations) if checkpoints is None else set(checkpoints)
    x0, target = jnp.asarray(initial), jnp.asarray(goal)

    @jax.jit
    def value_and_gradient(flat):
        controls = flat.reshape(p.steps, 2)
        value, gradient = jax.value_and_grad(
            lambda u: shooting_objective(u, x0, target, p))(controls)
        return value, gradient.ravel()

    roll = jax.jit(lambda flat: rollout(x0, flat.reshape(p.steps, 2), p))
    evaluations = 0

    def evaluate(flat):
        nonlocal evaluations
        evaluations += 1
        value, gradient = value_and_gradient(jnp.asarray(flat))
        return float(value), np.asarray(gradient)

    # Compile before timing.
    value_and_gradient(jnp.asarray(seed.ravel()))
    roll(jnp.asarray(seed.ravel()))
    history, cost_trace = [], []

    def record(iteration, flat, force=False):
        value, gradient = evaluate(flat)
        cost_trace.append(value)
        if force or iteration in checkpoints:
            history.append(dict(
                iteration=iteration, cost=value, states=np.asarray(roll(jnp.asarray(flat))),
                controls=np.asarray(flat).reshape(p.steps, 2).copy(),
                projected_gradient=_projected_gradient(flat, gradient, lower, upper)))

    count = 0

    def callback(flat):
        nonlocal count
        count += 1
        record(count, flat)

    started = perf_counter()
    record(0, seed.ravel(), force=True)
    outcome = minimize(evaluate, seed.ravel(), jac=True, method="L-BFGS-B",
                       bounds=list(zip(lower, upper)), callback=callback,
                       options={"maxiter": max_iterations, "gtol": gradient_tolerance,
                                "ftol": relative_tolerance, "maxcor": 30,
                                "maxfun": 20 * max_iterations})
    elapsed = perf_counter() - started
    final, iterations = np.asarray(outcome.x), int(outcome.nit)
    if history[-1]["iteration"] != iterations:
        record(iterations, final, force=True)
        cost_trace.pop()
    norm = history[-1]["projected_gradient"]
    if not np.all(np.isfinite(final)):
        status = "nonfinite"
    elif norm <= gradient_tolerance:
        status = "converged"
    elif iterations >= max_iterations:
        status = "iteration_limit"
    elif outcome.success:
        status = "small_decrease"
    else:
        status = "stalled"
    return ShootingResult(np.asarray(roll(jnp.asarray(final))), final.reshape(p.steps, 2),
                          history, np.asarray(cost_trace), status, iterations, evaluations,
                          float(norm), elapsed)


def terminal_sensitivity(initial, controls, parameters=TruckParameters()):
    """Reverse-mode Jacobian of the terminal state with respect to every control.

    Returns an array of shape (4, steps, 2): entry [i, t, j] is the derivative
    of the i-th terminal state coordinate with respect to control j at
    interval t. Its slices measure how strongly each interval's decision
    still influences where the rig ends up.
    """
    x0, us = jnp.asarray(initial), jnp.asarray(controls)
    jacobian = jax.jacrev(lambda u: rollout(x0, u, parameters)[-1])(us)
    return np.asarray(jacobian)


def hitch_sensitivity(initial, controls, parameters=TruckParameters()):
    """Derivative of the terminal hitch angle with respect to each steering angle."""
    jacobian = terminal_sensitivity(initial, controls, parameters)
    return jacobian[2, :, 1] - jacobian[3, :, 1]


def steering_amplification(initial, controls, parameters=TruckParameters()):
    """How much the final hitch angle magnifies each interval's steering effect.

    A steering perturbation at interval t first turns the tractor by about
    v_t dt / (L0 cos^2 delta_t). The ratio of the terminal hitch-angle
    derivative to that immediate turn is the amplification the rest of the
    rollout applies to it. Intervals with |v_t| below 0.1 m/s are masked
    with NaN because a stationary tractor cannot steer.
    """
    p = parameters
    controls = np.asarray(controls)
    immediate = np.abs(controls[:, 0]) * p.dt / (p.tractor_wheelbase * np.cos(controls[:, 1]) ** 2)
    ratio = np.abs(hitch_sensitivity(initial, controls, p)) / immediate
    return np.where(np.abs(controls[:, 0]) > 0.1, ratio, np.nan)


def linearized_amplification(states, controls, parameters=TruckParameters()):
    """Product of the per-interval hitch Jacobians from the end of interval t.

    Along the plan, the hitch angle beta = theta_0 - theta_1 obeys
    beta' = (v / L0) tan delta - (v / L1) sin beta, whose homogeneous part
    linearizes to -(v cos beta / L1) beta. Each backing interval therefore
    multiplies an earlier hitch perturbation by exp(-v cos(beta) dt / L1) > 1
    and each forward interval divides it. About a straight rig this is
    exp(d / L1), with d the distance still to be backed.
    """
    p = parameters
    states, controls = np.asarray(states), np.asarray(controls)
    beta = states[:-1, 2] - states[:-1, 3]
    factors = np.exp(-controls[:, 0] * np.cos(beta) * p.dt / p.trailer_length)
    # Half of interval t itself, then all later intervals.
    return np.sqrt(factors) * np.cumprod(factors[::-1])[::-1] / factors


def pull_out_plan(states, controls):
    """Time-reverse a backing plan: start docked, drive forward along the same path."""
    reversed_controls = np.asarray(controls)[::-1].copy()
    reversed_controls[:, 0] *= -1
    return np.asarray(states)[-1].copy(), reversed_controls


def audit(initial, controls, goal, parameters=TruckParameters(), subdivisions=4):
    """Independent, finer RK4 replay with arrival, clearance, and hitch checks."""
    p = parameters
    controls, goal = np.asarray(controls), np.asarray(goal)
    fine_controls = np.repeat(controls, subdivisions, axis=0)
    fine = np.asarray(jax.jit(lambda x, u: rollout(x, u, p, p.dt / subdivisions))(
        jnp.asarray(initial), jnp.asarray(fine_controls)))
    coarse = np.asarray(rollout(jnp.asarray(initial), jnp.asarray(controls), p))
    corners = np.asarray(jax.vmap(lambda x: body_corners(x, p))(jnp.asarray(fine)))
    end = fine[-1]
    axle_error = float(np.linalg.norm(np.asarray(trailer_axle(end, p))
                                      - np.asarray(trailer_axle(goal, p))))
    wrap = lambda a: np.arctan2(np.sin(a), np.cos(a))
    trailer_heading_error = float(abs(np.rad2deg(wrap(end[3] - goal[3]))))
    hitch_error = float(abs(np.rad2deg(wrap(end[2] - end[3]))))
    clearance = float(corners[:, :, 1].min())
    hitch_max = float(np.rad2deg(np.max(np.abs(wrap(fine[:, 2] - fine[:, 3])))))
    lower, upper = control_bounds(p)
    control_violation = float(max(0., np.max(controls.ravel() - upper),
                                  np.max(lower - controls.ravel())))
    difference = float(np.max(np.linalg.norm(fine[::subdivisions, :2] - coarse[:, :2], axis=1)))
    distance = float(np.sum(np.abs(controls[:, 0])) * p.dt)
    moving = controls[np.abs(controls[:, 0]) > 0.1, 0]
    reversals = int(np.sum(np.diff(np.sign(moving)) != 0))
    final_speed = float(abs(controls[-1, 0]))
    passed = bool(np.all(np.isfinite(fine)) and axle_error < .25 and trailer_heading_error < 2.
                  and hitch_error < 3. and clearance > 0. and final_speed < .1
                  and hitch_max < np.rad2deg(p.hitch_limit) and control_violation < 1e-9)
    return dict(axle_error_m=axle_error, trailer_heading_error_deg=trailer_heading_error,
                hitch_error_deg=hitch_error, minimum_clearance_m=clearance,
                maximum_hitch_deg=hitch_max, final_speed_mps=final_speed,
                control_violation=control_violation,
                fine_replay_max_position_difference_m=difference, distance_driven_m=distance,
                direction_reversals=reversals, fine_dt=p.dt / subdivisions, docking_pass=passed)
