"""Synthetic radiotherapy fractionation model on a two-dimensional thoracic slice.

Adapted from https://github.com/pierrelux/ddp-fractionation, which solves the
same kind of problem with box-constrained iLQR on a clinical phantom. Here the
phantom has four structures and three beams so that the chapter's teaching
solver applies directly. All coefficients are teaching values, not clinical
ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from trajectory_optimization import DifferentiableProblem

STRUCTURES = ("tumor", "spinal cord", "heart", "lung")
BEAMS = ("cord", "heart", "lung")                 # each beam named by the organ it crosses
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")


@dataclass(frozen=True)
class Phantom:
    """Shapes in centimetres; each organ sits in the path of one beam."""
    body_axes: tuple = (15.0, 12.0)
    tumor_radius: float = 2.0
    cord: tuple = (0.0, -7.5, 1.1)                  # centre x, centre y, radius
    heart: tuple = (-6.06, 3.5, 3.0, 2.6)           # centre x, centre y, semi-axes
    lung: tuple = (6.5, 3.75, 3.4, 2.6)
    beam_sources: tuple = (-90.0, 150.0, 30.0)      # degrees, side the beam enters from
    beam_half_width: float = 2.2
    penumbra: float = 0.3
    attenuation: float = 0.05                       # per cm of travel
    grid: int = 161
    extent: float = 16.0


@dataclass(frozen=True)
class CourseParameters:
    sessions: int = 20                               # four weeks, Monday to Friday
    alpha: tuple = (0.30, 0.10, 0.10, 0.10)          # 1/Gy
    beta: tuple = (0.030, 0.050, 0.030, 0.020)       # 1/Gy^2
    effect_scale: tuple = (22.0, 2.0, 2.0, 2.0)      # effect that cures / reaches tolerance
    tumor_growth: float = 0.005                      # per day
    repair: tuple = (0.0, 0.01, 0.04, 0.10)          # per day
    max_beam: float = 3.0                            # Gy at the tumor per beam and session
    dose_weight: float = 0.01
    tolerance_weight: float = 50.0
    cure_weight: float = 200.0
    sparing_weight: float = 1.0
    sharpness: float = 25.0
    uniform_beam: float = 0.7                        # initial plan, 2.1 Gy per session


def grid(phantom=Phantom()):
    axis = np.linspace(-phantom.extent, phantom.extent, phantom.grid)
    return np.meshgrid(axis, axis)


def masks(phantom=Phantom()):
    """Boolean masks for the body and the four structures."""
    X, Y = grid(phantom)
    a, b = phantom.body_axes
    body = (X / a) ** 2 + (Y / b) ** 2 <= 1
    tumor = X ** 2 + Y ** 2 <= phantom.tumor_radius ** 2
    cx, cy, r = phantom.cord
    cord = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2

    def ellipse(cx, cy, ax, ay):
        return ((X - cx) / ax) ** 2 + ((Y - cy) / ay) ** 2 <= 1

    return body, (tumor, cord, ellipse(*phantom.heart), ellipse(*phantom.lung))


def beam_doses(phantom=Phantom()):
    """Dose maps per unit intensity, normalized to a mean of 1 Gy on the tumor.

    Every structure lies inside the body, so the maps are not masked by it.
    """
    X, Y = grid(phantom)
    _, (tumor, *_) = masks(phantom)
    maps = []
    for source in phantom.beam_sources:
        travel = np.deg2rad(source + 180.0)
        dx, dy = np.cos(travel), np.sin(travel)
        across = -X * dy + Y * dx
        along = X * dx + Y * dy
        profile = 1 / (1 + np.exp((np.abs(across) - phantom.beam_half_width)
                                  / phantom.penumbra))
        dose = profile * np.exp(-phantom.attenuation * (along + phantom.extent))
        maps.append(dose / dose[tumor].mean())
    return np.array(maps)


def dose_matrix(phantom=Phantom()):
    """Mean dose to each structure per unit beam intensity, shape (4, 3)."""
    _, structures = masks(phantom)
    maps = beam_doses(phantom)
    return np.array([[m[s].mean() for m in maps] for s in structures])


def gaps(p=CourseParameters()):
    """Days from each session to the next; Friday is followed by a weekend."""
    return np.array([3.0 if t % 5 == 4 else 1.0 for t in range(p.sessions)])


def make_step(A, p=CourseParameters()):
    """Health after one session: linear-quadratic damage, then regrowth or repair."""
    A = jnp.asarray(A)
    alpha, beta = jnp.asarray(p.alpha), jnp.asarray(p.beta)
    scale, repair = jnp.asarray(p.effect_scale), jnp.asarray(p.repair)
    days = jnp.asarray(gaps(p))

    def step(health, beams, t):
        dose = A @ beams
        damaged = health - (alpha * dose + beta * dose ** 2) / scale
        factor = jnp.exp(-repair * days[t]).at[0].set(jnp.exp(p.tumor_growth * days[t]))
        return damaged * factor

    return step


def smooth_hinge(x, sharpness):
    return jax.nn.softplus(sharpness * x) / sharpness


def stage_cost(health, beams, p=CourseParameters()):
    excess = smooth_hinge(-(health[1:] + 1.0), p.sharpness)
    return 0.5 * p.dose_weight * jnp.sum(beams ** 2) + p.tolerance_weight * jnp.sum(excess)


def terminal_cost(health, p=CourseParameters()):
    excess = smooth_hinge(-(health[1:] + 1.0), p.sharpness)
    return (p.cure_weight * smooth_hinge(health[0], p.sharpness) ** 2
            + p.tolerance_weight * jnp.sum(excess)
            - p.sparing_weight * jnp.sum(health[1:]))


def make_problem(A, p=CourseParameters()):
    return DifferentiableProblem(
        4, make_step(A, p), lambda x, u, t: stage_cost(x, u, p),
        lambda x: terminal_cost(x, p), [0.0] * 3, [p.max_beam] * 3)


def initial_state():
    return np.array([1.0, 0.0, 0.0, 0.0])


def initial_controls(p=CourseParameters()):
    return np.full((p.sessions, 3), p.uniform_beam)


def uniform_curative_plan(problem, p=CourseParameters()):
    """Equal beams, the same every session, just strong enough to reach zero tumor health."""
    low, high = 0.0, p.max_beam
    for _ in range(60):
        level = 0.5 * (low + high)
        states = problem.rollout(initial_state(), np.full((p.sessions, 3), level))
        low, high = (level, high) if states[-1, 0] > 0 else (low, level)
    controls = np.full((p.sessions, 3), high)
    return problem.rollout(initial_state(), controls), controls
