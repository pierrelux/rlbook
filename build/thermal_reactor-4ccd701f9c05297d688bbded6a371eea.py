"""Illustrative 2D flow-through thermal reactor; all public fields use SI units.

Fields are (..., 2, ny, nx), with temperature [K] then unreacted fraction.
Four heater fractions are ordered upstream-bottom, downstream-bottom,
upstream-top, downstream-top. This is not a calibrated calciner model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import math
import numpy as np
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ReactorConfig:
    nx: int = 64
    ny: int = 32
    length: float = 8.0
    width: float = 2.0
    velocity: float = 0.2
    diffusivity: float = 0.0005
    loss: float = 0.01
    wall_temperature: float = 700.0
    reaction_cooling: float = 60.0
    heating_rate: float = 20.0
    inlet_temperature: float = 800.0
    transverse_amplitude: float = 10.0
    max_dt: float = 0.1
    control_dt: float = 2.0
    ou_tau: float = 20.0
    ou_sigma: float = 15.0
    temperature_limit: float = 1070.0
    conversion_target: float = 0.95
    velocity_shear: float = 0.0

    @property
    def state_dim(self):
        return 2 * self.nx * self.ny

    def to_dict(self):
        return asdict(self)


def reaction_rate(temperature):
    return 0.1 * jnp.exp(jnp.clip(12000.0 * (1 / 950.0 - 1 / temperature), -60, 30))


def heater_masks(config=ReactorConfig()):
    yy, xx = np.indices((config.ny, config.nx))
    return jnp.asarray(np.stack([
        ((xx >= config.nx // 2) == right) & ((yy >= config.ny // 2) == top)
        for top in (False, True) for right in (False, True)
    ]), dtype=jnp.float32)


def heating_field(controls, config=ReactorConfig()):
    return jnp.einsum("...j,jyx->...yx", controls, heater_masks(config))


def inlet_profile(disturbance, config=ReactorConfig()):
    y = (jnp.arange(config.ny) + 0.5) / config.ny
    return (config.inlet_temperature + jnp.asarray(disturbance)[..., None]
            + config.transverse_amplitude * jnp.cos(jnp.pi * y))


def velocity_profile(config=ReactorConfig()):
    """Positive downstream face velocity [m/s]; no transverse velocity.

    The continuous transverse mean is ``velocity``. Zero shear reproduces
    the original uniform flow; 0.2 gives 0.16 + 0.24 eta (1-eta) m/s.
    """
    eta = (jnp.arange(config.ny) + .5) / config.ny
    return config.velocity * (1 + config.velocity_shear * (6 * eta * (1 - eta) - 1))


def transport_rhs(field, inlet, config=ReactorConfig()):
    """Conservative upwind advection; zero diffusive boundary fluxes."""
    dx, dy = config.length / config.nx, config.width / config.ny
    inlet = jnp.broadcast_to(inlet, field.shape[:-1])
    up = jnp.concatenate([inlet[..., None], field[..., :-1]], axis=-1)
    left = jnp.concatenate([field[..., :1], field[..., :-1]], axis=-1)
    right = jnp.concatenate([field[..., 1:], field[..., -1:]], axis=-1)
    below = jnp.concatenate([field[..., :1, :], field[..., :-1, :]], axis=-2)
    above = jnp.concatenate([field[..., 1:, :], field[..., -1:, :]], axis=-2)
    lap = (left - 2 * field + right) / dx**2 + (below - 2 * field + above) / dy**2
    # Each row has the same velocity at its two streamwise faces, so this
    # difference is exactly the divergence of conservative advective fluxes.
    return -velocity_profile(config)[:, None] * (field - up) / dx + config.diffusivity * lap


def rhs(fields, controls, disturbance, config=ReactorConfig()):
    return rhs_with_inlet(fields, controls, inlet_profile(disturbance, config), config)


def rhs_with_inlet(fields, controls, inlet_temperature, config=ReactorConfig()):
    """Field derivative for explicit inlet profiles (..., ny), in kelvin."""
    T, c = fields[..., 0, :, :], fields[..., 1, :, :]
    r = reaction_rate(T) * c
    dT = (transport_rhs(T, inlet_temperature, config)
          - config.loss * (T - config.wall_temperature)
          - config.reaction_cooling * r
          + config.heating_rate * heating_field(controls, config))
    dc = transport_rhs(c, jnp.ones_like(c[..., :, 0]), config) - r
    return jnp.stack([dT, dc], axis=-3)


def stable_step_size(fields, config=ReactorConfig()):
    dx, dy = config.length / config.nx, config.width / config.ny
    transport = jnp.max(velocity_profile(config)) / dx + 2 * config.diffusivity * (1 / dx**2 + 1 / dy**2)
    max_rate = jnp.max(reaction_rate(fields[..., 0, :, :] + config.heating_rate * config.max_dt))
    return jnp.minimum(config.max_dt, .75 / (transport + config.loss + max_rate))


@partial(jax.jit, static_argnames=("config",))
def physics_step(fields, controls, disturbance, elapsed=2.0, config=ReactorConfig()):
    """Advance a batch with SSPRK2 and a reaction/transport stability bound.

    The inlet disturbance and heating controls are held over ``elapsed``.
    No state clipping or projection is used. Outputs are checked by callers.
    """
    fields = jnp.asarray(fields, dtype=jnp.float32)
    controls, disturbance = jnp.asarray(controls, dtype=jnp.float32), jnp.asarray(disturbance, dtype=jnp.float32)
    def advance(carry):
        t, x = carry
        # Include a full maximum-step heating rise when bounding reaction rates.
        h = stable_step_size(x, config)
        h = jnp.minimum(h, elapsed - t)
        first = x + h * rhs(x, controls, disturbance, config)
        second = .5 * x + .5 * (first + h * rhs(first, controls, disturbance, config))
        return t + h, second

    return jax.lax.while_loop(lambda z: z[0] < elapsed - 1e-6, advance,
                              (jnp.float32(0), fields))[1]


@partial(jax.jit, static_argnames=("config",))
def physics_rollout(initial, controls, disturbances, config=ReactorConfig()):
    """Time-major controls and inlet disturbances; returns subsequent fields."""
    def advance(x, ud):
        following = physics_step(x, ud[0], ud[1], config.control_dt, config)
        return following, following
    return jax.lax.scan(advance, initial, (controls, disturbances))[1]


def field_metrics(fields, config=None):
    """Outlet conversion is flux-weighted when a flow configuration is given.

    Omitting config preserves the original uniform-flow metric.
    """
    fields = np.asarray(fields)
    conversion = (1 - np.mean(fields[..., 1, :, -1], axis=-1) if config is None else
                  1 - np.average(fields[..., 1, :, -1], axis=-1, weights=np.asarray(velocity_profile(config))))
    return {"peak_temperature_K": np.max(fields[..., 0, :, :], axis=(-2, -1)),
            "outlet_conversion": conversion}


def valid_fields(fields):
    x = np.asarray(fields)
    return bool(np.isfinite(x).all() and np.min(x[..., 0, :, :]) > 0
                and np.min(x[..., 1, :, :]) >= -2e-5
                and np.max(x[..., 1, :, :]) <= 1 + 2e-5)


def warm_state(config=ReactorConfig(), heating=.48):
    x = jnp.stack([jnp.full((config.ny, config.nx), 850.),
                   jnp.ones((config.ny, config.nx))])
    return np.asarray(physics_step(x, jnp.full(4, heating), 0., 400., config))


def ou_forecasts(current, steps, ensemble, rng, config=ReactorConfig(), mean_only=False):
    """Forecast held inlet values: the first interval uses the observation.

    Returns (steps, ensemble), independently generated by the caller's RNG.
    No realized future disturbance is an argument to this function.
    """
    a = math.exp(-config.control_dt / config.ou_tau)
    sigma = config.ou_sigma * math.sqrt(1 - a * a)
    g = np.full(ensemble, current, dtype=float)
    result = []
    for _ in range(steps):
        result.append(g.copy())
        g = a * g + (0 if mean_only else sigma * rng.normal(size=ensemble))
    return np.asarray(result, dtype=np.float32)


def random_streams(seed):
    """Independent physical, proposal, and forecast streams."""
    return tuple(np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))


class Physics:
    """Reference model with the same batched transition contract as Surrogate."""
    def __init__(self, config=ReactorConfig()):
        self.config = config

    def step(self, fields, controls, disturbance, elapsed=2.0):
        return physics_step(fields, controls, disturbance, elapsed, self.config)

    def rollout(self, initial, controls, disturbances):
        return physics_rollout(initial, controls, disturbances, self.config)


@partial(jax.jit, static_argnames=("config", "resolution"))
def fine_rollout_diagnostics(initial, controls, disturbances, config=ReactorConfig(), resolution=.1):
    """Replay controls on a finer observation grid; keep scalar traces only.

    These are discrete residual checks, not continuous-time guarantees.
    Each physical integration step still obeys the SSPRK2 stability bound.
    """
    per_interval = int(round(config.control_dt / resolution))
    if abs(per_interval * resolution - config.control_dt) > 1e-6:
        raise ValueError("diagnostic resolution must divide the control interval")
    us = jnp.repeat(controls, per_interval, axis=0)
    gs = jnp.repeat(disturbances, per_interval, axis=0)
    def advance(x, ug):
        following = physics_step(x, ug[0], ug[1], resolution, config)
        return following, jnp.stack([jnp.max(following[0]), 1 - jnp.mean(following[1, :, -1])])
    return jax.lax.scan(advance, initial, (us, gs))[1]
