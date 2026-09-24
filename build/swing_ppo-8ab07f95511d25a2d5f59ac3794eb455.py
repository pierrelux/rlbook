"""Train, evaluate, and render PPO on the actual SwingRL standing model.

The textbook build only reads artifacts produced by this module.  Training is
an explicit offline step so that a documentation rebuild never launches a
long experiment or replaces a recorded result.

The implementation deliberately has few dependencies.  SwingRL supplies the
plant and renderer, JAX evaluates the plant in batches, and Optax applies PPO
updates.  No separate deep-RL framework is used.
"""

from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import distribution, version
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np
import optax

from swing_rl.envs import SwingEnv
from swing_rl.jaxsim import rider_for
from swing_rl.physics import RewardParams, SwingParams
from swing_rl.physics.models import articulated_standing
from swing_rl.viz import SwingAnimation, record_episode


ArrayTree = Mapping[str, Any]

OBSERVATION_SCALE = jnp.asarray(
    [1.0, 1.0, 25.0, 1.0, 1.0, 25.0, 5.0, 25.0, 25.0, 25.0, 25.0],
    dtype=jnp.float64,
)

OI = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
}

FIGURE_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
}


@dataclass(frozen=True)
class PPOConfig:
    """Fixed protocol used by the textbook experiment."""

    seed: int = 0
    total_environment_steps: int = 1_000_000
    number_of_environments: int = 8
    rollout_steps: int = 256
    episode_steps: int = 1_600
    reset_noise: float = 0.01
    hidden_units: int = 64
    learning_rate: float = 3e-4
    discount: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.001
    maximum_gradient_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    checkpoint_interval: int = 50_000
    evaluation_states: int = 100
    evaluation_seed: int = 20_260_901

    @property
    def batch_size(self) -> int:
        return self.number_of_environments * self.rollout_steps

    @property
    def number_of_updates(self) -> int:
        return math.ceil(self.total_environment_steps / self.batch_size)

    @property
    def actual_environment_steps(self) -> int:
        return self.number_of_updates * self.batch_size

    @property
    def gradient_steps(self) -> int:
        minibatches = math.ceil(self.batch_size / self.minibatch_size)
        return self.number_of_updates * self.update_epochs * minibatches


def make_swing_components(
    *, reset_noise: float = 0.0, episode_steps: int = 1_600
) -> tuple[SwingParams, Any, Any, RewardParams]:
    """Create the plant and reward shared by training and both baselines."""

    parameters = SwingParams()
    model = articulated_standing(parameters)
    rider = rider_for(model, parameters)
    reward = RewardParams(success_angle=2.0 * np.pi)
    environment = SwingEnv(
        swing=parameters,
        model=model,
        rider=rider,
        reward=reward,
        max_episode_steps=episode_steps,
        reset_noise=reset_noise,
    )
    return parameters, model, environment, reward


def _dense_init(
    generator: np.random.Generator,
    input_size: int,
    output_size: int,
    *,
    gain: float,
) -> dict[str, jax.Array]:
    """Initialize one dense layer with scaled Gaussian weights."""

    scale = gain / math.sqrt(float(input_size))
    return {
        "weight": jnp.asarray(
            generator.normal(0.0, scale, size=(input_size, output_size)),
            dtype=jnp.float64,
        ),
        "bias": jnp.zeros((output_size,), dtype=jnp.float64),
    }


def initialize_parameters(config: PPOConfig) -> dict[str, Any]:
    """Initialize separate two-layer actor and critic networks."""

    generator = np.random.default_rng(config.seed)
    hidden = config.hidden_units
    actor = {
        "layer_1": _dense_init(generator, 11, hidden, gain=math.sqrt(2.0)),
        "layer_2": _dense_init(generator, hidden, hidden, gain=math.sqrt(2.0)),
        "output": _dense_init(generator, hidden, 2, gain=0.01),
    }
    critic = {
        "layer_1": _dense_init(generator, 11, hidden, gain=math.sqrt(2.0)),
        "layer_2": _dense_init(generator, hidden, hidden, gain=math.sqrt(2.0)),
        "output": _dense_init(generator, hidden, 1, gain=1.0),
    }
    return {
        "actor": actor,
        "critic": critic,
        "log_standard_deviation": jnp.full((2,), -0.5, dtype=jnp.float64),
    }


def _dense(layer: ArrayTree, inputs: jax.Array) -> jax.Array:
    return inputs @ layer["weight"] + layer["bias"]


def _mlp(network: ArrayTree, observations: jax.Array) -> jax.Array:
    normalized = observations / OBSERVATION_SCALE
    hidden = jnp.tanh(_dense(network["layer_1"], normalized))
    hidden = jnp.tanh(_dense(network["layer_2"], hidden))
    return _dense(network["output"], hidden)


def policy_mean(parameters: ArrayTree, observations: jax.Array) -> jax.Array:
    return _mlp(parameters["actor"], observations)


def value_estimate(parameters: ArrayTree, observations: jax.Array) -> jax.Array:
    return jnp.squeeze(_mlp(parameters["critic"], observations), axis=-1)


def tanh_log_probability(
    mean: jax.Array,
    log_standard_deviation: jax.Array,
    latent_action: jax.Array,
) -> jax.Array:
    """Log density after a tanh transform, evaluated from the latent action."""

    standardized = (latent_action - mean) * jnp.exp(-log_standard_deviation)
    gaussian = (
        -0.5 * jnp.square(standardized)
        - log_standard_deviation
        - 0.5 * math.log(2.0 * math.pi)
    )
    # log(1 - tanh(z)^2), written without subtracting nearly equal numbers.
    log_jacobian = 2.0 * (
        math.log(2.0) - latent_action - jax.nn.softplus(-2.0 * latent_action)
    )
    return jnp.sum(gaussian - log_jacobian, axis=-1)


def policy_log_probability(
    parameters: ArrayTree, observations: jax.Array, latent_actions: jax.Array
) -> jax.Array:
    return tanh_log_probability(
        policy_mean(parameters, observations),
        parameters["log_standard_deviation"],
        latent_actions,
    )


def sample_policy(
    parameters: ArrayTree, observations: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample latent and bounded actions and return their log density and value."""

    mean = policy_mean(parameters, observations)
    noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    latent = mean + jnp.exp(parameters["log_standard_deviation"]) * noise
    actions = jnp.tanh(latent)
    log_probability = tanh_log_probability(
        mean, parameters["log_standard_deviation"], latent
    )
    return latent, actions, log_probability, value_estimate(parameters, observations)


def deterministic_policy(parameters: ArrayTree, observations: jax.Array) -> jax.Array:
    return jnp.tanh(policy_mean(parameters, observations))


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    episode_ended: np.ndarray,
    *,
    discount: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE without carrying estimates across episode boundaries."""

    advantages = np.zeros_like(rewards, dtype=np.float64)
    accumulator = np.zeros(rewards.shape[1], dtype=np.float64)
    for step in range(rewards.shape[0] - 1, -1, -1):
        bootstrap = 1.0 - terminated[step].astype(np.float64)
        continuation = 1.0 - episode_ended[step].astype(np.float64)
        delta = (
            rewards[step]
            + discount * bootstrap * next_values[step]
            - values[step]
        )
        accumulator = delta + discount * gae_lambda * continuation * accumulator
        advantages[step] = accumulator
    return advantages, advantages + values


def _select_tree(mask: jax.Array, when_true: Any, when_false: Any) -> Any:
    def select(true_leaf: jax.Array, false_leaf: jax.Array) -> jax.Array:
        shape = (mask.shape[0],) + (1,) * (true_leaf.ndim - 1)
        return jnp.where(mask.reshape(shape), true_leaf, false_leaf)

    return jax.tree_util.tree_map(select, when_true, when_false)


def _initial_batch(simulator: Any, key: jax.Array, count: int, noise: float) -> Any:
    positions = noise * jax.random.normal(key, (count, simulator.n_free, 2))
    q = positions[:, :, 0]
    q_dot = positions[:, :, 1]
    states = jax.vmap(simulator.initial_state)(q, q_dot)
    return states._replace(wound=q[:, 0])


def _observations_and_potential(simulator: Any, states: Any) -> tuple[jax.Array, jax.Array]:
    actions = jnp.zeros((states.time.shape[0], simulator.n_actions), dtype=jnp.float64)
    observations = jax.vmap(simulator.observe)(states, actions)
    summaries = jax.vmap(simulator.summary)(states, actions)
    energy_index = simulator.summary_fields.index("energy_norm")
    return observations, jnp.clip(summaries[:, energy_index], -0.5, 1.5)


def initialize_training_state(
    simulator: Any, config: PPOConfig, key: jax.Array
) -> dict[str, Any]:
    key, reset_key = jax.random.split(key)
    states = _initial_batch(
        simulator, reset_key, config.number_of_environments, config.reset_noise
    )
    observations, potential = _observations_and_potential(simulator, states)
    return {
        "states": states,
        "observations": observations,
        "last_potential": potential,
        "episode_returns": jnp.zeros(config.number_of_environments),
        "episode_lengths": jnp.zeros(config.number_of_environments, dtype=jnp.int32),
        "key": key,
    }


def make_rollout_function(
    simulator: Any, reward_parameters: RewardParams, config: PPOConfig
) -> Callable[[ArrayTree, Mapping[str, Any]], tuple[dict[str, Any], dict[str, jax.Array]]]:
    """Compile collection on the exact SwingRL simulator."""

    energy_index = simulator.summary_fields.index("energy_norm")
    angle_index = simulator.summary_fields.index("swing_angle")
    tension_index = simulator.summary_fields.index("tension")
    effort_index = simulator.summary_fields.index("command_speed")
    radius_index = simulator.summary_fields.index("seat_radius")
    minimum_radius = (
        reward_parameters.success_radius_fraction * simulator.spec.nominal_length
    )
    number_of_environments = config.number_of_environments

    def one_step(
        supplied_parameters: ArrayTree, carry: Mapping[str, Any], _: None
    ):
        key, sample_key, reset_key = jax.random.split(carry["key"], 3)
        latent, actions, log_probability, values = sample_policy(
            supplied_parameters, carry["observations"], sample_key
        )
        next_states, next_observations, summaries = jax.vmap(simulator.advance)(
            carry["states"], actions
        )

        potential = jnp.clip(summaries[:, energy_index], -0.5, 1.5)
        effort = summaries[:, effort_index]
        reward = reward_parameters.energy_shaping * (
            reward_parameters.gamma * potential - carry["last_potential"]
        )
        reward = reward - reward_parameters.effort_cost * effort
        reward = reward - reward_parameters.time_cost

        success = jnp.logical_and(
            jnp.abs(summaries[:, angle_index]) >= reward_parameters.success_angle,
            summaries[:, radius_index] >= minimum_radius,
        )
        reward = reward + reward_parameters.success_bonus * success
        episode_lengths = carry["episode_lengths"] + 1
        truncated = jnp.logical_and(
            episode_lengths >= config.episode_steps, jnp.logical_not(success)
        )
        ended = jnp.logical_or(success, truncated)
        episode_returns = carry["episode_returns"] + reward
        completed_returns = jnp.where(ended, episode_returns, jnp.nan)

        # GAE bootstraps time-limit truncations from the last physical state,
        # but does not bootstrap a successful terminal transition.
        next_values = value_estimate(supplied_parameters, next_observations)

        reset_states = _initial_batch(
            simulator, reset_key, number_of_environments, config.reset_noise
        )
        reset_observations, reset_potential = _observations_and_potential(
            simulator, reset_states
        )
        states = _select_tree(ended, reset_states, next_states)
        observations = jnp.where(
            ended[:, None], reset_observations, next_observations
        )
        last_potential = jnp.where(ended, reset_potential, potential)

        next_carry = {
            "states": states,
            "observations": observations,
            "last_potential": last_potential,
            "episode_returns": jnp.where(ended, 0.0, episode_returns),
            "episode_lengths": jnp.where(ended, 0, episode_lengths),
            "key": key,
        }
        transition = {
            "observations": carry["observations"],
            "latent_actions": latent,
            "actions": actions,
            "log_probabilities": log_probability,
            "values": values,
            "next_values": next_values,
            "rewards": reward,
            "terminated": success,
            "episode_ended": ended,
            "completed_returns": completed_returns,
            "tensions": summaries[:, tension_index],
            "effort": effort,
        }
        return next_carry, transition

    def collect(
        supplied_parameters: ArrayTree, training_state: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, jax.Array]]:
        return jax.lax.scan(
            lambda carry, item: one_step(supplied_parameters, carry, item),
            training_state,
            None,
            length=config.rollout_steps,
        )

    return jax.jit(collect)


def ppo_loss(
    parameters: ArrayTree,
    batch: Mapping[str, jax.Array],
    config: PPOConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    log_probability = policy_log_probability(
        parameters, batch["observations"], batch["latent_actions"]
    )
    log_ratio = jnp.clip(log_probability - batch["old_log_probabilities"], -20.0, 20.0)
    ratio = jnp.exp(log_ratio)
    unclipped = ratio * batch["advantages"]
    clipped = jnp.clip(
        ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
    ) * batch["advantages"]
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    values = value_estimate(parameters, batch["observations"])
    value_loss = 0.5 * jnp.mean(jnp.square(values - batch["returns"]))
    sampled_entropy = -jnp.mean(log_probability)
    total_loss = (
        policy_loss
        + config.value_coefficient * value_loss
        - config.entropy_coefficient * sampled_entropy
    )
    statistics = {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": sampled_entropy,
        "approximate_kl": jnp.mean((ratio - 1.0) - log_ratio),
        "clip_fraction": jnp.mean(
            jnp.abs(ratio - 1.0) > config.clip_epsilon
        ),
    }
    return total_loss, statistics


def make_update_function(
    optimizer: optax.GradientTransformation, config: PPOConfig
) -> Callable:
    def update(parameters, optimizer_state, batch):
        (_, statistics), gradients = jax.value_and_grad(ppo_loss, has_aux=True)(
            parameters, batch, config
        )
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, parameters
        )
        parameters = optax.apply_updates(parameters, updates)
        statistics = dict(statistics)
        squared_norm = sum(
            jnp.sum(jnp.square(leaf))
            for leaf in jax.tree_util.tree_leaves(gradients)
        )
        statistics["gradient_norm"] = jnp.sqrt(squared_norm)
        return parameters, optimizer_state, statistics

    return jax.jit(update)


def held_out_initial_states(config: PPOConfig) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(config.evaluation_seed)
    theta = generator.uniform(
        np.deg2rad(-5.0), np.deg2rad(5.0), size=config.evaluation_states
    )
    theta_dot = generator.uniform(-0.1, 0.1, size=config.evaluation_states)
    return theta, theta_dot


def _structured_actions(
    observations: jax.Array, time_step: jax.Array, natural_frequency: float
) -> jax.Array:
    psi = jnp.arctan2(observations[:, 4], observations[:, 3])
    psi_dot = observations[:, 5] * natural_frequency
    cosine_amplitude = jnp.cos(psi) - jnp.square(psi_dot) / (
        2.0 * natural_frequency**2
    )
    amplitude = jnp.arccos(jnp.clip(cosine_amplitude, -1.0, 1.0))
    phase = jnp.arctan2(-psi_dot / natural_frequency, psi)
    clock = natural_frequency * (time_step + 1) * 0.02
    source = jnp.where(amplitude < 0.03, clock, phase)
    return jnp.stack(
        [
            jnp.sin(2.0 * source + jnp.deg2rad(67.5)),
            jnp.sin(source + jnp.deg2rad(247.5)),
        ],
        axis=-1,
    )


def evaluate_policy_batch(
    parameters: ArrayTree | None,
    config: PPOConfig,
    *,
    structured: bool = False,
) -> dict[str, Any]:
    """Evaluate one policy on 100 fixed starts using the exact SwingRL plant."""

    swing, _, environment, reward_parameters = make_swing_components(
        episode_steps=config.episode_steps
    )
    simulator = environment.sim
    theta, theta_dot = held_out_initial_states(config)
    q = jnp.asarray(theta[:, None])
    q_dot = jnp.asarray(theta_dot[:, None])
    states = jax.vmap(simulator.initial_state)(q, q_dot)
    states = states._replace(wound=q[:, 0])
    observations, potential = _observations_and_potential(simulator, states)

    energy_index = simulator.summary_fields.index("energy_norm")
    angle_index = simulator.summary_fields.index("swing_angle")
    tension_index = simulator.summary_fields.index("tension")
    effort_index = simulator.summary_fields.index("command_speed")
    radius_index = simulator.summary_fields.index("seat_radius")
    minimum_radius = (
        reward_parameters.success_radius_fraction * simulator.spec.nominal_length
    )
    count = config.evaluation_states

    initial_carry = {
        "states": states,
        "observations": observations,
        "potential": potential,
        "done": jnp.zeros(count, dtype=bool),
        "returns": jnp.zeros(count),
        "success_time": jnp.full(count, jnp.nan),
        "effort": jnp.zeros(count),
        "minimum_tension": jnp.full(count, jnp.inf),
        "negative_tension_steps": jnp.zeros(count),
        "active_steps": jnp.zeros(count),
    }

    def step(carry, time_step):
        active = jnp.logical_not(carry["done"])
        if structured:
            actions = _structured_actions(
                carry["observations"], time_step, swing.natural_frequency
            )
        else:
            assert parameters is not None
            actions = deterministic_policy(parameters, carry["observations"])

        next_states, next_observations, summaries = jax.vmap(simulator.advance)(
            carry["states"], actions
        )
        next_potential = jnp.clip(summaries[:, energy_index], -0.5, 1.5)
        rewards = reward_parameters.energy_shaping * (
            reward_parameters.gamma * next_potential - carry["potential"]
        )
        rewards = rewards - reward_parameters.effort_cost * summaries[:, effort_index]
        rewards = rewards - reward_parameters.time_cost
        success = jnp.logical_and(
            active,
            jnp.logical_and(
                jnp.abs(summaries[:, angle_index]) >= reward_parameters.success_angle,
                summaries[:, radius_index] >= minimum_radius,
            ),
        )
        rewards = rewards + reward_parameters.success_bonus * success
        done = jnp.logical_or(carry["done"], success)
        next_states = _select_tree(carry["done"], carry["states"], next_states)
        next_observations = jnp.where(
            carry["done"][:, None], carry["observations"], next_observations
        )
        next_potential = jnp.where(
            carry["done"], carry["potential"], next_potential
        )
        success_time = jnp.where(
            success, summaries[:, simulator.summary_fields.index("time")], carry["success_time"]
        )
        next_carry = {
            "states": next_states,
            "observations": next_observations,
            "potential": next_potential,
            "done": done,
            "returns": carry["returns"] + jnp.where(active, rewards, 0.0),
            "success_time": success_time,
            "effort": carry["effort"]
            + jnp.where(active, 0.02 * summaries[:, effort_index], 0.0),
            "minimum_tension": jnp.where(
                active,
                jnp.minimum(carry["minimum_tension"], summaries[:, tension_index]),
                carry["minimum_tension"],
            ),
            "negative_tension_steps": carry["negative_tension_steps"]
            + jnp.where(
                jnp.logical_and(active, summaries[:, tension_index] < 0.0), 1.0, 0.0
            ),
            "active_steps": carry["active_steps"] + active.astype(jnp.float64),
        }
        return next_carry, None

    final, _ = jax.jit(
        lambda carry: jax.lax.scan(step, carry, jnp.arange(config.episode_steps))
    )(initial_carry)
    success = np.asarray(final["done"], dtype=bool)
    success_times = np.asarray(final["success_time"], dtype=float)
    returns = np.asarray(final["returns"], dtype=float)
    effort = np.asarray(final["effort"], dtype=float)
    minimum_tension = np.asarray(final["minimum_tension"], dtype=float)
    negative_rate = np.asarray(
        final["negative_tension_steps"] / jnp.maximum(final["active_steps"], 1.0),
        dtype=float,
    )
    conditional_time = float(np.nanmean(success_times)) if np.any(success) else math.nan
    return {
        "episode_count": int(count),
        "success_rate": float(np.mean(success)),
        "return_mean": float(np.mean(returns)),
        "return_standard_deviation": float(np.std(returns, ddof=1)),
        "conditional_time_to_rotation": conditional_time,
        "integrated_effort_mean": float(np.mean(effort)),
        "minimum_tension_mean": float(np.mean(minimum_tension)),
        "minimum_tension_worst": float(np.min(minimum_tension)),
        "negative_tension_rate_mean": float(np.mean(negative_rate)),
    }


def _flatten_transitions(transitions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: values.reshape((-1,) + values.shape[2:])
        for name, values in transitions.items()
    }


def _parameter_arrays(parameters: ArrayTree) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for name, child in value.items():
                visit(f"{prefix}__{name}" if prefix else name, child)
        else:
            arrays[prefix] = np.asarray(value)

    visit("", parameters)
    return arrays


def save_parameters(parameters: ArrayTree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **_parameter_arrays(parameters))


def load_parameters(path: Path) -> dict[str, Any]:
    loaded = np.load(path)
    parameters: dict[str, Any] = {}
    for flat_name in loaded.files:
        cursor = parameters
        pieces = flat_name.split("__")
        for piece in pieces[:-1]:
            cursor = cursor.setdefault(piece, {})
        cursor[pieces[-1]] = jnp.asarray(loaded[flat_name])
    return parameters


class PPOPolicy:
    """NumPy-callable deterministic policy for SwingRL's recorder."""

    def __init__(self, parameters: ArrayTree):
        self.parameters = parameters

    def reset(self) -> None:
        return None

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        batch = jnp.asarray(observation, dtype=jnp.float64)[None, :]
        return np.asarray(deterministic_policy(self.parameters, batch)[0], dtype=np.float32)


def record_showcase(parameters: ArrayTree, config: PPOConfig):
    _, _, environment, _ = make_swing_components(episode_steps=config.episode_steps)
    return record_episode(
        environment,
        PPOPolicy(parameters),
        seed=0,
        options={"theta": np.deg2rad(3.0), "theta_dot": 0.0, "noise": 0.0},
    )


def save_showcase_trace(rollout: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    commands = np.stack([frame.commands for frame in rollout.frames])
    np.savez_compressed(
        path,
        time=rollout.times,
        theta=rollout.thetas,
        energy=rollout.energies,
        tension=rollout.tensions,
        commands=commands,
        success=np.asarray(rollout.success),
        success_time=np.asarray(
            math.nan if rollout.success_time is None else rollout.success_time
        ),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _swing_rl_metadata() -> dict[str, Any]:
    package = distribution("swing-rl")
    direct_url = package.read_text("direct_url.json")
    return {
        "version": version("swing-rl"),
        "direct_url": json.loads(direct_url) if direct_url else None,
    }


def _checkpoint_targets(config: PPOConfig) -> list[int]:
    return list(
        range(0, config.total_environment_steps + 1, config.checkpoint_interval)
    )


def train_seed(config: PPOConfig, output_directory: Path) -> dict[str, Any]:
    """Train one prespecified seed and persist every checkpoint and raw log."""

    seed_directory = output_directory / f"seed_{config.seed}"
    checkpoint_directory = seed_directory / "checkpoints"
    trace_directory = seed_directory / "showcase"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    trace_directory.mkdir(parents=True, exist_ok=True)

    _, _, environment, reward_parameters = make_swing_components(
        reset_noise=config.reset_noise, episode_steps=config.episode_steps
    )
    simulator = environment.sim
    parameters = initialize_parameters(config)
    key = jax.random.PRNGKey(config.seed)
    training_state = initialize_training_state(simulator, config, key)
    collect = make_rollout_function(simulator, reward_parameters, config)

    schedule = optax.linear_schedule(
        init_value=config.learning_rate,
        end_value=0.0,
        transition_steps=max(config.gradient_steps, 1),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.maximum_gradient_norm),
        optax.adam(schedule),
    )
    optimizer_state = optimizer.init(parameters)
    update = make_update_function(optimizer, config)
    shuffle_generator = np.random.default_rng(config.seed + 100_000)

    update_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    targets = _checkpoint_targets(config)
    target_index = 0
    start_time = time.perf_counter()

    def save_checkpoint(target: int, actual_steps: int) -> None:
        elapsed = time.perf_counter() - start_time
        metrics = evaluate_policy_batch(parameters, config)
        checkpoint_path = checkpoint_directory / f"step_{actual_steps:07d}.npz"
        save_parameters(parameters, checkpoint_path)
        showcase = record_showcase(parameters, config)
        trace_path = trace_directory / f"step_{actual_steps:07d}.npz"
        save_showcase_trace(showcase, trace_path)
        row = {
            "seed": config.seed,
            "checkpoint_target": target,
            "environment_steps": actual_steps,
            "elapsed_seconds": elapsed,
            **metrics,
            "checkpoint": str(checkpoint_path.relative_to(output_directory)),
            "showcase_trace": str(trace_path.relative_to(output_directory)),
        }
        checkpoint_rows.append(row)
        _write_csv(seed_directory / "checkpoints.csv", checkpoint_rows)

    save_checkpoint(targets[0], 0)
    target_index = 1

    for update_index in range(config.number_of_updates):
        training_state, transitions_jax = collect(parameters, training_state)
        transitions = {
            name: np.asarray(value) for name, value in transitions_jax.items()
        }
        advantages, returns = compute_gae(
            transitions["rewards"],
            transitions["values"],
            transitions["next_values"],
            transitions["terminated"],
            transitions["episode_ended"],
            discount=config.discount,
            gae_lambda=config.gae_lambda,
        )
        flat = _flatten_transitions(
            {
                "observations": transitions["observations"],
                "latent_actions": transitions["latent_actions"],
                "old_log_probabilities": transitions["log_probabilities"],
                "advantages": advantages,
                "returns": returns,
            }
        )
        normalized_advantages = flat["advantages"]
        normalized_advantages = (
            normalized_advantages - normalized_advantages.mean()
        ) / (normalized_advantages.std() + 1e-8)
        flat["advantages"] = normalized_advantages

        statistics_accumulator: list[dict[str, float]] = []
        for _ in range(config.update_epochs):
            permutation = shuffle_generator.permutation(config.batch_size)
            for start in range(0, config.batch_size, config.minibatch_size):
                indices = permutation[start : start + config.minibatch_size]
                batch = {
                    name: jnp.asarray(values[indices]) for name, values in flat.items()
                }
                parameters, optimizer_state, statistics = update(
                    parameters, optimizer_state, batch
                )
                statistics_accumulator.append(
                    {name: float(value) for name, value in statistics.items()}
                )

        completed = transitions["completed_returns"]
        completed = completed[np.isfinite(completed)]
        actual_steps = (update_index + 1) * config.batch_size
        row = {
            "seed": config.seed,
            "update": update_index + 1,
            "environment_steps": actual_steps,
            "elapsed_seconds": time.perf_counter() - start_time,
            "mean_step_reward": float(np.mean(transitions["rewards"])),
            "mean_episode_return": (
                float(np.mean(completed)) if completed.size else math.nan
            ),
            "episodes_completed": int(completed.size),
            "minimum_tension": float(np.min(transitions["tensions"])),
            "mean_command_speed": float(np.mean(transitions["effort"])),
        }
        for name in statistics_accumulator[0]:
            row[name] = float(
                np.mean([statistics[name] for statistics in statistics_accumulator])
            )
        update_rows.append(row)
        _write_csv(seed_directory / "updates.csv", update_rows)

        while target_index < len(targets) and actual_steps >= targets[target_index]:
            save_checkpoint(targets[target_index], actual_steps)
            target_index += 1

        if (update_index + 1) % 10 == 0 or update_index == 0:
            print(
                f"seed={config.seed} update={update_index + 1}/{config.number_of_updates} "
                f"steps={actual_steps} reward/step={row['mean_step_reward']:.3f}",
                flush=True,
            )

    structured_metrics = evaluate_policy_batch(None, config, structured=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "optax": optax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "swing_rl": _swing_rl_metadata(),
        "config": asdict(config),
        "requested_environment_steps": config.total_environment_steps,
        "actual_environment_steps": config.actual_environment_steps,
        "checkpoint_policy": (
            "The first complete 2,048-transition update at or beyond each "
            "50,000-transition target is saved; both target and actual counts are logged."
        ),
        "evaluation_distribution": {
            "states": config.evaluation_states,
            "seed": config.evaluation_seed,
            "theta_degrees": [-5.0, 5.0],
            "theta_dot": [-0.1, 0.1],
            "policy": "deterministic tanh of the Gaussian mean",
        },
        "showcase_state": {"theta_degrees": 3.0, "theta_dot": 0.0},
        "structured_controller_metrics": structured_metrics,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    (seed_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _checkpoint_rows(artifact_directory: Path, seeds: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for row in _read_csv(artifact_directory / f"seed_{seed}" / "checkpoints.csv"):
            converted: dict[str, Any] = {}
            for name, value in row.items():
                if name in {"checkpoint", "showcase_trace"}:
                    converted[name] = value
                elif name in {"seed", "checkpoint_target", "environment_steps", "episode_count"}:
                    converted[name] = int(float(value))
                else:
                    converted[name] = float(value)
            rows.append(converted)
    return rows


def make_aggregate_figure(
    artifact_directory: Path, seeds: Sequence[int]
) -> tuple[plt.Figure, list[dict[str, Any]]]:
    """Plot the five-seed held-out evidence and return final metric rows."""

    rows = _checkpoint_rows(artifact_directory, seeds)
    targets = sorted({row["checkpoint_target"] for row in rows})
    metrics = (
        ("success_rate", "success rate"),
        ("return_mean", "mean episode return"),
    )
    with mpl.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1, 2, figsize=(7.2, 2.8), constrained_layout=True
        )
        for axis, (metric, label) in zip(axes, metrics):
            values = np.asarray(
                [
                    [
                        next(
                            row[metric]
                            for row in rows
                            if row["seed"] == seed
                            and row["checkpoint_target"] == target
                        )
                        for target in targets
                    ]
                    for seed in seeds
                ]
            )
            x = np.asarray(targets, dtype=float)
            mean = values.mean(axis=0)
            if len(seeds) > 1:
                interval = 2.776 * values.std(axis=0, ddof=1) / math.sqrt(len(seeds))
            else:
                interval = np.zeros_like(mean)
            for seed_values in values:
                axis.plot(x, seed_values, color=OI["blue"], alpha=0.18, linewidth=0.8)
            axis.fill_between(
                x,
                mean - interval,
                mean + interval,
                color=OI["blue"],
                alpha=0.18,
                linewidth=0.0,
            )
            axis.plot(
                x,
                mean,
                color=OI["blue"],
                linewidth=1.8,
                marker="o",
                markevery=4,
                markersize=2.7,
                label="PPO",
            )

            structured = json.loads(
                (artifact_directory / f"seed_{seeds[0]}" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )["structured_controller_metrics"][metric]
            if metric == "success_rate":
                axis.axhline(
                    structured,
                    color=OI["vermilion"],
                    linestyle="--",
                    linewidth=1.3,
                    label="structured controller",
                )
                axis.set_ylim(-0.03, 1.03)
            else:
                lower = float(np.min(values))
                upper = float(np.max(values))
                margin = max(0.4, 0.12 * (upper - lower))
                axis.set_ylim(lower - margin, upper + margin)
                axis.text(
                    0.98,
                    0.96,
                    f"structured controller: {structured:.1f} (off scale)",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    color=OI["vermilion"],
                    fontsize=7.5,
                )
            axis.set_xlabel("environment interactions")
            axis.set_ylabel(label)
            axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
            axis.grid(axis="y", color="0.9", linewidth=0.6)
        axes[0].legend(loc="lower right")

    final_rows = [
        row
        for row in rows
        if row["checkpoint_target"] == max(targets)
    ]
    return figure, final_rows


def _rolling_mean(values: np.ndarray, window: int = 15) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for index in range(values.size):
        start = max(0, index - window + 1)
        result[index] = np.nanmean(values[start : index + 1])
    return result


def make_training_replay(
    artifact_directory: Path,
    output_path: Path,
    *,
    seed: int = 0,
    fps: int = 15,
    frames_per_checkpoint: int = 40,
) -> list[dict[str, Any]]:
    """Render one time-compressed movie from the saved seed-0 checkpoints."""

    seed_directory = artifact_directory / f"seed_{seed}"
    checkpoints = _read_csv(seed_directory / "checkpoints.csv")
    updates = _read_csv(seed_directory / "updates.csv")
    completed_episode_rows = [
        row
        for row in updates
        if row["mean_episode_return"]
        and math.isfinite(float(row["mean_episode_return"]))
    ]
    update_steps = np.asarray(
        [float(row["environment_steps"]) for row in completed_episode_rows]
    )
    raw_reward = np.asarray(
        [float(row["mean_episode_return"]) for row in completed_episode_rows]
    )
    smooth_reward = _rolling_mean(raw_reward, window=8)

    rollouts = []
    checkpoint_parameters = []
    for row in checkpoints:
        path = artifact_directory / row["checkpoint"]
        parameters = load_parameters(path)
        checkpoint_parameters.append(parameters)
        rollouts.append(record_showcase(parameters, PPOConfig(seed=seed)))

    view = SwingAnimation(
        rollouts[0], stride=1, figsize=(10.0, 4.8), show_diagnostics=False
    )
    if view.ax.get_legend() is not None:
        view.ax.get_legend().remove()
    view.ax.set_position([0.02, 0.07, 0.47, 0.88])
    curve_axis = view.fig.add_axes([0.57, 0.19, 0.40, 0.66])
    curve_axis.set_xlabel("environment interactions")
    curve_axis.set_ylabel("completed-episode return")
    curve_axis.set_xlim(0.0, max(update_steps.max(), 1.0))
    reward_margin = max(0.1, 0.08 * np.ptp(raw_reward))
    curve_axis.set_ylim(raw_reward.min() - reward_margin, raw_reward.max() + reward_margin)
    curve_axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    curve_axis.grid(axis="y", color="0.9", linewidth=0.6)
    (raw_line,) = curve_axis.plot(
        [],
        [],
        color="0.55",
        linewidth=0.7,
        alpha=0.7,
        label="raw completed-episode mean",
    )
    (smooth_line,) = curve_axis.plot(
        [], [], color=OI["blue"], linewidth=1.8, label="8-point trailing mean"
    )
    curve_axis.text(
        0.98,
        0.04,
        "gray: raw episode mean\nblue: 8-point mean",
        transform=curve_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="0.3",
    )
    checkpoint_text = curve_axis.text(
        0.98,
        0.98,
        "",
        transform=curve_axis.transAxes,
        va="top",
        ha="right",
        fontsize=8.5,
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.8", "alpha": 0.92},
    )

    sampled_indices: list[np.ndarray] = []
    timeline: list[dict[str, Any]] = []
    elapsed_video = 0.0
    for index, (row, rollout) in enumerate(zip(checkpoints, rollouts)):
        indices = np.linspace(
            0, max(len(rollout.frames) - 1, 0), frames_per_checkpoint
        ).round().astype(int)
        sampled_indices.append(indices)
        timeline.append(
            {
                "index": index,
                "checkpoint_target": int(float(row["checkpoint_target"])),
                "environment_steps": int(float(row["environment_steps"])),
                "video_seconds": elapsed_video,
            }
        )
        elapsed_video += len(indices) / fps

    def update_frame(global_index: int):
        checkpoint_index = global_index // frames_per_checkpoint
        local_index = global_index % frames_per_checkpoint
        row = checkpoints[checkpoint_index]
        rollout = rollouts[checkpoint_index]
        frame_index = int(sampled_indices[checkpoint_index][local_index])
        view.roll = rollout
        view.indices = list(range(len(rollout.frames)))
        artists = list(view._update(frame_index))

        steps = float(row["environment_steps"])
        prefix = update_steps <= steps
        raw_line.set_data(update_steps[prefix], raw_reward[prefix])
        smooth_line.set_data(update_steps[prefix], smooth_reward[prefix])
        checkpoint_text.set_text(
            f"recorded PPO run, seed {seed}\n"
            f"interactions: {int(steps):,}\n"
            f"training time: {float(row['elapsed_seconds']) / 60.0:.1f} min\n"
            f"held-out success: {100.0 * float(row['success_rate']):.0f}%\n"
            f"worst tension: {float(row['minimum_tension_worst']):.0f} N"
        )
        artists.extend([raw_line, smooth_line, checkpoint_text])
        return artists

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        view.fig,
        update_frame,
        frames=len(checkpoints) * frames_per_checkpoint,
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "22"],
    )
    animation.save(output_path, writer=writer, dpi=110)
    poster_frame = min(
        2 * frames_per_checkpoint - 1,
        len(checkpoints) * frames_per_checkpoint - 1,
    )
    update_frame(poster_frame)
    view.fig.savefig(output_path.with_name("training_replay_poster.png"), dpi=180)
    # MyST's wildcard video syntax pairs this same-stem image with the MP4 for
    # static exports while retaining the more explicit poster filename for
    # direct reuse outside the book.
    view.fig.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.close(view.fig)
    output_path.with_name("training_replay_checkpoints.json").write_text(
        json.dumps(timeline, indent=2) + "\n", encoding="utf-8"
    )
    return timeline


def replay_player_html(
    output_directory: Path,
    *,
    player_id: str = "swing-ppo-replay",
) -> str:
    """Return a self-contained player for the recorded replay and checkpoints.

    MyST's native video figure is designed for short looping clips.  The
    recorded training replay instead needs ordinary playback controls and a
    checkpoint seeker.  Embedding the already-rendered assets in the output
    keeps the MyST build independent of a development-server asset URL.
    """

    video_path = output_directory / "training_replay.mp4"
    poster_path = output_directory / "training_replay.png"
    checkpoints_path = output_directory / "training_replay_checkpoints.json"
    for path in (video_path, poster_path, checkpoints_path):
        if not path.exists():
            raise FileNotFoundError(f"Recorded replay asset is missing: {path}")

    video_uri = (
        "data:video/mp4;base64,"
        + base64.b64encode(video_path.read_bytes()).decode("ascii")
    )
    poster_uri = (
        "data:image/png;base64,"
        + base64.b64encode(poster_path.read_bytes()).decode("ascii")
    )
    checkpoints = json.loads(checkpoints_path.read_text(encoding="utf-8"))
    checkpoints_json = json.dumps(checkpoints, separators=(",", ":"))
    video_id = f"{player_id}-video"
    slider_id = f"{player_id}-checkpoint"
    label_id = f"{player_id}-label"

    return f"""
<div id="{player_id}" style="max-width: 920px; margin: 0 auto;">
  <video id="{video_id}" controls preload="metadata" playsinline
         poster="{poster_uri}" style="display: block; width: 100%;">
    <source src="{video_uri}" type="video/mp4">
    The recorded replay is available as an MP4 download below.
  </video>
  <div style="display: grid; grid-template-columns: minmax(8rem, auto) 1fr;
              gap: 0.6rem; align-items: center; margin-top: 0.6rem;">
    <label for="{slider_id}">Training checkpoint</label>
    <input id="{slider_id}" type="range" min="0"
           max="{len(checkpoints) - 1}" value="0" step="1">
    <span aria-hidden="true"></span>
    <output id="{label_id}">0 requested interactions (0 recorded)</output>
  </div>
</div>
<script>
(() => {{
  const video = document.getElementById("{video_id}");
  const slider = document.getElementById("{slider_id}");
  const label = document.getElementById("{label_id}");
  const points = {checkpoints_json};
  const describe = (point) =>
    point.checkpoint_target.toLocaleString() + " requested interactions (" +
    point.environment_steps.toLocaleString() + " recorded)";
  slider.addEventListener("input", () => {{
    const point = points[Number(slider.value)];
    video.currentTime = point.video_seconds;
    label.textContent = describe(point);
  }});
  video.addEventListener("timeupdate", () => {{
    let index = 0;
    for (let i = 1; i < points.length; i += 1) {{
      if (points[i].video_seconds <= video.currentTime) index = i;
    }}
    slider.value = String(index);
    label.textContent = describe(points[index]);
  }});
}})();
</script>
""".strip()


def render_artifacts(
    artifact_directory: Path,
    output_directory: Path,
    seeds: Sequence[int],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    figure, final_rows = make_aggregate_figure(artifact_directory, seeds)
    figure.savefig(output_directory / "swing_ppo_learning.pdf")
    figure.savefig(output_directory / "swing_ppo_learning.png", dpi=220)
    plt.close(figure)
    _write_csv(output_directory / "swing_ppo_final_metrics.csv", final_rows)
    timeline = make_training_replay(
        artifact_directory, output_directory / "training_replay.mp4", seed=0
    )
    combined_manifest = {
        "seeds": list(seeds),
        "source_artifacts": str(artifact_directory),
        "replay_seed": 0,
        "replay_checkpoints": timeline,
        "curve": (
            "raw mean return over episodes completed in each logged update "
            "and an 8-point trailing mean"
        ),
        "aggregate_interval": "two-sided 95% t interval with seed as the unit",
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(combined_manifest, indent=2) + "\n", encoding="utf-8"
    )


def run_protocol(
    artifact_directory: Path,
    output_directory: Path,
    seeds: Sequence[int],
    *,
    steps: int = 1_000_000,
) -> None:
    for seed in seeds:
        config = PPOConfig(seed=seed, total_environment_steps=steps)
        print(f"training prespecified seed {seed}", flush=True)
        train_seed(config, artifact_directory)
    render_artifacts(artifact_directory, output_directory, seeds)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train and save one or more seeds")
    train.add_argument("--artifacts", type=Path, default=Path("artifacts/swing_ppo"))
    train.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    train.add_argument("--steps", type=int, default=1_000_000)

    render = subparsers.add_parser("render", help="render saved checkpoints")
    render.add_argument("--artifacts", type=Path, default=Path("artifacts/swing_ppo"))
    render.add_argument("--output", type=Path, default=Path("_static/swing_ppo"))
    render.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    complete = subparsers.add_parser("all", help="train, evaluate, and render")
    complete.add_argument("--artifacts", type=Path, default=Path("artifacts/swing_ppo"))
    complete.add_argument("--output", type=Path, default=Path("_static/swing_ppo"))
    complete.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    complete.add_argument("--steps", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    if arguments.command == "train":
        for seed in arguments.seeds:
            train_seed(
                PPOConfig(seed=seed, total_environment_steps=arguments.steps),
                arguments.artifacts,
            )
    elif arguments.command == "render":
        render_artifacts(arguments.artifacts, arguments.output, arguments.seeds)
    else:
        run_protocol(
            arguments.artifacts,
            arguments.output,
            arguments.seeds,
            steps=arguments.steps,
        )


if __name__ == "__main__":
    # SwingRL enables float64 dynamics.  Matching that precision in the policy
    # avoids repeated casts inside the compiled simulator loop.
    jax.config.update("jax_enable_x64", True)
    main()
