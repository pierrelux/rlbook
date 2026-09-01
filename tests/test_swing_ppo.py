"""Focused numerical tests for the SwingRL PPO experiment."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

from swing_ppo import (  # noqa: E402
    PPOConfig,
    compute_gae,
    deterministic_policy,
    initialize_parameters,
    load_parameters,
    ppo_loss,
    replay_player_html,
    save_parameters,
    tanh_log_probability,
    train_seed,
)


class SwingPPOTests(unittest.TestCase):
    def test_tanh_log_probability_matches_change_of_variables(self) -> None:
        mean = jnp.asarray([[0.1, -0.2]])
        log_standard_deviation = jnp.log(jnp.asarray([0.7, 1.2]))
        latent = jnp.asarray([[0.3, -0.4]])
        actual = float(tanh_log_probability(mean, log_standard_deviation, latent)[0])

        mean_np = np.asarray(mean[0])
        std_np = np.exp(np.asarray(log_standard_deviation))
        latent_np = np.asarray(latent[0])
        action_np = np.tanh(latent_np)
        gaussian = np.sum(
            -0.5 * np.square((latent_np - mean_np) / std_np)
            - np.log(std_np)
            - 0.5 * np.log(2.0 * np.pi)
        )
        expected = gaussian - np.sum(np.log(1.0 - np.square(action_np)))
        self.assertAlmostEqual(actual, float(expected), places=6)

    def test_gae_stops_at_episode_boundary_and_bootstraps_truncation(self) -> None:
        rewards = np.asarray([[1.0], [2.0], [3.0]])
        values = np.asarray([[0.2], [0.4], [0.6]])
        next_values = np.asarray([[0.4], [10.0], [0.0]])
        terminated = np.asarray([[False], [False], [True]])
        ended = np.asarray([[False], [True], [True]])
        advantages, returns = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            ended,
            discount=0.9,
            gae_lambda=0.8,
        )

        final_delta = 3.0 - 0.6
        truncated_delta = 2.0 + 0.9 * 10.0 - 0.4
        first_delta = 1.0 + 0.9 * 0.4 - 0.2
        self.assertAlmostEqual(advantages[2, 0], final_delta)
        self.assertAlmostEqual(advantages[1, 0], truncated_delta)
        self.assertAlmostEqual(
            advantages[0, 0], first_delta + 0.9 * 0.8 * truncated_delta
        )
        np.testing.assert_allclose(returns, advantages + values)

    def test_checkpoint_round_trip_preserves_policy(self) -> None:
        config = PPOConfig(seed=3, hidden_units=8)
        parameters = initialize_parameters(config)
        observations = jnp.linspace(-0.5, 0.5, 22).reshape(2, 11)
        before = np.asarray(deterministic_policy(parameters, observations))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_parameters(parameters, path)
            restored = load_parameters(path)
            after = np.asarray(deterministic_policy(restored, observations))
        np.testing.assert_allclose(after, before, rtol=0.0, atol=0.0)

    def test_ppo_loss_and_gradients_are_finite(self) -> None:
        config = PPOConfig(seed=5, hidden_units=8)
        parameters = initialize_parameters(config)
        observations = jnp.zeros((16, 11))
        latent = jnp.zeros((16, 2))
        old_log_probability = tanh_log_probability(
            jnp.zeros((16, 2)), parameters["log_standard_deviation"], latent
        )
        batch = {
            "observations": observations,
            "latent_actions": latent,
            "old_log_probabilities": old_log_probability,
            "advantages": jnp.linspace(-1.0, 1.0, 16),
            "returns": jnp.linspace(-0.5, 0.5, 16),
        }
        (loss, statistics), gradients = jax.value_and_grad(
            ppo_loss, has_aux=True
        )(parameters, batch, config)
        self.assertTrue(math.isfinite(float(loss)))
        self.assertTrue(all(math.isfinite(float(value)) for value in statistics.values()))
        leaves = jax.tree_util.tree_leaves(gradients)
        self.assertTrue(all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves))

    def test_short_training_run_writes_finite_checkpoints(self) -> None:
        config = PPOConfig(
            seed=0,
            total_environment_steps=64,
            number_of_environments=2,
            rollout_steps=16,
            episode_steps=40,
            hidden_units=8,
            update_epochs=1,
            minibatch_size=16,
            checkpoint_interval=32,
            evaluation_states=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = train_seed(config, root)
            self.assertEqual(manifest["actual_environment_steps"], 64)
            checkpoints = sorted((root / "seed_0" / "checkpoints").glob("*.npz"))
            self.assertEqual(len(checkpoints), 3)
            for checkpoint in checkpoints:
                for leaf in jax.tree_util.tree_leaves(load_parameters(checkpoint)):
                    self.assertTrue(np.isfinite(np.asarray(leaf)).all())

    def test_replay_player_uses_recorded_checkpoint_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training_replay.mp4").write_bytes(b"recorded-video")
            (root / "training_replay.png").write_bytes(b"recorded-poster")
            (root / "training_replay_checkpoints.json").write_text(
                '[{"checkpoint_target":50000,"environment_steps":51200,'
                '"video_seconds":2.666667}]\n',
                encoding="utf-8",
            )
            player = replay_player_html(root)

        self.assertIn("controls", player)
        self.assertIn("data:video/mp4;base64,", player)
        self.assertIn('"environment_steps":51200', player)
        self.assertIn('max="0"', player)


if __name__ == "__main__":
    unittest.main()
