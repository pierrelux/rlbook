"""Focused tests for the SwingRL model-audit experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

from swing_control import (  # noqa: E402
    DEFAULT_SWING_SCENARIO,
    audit_metrics,
    make_environment,
    make_model_audit_animation,
    make_model_audit_figure,
    make_structured_controller,
    model_audit_player_html,
    mode_events,
    run_model_audit,
)


class SwingModelAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.traces = run_model_audit(DEFAULT_SWING_SCENARIO)
        cls.metrics = audit_metrics(cls.traces, DEFAULT_SWING_SCENARIO)

    def test_scenario_uses_matched_numerics_and_feedback_law(self) -> None:
        scenario = DEFAULT_SWING_SCENARIO
        rod_parameters, rod = make_environment(scenario, suspension="rigid_rod")
        chain_parameters, chain = make_environment(
            scenario,
            suspension="unilateral_chain",
        )

        self.assertAlmostEqual(
            rod.integrator_params.dt,
            scenario.control_interval,
        )
        self.assertAlmostEqual(
            chain.integrator_params.dt,
            scenario.control_interval,
        )
        self.assertEqual(
            rod.integrator_params.substeps,
            scenario.integrator_substeps,
        )
        self.assertEqual(
            chain.integrator_params.substeps,
            scenario.integrator_substeps,
        )
        self.assertAlmostEqual(
            rod_parameters.natural_frequency,
            chain_parameters.natural_frequency,
        )

        observation, _ = rod.reset(
            seed=scenario.seed,
            options={"theta": 0.0, "theta_dot": 0.0, "noise": 0.0},
        )
        controller = make_structured_controller(rod_parameters, scenario)
        first_action = controller(observation)
        self.assertEqual(first_action.shape, rod.action_space.shape)
        self.assertTrue(np.all(np.abs(first_action) <= 1.0))
        self.assertLess(np.max(np.abs(first_action)), 0.1)

    def test_matched_controller_exposes_model_class_failure(self) -> None:
        rod = self.metrics["rigid_rod"]
        chain = self.metrics["unilateral_chain"]

        self.assertTrue(rod["success"])
        self.assertAlmostEqual(rod["time_to_rotation_seconds"], 26.08, places=2)
        self.assertGreater(rod["maximum_compression_demand_newtons"], 390.0)
        self.assertLess(rod["minimum_tension_newtons"], 0.0)
        self.assertAlmostEqual(rod["minimum_seat_radius_fraction"], 1.0)

        self.assertFalse(chain["success"])
        self.assertLess(chain["peak_absolute_angle_degrees"], 135.0)
        self.assertGreater(chain["peak_absolute_angle_degrees"], 125.0)
        self.assertEqual(chain["minimum_tension_newtons"], 0.0)
        self.assertLess(chain["minimum_seat_radius_fraction"], 0.40)
        self.assertGreater(chain["slack_time_seconds"], 1.5)
        self.assertGreater(chain["snap_energy_loss_joules"], 1300.0)

    def test_chain_mode_events_are_ordered_and_physical(self) -> None:
        events = mode_events(self.traces["unilateral_chain"])
        self.assertEqual(
            [event["kind"] for event in events],
            ["release", "reattachment", "release", "reattachment"],
        )
        event_times = [event["time_seconds"] for event in events]
        self.assertEqual(event_times, sorted(event_times))
        self.assertAlmostEqual(event_times[0], 20.84, places=2)
        self.assertAlmostEqual(event_times[-1], 23.68, places=2)
        self.assertGreaterEqual(
            min(event["seat_radius_fraction"] for event in events),
            0.999,
        )

    def test_traces_are_finite_and_aligned_with_recorded_frames(self) -> None:
        for trace in self.traces.values():
            length = len(trace.rollout.frames)
            self.assertEqual(trace.target_actions.shape, (length, 2))
            self.assertEqual(trace.realized_commands.shape, (length, 2))
            self.assertEqual(trace.taut.shape, (length,))
            self.assertTrue(np.isfinite(trace.target_actions).all())
            self.assertTrue(np.isfinite(trace.realized_commands).all())
            self.assertTrue(np.isfinite(trace.seat_radius).all())
            self.assertTrue(np.isfinite(trace.tensions).all())
            self.assertTrue(np.all(np.diff(trace.times) > 0.0))

    def test_static_and_animated_views_use_prefixes_of_real_rollouts(self) -> None:
        figure = make_model_audit_figure(self.traces)
        self.assertEqual(len(figure.axes), 5)
        plt.close(figure)

        view = make_model_audit_animation(
            self.traces,
            fps=10,
            speed=4.0,
        )
        view._update(0)
        self.assertEqual(len(view.rod_angle_line.get_xdata()), 1)
        view._update(len(view.timeline) - 1)
        self.assertEqual(
            len(view.chain_angle_line.get_xdata()),
            len(self.traces["unilateral_chain"].times),
        )
        view.animation()._draw_was_started = True
        plt.close(view.figure)


class SwingModelAuditPlayerTests(unittest.TestCase):
    def test_player_embeds_records_and_only_seeks_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static = root / "static"
            records = root / "records"
            static.mkdir()
            records.mkdir()
            (static / "model_audit.mp4").write_bytes(b"immutable-movie")
            (static / "model_audit_poster.png").write_bytes(b"recorded-poster")
            events = [
                {
                    "kind": "release",
                    "label": "chain release 1",
                    "time_seconds": 20.84,
                    "video_seconds": 10.45,
                },
                {
                    "kind": "rod_rotation",
                    "label": "nominal rod rotation",
                    "time_seconds": 26.08,
                    "video_seconds": 15.84,
                },
            ]
            event_path = records / "events.json"
            event_path.write_text(json.dumps(events), encoding="utf-8")
            player = model_audit_player_html(
                static,
                event_path,
                player_id="audit-test",
                fallback_id="audit-fallback",
            )

        self.assertIn("data:video/mp4;base64,", player)
        self.assertIn("data:image/png;base64,", player)
        self.assertEqual(player.count('data-seek="'), 2)
        self.assertIn("loadedmetadata", player)
        self.assertIn('fallback.hidden = true', player)
        self.assertIn("video.currentTime", player)
        self.assertNotIn("requestAnimationFrame", player)
        self.assertNotIn("setInterval", player)


if __name__ == "__main__":
    unittest.main()
