"""Tests for the recorded inference-serving replay."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import sys
import tempfile
import unittest

import matplotlib.pyplot as plt


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

from inference_replay import (  # noqa: E402
    ReplayDataError,
    _normalise_replay,
    render_serving_replay,
    render_static_figure,
)


def sample_artifact() -> dict[str, object]:
    time_s = [0.0, 1.0, 2.0, 3.0]
    common = {
        "time_s": time_s,
        "prefill_queue": [2, 1, 0, 0],
        "decode_active": [0, 1, 1, 0],
        "completed_requests": [0, 0, 0, 2],
        "kv_tokens": [0, 512, 768, 0],
        "temperature_c": [35.0, 38.0, 41.0, 39.0],
        "power_w": [45.0, 62.0, 70.0, 48.0],
        "requested_clock_mhz": [600, 900, 1200, 600],
        "realized_clock_mhz": [585, 885, 1180, 590],
        "energy_j": [0.0, 52.0, 118.0, 164.0],
        "requests": [
            {
                "request_id": "request-0",
                "arrival_time_s": 0.0,
                "prefill_start_s": 1.0,
                "first_token_time_s": 2.0,
                "completion_time_s": 3.0,
                "prompt_tokens": 512,
                "output_tokens": 32,
            }
        ],
    }
    return {
        "metadata": {
            "profile_status": "engineering_proxy_not_measured",
            "profile_source": "Integration profile",
        },
        "modeling": {
            "title": "Recorded serving process",
            "default_controller": "chunked prefill",
            "controllers": {
                "chunked prefill": common,
                "maximum clock": {
                    **common,
                    "requested_clock_mhz": [1200, 1200, 1200, 1200],
                    "realized_clock_mhz": [1180, 1182, 1177, 1181],
                },
            },
        },
        "open_loop": {
            "controllers": {
                "nominal": {
                    **common,
                    "planned_clock_mhz": [
                        [600, 900, 1200, 600],
                        [],
                        [],
                        [],
                    ],
                    "planned_clock_start_time_s": [0.0, None, None, None],
                    "plan_dt_s": 1.0,
                }
            }
        },
        "mpc": {
            "controllers": {
                "MPC": {
                    **common,
                    "planned_clock_mhz": [
                        [600, 900, 1200],
                        [900, 1200, 1200],
                        [1200, 900],
                        [600],
                    ],
                }
            }
        },
    }


class InferenceReplayTests(unittest.TestCase):
    def test_player_is_scoped_accessible_and_causal(self) -> None:
        rendered = render_serving_replay(sample_artifact(), view="modeling")

        root_match = re.search(r'<section id="([^"]+)"', rendered)
        self.assertIsNotNone(root_match)
        root_id = root_match.group(1)
        self.assertIn(f"#{root_id} .controls", rendered)
        self.assertNotIn("__ROOT__", rendered)
        self.assertIn('data-action="play"', rendered)
        self.assertIn('data-action="step"', rendered)
        self.assertIn('data-action="reset"', rendered)
        self.assertIn('type="range"', rendered)
        self.assertIn('aria-live="polite"', rendered)
        self.assertIn('role="img"', rendered)
        self.assertIn("prefers-reduced-motion: reduce", rendered)
        self.assertIn("frames.slice(0, frameIndex + 1)", rendered)
        self.assertIn("MutationObserver", rendered)
        self.assertIn("window.parent && window.parent !== window", rendered)
        self.assertNotIn("fetch(", rendered)
        self.assertNotIn("XMLHttpRequest", rendered)
        self.assertNotIn("WebSocket", rendered)
        self.assertNotIn("autoplay", rendered.lower())
        self.assertIn("Profile: Engineering surrogate, not L4 measurements", rendered)
        self.assertIn("fig-inference-serving-model-fallback", rendered)
        self.assertIn('fallback.hidden = true', rendered)
        self.assertIn('fallback.setAttribute("aria-hidden", "true")', rendered)
        self.assertIn("fallbackObserver.observe", rendered)

    def test_each_player_receives_a_unique_root(self) -> None:
        first = render_serving_replay(sample_artifact(), replay_id="lecture")
        second = render_serving_replay(sample_artifact(), replay_id="lecture")
        first_id = re.search(r'<section id="([^"]+)"', first).group(1)
        second_id = re.search(r'<section id="([^"]+)"', second).group(1)
        self.assertNotEqual(first_id, second_id)
        self.assertTrue(first_id.startswith("lecture-"))

    def test_embedded_json_escapes_script_termination(self) -> None:
        artifact = sample_artifact()
        artifact["modeling"]["controllers"]["bad</script><script>alert(1)</script>"] = (
            artifact["modeling"]["controllers"].pop("maximum clock")
        )
        rendered = render_serving_replay(artifact)
        self.assertNotIn("</script><script>alert(1)</script>", rendered)
        self.assertIn("\\u003c/script", rendered)
        self.assertNotIn("NaN", rendered)

    def test_path_input_and_view_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "textbook_results.json"
            path.write_text(json.dumps(sample_artifact()), encoding="utf-8")
            rendered = render_serving_replay(path, view="mpc")
        self.assertIn("Receding-horizon control under trace replay", rendered)
        self.assertIn("planned_clock_mhz", rendered)

    def test_measured_badge_requires_complete_validation_attestation(self) -> None:
        artifact = sample_artifact()
        artifact["metadata"] = {
            "profile_status": "measured_l4",
            "profile_source": "Measured NVIDIA L4 profile",
        }
        rendered = render_serving_replay(artifact)
        self.assertIn("Profile: Unverified measured-profile claim", rendered)
        self.assertNotIn('<p class="provenance" data-verdict="stands">', rendered)

        artifact["metadata"]["measurement_validation"] = {
            "validated": True,
            "manifest_status": "complete",
            "manifest_sha256": "a" * 64,
        }
        rendered = render_serving_replay(artifact)
        self.assertIn("Profile: Measured NVIDIA L4 profile", rendered)
        self.assertIn('<p class="provenance" data-verdict="stands">', rendered)

    def test_path_input_rejects_artifact_stale_against_profile_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "data" / "inference_serving" / "l4_profile.csv"
            profile.parent.mkdir(parents=True)
            profile.write_text("first profile\n", encoding="utf-8")
            manifest = profile.with_name("profile_manifest.json")
            manifest.write_text('{"profile_status":"test"}\n', encoding="utf-8")
            artifact = sample_artifact()
            artifact["metadata"].update(
                {
                    "profile_csv_path": "data/inference_serving/l4_profile.csv",
                    "profile_csv_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                    "profile_manifest_path": "data/inference_serving/profile_manifest.json",
                    "measurement_validation": {
                        "validated": False,
                        "manifest_status": "not_measured",
                        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    },
                }
            )
            artifact_path = root / "artifacts" / "inference_serving" / "textbook_results.json"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            render_serving_replay(artifact_path)

            profile.write_text("changed profile\n", encoding="utf-8")
            with self.assertRaisesRegex(ReplayDataError, "artifact is stale"):
                render_serving_replay(artifact_path)

            profile.write_text("first profile\n", encoding="utf-8")
            manifest.write_text('{"profile_status":"changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ReplayDataError, "manifest has changed"):
                render_serving_replay(artifact_path)

    def test_open_loop_plan_is_full_fixed_and_anchored_at_time_zero(self) -> None:
        replay = _normalise_replay(
            sample_artifact(), view="open_loop", maximum_frames=3
        )
        run = replay["controllers"]["nominal"]

        self.assertEqual(run["plan_dt_s"], 1.0)
        self.assertEqual(len(run["frames"]), 3)
        for frame in run["frames"]:
            self.assertEqual(frame["planned_clock_mhz"], [600.0, 900.0, 1200.0, 600.0])
            self.assertEqual(frame["plan_start_time_s"], 0.0)

    def test_downsampling_retains_every_mpc_plan_update(self) -> None:
        time_s = [0.1 * (index + 1) for index in range(100)]
        plans = [[] for _ in time_s]
        starts = [None for _ in time_s]
        for index in range(0, 100, 10):
            plans[index] = [600 + index, 900 + index]
            starts[index] = 0.1 * index
        trajectory = {
            "time_s": time_s,
            "requested_clock_mhz": [900] * 100,
            "planned_clock_mhz": plans,
            "planned_clock_start_time_s": starts,
            "plan_dt_s": 1.0,
        }
        replay = _normalise_replay(trajectory, view="mpc", maximum_frames=12)
        frames = replay["controllers"]["recorded run"]["frames"]

        self.assertEqual(sum(bool(frame["plan_update"]) for frame in frames), 10)
        self.assertEqual(
            [frame["plan_start_time_s"] for frame in frames if frame["plan_update"]],
            [float(index) for index in range(10)],
        )
        self.assertEqual(replay["controllers"]["recorded run"]["plan_dt_s"], 1.0)

    def test_static_svg_uses_the_same_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "modeling.svg"
            figure = render_static_figure(sample_artifact(), output, view="modeling")
            plt.close(figure)
            svg = output.read_text(encoding="utf-8")
        self.assertIn("<svg", svg)
        self.assertIn("unfinished requests", svg)
        self.assertIn("realized clock", svg)
        self.assertIn("Engineering surrogate", svg)

    def test_policy_slices_and_coverage_have_interactive_and_static_views(self) -> None:
        artifact = sample_artifact()
        base = artifact["modeling"]["controllers"]["chunked prefill"]
        matrix = [[(p + d) % 3 for d in range(7)] for p in range(7)]
        coverage = [[p * 7 + d for d in range(7)] for p in range(7)]
        artifact["fqi"] = {
            "controllers": {
                "exact_dp": base,
                "broad_fqi": base,
                "narrow_fqi": base,
            },
            "policy_slices": {
                name: {str(age): matrix for age in range(5)}
                for name in ("exact_dp", "broad_fqi", "narrow_fqi")
            },
            "coverage_slices": {
                "broad_fqi": {str(age): coverage for age in range(5)},
                "narrow_fqi": {str(age): coverage for age in range(5)},
            },
            "metrics": {
                "broad_fqi": {"coverage_fraction": 1.0},
                "narrow_fqi": {"coverage_fraction": 0.2},
            },
        }
        rendered = render_serving_replay(artifact, view="fqi")
        self.assertIn("data-policy-view", rendered)
        self.assertIn("data-age-field", rendered)
        self.assertIn("logged visits", rendered)
        self.assertIn('"coverage_fraction":{"broad_fqi":1.0', rendered)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fqi.svg"
            figure = render_static_figure(artifact, output, view="fqi")
            plt.close(figure)
            svg = output.read_text(encoding="utf-8")
        self.assertIn("Oldest", rendered)
        self.assertIn("[hidden] { display: none !important; }", rendered)
        self.assertIn('policyView.removeAttribute("hidden")', rendered)
        self.assertIn('policyEmpty.toggleAttribute("hidden"', rendered)
        self.assertIn("broad fqi", svg)
        self.assertIn("logged visits", svg)

    def test_missing_time_has_a_useful_error(self) -> None:
        with self.assertRaisesRegex(ReplayDataError, "time_s"):
            render_serving_replay(
                {"modeling": {"controllers": {"controller": {"power_w": [1, 2]}}}}
            )

    def test_rejects_nonfinite_or_misaligned_series(self) -> None:
        with self.assertRaisesRegex(ReplayDataError, "non-finite"):
            render_serving_replay({"time_s": [0.0, 1.0], "power_w": [1.0, float("nan")]})
        with self.assertRaisesRegex(ReplayDataError, "has 1 values"):
            render_serving_replay({"time_s": [0.0, 1.0], "power_w": [1.0]})


if __name__ == "__main__":
    unittest.main()
