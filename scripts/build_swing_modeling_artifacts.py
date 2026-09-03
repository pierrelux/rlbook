#!/usr/bin/env python3
"""Build the SwingRL model-audit records and book-ready media."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CODE_DIRECTORY = ROOT / "code"
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

from swing_control import (  # noqa: E402
    DEFAULT_SWING_SCENARIO,
    audit_metrics,
    run_model_audit,
    write_model_audit_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-directory",
        type=Path,
        default=ROOT / "_static" / "swing_modeling",
        help="directory for the SVG, PNG, and MP4 used by the book",
    )
    parser.add_argument(
        "--record-directory",
        type=Path,
        default=ROOT / "artifacts" / "swing_modeling",
        help="directory for raw traces, metrics, and the manifest",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--speed",
        type=float,
        default=2.0,
        help="simulated seconds shown per video second",
    )
    arguments = parser.parse_args()

    traces = run_model_audit(DEFAULT_SWING_SCENARIO)
    outputs = write_model_audit_artifacts(
        traces,
        static_directory=arguments.static_directory,
        record_directory=arguments.record_directory,
        scenario=DEFAULT_SWING_SCENARIO,
        fps=arguments.fps,
        speed=arguments.speed,
    )

    print("SwingRL model audit")
    for name, values in audit_metrics(traces).items():
        outcome = (
            f"rotation at {values['time_to_rotation_seconds']:.2f} s"
            if values["success"]
            else "no full rotation"
        )
        print(
            f"  {name}: {outcome}; "
            f"peak={values['peak_absolute_angle_degrees']:.2f} deg; "
            f"min axial force={values['minimum_tension_newtons']:.1f} N; "
            f"min r/L={values['minimum_seat_radius_fraction']:.3f}"
        )
    print("Wrote:")
    for path in outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
