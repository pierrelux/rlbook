"""Accessible browser replay for the recorded CubeSat formation experiment.

Python supplies every command and state sample.  The embedded JavaScript only
selects recorded daily frames, reveals trajectory prefixes, and draws the
already-computed open-loop command plan.  It contains no optimizer, orbital
dynamics, numerical integrator, or network access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import html
import json
import math
from pathlib import Path
import re
import uuid
from typing import Any


RUN_ORDER = ("nominal_linear", "nonlinear_replay")
RUN_LABELS = {
    "nominal_linear": "Nominal linear model",
    "nonlinear_replay": "Nonlinear replay",
}
FALLBACK_ID = "fig-cubesat-formation-fallback"
_SPACECRAFT_COUNT = 3
_FRAME_FIELDS = (
    "phase_deg",
    "cyclic_gap_deg",
    "cyclic_gap_error_deg",
    "relative_rate_deg_per_day",
    "altitude_km",
    "extra_altitude_loss_km",
)
_UNIT_CONTRACT = {
    "command_fraction": "unitless",
    "angle": "deg",
    "angular_rate": "deg/day",
    "altitude": "km",
    "density": "kg/m^3",
    "time": "day",
}


class CubeSatReplayError(ValueError):
    """Raised when a CubeSat artifact violates the replay contract."""


def _load_source(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"CubeSat replay artifact is missing: {path}") from None
    except json.JSONDecodeError as error:
        raise CubeSatReplayError(f"invalid CubeSat replay JSON: {error}") from error
    if not isinstance(value, dict):
        raise CubeSatReplayError("CubeSat replay root must be an object")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise CubeSatReplayError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise CubeSatReplayError(f"{field} must be numeric") from None
    if not math.isfinite(number):
        raise CubeSatReplayError(f"{field} must be finite")
    return number


def _vector(value: Any, field: str, *, length: int = _SPACECRAFT_COUNT) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CubeSatReplayError(f"{field} must be an array")
    if len(value) != length:
        raise CubeSatReplayError(f"{field} must contain exactly {length} values")
    return [_finite_number(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _close(left: float, right: float, *, tolerance: float = 1e-7) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _axis_range(values: Sequence[float], *, minimum_span: float) -> list[float]:
    low = min(values)
    high = max(values)
    span = max(high - low, minimum_span)
    padding = 0.08 * span
    return [low - padding, high + padding]


def _normalise(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    artifact = _load_source(source)
    if artifact.get("schema_version") != 1:
        raise CubeSatReplayError("unsupported CubeSat replay schema version")
    if artifact.get("status") != "complete":
        raise CubeSatReplayError("CubeSat replay artifact must have status 'complete'")

    raw_spacecraft = artifact.get("spacecraft")
    if not isinstance(raw_spacecraft, Sequence) or isinstance(raw_spacecraft, (str, bytes)):
        raise CubeSatReplayError("spacecraft must be an array")
    if len(raw_spacecraft) != _SPACECRAFT_COUNT:
        raise CubeSatReplayError("spacecraft must contain exactly three labels")
    spacecraft = [str(label).strip() for label in raw_spacecraft]
    if any(not label for label in spacecraft) or len(set(spacecraft)) != len(spacecraft):
        raise CubeSatReplayError("spacecraft labels must be nonempty and unique")

    raw_units = artifact.get("units")
    if not isinstance(raw_units, Mapping):
        raise CubeSatReplayError("units must be an object")
    for field, expected in _UNIT_CONTRACT.items():
        if raw_units.get(field) != expected:
            raise CubeSatReplayError(f"units.{field} must be {expected!r}")

    raw_scenario = artifact.get("scenario")
    if not isinstance(raw_scenario, Mapping):
        raise CubeSatReplayError("scenario must be an object")
    horizon_days = _finite_number(raw_scenario.get("horizon_days"), "scenario.horizon_days")
    interval_days = _finite_number(
        raw_scenario.get("control_interval_days"), "scenario.control_interval_days"
    )
    if not _close(horizon_days, 180.0) or not _close(interval_days, 1.0):
        raise CubeSatReplayError("replay requires the recorded 180-day daily plan")
    leader_index_raw = raw_scenario.get("leader_index")
    if isinstance(leader_index_raw, bool) or leader_index_raw != 0:
        raise CubeSatReplayError("scenario.leader_index must identify spacecraft 0")
    initial_altitude_km = _finite_number(
        raw_scenario.get("initial_altitude_km"), "scenario.initial_altitude_km"
    )
    if initial_altitude_km <= 0.0:
        raise CubeSatReplayError("scenario.initial_altitude_km must be positive")
    initial_phase = _vector(raw_scenario.get("initial_phase_deg"), "scenario.initial_phase_deg")
    initial_rate = _vector(
        raw_scenario.get("initial_relative_rate_deg_per_day"),
        "scenario.initial_relative_rate_deg_per_day",
    )
    target_slot = _vector(raw_scenario.get("target_slot_deg"), "scenario.target_slot_deg")
    target_gap = _vector(raw_scenario.get("target_gap_deg"), "scenario.target_gap_deg")
    gap_tolerance = _finite_number(
        raw_scenario.get("gap_tolerance_deg"), "scenario.gap_tolerance_deg"
    )
    rate_tolerance = _finite_number(
        raw_scenario.get("relative_rate_tolerance_deg_per_day"),
        "scenario.relative_rate_tolerance_deg_per_day",
    )
    if gap_tolerance <= 0.0 or rate_tolerance <= 0.0:
        raise CubeSatReplayError("scenario tolerances must be positive")
    expected_target_gap = [
        target_slot[1] - target_slot[0],
        target_slot[2] - target_slot[1],
        target_slot[0] - target_slot[2],
    ]
    if any(not _close(value, expected) for value, expected in zip(target_gap, expected_target_gap)):
        raise CubeSatReplayError("scenario target gaps must match the cyclic target slots")

    raw_plan = artifact.get("plan")
    if not isinstance(raw_plan, Mapping):
        raise CubeSatReplayError("plan must be an object")
    raw_days = raw_plan.get("day")
    if not isinstance(raw_days, Sequence) or isinstance(raw_days, (str, bytes)):
        raise CubeSatReplayError("plan.day must be an array")
    plan_days = [_finite_number(value, f"plan.day[{index}]") for index, value in enumerate(raw_days)]
    if len(plan_days) != 180 or any(
        not _close(day, float(index)) for index, day in enumerate(plan_days)
    ):
        raise CubeSatReplayError("plan.day must be the recorded grid 0, ..., 179")

    raw_u = raw_plan.get("U")
    if not isinstance(raw_u, Sequence) or isinstance(raw_u, (str, bytes)):
        raise CubeSatReplayError("plan.U must be a satellite-major array")
    if len(raw_u) != _SPACECRAFT_COUNT:
        raise CubeSatReplayError("plan.U must contain one row per spacecraft")
    commands: list[list[float]] = []
    for satellite, raw_row in enumerate(raw_u):
        row = _vector(raw_row, f"plan.U[{satellite}]", length=len(plan_days))
        if any(value < -1e-9 or value > 1.0 + 1e-9 for value in row):
            raise CubeSatReplayError("plan.U values must lie in [0, 1]")
        commands.append(row)

    duty_fraction = _vector(raw_plan.get("duty_fraction"), "plan.duty_fraction")
    equivalent_days = _vector(
        raw_plan.get("equivalent_high_drag_days"), "plan.equivalent_high_drag_days"
    )
    final_loss = _vector(
        raw_plan.get("final_extra_altitude_loss_km"),
        "plan.final_extra_altitude_loss_km",
    )
    if any(value < -1e-9 or value > 1.0 + 1e-9 for value in duty_fraction):
        raise CubeSatReplayError("plan.duty_fraction values must lie in [0, 1]")
    if any(value < -1e-9 for value in equivalent_days + final_loss):
        raise CubeSatReplayError("plan duty days and altitude losses cannot be negative")
    for satellite, row in enumerate(commands):
        expected_duty = sum(row) / len(row)
        expected_days = sum(row) * interval_days
        if not _close(duty_fraction[satellite], expected_duty, tolerance=5e-7):
            raise CubeSatReplayError("plan.duty_fraction disagrees with plan.U")
        if not _close(equivalent_days[satellite], expected_days, tolerance=5e-7):
            raise CubeSatReplayError("plan.equivalent_high_drag_days disagrees with plan.U")
    identity = raw_plan.get("identity_sha256")
    if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
        raise CubeSatReplayError("plan.identity_sha256 must be a lowercase SHA-256 digest")

    raw_runs = artifact.get("runs")
    if not isinstance(raw_runs, Mapping):
        raise CubeSatReplayError("runs must be an object")
    runs: dict[str, dict[str, Any]] = {}
    expected_frame_days = [float(index) for index in range(181)]
    altitude_values: list[float] = []
    error_values: list[float] = []
    for run_name in RUN_ORDER:
        raw_run = raw_runs.get(run_name)
        if not isinstance(raw_run, Mapping):
            raise CubeSatReplayError(f"missing run: {run_name}")
        if "status" in raw_run and raw_run.get("status") != "complete":
            raise CubeSatReplayError(f"{run_name}.status must be 'complete'")
        raw_frames = raw_run.get("frames")
        if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, (str, bytes)):
            raise CubeSatReplayError(f"{run_name}.frames must be an array")
        if len(raw_frames) != len(expected_frame_days):
            raise CubeSatReplayError(f"{run_name}.frames must contain days 0, ..., 180")
        frames: list[dict[str, Any]] = []
        for frame_index, raw_frame in enumerate(raw_frames):
            prefix = f"{run_name}.frames[{frame_index}]"
            if not isinstance(raw_frame, Mapping):
                raise CubeSatReplayError(f"{prefix} must be an object")
            day = _finite_number(raw_frame.get("day"), f"{prefix}.day")
            if not _close(day, expected_frame_days[frame_index]):
                raise CubeSatReplayError(f"{run_name} frame days must be 0, ..., 180")
            frame = {"day": day}
            for field in _FRAME_FIELDS:
                frame[field] = _vector(raw_frame.get(field), f"{prefix}.{field}")
            if any(value <= 0.0 for value in frame["altitude_km"]):
                raise CubeSatReplayError(f"{prefix}.altitude_km must be positive")
            expected_gap = [
                frame["phase_deg"][1] - frame["phase_deg"][0],
                frame["phase_deg"][2] - frame["phase_deg"][1],
                frame["phase_deg"][0] - frame["phase_deg"][2],
            ]
            if any(
                not _close(value, expected, tolerance=2e-6)
                for value, expected in zip(frame["cyclic_gap_deg"], expected_gap)
            ):
                raise CubeSatReplayError(f"{prefix}.cyclic_gap_deg disagrees with phase_deg")
            expected_error = [
                value - target
                for value, target in zip(frame["cyclic_gap_deg"], target_gap)
            ]
            if any(
                not _close(value, expected, tolerance=2e-6)
                for value, expected in zip(frame["cyclic_gap_error_deg"], expected_error)
            ):
                raise CubeSatReplayError(
                    f"{prefix}.cyclic_gap_error_deg disagrees with the target gaps"
                )
            altitude_values.extend(frame["altitude_km"])
            error_values.extend(frame["cyclic_gap_error_deg"])
            frames.append(frame)
        if any(not _close(value, initial_phase[index], tolerance=2e-6) for index, value in enumerate(frames[0]["phase_deg"])):
            raise CubeSatReplayError(f"{run_name} initial phase disagrees with scenario")
        if any(not _close(value, initial_rate[index], tolerance=2e-6) for index, value in enumerate(frames[0]["relative_rate_deg_per_day"])):
            raise CubeSatReplayError(f"{run_name} initial relative rate disagrees with scenario")
        runs[run_name] = {"label": RUN_LABELS[run_name], "frames": frames}

    raw_metrics = artifact.get("metrics")
    if not isinstance(raw_metrics, Mapping) or any(
        not isinstance(raw_metrics.get(name), Mapping)
        for name in (*RUN_ORDER, "replay_refinement", "validation")
    ):
        raise CubeSatReplayError("metrics must contain both runs, refinement, and validation")

    replay = {
        "spacecraft": spacecraft,
        "scenario": {
            "horizon_days": horizon_days,
            "leader_index": 0,
            "target_slot_deg": target_slot,
            "target_gap_deg": target_gap,
            "gap_tolerance_deg": gap_tolerance,
            "relative_rate_tolerance_deg_per_day": rate_tolerance,
        },
        "plan": {
            "day": plan_days,
            "U": commands,
            "duty_fraction": duty_fraction,
            "equivalent_high_drag_days": equivalent_days,
            "final_extra_altitude_loss_km": final_loss,
        },
        "runs": runs,
        "ranges": {
            "altitude_km": _axis_range(altitude_values, minimum_span=0.2),
            "gap_error_deg": _axis_range(
                [*error_values, -gap_tolerance, gap_tolerance], minimum_span=0.2
            ),
        },
    }
    return replay


def _dom_id(prefix: str | None) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", prefix or "cubesat-replay").strip("-")
    if not stem:
        stem = "cubesat-replay"
    return f"{stem}-{uuid.uuid4().hex[:10]}"


def _safe_json(value: Any) -> str:
    return (
        json.dumps(value, separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_cubesat_replay(
    source: Path | str | Mapping[str, Any],
    *,
    replay_id: str | None = None,
    fallback_id: str = FALLBACK_ID,
) -> str:
    """Return a dependency-free player for immutable CubeSat results."""

    replay = _normalise(source)
    root = _dom_id(replay_id)
    title_id = f"{root}-title"
    description_id = f"{root}-description"
    data_id = f"{root}-data"

    satellite_rows = "".join(
        f"""
        <div class="duty-card" data-satellite="{index}">
          <div class="duty-head"><span class="sat-mark sat-{index}" aria-hidden="true"></span><strong>{html.escape(label)}</strong><output data-duty-value>0%</output></div>
          <meter min="0" max="1" value="0" data-duty-meter aria-label="Recorded high-drag command for {html.escape(label)} on the selected day">0%</meter>
          <span class="duty-summary">whole-plan duty {100 * replay['plan']['duty_fraction'][index]:.1f}% · {replay['plan']['equivalent_high_drag_days'][index]:.1f} equivalent days</span>
        </div>
        """
        for index, label in enumerate(replay["spacecraft"])
    )
    gap_rows = "".join(
        f"""
        <div class="gap-row" data-gap="{index}">
          <span class="gap-edge edge-{index}" aria-hidden="true"></span>
          <span>{html.escape(replay['spacecraft'][index])} → {html.escape(replay['spacecraft'][(index + 1) % _SPACECRAFT_COUNT])}</span>
          <output data-gap-value>—</output>
        </div>
        """
        for index in range(_SPACECRAFT_COUNT)
    )

    template = r'''
<section id="__ROOT__" class="cubesat-replay" tabindex="0" aria-labelledby="__TITLE_ID__" aria-describedby="__DESCRIPTION_ID__">
  <style>
    #__ROOT__ {
      --paper:#F6F7F4; --raised:#FFFFFF; --ink:#1B2430; --muted:#66727D; --rule:#CDD5D5;
      --blue:#0072B2; --orange:#E69F00; --green:#009E73; --accent:#2F6F8F; --error:#A83A32;
      background:var(--paper); border:1px solid var(--rule); border-radius:9px; color:var(--ink);
      color-scheme:light; box-sizing:border-box; font:14px/1.4 "IBM Plex Sans",system-ui,sans-serif;
      margin-inline:auto; max-width:64rem; padding:clamp(.7rem,2vw,1.1rem); width:100%;
    }
    #__ROOT__[data-theme="dark"] {
      --paper:#121920; --raised:#1B2430; --ink:#EDF1F1; --muted:#A8B2B9; --rule:#34434D;
      --blue:#56B4E9; --orange:#F0B94A; --green:#45B999; --accent:#72A8C2; --error:#DF786F;
      color-scheme:dark;
    }
    #__ROOT__ *, #__ROOT__ *::before, #__ROOT__ *::after { box-sizing:border-box; }
    #__ROOT__ [hidden] { display:none !important; }
    #__ROOT__:focus-visible { outline:3px solid color-mix(in srgb,var(--accent) 35%,transparent); outline-offset:3px; }
    #__ROOT__ h3 { font:500 clamp(1.2rem,3vw,1.5rem)/1.12 Newsreader,Georgia,serif; margin:0; }
    #__ROOT__ h4 { font-size:.82rem; margin:0 0 .35rem; }
    #__ROOT__ .lede { color:var(--muted); font-size:.88rem; margin:.28rem 0 .7rem; max-width:78ch; }
    #__ROOT__ .toolbar { align-items:center; border-block:1px solid var(--rule); display:flex; flex-wrap:wrap; gap:.45rem; padding:.55rem 0; }
    #__ROOT__ button { appearance:none; background:var(--raised); border:1px solid #98A5A9; border-radius:5px; color:var(--ink); font:inherit; font-size:.8rem; min-height:2rem; padding:.3rem .58rem; }
    #__ROOT__ button:hover, #__ROOT__ button:focus-visible { border-color:var(--accent); outline:2px solid color-mix(in srgb,var(--accent) 23%,transparent); outline-offset:1px; }
    #__ROOT__ .scrubber { accent-color:var(--accent); flex:1 1 15rem; min-width:10rem; }
    #__ROOT__ [data-time] { font:600 .8rem "IBM Plex Mono",ui-monospace,monospace; min-width:5.7rem; text-align:right; }
    #__ROOT__ .orbit-grid { display:grid; gap:.75rem; grid-template-columns:repeat(2,minmax(0,1fr)); margin-top:.75rem; }
    #__ROOT__ .orbit-card, #__ROOT__ .chart-card, #__ROOT__ .plan-card { background:color-mix(in srgb,var(--raised) 62%,var(--paper)); border:1px solid var(--rule); border-radius:7px; min-width:0; padding:.55rem; }
    #__ROOT__ .orbit-card > p, #__ROOT__ .chart-note, #__ROOT__ .plan-note { color:var(--muted); font-size:.7rem; margin:.1rem 0 .3rem; }
    #__ROOT__ .orbit { display:block; height:auto; margin:auto; max-width:22rem; overflow:visible; width:100%; }
    #__ROOT__ .orbit-ring { fill:none; stroke:var(--rule); stroke-width:2; }
    #__ROOT__ .reference-ray { stroke:var(--muted); stroke-dasharray:3 4; stroke-width:1; }
    #__ROOT__ .target-slot { fill:var(--paper); stroke:var(--muted); stroke-dasharray:2 2; stroke-width:1.5; }
    #__ROOT__ .target-label { fill:var(--muted); font:10px "IBM Plex Mono",ui-monospace,monospace; }
    #__ROOT__ .satellite { stroke:var(--paper); stroke-width:2.2; vector-effect:non-scaling-stroke; }
    #__ROOT__ .satellite.sat-0 { fill:var(--blue); }
    #__ROOT__ .satellite.sat-1 { fill:var(--orange); }
    #__ROOT__ .satellite.sat-2 { fill:var(--green); }
    #__ROOT__ .sat-label { fill:var(--ink); font:600 10px "IBM Plex Sans",sans-serif; paint-order:stroke; stroke:var(--paper); stroke-width:3px; }
    #__ROOT__ .gap-list { border-top:1px solid var(--rule); display:grid; gap:.18rem; margin-top:.25rem; padding-top:.35rem; }
    #__ROOT__ .gap-row { align-items:center; display:grid; font-size:.68rem; gap:.35rem; grid-template-columns:.7rem 1fr auto; }
    #__ROOT__ .gap-row output { font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums; }
    #__ROOT__ .gap-edge { border-top:3px solid; display:block; width:.7rem; }
    #__ROOT__ .edge-0 { color:var(--blue); } #__ROOT__ .edge-1 { color:var(--orange); } #__ROOT__ .edge-2 { color:var(--green); }
    #__ROOT__ .trace-grid { display:grid; gap:.75rem; grid-template-columns:repeat(2,minmax(0,1fr)); margin-top:.75rem; }
    #__ROOT__ .chart { display:block; height:auto; overflow:visible; width:100%; }
    #__ROOT__ .chart .axis, #__ROOT__ .chart .grid { stroke:var(--rule); stroke-width:1; vector-effect:non-scaling-stroke; }
    #__ROOT__ .chart .grid { stroke-dasharray:2 5; }
    #__ROOT__ .chart .zero { stroke:var(--muted); stroke-dasharray:4 4; stroke-width:1; }
    #__ROOT__ .chart .tolerance { fill:color-mix(in srgb,var(--green) 10%,transparent); }
    #__ROOT__ .chart .trace { fill:none; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; vector-effect:non-scaling-stroke; }
    #__ROOT__ .chart .nominal { opacity:.72; }
    #__ROOT__ .chart .nonlinear { stroke-dasharray:5 3; }
    #__ROOT__ .chart .series-0 { stroke:var(--blue); } #__ROOT__ .chart .series-1 { stroke:var(--orange); } #__ROOT__ .chart .series-2 { stroke:var(--green); }
    #__ROOT__ .chart text { fill:var(--muted); font:10px "IBM Plex Mono",ui-monospace,monospace; }
    #__ROOT__ .encoding { color:var(--muted); display:flex; flex-wrap:wrap; font-size:.68rem; gap:.55rem; margin-top:.25rem; }
    #__ROOT__ .encoding span::before { border-top:2px solid var(--ink); content:""; display:inline-block; margin-right:.25rem; transform:translateY(-.12rem); width:1rem; }
    #__ROOT__ .encoding .dash::before { border-top-style:dashed; }
    #__ROOT__ .plan-card { margin-top:.75rem; }
    #__ROOT__ .duty-grid { display:grid; gap:.45rem; grid-template-columns:repeat(3,minmax(0,1fr)); margin:.45rem 0 .6rem; }
    #__ROOT__ .duty-card { min-width:0; }
    #__ROOT__ .duty-head { align-items:center; display:grid; font-size:.72rem; gap:.28rem; grid-template-columns:.7rem 1fr auto; }
    #__ROOT__ .sat-mark { border-radius:50%; display:block; height:.55rem; width:.55rem; }
    #__ROOT__ .sat-mark.sat-0 { background:var(--blue); } #__ROOT__ .sat-mark.sat-1 { background:var(--orange); } #__ROOT__ .sat-mark.sat-2 { background:var(--green); }
    #__ROOT__ meter { accent-color:var(--accent); display:block; height:.72rem; width:100%; }
    #__ROOT__ .duty-summary { color:var(--muted); display:block; font-size:.62rem; line-height:1.2; }
    #__ROOT__ .heatmap { display:block; height:auto; width:100%; }
    #__ROOT__ .heatmap .off { fill:color-mix(in srgb,var(--rule) 52%,transparent); }
    #__ROOT__ .heatmap .on-0 { fill:var(--blue); } #__ROOT__ .heatmap .on-1 { fill:var(--orange); } #__ROOT__ .heatmap .on-2 { fill:var(--green); }
    #__ROOT__ .heatmap .cursor { fill:none; stroke:var(--ink); stroke-width:1.5; vector-effect:non-scaling-stroke; }
    #__ROOT__ .heatmap text { fill:var(--muted); font:10px "IBM Plex Mono",ui-monospace,monospace; }
    #__ROOT__ .sr-only { clip:rect(0 0 0 0); clip-path:inset(50%); height:1px; overflow:hidden; position:absolute; white-space:nowrap; width:1px; }
    @media (max-width:720px) {
      #__ROOT__ .orbit-grid, #__ROOT__ .trace-grid { grid-template-columns:1fr; }
      #__ROOT__ .duty-grid { grid-template-columns:1fr; }
      #__ROOT__ .toolbar { align-items:stretch; }
      #__ROOT__ [data-time] { text-align:left; }
    }
    @media (prefers-reduced-motion:reduce) {
      #__ROOT__ *, #__ROOT__ *::before, #__ROOT__ *::after { animation-duration:.001ms !important; scroll-behavior:auto !important; transition-duration:.001ms !important; }
    }
    @media print { #__ROOT__ .toolbar { display:none; } }
  </style>
  <header>
    <h3 id="__TITLE_ID__">Differential drag: plan once, verify twice</h3>
    <p id="__DESCRIPTION_ID__" class="lede">Both formation views use the same fixed-radius orbit and literal recorded angles. The state traces reveal only history through the selected day; the complete open-loop drag plan is visible from the start.</p>
  </header>
  <div class="toolbar" aria-label="Replay controls">
    <button type="button" data-action="play" aria-keyshortcuts="Space">Play</button>
    <button type="button" data-action="step-back" aria-label="Previous recorded day" aria-keyshortcuts="ArrowLeft">−1 day</button>
    <button type="button" data-action="step" aria-label="Next recorded day" aria-keyshortcuts="ArrowRight">+1 day</button>
    <button type="button" data-action="reset" aria-keyshortcuts="Home">Reset</button>
    <input class="scrubber" type="range" min="0" max="180" step="1" value="0" aria-label="Recorded day">
    <span role="timer" data-time aria-live="polite">day 0 / 180</span>
  </div>
  <div class="orbit-grid">
    __ORBIT_CARDS__
  </div>
  <div class="trace-grid">
    <article class="chart-card">
      <h4>Model-reported altitude</h4>
      <p class="chart-note">Nominal is 475 km minus extra loss and omits common low-drag decay; nonlinear is absolute. Traces stop at the playhead.</p>
      <svg class="chart" data-chart="altitude_km" viewBox="0 0 520 205" role="img" aria-label="Nominal altitude proxy and nonlinear absolute altitude histories through the selected day">
        <line class="axis" x1="42" y1="12" x2="42" y2="174"></line><line class="axis" x1="42" y1="174" x2="510" y2="174"></line>
        <line class="grid" x1="42" y1="12" x2="510" y2="12"></line><line class="grid" x1="42" y1="93" x2="510" y2="93"></line>
        <text x="3" y="17" data-ymax></text><text x="3" y="178" data-ymin></text><text x="42" y="195">0</text><text x="483" y="195">180 d</text>
        __TRACE_PATHS_ALTITUDE__
        <line class="zero" data-cursor x1="42" y1="12" x2="42" y2="174"></line>
      </svg>
      <div class="encoding"><span>nominal, solid</span><span class="dash">nonlinear, dashed</span></div>
    </article>
    <article class="chart-card">
      <h4>Cyclic-gap error</h4>
      <p class="chart-note">Each trace is recorded gap minus its fixed target; all traces stop at the playhead.</p>
      <svg class="chart" data-chart="cyclic_gap_error_deg" viewBox="0 0 520 205" role="img" aria-label="Recorded nominal and nonlinear cyclic gap errors through the selected day">
        <rect class="tolerance" data-tolerance x="42" y="90" width="468" height="6"></rect>
        <line class="axis" x1="42" y1="12" x2="42" y2="174"></line><line class="axis" x1="42" y1="174" x2="510" y2="174"></line>
        <line class="grid" x1="42" y1="12" x2="510" y2="12"></line><line class="zero" data-zero x1="42" y1="93" x2="510" y2="93"></line>
        <text x="3" y="17" data-ymax></text><text x="3" y="178" data-ymin></text><text x="42" y="195">0</text><text x="483" y="195">180 d</text>
        __TRACE_PATHS_ERROR__
        <line class="zero" data-cursor x1="42" y1="12" x2="42" y2="174"></line>
      </svg>
      <div class="encoding"><span>nominal, solid</span><span class="dash">nonlinear, dashed</span></div>
    </article>
  </div>
  <article class="plan-card">
    <h4>Complete open-loop high-drag plan</h4>
    <p class="plan-note">Every column is one recorded daily command interval. The outline follows the selected day; the plan itself is never censored by the playhead.</p>
    <div class="duty-grid">__DUTY_ROWS__</div>
    <svg class="heatmap" viewBox="0 0 760 104" role="img" aria-label="Complete 180-day high-drag command heatmap available from day zero">
      <g data-heatmap-cells></g><rect class="cursor" data-plan-cursor x="112" y="10" width="3" height="67"></rect>
      <text x="112" y="97">day 0</text><text x="700" y="97">day 180</text>
    </svg>
  </article>
  <p class="sr-only" data-status aria-live="polite"></p>
  <script type="application/json" id="__DATA_ID__">__DATA__</script>
  <script>
  (() => {
    const root=document.getElementById("__ROOT__");
    const replay=JSON.parse(document.getElementById("__DATA_ID__").textContent);
    const NS="http://www.w3.org/2000/svg";
    const scrubber=root.querySelector(".scrubber"), timeOutput=root.querySelector("[data-time]");
    const playButton=root.querySelector('[data-action="play"]'), statusOutput=root.querySelector("[data-status]");
    const colors=["var(--blue)","var(--orange)","var(--green)"];
    let frameIndex=0, playing=false, lastTick=0;
    const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));
    const svgElement=(name,attributes={}) => { const node=document.createElementNS(NS,name); Object.entries(attributes).forEach(([key,value])=>node.setAttribute(key,String(value))); return node; };
    const orbitPoint=(relativeAngleDeg,radius=112) => { const angle=(relativeAngleDeg-90)*Math.PI/180; return [160+radius*Math.cos(angle),160+radius*Math.sin(angle)]; };
    const relativePhase=(frame,index) => frame.phase_deg[index]-frame.phase_deg[replay.scenario.leader_index];
    const xScale=day => 42+468*day/replay.scenario.horizon_days;
    const yScale=(field,value) => { const range=field === "altitude_km" ? replay.ranges.altitude_km : replay.ranges.gap_error_deg; return 174-162*(value-range[0])/(range[1]-range[0]); };
    const frame=(runName) => replay.runs[runName].frames[frameIndex];

    const makeOrbit = runName => {
      const svg=root.querySelector(`[data-orbit="${runName}"]`), targets=svg.querySelector("[data-targets]"), satellites=svg.querySelector("[data-satellites]");
      replay.scenario.target_slot_deg.forEach((angle,index) => {
        const [x,y]=orbitPoint(angle-replay.scenario.target_slot_deg[replay.scenario.leader_index]);
        const [labelX,labelY]=orbitPoint(angle-replay.scenario.target_slot_deg[replay.scenario.leader_index],88);
        targets.append(svgElement("circle",{cx:x,cy:y,r:8,class:"target-slot"}));
        const label=svgElement("text",{x:labelX,y:labelY+3,class:"target-label","text-anchor":"middle"}); label.textContent=`slot ${index+1}`; targets.append(label);
        const dot=svgElement("circle",{cx:160,cy:48,r:index===0?8:7,class:`satellite sat-${index}`,"data-satellite":index}); satellites.append(dot);
        const name=svgElement("text",{x:160,y:35,class:"sat-label","text-anchor":"middle","data-satellite-label":index}); name.textContent=replay.spacecraft[index]; satellites.append(name);
      });
    };
    const drawHeatmap = () => {
      const layer=root.querySelector("[data-heatmap-cells]"), left=112, top=10, width=600, rowHeight=21, cellWidth=width/replay.plan.day.length;
      replay.spacecraft.forEach((label,satellite) => {
        const text=svgElement("text",{x:105,y:top+satellite*rowHeight+14,"text-anchor":"end"}); text.textContent=label; layer.append(text);
        replay.plan.U[satellite].forEach((command,day) => layer.append(svgElement("rect",{x:left+day*cellWidth,y:top+satellite*rowHeight,width:cellWidth+.15,height:17,class:command>0?`on-${satellite}`:"off",opacity:command>0?clamp(.3+.7*command,.3,1):1})));
      });
    };
    const tracePoints = (runName,field,series) => replay.runs[runName].frames.slice(0,frameIndex+1).map(sample => `${xScale(sample.day).toFixed(2)},${yScale(field,sample[field][series]).toFixed(2)}`).join(" ");
    const configureCharts = () => {
      root.querySelectorAll("[data-chart]").forEach(chart => {
        const field=chart.dataset.chart, range=field === "altitude_km" ? replay.ranges.altitude_km : replay.ranges.gap_error_deg;
        chart.querySelector("[data-ymax]").textContent=field === "altitude_km" ? `${range[1].toFixed(2)} km` : `${range[1].toFixed(1)}°`;
        chart.querySelector("[data-ymin]").textContent=field === "altitude_km" ? `${range[0].toFixed(2)} km` : `${range[0].toFixed(1)}°`;
        if (field === "cyclic_gap_error_deg") {
          const zero=chart.querySelector("[data-zero]"), tolerance=chart.querySelector("[data-tolerance]");
          const zeroY=yScale(field,0), top=yScale(field,replay.scenario.gap_tolerance_deg), bottom=yScale(field,-replay.scenario.gap_tolerance_deg);
          zero.setAttribute("y1",zeroY); zero.setAttribute("y2",zeroY); tolerance.setAttribute("y",Math.min(top,bottom)); tolerance.setAttribute("height",Math.abs(bottom-top));
        }
      });
    };
    const renderOrbit = runName => {
      const current=frame(runName), svg=root.querySelector(`[data-orbit="${runName}"]`);
      replay.spacecraft.forEach((_,index) => {
        const [x,y]=orbitPoint(relativePhase(current,index));
        const dot=svg.querySelector(`[data-satellite="${index}"]`), label=svg.querySelector(`[data-satellite-label="${index}"]`);
        const labelDy=[-8,4,16][index];
        dot.setAttribute("cx",x); dot.setAttribute("cy",y); label.setAttribute("x",x+(x>=160?10:-10)); label.setAttribute("y",y+labelDy); label.setAttribute("text-anchor",x>=160?"start":"end");
      });
      svg.closest(".orbit-card").querySelectorAll("[data-gap]").forEach((row,index) => {
        const gap=current.cyclic_gap_deg[index], error=current.cyclic_gap_error_deg[index], sign=error>=0?"+":"";
        row.querySelector("[data-gap-value]").textContent=`${gap.toFixed(2)}° (${sign}${error.toFixed(2)}°)`;
      });
    };
    const render = () => {
      scrubber.value=String(frameIndex); timeOutput.textContent=`day ${frameIndex} / 180`;
      RUNS.forEach(renderOrbit);
      root.querySelectorAll("[data-chart]").forEach(chart => {
        const field=chart.dataset.chart;
        RUNS.forEach(runName => replay.spacecraft.forEach((_,series) => chart.querySelector(`[data-trace="${runName}-${series}"]`).setAttribute("points",tracePoints(runName,field,series))));
        const cursorX=xScale(frameIndex); const cursor=chart.querySelector("[data-cursor]"); cursor.setAttribute("x1",cursorX); cursor.setAttribute("x2",cursorX);
      });
      const actionDay=Math.min(frameIndex,replay.plan.day.length-1), cellWidth=600/replay.plan.day.length;
      const planCursor=root.querySelector("[data-plan-cursor]"); planCursor.setAttribute("x",112+actionDay*cellWidth); planCursor.setAttribute("width",Math.max(cellWidth,1.5));
      root.querySelectorAll("[data-satellite]").forEach(card => {
        if (!card.classList.contains("duty-card")) return;
        const satellite=Number(card.dataset.satellite), command=replay.plan.U[satellite][actionDay];
        const meter=card.querySelector("[data-duty-meter]"); meter.value=command; meter.textContent=`${(100*command).toFixed(0)}%`;
        card.querySelector("[data-duty-value]").textContent=`${(100*command).toFixed(0)}%`;
      });
      statusOutput.textContent=`Showing recorded day ${frameIndex}. State histories end at this day; the full command plan remains visible.`;
    };
    const stop = () => { playing=false; playButton.textContent="Play"; };
    const tick = timestamp => {
      if (!playing) return;
      if (!lastTick || timestamp-lastTick>=90) {
        if (frameIndex>=180) { stop(); return; }
        frameIndex+=1; lastTick=timestamp; render();
      }
      requestAnimationFrame(tick);
    };
    const togglePlay = () => {
      if (playing) { stop(); return; }
      if (frameIndex>=180) frameIndex=0;
      playing=true; lastTick=0; playButton.textContent="Pause"; requestAnimationFrame(tick);
    };
    playButton.addEventListener("click",togglePlay);
    root.querySelector('[data-action="step-back"]').addEventListener("click",() => { stop(); frameIndex=Math.max(0,frameIndex-1); render(); });
    root.querySelector('[data-action="step"]').addEventListener("click",() => { stop(); frameIndex=Math.min(180,frameIndex+1); render(); });
    root.querySelector('[data-action="reset"]').addEventListener("click",() => { stop(); frameIndex=0; render(); });
    scrubber.addEventListener("input",() => { stop(); frameIndex=Number(scrubber.value); render(); });
    root.addEventListener("keydown",event => {
      if (event.target!==root) return;
      if (event.key===" ") { event.preventDefault(); togglePlay(); }
      else if (event.key==="ArrowRight") { event.preventDefault(); stop(); frameIndex=Math.min(180,frameIndex+1); render(); }
      else if (event.key==="ArrowLeft") { event.preventDefault(); stop(); frameIndex=Math.max(0,frameIndex-1); render(); }
      else if (event.key==="Home") { event.preventDefault(); stop(); frameIndex=0; render(); }
      else if (event.key==="End") { event.preventDefault(); stop(); frameIndex=180; render(); }
    });
    const applyTheme = () => {
      let dark=false;
      try {
        const themeRoot=window.parent&&window.parent!==window?window.parent.document.documentElement:document.documentElement;
        const declared=String(themeRoot.dataset.theme||themeRoot.getAttribute("data-mode")||"").toLowerCase();
        dark=declared==="dark"||themeRoot.classList.contains("dark")||getComputedStyle(themeRoot).colorScheme==="dark";
        if (!dark&&window.matchMedia) dark=window.matchMedia("(prefers-color-scheme: dark)").matches;
      } catch (_) { dark=window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)").matches; }
      root.dataset.theme=dark?"dark":"light";
    };
    const RUNS=["nominal_linear","nonlinear_replay"];
    applyTheme();
    try {
      const themeRoot=window.parent&&window.parent!==window?window.parent.document.documentElement:document.documentElement;
      if (typeof MutationObserver!=="undefined") new MutationObserver(applyTheme).observe(themeRoot,{attributes:true,attributeFilter:["class","data-theme","data-mode","style"]});
    } catch (_) {}
    if (window.matchMedia) { const query=window.matchMedia("(prefers-color-scheme: dark)"); if (typeof query.addEventListener==="function") query.addEventListener("change",applyTheme); }
    RUNS.forEach(makeOrbit); drawHeatmap(); configureCharts(); render();

    const fallbackId=__FALLBACK_JSON__;
    const hideFallback = doc => {
      if (!doc) return false; const fallback=doc.getElementById(fallbackId); if (!fallback) return false;
      fallback.hidden=true; fallback.setAttribute("aria-hidden","true"); return true;
    };
    hideFallback(document);
    try { if (window.parent&&window.parent!==window) hideFallback(window.parent.document); } catch (_) {}
    if (typeof MutationObserver!=="undefined") {
      const fallbackObserver=new MutationObserver(() => { if (hideFallback(document)) fallbackObserver.disconnect(); });
      fallbackObserver.observe(document.documentElement,{childList:true,subtree:true});
    }
  })();
  </script>
</section>
'''

    orbit_card_template = """
    <article class="orbit-card" data-run="{run_name}">
      <h4>{label}</h4>
      <p>Leader-relative recorded phase · fixed radius · altitude not encoded radially</p>
      <svg class="orbit" data-orbit="{run_name}" viewBox="0 0 320 320" role="img" aria-label="{label} recorded CubeSat phases on a fixed-radius leader-relative orbit">
        <circle class="orbit-ring" cx="160" cy="160" r="112"></circle>
        <line class="reference-ray" x1="160" y1="160" x2="160" y2="48"></line>
        <g data-targets></g><g data-satellites></g>
      </svg>
      <div class="gap-list" aria-label="Direct cyclic-gap readings">{gap_rows}</div>
    </article>
    """
    orbit_cards = "".join(
        orbit_card_template.format(
            run_name=run_name,
            label=html.escape(RUN_LABELS[run_name]),
            gap_rows=gap_rows,
        )
        for run_name in RUN_ORDER
    )
    trace_paths = "".join(
        f'<polyline class="trace {"nominal" if run_name == "nominal_linear" else "nonlinear"} series-{series}" data-trace="{run_name}-{series}" points=""></polyline>'
        for run_name in RUN_ORDER
        for series in range(_SPACECRAFT_COUNT)
    )
    return (
        template.replace("__ROOT__", root)
        .replace("__TITLE_ID__", title_id)
        .replace("__DESCRIPTION_ID__", description_id)
        .replace("__DATA_ID__", data_id)
        .replace("__ORBIT_CARDS__", orbit_cards)
        .replace("__DUTY_ROWS__", satellite_rows)
        .replace("__TRACE_PATHS_ALTITUDE__", trace_paths)
        .replace("__TRACE_PATHS_ERROR__", trace_paths)
        .replace("__DATA__", _safe_json(replay))
        .replace("__FALLBACK_JSON__", _safe_json(str(fallback_id)))
    )


__all__ = [
    "CubeSatReplayError",
    "FALLBACK_ID",
    "RUN_ORDER",
    "render_cubesat_replay",
]
