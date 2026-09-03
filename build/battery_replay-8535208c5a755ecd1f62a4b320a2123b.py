"""Browser-native replay for the recorded battery fast-charging audit.

The renderer accepts only completed Python trajectories. Its script selects a
recorded run, advances a playhead, and redraws trajectory prefixes. It contains
no battery model, parameter fit, controller, or network access.
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


RUN_ORDER = (
    "fresh_nominal",
    "high_resistance_stale",
    "high_resistance_calibrated",
)
FALLBACK_ID = "fig-battery-fast-charging-fallback"

_DEFAULT_LABELS = {
    "fresh_nominal": "Fresh plant, nominal model",
    "high_resistance_stale": "High resistance, stale model",
    "high_resistance_calibrated": "High resistance, fitted model",
}
_DEFAULT_VERDICTS = {
    "fresh_nominal": "stands",
    "high_resistance_stale": "withdrawn",
    "high_resistance_calibrated": "stands",
}
_VERDICT_LABELS = {
    "stands": "within plant bounds",
    "caveat": "inspect the recorded bound margin",
    "withdrawn": "plant bound crossed",
}


class BatteryReplayError(ValueError):
    """Raised when a battery replay artifact violates the player contract."""


def _load_source(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"battery replay artifact is missing: {path}") from None
    except json.JSONDecodeError as error:
        raise BatteryReplayError(f"invalid battery replay JSON: {error}") from error
    if not isinstance(value, dict):
        raise BatteryReplayError("battery replay root must be an object")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise BatteryReplayError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise BatteryReplayError(f"{field} must be numeric") from None
    if not math.isfinite(number):
        raise BatteryReplayError(f"{field} must be finite")
    return number


def _optional_time(value: Any, field: str, final_time: float) -> float | None:
    if value is None:
        return None
    number = _finite_number(value, field)
    if number < 0.0 or number > final_time + 1e-9:
        raise BatteryReplayError(f"{field} must lie on its recorded run")
    return number


def _causal_index(frames: Sequence[Mapping[str, float]], at_s: float | None) -> int | None:
    if at_s is None:
        return None
    return next(
        (
            index
            for index, frame in enumerate(frames)
            if float(frame["time_s"]) >= at_s - 1e-9
        ),
        len(frames) - 1,
    )


def _axis_limits(values: Sequence[float], required: Sequence[float], pad: float) -> list[float]:
    low = min((*values, *required))
    high = max((*values, *required))
    span = max(high - low, pad)
    return [low - 0.06 * span, high + 0.08 * span]


def _normalise(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    artifact = _load_source(source)
    if artifact.get("schema_version") != 1:
        raise BatteryReplayError("unsupported battery replay schema version")

    raw_scenario = artifact.get("scenario")
    raw_runs = artifact.get("runs")
    if not isinstance(raw_scenario, Mapping):
        raise BatteryReplayError("battery replay must contain a scenario object")
    if not isinstance(raw_runs, Mapping):
        raise BatteryReplayError("battery replay must contain a runs object")

    scenario_fields = (
        "initial_soc",
        "target_soc",
        "current_limit_a",
        "voltage_limit_v",
        "temperature_limit_c",
        "voltage_guard_v",
        "temperature_guard_c",
        "duration_s",
        "control_period_s",
    )
    scenario = {
        field: _finite_number(raw_scenario.get(field), f"scenario.{field}")
        for field in scenario_fields
    }
    if not 0.0 <= scenario["initial_soc"] < scenario["target_soc"] <= 1.0:
        raise BatteryReplayError("scenario SOC values must satisfy 0 <= initial < target <= 1")
    if scenario["current_limit_a"] <= 0.0:
        raise BatteryReplayError("scenario.current_limit_a must be positive")
    if scenario["voltage_guard_v"] > scenario["voltage_limit_v"]:
        raise BatteryReplayError("voltage guard cannot exceed the plant limit")
    if scenario["temperature_guard_c"] > scenario["temperature_limit_c"]:
        raise BatteryReplayError("temperature guard cannot exceed the plant limit")
    if scenario["duration_s"] <= 0.0 or scenario["control_period_s"] <= 0.0:
        raise BatteryReplayError("scenario times must be positive")

    frame_fields = (
        "time_s",
        "soc",
        "current_a",
        "terminal_voltage_v",
        "cell_temperature_c",
        "jig_temperature_c",
    )
    metric_fields = (
        "target_time_s",
        "max_voltage_v",
        "voltage_violation_duration_s",
        "max_cell_temperature_c",
    )
    event_fields = (
        "first_taper_time_s",
        "first_violation_time_s",
        "target_time_s",
    )
    runs: dict[str, Any] = {}
    all_voltage: list[float] = []
    all_temperature: list[float] = []
    all_soc: list[float] = []
    maximum_time = 0.0

    for name in RUN_ORDER:
        raw_run = raw_runs.get(name)
        if not isinstance(raw_run, Mapping):
            raise BatteryReplayError(f"missing run: {name}")
        raw_frames = raw_run.get("frames")
        if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, (str, bytes)):
            raise BatteryReplayError(f"{name}.frames must be an array")
        if len(raw_frames) < 2:
            raise BatteryReplayError(f"{name}.frames must contain at least two samples")

        frames: list[dict[str, float]] = []
        for index, raw_frame in enumerate(raw_frames):
            if not isinstance(raw_frame, Mapping):
                raise BatteryReplayError(f"{name}.frames[{index}] must be an object")
            frame = {
                field: _finite_number(
                    raw_frame.get(field), f"{name}.frames[{index}].{field}"
                )
                for field in frame_fields
            }
            if not -1e-9 <= frame["soc"] <= 1.0 + 1e-9:
                raise BatteryReplayError(f"{name}.frames[{index}].soc must lie in [0, 1]")
            if not -1e-7 <= frame["current_a"] <= scenario["current_limit_a"] + 1e-7:
                raise BatteryReplayError(
                    f"{name}.frames[{index}].current_a exceeds the action bounds"
                )
            frames.append(frame)

        times = [frame["time_s"] for frame in frames]
        if times[0] < 0.0:
            raise BatteryReplayError(f"{name} replay cannot start before time zero")
        if any(next_time <= time for time, next_time in zip(times, times[1:])):
            raise BatteryReplayError(f"{name} replay times must be strictly increasing")
        if times[-1] > scenario["duration_s"] + scenario["control_period_s"] + 1e-9:
            raise BatteryReplayError(f"{name} replay exceeds scenario.duration_s")

        raw_metrics = raw_run.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise BatteryReplayError(f"{name}.metrics must be an object")
        metrics = {
            field: _finite_number(raw_metrics.get(field), f"{name}.metrics.{field}")
            for field in metric_fields
        }
        if metrics["target_time_s"] < 0.0 or metrics["target_time_s"] > times[-1] + 1e-9:
            raise BatteryReplayError(f"{name}.metrics.target_time_s must lie on its run")
        if metrics["voltage_violation_duration_s"] < 0.0:
            raise BatteryReplayError(
                f"{name}.metrics.voltage_violation_duration_s cannot be negative"
            )

        raw_events = raw_run.get("events")
        if not isinstance(raw_events, Mapping):
            raise BatteryReplayError(f"{name}.events must be an object")
        events = {
            field: _optional_time(raw_events.get(field), f"{name}.events.{field}", times[-1])
            for field in event_fields
        }
        if events["target_time_s"] is None:
            raise BatteryReplayError(f"{name}.events.target_time_s is required")
        if not math.isclose(
            events["target_time_s"], metrics["target_time_s"], rel_tol=0.0, abs_tol=1e-6
        ):
            raise BatteryReplayError(f"{name} target event and metric disagree")

        violating_indices = [
            index
            for index, frame in enumerate(frames)
            if frame["terminal_voltage_v"] > scenario["voltage_limit_v"] + 1e-8
            or frame["cell_temperature_c"]
            > scenario["temperature_limit_c"] + 1e-8
        ]
        recorded_violation = bool(violating_indices)
        if recorded_violation != (events["first_violation_time_s"] is not None):
            raise BatteryReplayError(f"{name} violation event disagrees with recorded frames")
        if recorded_violation:
            first_index = violating_indices[0]
            lower_time = times[max(first_index - 1, 0)]
            upper_time = times[first_index]
            event_time = events["first_violation_time_s"]
            if event_time is None or not (
                lower_time - 1e-9 <= event_time <= upper_time + 1e-9
            ):
                raise BatteryReplayError(
                    f"{name} first violation time does not bracket the recorded crossing"
                )

        if not math.isclose(
            events["target_time_s"], times[-1], rel_tol=0.0, abs_tol=1e-6
        ) or not math.isclose(
            frames[-1]["soc"], scenario["target_soc"], rel_tol=0.0, abs_tol=1e-6
        ):
            raise BatteryReplayError(f"{name} target event must end at the target state")

        verdict = str(raw_run.get("verdict") or _DEFAULT_VERDICTS[name]).strip().lower()
        if verdict not in _VERDICT_LABELS:
            raise BatteryReplayError(f"{name}.verdict must be stands, caveat, or withdrawn")
        if verdict == "withdrawn" and not recorded_violation:
            raise BatteryReplayError(f"{name} is withdrawn without a recorded plant violation")
        if verdict == "stands" and recorded_violation:
            raise BatteryReplayError(f"{name} cannot stand after a recorded plant violation")

        event_indices = {
            field.removesuffix("_time_s"): _causal_index(frames, value)
            for field, value in events.items()
        }
        runs[name] = {
            "label": str(raw_run.get("label") or _DEFAULT_LABELS[name]),
            "verdict": verdict,
            "verdict_label": _VERDICT_LABELS[verdict],
            "frames": frames,
            "metrics": metrics,
            "events": events,
            "event_indices": event_indices,
        }
        maximum_time = max(maximum_time, times[-1])
        all_voltage.extend(frame["terminal_voltage_v"] for frame in frames)
        all_temperature.extend(frame["cell_temperature_c"] for frame in frames)
        all_soc.extend(frame["soc"] for frame in frames)

    replay = {
        "title": str(artifact.get("title") or "Fast charging when resistance drifts"),
        "description": str(
            artifact.get("description")
            or "Recorded current-governor trajectories from the matched battery audit."
        ),
        "fps": max(1.0, _finite_number(artifact.get("playback_fps", 25), "playback_fps")),
        "scenario": scenario,
        "maximum_time_s": maximum_time,
        "ranges": {
            "current_a": [0.0, scenario["current_limit_a"] * 1.05],
            "terminal_voltage_v": _axis_limits(
                all_voltage,
                [scenario["voltage_guard_v"], scenario["voltage_limit_v"]],
                0.2,
            ),
            "cell_temperature_c": _axis_limits(
                all_temperature,
                [scenario["temperature_guard_c"], scenario["temperature_limit_c"]],
                4.0,
            ),
            "soc": _axis_limits(
                all_soc,
                [scenario["initial_soc"], scenario["target_soc"]],
                0.2,
            ),
        },
        "runs": runs,
    }
    return replay


def _dom_id(prefix: str | None) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", prefix or "battery-replay").strip("-")
    if not stem:
        stem = "battery-replay"
    return f"{stem}-{uuid.uuid4().hex[:10]}"


def _safe_json(value: Any) -> str:
    return (
        json.dumps(value, separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_battery_replay(
    source: Path | str | Mapping[str, Any],
    *,
    replay_id: str | None = None,
    fallback_id: str = FALLBACK_ID,
) -> str:
    """Return an accessible player for immutable, Python-generated trajectories."""

    replay = _normalise(source)
    root = _dom_id(replay_id)
    title_id = f"{root}-title"
    description_id = f"{root}-description"
    data_id = f"{root}-data"
    arrow_id = f"{root}-current-arrow"

    run_options = "".join(
        f'<option value="{name}">{html.escape(replay["runs"][name]["label"])}</option>'
        for name in RUN_ORDER
    )
    metric_rows = "".join(
        f"""
        <tr data-run="{name}" data-verdict="{run['verdict']}">
          <th scope="row">{html.escape(run['label'])}</th>
          <td>{run['metrics']['target_time_s'] / 60.0:.1f} min</td>
          <td>{run['metrics']['max_voltage_v']:.3f} V</td>
          <td>{run['metrics']['voltage_violation_duration_s']:.0f} s</td>
          <td>{run['metrics']['max_cell_temperature_c']:.2f} °C</td>
          <td><span class="table-verdict">{html.escape(run['verdict_label'])}</span></td>
        </tr>
        """
        for name in RUN_ORDER
        for run in (replay["runs"][name],)
    )

    template = r'''
<section id="__ROOT__" class="battery-replay" tabindex="0" aria-labelledby="__TITLE_ID__" aria-describedby="__DESCRIPTION_ID__">
  <style>
    #__ROOT__ {
      --paper:#F6F7F4; --raised:#FFFFFF; --ink:#1B2430; --muted:#66727D;
      --rule:#CDD5D5; --teal:#2F6F8F; --stands:#2E7D5B; --caveat:#B8860B;
      --withdrawn:#A83A32; color:var(--ink); background:var(--paper); color-scheme:light;
      border:1px solid var(--rule); border-radius:9px; box-shadow:0 1px 2px rgba(20,30,40,.05);
      box-sizing:border-box; font:14px/1.38 "IBM Plex Sans",system-ui,sans-serif;
      margin-inline:auto; max-width:58rem; padding:clamp(.75rem,2vw,1.15rem); width:100%;
    }
    #__ROOT__[data-theme="dark"] {
      --paper:#121920; --raised:#1B2430; --ink:#EDF1F1; --muted:#A8B2B9;
      --rule:#34434D; --teal:#72A8C2; --stands:#69B18F; --caveat:#D5B452;
      --withdrawn:#DF786F; color-scheme:dark;
    }
    #__ROOT__ *, #__ROOT__ *::before, #__ROOT__ *::after { box-sizing:border-box; }
    #__ROOT__ [hidden] { display:none !important; }
    #__ROOT__:focus-visible { outline:3px solid color-mix(in srgb,var(--teal) 30%,transparent); outline-offset:3px; }
    #__ROOT__ h3 { font:500 clamp(1.2rem,3vw,1.45rem)/1.12 Newsreader,Georgia,serif; margin:0; }
    #__ROOT__ .lede { color:var(--muted); font-size:.9rem; margin:.3rem 0 .75rem; max-width:75ch; }
    #__ROOT__ .toolbar, #__ROOT__ .transport, #__ROOT__ .seeks { align-items:center; display:flex; flex-wrap:wrap; gap:.42rem; }
    #__ROOT__ .toolbar { border-block:1px solid var(--rule); justify-content:space-between; padding:.55rem 0; }
    #__ROOT__ .transport { flex:1 1 29rem; }
    #__ROOT__ button, #__ROOT__ select { appearance:none; background:var(--raised); border:1px solid #9EA9AC; border-radius:5px; color:var(--ink); font:inherit; font-size:.8rem; min-height:2rem; padding:.3rem .58rem; }
    #__ROOT__ button:hover:not(:disabled), #__ROOT__ button:focus-visible, #__ROOT__ select:focus-visible { border-color:var(--teal); outline:2px solid color-mix(in srgb,var(--teal) 22%,transparent); outline-offset:1px; }
    #__ROOT__ button:disabled { cursor:not-allowed; opacity:.42; }
    #__ROOT__ .scrubber { accent-color:var(--teal); flex:1 1 10rem; min-width:8rem; }
    #__ROOT__ [data-time], #__ROOT__ .number { font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums; }
    #__ROOT__ [data-time] { font-size:.8rem; min-width:5.7rem; }
    #__ROOT__ .run-control { align-items:center; display:flex; gap:.42rem; }
    #__ROOT__ .run-control > span { color:var(--muted); font-size:.76rem; }
    #__ROOT__ .badge { border-radius:999px; display:inline-flex; font-size:.73rem; font-weight:600; padding:.24rem .52rem; }
    #__ROOT__ [data-verdict="structure"] { background:color-mix(in srgb,var(--teal) 11%,var(--paper)); color:var(--teal); }
    #__ROOT__ [data-verdict="stands"] { background:color-mix(in srgb,var(--stands) 12%,var(--paper)); color:var(--stands); }
    #__ROOT__ [data-verdict="caveat"] { background:color-mix(in srgb,var(--caveat) 13%,var(--paper)); color:var(--caveat); }
    #__ROOT__ [data-verdict="withdrawn"] { background:color-mix(in srgb,var(--withdrawn) 11%,var(--paper)); color:var(--withdrawn); }
    #__ROOT__ .seeks { margin:.52rem 0 .25rem; }
    #__ROOT__ .seeks > span { color:var(--muted); font-size:.74rem; margin-right:.1rem; }
    #__ROOT__ .stage { align-items:start; display:grid; gap:1rem; grid-template-columns:minmax(14rem,.72fr) minmax(22rem,1.5fr); margin-top:.7rem; }
    #__ROOT__ .battery-panel { background:color-mix(in srgb,var(--raised) 58%,var(--paper)); border:1px solid var(--rule); border-radius:7px; min-width:0; padding:.55rem; }
    #__ROOT__ .battery-drawing { display:block; height:auto; margin:auto; max-width:21rem; overflow:visible; width:100%; }
    #__ROOT__ .battery-shell { fill:var(--raised); stroke:var(--ink); stroke-width:3; }
    #__ROOT__ .battery-terminal { fill:var(--ink); }
    #__ROOT__ .battery-fill { fill:var(--stands); opacity:.9; }
    #__ROOT__ .target-line { stroke:var(--stands); stroke-dasharray:5 4; stroke-width:1.5; }
    #__ROOT__ .thermal-halo { fill:none; stroke:var(--caveat); stroke-width:11; }
    #__ROOT__ .current-arrow { stroke:var(--teal); stroke-linecap:round; }
    #__ROOT__ .diagram-label { fill:var(--muted); font:11px "IBM Plex Sans",sans-serif; }
    #__ROOT__ .diagram-number { fill:var(--ink); font:600 13px "IBM Plex Mono",monospace; }
    #__ROOT__ .readouts { display:grid; gap:.26rem .55rem; grid-template-columns:repeat(2,minmax(0,1fr)); margin:.3rem 0 0; }
    #__ROOT__ .readouts div { border-top:1px solid var(--rule); min-width:0; padding-top:.26rem; }
    #__ROOT__ .readouts dt { color:var(--muted); font-size:.68rem; }
    #__ROOT__ .readouts dd { font:.78rem "IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums; margin:0; white-space:nowrap; }
    #__ROOT__ .active-verdict { margin-top:.48rem; }
    #__ROOT__ .chart-panel { min-width:0; overflow:hidden; }
    #__ROOT__ .chart-title { font-size:.78rem; font-weight:600; margin:0 0 .15rem; }
    #__ROOT__ .chart { display:block; height:auto; overflow:visible; width:100%; }
    #__ROOT__ .chart .grid { stroke:var(--rule); stroke-width:1; }
    #__ROOT__ .chart .bound { stroke:var(--withdrawn); stroke-dasharray:5 4; stroke-width:1.2; }
    #__ROOT__ .chart .guard { stroke:var(--caveat); stroke-dasharray:2 4; stroke-width:1; }
    #__ROOT__ .chart .target { stroke:var(--stands); stroke-dasharray:5 4; stroke-width:1.2; }
    #__ROOT__ .chart .trace { fill:none; stroke:var(--teal); stroke-linejoin:round; stroke-linecap:round; stroke-width:2.15; vector-effect:non-scaling-stroke; }
    #__ROOT__ .chart .trace-dot { fill:var(--teal); stroke:var(--paper); stroke-width:1.5; vector-effect:non-scaling-stroke; }
    #__ROOT__ .chart .cursor { stroke:var(--ink); stroke-width:1; opacity:.32; }
    #__ROOT__ .chart text { fill:var(--muted); font:11px "IBM Plex Sans",sans-serif; }
    #__ROOT__ .chart .numeric { font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums; }
    #__ROOT__ .chart .bound-label { fill:var(--withdrawn); font-size:10px; paint-order:stroke; stroke:var(--paper); stroke-width:4px; }
    #__ROOT__ .chart .guard-label { fill:var(--caveat); font-size:10px; paint-order:stroke; stroke:var(--paper); stroke-width:4px; }
    #__ROOT__ .chart .target-label { fill:var(--stands); font-size:10px; paint-order:stroke; stroke:var(--paper); stroke-width:4px; }
    #__ROOT__ .metrics-wrap { margin-top:.7rem; overflow-x:auto; }
    #__ROOT__ .mobile-chart-note { color:var(--muted); display:none; font-size:.7rem; margin:.1rem 0 .25rem; }
    #__ROOT__ table { border-collapse:collapse; font-size:.73rem; min-width:43rem; width:100%; }
    #__ROOT__ th, #__ROOT__ td { border-top:1px solid var(--rule); padding:.34rem .42rem; text-align:right; }
    #__ROOT__ th:first-child { text-align:left; }
    #__ROOT__ thead th { color:var(--muted); font-weight:500; }
    #__ROOT__ tbody td { font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums; }
    #__ROOT__ tbody tr[data-verdict="stands"] th, #__ROOT__ tbody tr[data-verdict="stands"] .table-verdict { color:var(--stands); }
    #__ROOT__ tbody tr[data-verdict="withdrawn"] th, #__ROOT__ tbody tr[data-verdict="withdrawn"] .table-verdict { color:var(--withdrawn); }
    #__ROOT__ .provenance { color:var(--muted); font-size:.72rem; margin:.5rem 0 0; }
    @media (max-width:900px) {
      #__ROOT__ .stage { grid-template-columns:1fr; }
      #__ROOT__ .battery-drawing { max-width:17rem; }
    }
    @media (max-width:700px) {
      #__ROOT__ .toolbar { align-items:flex-start; }
      #__ROOT__ .run-control { align-items:flex-start; flex-direction:column; width:100%; }
      #__ROOT__ select { max-width:100%; width:100%; }
      #__ROOT__ .chart-panel { overflow-x:auto; }
      #__ROOT__ .chart { min-width:44rem; }
      #__ROOT__ .chart-title { left:0; position:sticky; width:max-content; }
      #__ROOT__ .mobile-chart-note { display:block; left:0; position:sticky; width:max-content; }
    }
    @media (prefers-reduced-motion:reduce) {
      #__ROOT__ *, #__ROOT__ *::before, #__ROOT__ *::after { animation:none !important; scroll-behavior:auto !important; transition:none !important; }
    }
    @media print { #__ROOT__ { display:none !important; } }
  </style>
  <svg width="0" height="0" aria-hidden="true" style="position:absolute">
    <defs><marker id="__ARROW_ID__" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2F6F8F"></path></marker></defs>
  </svg>
  <header>
    <h3 id="__TITLE_ID__">__TITLE__</h3>
    <p class="lede" id="__DESCRIPTION_ID__">__DESCRIPTION__</p>
  </header>
  <div class="toolbar">
    <div class="transport">
      <button type="button" data-action="play" aria-keyshortcuts="Space">Play</button>
      <button type="button" data-action="step-back" aria-label="Step one recorded sample backward">− Step</button>
      <button type="button" data-action="step" aria-label="Step one recorded sample forward">+ Step</button>
      <button type="button" data-action="reset" aria-keyshortcuts="Home">Reset</button>
      <input class="scrubber" data-scrubber type="range" min="0" max="1" step="1" value="0" aria-label="Recorded battery replay time">
      <span role="timer" data-time>0.0 min</span>
    </div>
    <label class="run-control"><span>recorded run</span><select data-run aria-label="Recorded charging run">__RUN_OPTIONS__</select></label>
  </div>
  <div class="seeks" aria-label="Recorded event seeks">
    <span>jump to</span>
    <button type="button" data-seek="first_taper">first taper</button>
    <button type="button" data-seek="first_violation">first violation</button>
    <button type="button" data-seek="target">__TARGET_PERCENT__% target</button>
  </div>
  <div class="stage">
    <div class="battery-panel">
      <svg class="battery-drawing" viewBox="0 0 330 270" role="img" aria-label="Recorded battery charge, temperature, and current">
        <defs><clipPath id="__ROOT__-fill-clip"><rect x="105" y="38" width="120" height="182" rx="13"></rect></clipPath></defs>
        <rect class="thermal-halo" data-thermal x="95" y="28" width="140" height="202" rx="22" opacity="0.08"></rect>
        <rect class="battery-terminal" x="140" y="20" width="50" height="15" rx="3"></rect>
        <rect class="battery-shell" x="100" y="33" width="130" height="192" rx="17"></rect>
        <rect class="battery-fill" data-fill x="105" y="216" width="120" height="0" clip-path="url(#__ROOT__-fill-clip)"></rect>
        <line class="target-line" data-battery-target x1="105" y1="70" x2="225" y2="70"></line>
        <text class="diagram-label" data-battery-target-label x="237" y="74">__TARGET_PERCENT__% target</text>
        <g data-current-arrows>
          <line class="current-arrow" x1="32" y1="91" x2="87" y2="91" marker-end="url(#__ARROW_ID__)"></line>
          <line class="current-arrow" x1="32" y1="128" x2="87" y2="128" marker-end="url(#__ARROW_ID__)"></line>
          <line class="current-arrow" x1="32" y1="165" x2="87" y2="165" marker-end="url(#__ARROW_ID__)"></line>
        </g>
        <text class="diagram-label" x="58" y="185" text-anchor="middle">charge current</text>
        <text class="diagram-number" data-soc-label x="165" y="135" text-anchor="middle">__INITIAL_PERCENT__%</text>
      </svg>
      <dl class="readouts">
        <div><dt>state of charge</dt><dd data-value="soc">__INITIAL_PERCENT__.0%</dd></div>
        <div><dt>charge current</dt><dd data-value="current">0.00 A</dd></div>
        <div><dt>terminal voltage</dt><dd data-value="voltage">0.000 V</dd></div>
        <div><dt>cell temperature</dt><dd data-value="temperature">0.00 °C</dd></div>
      </dl>
      <div class="active-verdict"><span class="badge" data-run-verdict data-verdict="stands" aria-live="polite">within plant bounds</span></div>
    </div>
    <div class="chart-panel">
      <p class="chart-title">Recorded trajectories, revealed only to the playhead</p>
      <p class="mobile-chart-note">Scroll horizontally to inspect the full time axis.</p>
      <svg class="chart" viewBox="0 0 820 460" role="img" aria-label="Recorded current, voltage, temperature, and state-of-charge trajectory prefixes">
        <g data-panel="current_a">
          <line class="grid" x1="86" y1="24" x2="790" y2="24"></line><line class="grid" x1="86" y1="103" x2="790" y2="103"></line>
          <text x="12" y="61">current</text><text class="numeric" x="12" y="77">A</text>
          <text class="numeric" data-ymax="current_a" x="78" y="28" text-anchor="end"></text><text class="numeric" data-ymin="current_a" x="78" y="103" text-anchor="end"></text>
          <polyline class="trace" data-trace="current_a"></polyline><circle class="trace-dot" data-dot="current_a" r="3"></circle>
        </g>
        <g data-panel="terminal_voltage_v">
          <line class="grid" x1="86" y1="126" x2="790" y2="126"></line><line class="grid" x1="86" y1="205" x2="790" y2="205"></line>
          <line class="guard" data-line="voltage_guard" x1="86" x2="790"></line><line class="bound" data-line="voltage_limit" x1="86" x2="790"></line>
          <text x="12" y="164">voltage</text><text class="numeric" x="12" y="180">V</text>
          <text class="numeric" data-ymax="terminal_voltage_v" x="78" y="130" text-anchor="end"></text><text class="numeric" data-ymin="terminal_voltage_v" x="78" y="205" text-anchor="end"></text>
          <text class="guard-label" data-line-label="voltage_guard" x="786" text-anchor="end">__VOLTAGE_GUARD__ V guard</text><text class="bound-label" data-line-label="voltage_limit" x="786" text-anchor="end">__VOLTAGE_LIMIT__ V plant bound</text>
          <polyline class="trace" data-trace="terminal_voltage_v"></polyline><circle class="trace-dot" data-dot="terminal_voltage_v" r="3"></circle>
        </g>
        <g data-panel="cell_temperature_c">
          <line class="grid" x1="86" y1="228" x2="790" y2="228"></line><line class="grid" x1="86" y1="307" x2="790" y2="307"></line>
          <line class="guard" data-line="temperature_guard" x1="86" x2="790"></line><line class="bound" data-line="temperature_limit" x1="86" x2="790"></line>
          <text x="12" y="265">cell temp.</text><text class="numeric" x="12" y="281">°C</text>
          <text class="numeric" data-ymax="cell_temperature_c" x="78" y="232" text-anchor="end"></text><text class="numeric" data-ymin="cell_temperature_c" x="78" y="307" text-anchor="end"></text>
          <text class="guard-label" data-line-label="temperature_guard" x="786" text-anchor="end">__TEMPERATURE_GUARD__ °C guard</text><text class="bound-label" data-line-label="temperature_limit" x="786" text-anchor="end">__TEMPERATURE_LIMIT__ °C plant bound</text>
          <polyline class="trace" data-trace="cell_temperature_c"></polyline><circle class="trace-dot" data-dot="cell_temperature_c" r="3"></circle>
        </g>
        <g data-panel="soc">
          <line class="grid" x1="86" y1="330" x2="790" y2="330"></line><line class="grid" x1="86" y1="409" x2="790" y2="409"></line>
          <line class="target" data-line="soc_target" x1="86" x2="790"></line>
          <text x="12" y="367">charge</text><text class="numeric" x="12" y="383">%</text>
          <text class="numeric" data-ymax="soc" x="78" y="334" text-anchor="end"></text><text class="numeric" data-ymin="soc" x="78" y="409" text-anchor="end"></text>
          <text class="target-label" data-line-label="soc_target" x="786" text-anchor="end">__TARGET_PERCENT__% target</text>
          <polyline class="trace" data-trace="soc"></polyline><circle class="trace-dot" data-dot="soc" r="3"></circle>
        </g>
        <line class="cursor" data-cursor x1="86" x2="86" y1="24" y2="409"></line>
        <text class="numeric" x="86" y="438" text-anchor="middle">0</text><text x="438" y="454" text-anchor="middle">elapsed time (minutes)</text><text class="numeric" data-duration-label x="790" y="438" text-anchor="middle"></text>
      </svg>
    </div>
  </div>
  <div class="metrics-wrap">
    <table>
      <thead><tr><th scope="col">recorded run</th><th scope="col">time to __TARGET_PERCENT__%</th><th scope="col">peak voltage</th><th scope="col">above __VOLTAGE_LIMIT__ V</th><th scope="col">peak cell temp.</th><th scope="col">verdict</th></tr></thead>
      <tbody>__METRIC_ROWS__</tbody>
    </table>
  </div>
  <p class="provenance">This player only seeks immutable, Python-generated arrays. It does not solve the plant, fit a parameter, or recompute the current governor.</p>
  <script type="application/json" id="__DATA_ID__">__DATA__</script>
  <script>
  (() => {
    const root = document.getElementById("__ROOT__");
    const replay = JSON.parse(document.getElementById("__DATA_ID__").textContent);
    const names = ["fresh_nominal", "high_resistance_stale", "high_resistance_calibrated"];
    const select = root.querySelector("[data-run]");
    const scrubber = root.querySelector("[data-scrubber]");
    const timeOutput = root.querySelector("[data-time]");
    const playButton = root.querySelector('[data-action="play"]');
    const runVerdict = root.querySelector("[data-run-verdict]");
    const chart = {left:86, right:790};
    const panels = {
      current_a:{top:24,bottom:103}, terminal_voltage_v:{top:126,bottom:205},
      cell_temperature_c:{top:228,bottom:307}, soc:{top:330,bottom:409}
    };
    let frameIndex = 0;
    let playing = false;
    let lastTick = 0;
    const activeRun = () => replay.runs[select.value];
    const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
    const xScale = time => chart.left + time / replay.maximum_time_s * (chart.right-chart.left);
    const yScale = (field, value) => {
      const [low, high] = replay.ranges[field]; const panel = panels[field];
      return panel.bottom - (value-low)/(high-low)*(panel.bottom-panel.top);
    };
    const pointsPrefix = (run, index, field) => run.frames.slice(0, index + 1).map(frame => `${xScale(frame.time_s).toFixed(1)},${yScale(field,frame[field]).toFixed(1)}`).join(" ");
    const stop = () => { playing=false; playButton.textContent="Play"; };
    const eventStatus = (run, index) => {
      const events = run.event_indices;
      if (events.target !== null && index >= events.target) {
        return run.verdict === "withdrawn" ? ["target reached after a bound crossing","withdrawn"] : ["target reached within plant bounds","stands"];
      }
      if (events.first_violation !== null && index >= events.first_violation) return ["plant bound crossed","withdrawn"];
      if (events.first_taper !== null && index >= events.first_taper) return ["current governor is tapering","caveat"];
      return ["charging at the recorded current","structure"];
    };
    const configureLines = () => {
      const scenario = replay.scenario;
      const lineValues = {
        voltage_guard:["terminal_voltage_v",scenario.voltage_guard_v], voltage_limit:["terminal_voltage_v",scenario.voltage_limit_v],
        temperature_guard:["cell_temperature_c",scenario.temperature_guard_c], temperature_limit:["cell_temperature_c",scenario.temperature_limit_c],
        soc_target:["soc",scenario.target_soc]
      };
      Object.entries(lineValues).forEach(([name,[field,value]]) => {
        const y = yScale(field,value);
        const line = root.querySelector(`[data-line="${name}"]`); line.setAttribute("y1",String(y)); line.setAttribute("y2",String(y));
        const labelOffset=name.endsWith("guard") ? 13 : -4;
        root.querySelector(`[data-line-label="${name}"]`).setAttribute("y",String(y+labelOffset));
      });
      Object.entries(replay.ranges).forEach(([field,[low,high]]) => {
        const format = field === "soc" ? value => `${(100*value).toFixed(0)}` : value => value.toFixed(field === "terminal_voltage_v" ? 2 : 1);
        root.querySelector(`[data-ymax="${field}"]`).textContent=format(high);
        root.querySelector(`[data-ymin="${field}"]`).textContent=format(low);
      });
      const batteryTargetY=216-replay.scenario.target_soc*178;
      const batteryTarget=root.querySelector("[data-battery-target]"); batteryTarget.setAttribute("y1",String(batteryTargetY)); batteryTarget.setAttribute("y2",String(batteryTargetY));
      root.querySelector("[data-battery-target-label]").setAttribute("y",String(batteryTargetY+4));
      root.querySelector("[data-duration-label]").textContent=`${(replay.maximum_time_s/60).toFixed(0)}`;
    };
    const configureRun = () => {
      const run = activeRun();
      scrubber.max=String(run.frames.length-1); frameIndex=0;
      root.querySelectorAll("[data-seek]").forEach(button => {
        const index=run.event_indices[button.dataset.seek]; button.disabled=index === null;
        button.title=index === null ? "No such event in this recorded run" : `${(run.frames[index].time_s/60).toFixed(1)} min`;
      });
      runVerdict.textContent=run.verdict_label; runVerdict.dataset.verdict=run.verdict;
    };
    const render = () => {
      const run=activeRun(); const frame=run.frames[frameIndex];
      scrubber.value=String(frameIndex); timeOutput.textContent=`${(frame.time_s/60).toFixed(1)} min`;
      const fillHeight=clamp(frame.soc,0,1)*178;
      const fill=root.querySelector("[data-fill]"); fill.setAttribute("height",String(fillHeight)); fill.setAttribute("y",String(216-fillHeight));
      root.querySelector("[data-soc-label]").textContent=`${(100*frame.soc).toFixed(0)}%`;
      const thermalBase=replay.ranges.cell_temperature_c[0];
      const thermalFraction=clamp((frame.cell_temperature_c-thermalBase)/(replay.scenario.temperature_limit_c-thermalBase),0,1);
      root.querySelector("[data-thermal]").setAttribute("opacity",String(.06+.72*thermalFraction));
      const currentFraction=clamp(frame.current_a/replay.scenario.current_limit_a,0,1);
      root.querySelectorAll(".current-arrow").forEach((arrow,index) => { arrow.style.opacity=String(.16+.84*currentFraction); arrow.style.strokeWidth=String(1.1+2.8*currentFraction); arrow.toggleAttribute("hidden",currentFraction < .02 && index > 0); });
      root.querySelector('[data-value="soc"]').textContent=`${(100*frame.soc).toFixed(1)}%`;
      root.querySelector('[data-value="current"]').textContent=`${frame.current_a.toFixed(2)} A`;
      root.querySelector('[data-value="voltage"]').textContent=`${frame.terminal_voltage_v.toFixed(4)} V`;
      root.querySelector('[data-value="temperature"]').textContent=`${frame.cell_temperature_c.toFixed(2)} °C`;
      Object.keys(panels).forEach(field => {
        root.querySelector(`[data-trace="${field}"]`).setAttribute("points",pointsPrefix(run,frameIndex,field));
        const dot=root.querySelector(`[data-dot="${field}"]`); dot.setAttribute("cx",String(xScale(frame.time_s))); dot.setAttribute("cy",String(yScale(field,frame[field])));
      });
      const cursorX=xScale(frame.time_s); const cursor=root.querySelector("[data-cursor]"); cursor.setAttribute("x1",String(cursorX)); cursor.setAttribute("x2",String(cursorX));
      const [status,verdict]=eventStatus(run,frameIndex);
      if (runVerdict.textContent !== status) runVerdict.textContent=status;
      if (runVerdict.dataset.verdict !== verdict) runVerdict.dataset.verdict=verdict;
    };
    const tick = timestamp => {
      if (!playing) return;
      if (!lastTick || timestamp-lastTick >= 1000/replay.fps) {
        const run=activeRun();
        if (frameIndex >= run.frames.length-1) { stop(); return; }
        frameIndex += 1; lastTick=timestamp; render();
      }
      requestAnimationFrame(tick);
    };
    const togglePlay = () => {
      if (playing) { stop(); return; }
      const run=activeRun(); if (frameIndex >= run.frames.length-1) frameIndex=0;
      playing=true; lastTick=0; playButton.textContent="Pause"; requestAnimationFrame(tick);
    };
    playButton.addEventListener("click",togglePlay);
    root.querySelector('[data-action="step"]').addEventListener("click",() => { stop(); frameIndex=Math.min(frameIndex+1,activeRun().frames.length-1); render(); });
    root.querySelector('[data-action="step-back"]').addEventListener("click",() => { stop(); frameIndex=Math.max(frameIndex-1,0); render(); });
    root.querySelector('[data-action="reset"]').addEventListener("click",() => { stop(); frameIndex=0; render(); });
    scrubber.addEventListener("input",() => { stop(); frameIndex=Number(scrubber.value); render(); });
    select.addEventListener("change",() => { stop(); configureRun(); render(); });
    root.querySelectorAll("[data-seek]").forEach(button => button.addEventListener("click",() => { const index=activeRun().event_indices[button.dataset.seek]; if (index === null) return; stop(); frameIndex=index; render(); }));
    root.addEventListener("keydown",event => {
      if (event.target !== root) return;
      if (event.key === " ") { event.preventDefault(); togglePlay(); }
      else if (event.key === "ArrowRight") { event.preventDefault(); stop(); frameIndex=Math.min(frameIndex+1,activeRun().frames.length-1); render(); }
      else if (event.key === "ArrowLeft") { event.preventDefault(); stop(); frameIndex=Math.max(frameIndex-1,0); render(); }
      else if (event.key === "Home") { event.preventDefault(); stop(); frameIndex=0; render(); }
    });

    const applyTheme = () => {
      let dark = false;
      try {
        const themeRoot = window.parent && window.parent !== window
          ? window.parent.document.documentElement
          : document.documentElement;
        const declared = String(themeRoot.dataset.theme || themeRoot.getAttribute("data-mode") || "").toLowerCase();
        dark = declared === "dark" || themeRoot.classList.contains("dark") || getComputedStyle(themeRoot).colorScheme === "dark";
        if (!dark && window.matchMedia) dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      } catch (error) {
        dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      }
      root.dataset.theme = dark ? "dark" : "light";
    };
    applyTheme();
    try {
      const themeRoot = window.parent && window.parent !== window
        ? window.parent.document.documentElement
        : document.documentElement;
      if (typeof MutationObserver !== "undefined") {
        new MutationObserver(applyTheme).observe(themeRoot,{attributes:true,attributeFilter:["class","data-theme","data-mode","style"]});
      }
    } catch (error) { /* Cross-origin embedding falls back to the media query. */ }
    if (window.matchMedia) {
      const themeQuery = window.matchMedia("(prefers-color-scheme: dark)");
      if (typeof themeQuery.addEventListener === "function") themeQuery.addEventListener("change",applyTheme);
    }
    configureLines(); configureRun(); render();

    const fallbackId=__FALLBACK_JSON__;
    const hideFallback = doc => {
      if (!doc) return false; const fallback=doc.getElementById(fallbackId); if (!fallback) return false;
      fallback.hidden=true; fallback.setAttribute("aria-hidden","true"); return true;
    };
    hideFallback(document);
    try { if (window.parent && window.parent !== window) hideFallback(window.parent.document); } catch (_) {}
    const fallbackObserver=new MutationObserver(() => { if (hideFallback(document)) fallbackObserver.disconnect(); });
    fallbackObserver.observe(document.documentElement,{childList:true,subtree:true});
  })();
  </script>
</section>
'''
    return (
        template.replace("__ROOT__", root)
        .replace("__TITLE_ID__", title_id)
        .replace("__DESCRIPTION_ID__", description_id)
        .replace("__ARROW_ID__", arrow_id)
        .replace("__TITLE__", html.escape(replay["title"]))
        .replace("__DESCRIPTION__", html.escape(replay["description"]))
        .replace("__RUN_OPTIONS__", run_options)
        .replace("__METRIC_ROWS__", metric_rows)
        .replace("__DATA_ID__", data_id)
        .replace("__DATA__", _safe_json(replay))
        .replace("__FALLBACK_JSON__", _safe_json(str(fallback_id)))
        .replace("__VOLTAGE_GUARD__", f'{replay["scenario"]["voltage_guard_v"]:.2f}')
        .replace("__VOLTAGE_LIMIT__", f'{replay["scenario"]["voltage_limit_v"]:.2f}')
        .replace(
            "__TEMPERATURE_GUARD__",
            f'{replay["scenario"]["temperature_guard_c"]:.1f}',
        )
        .replace(
            "__TEMPERATURE_LIMIT__",
            f'{replay["scenario"]["temperature_limit_c"]:.0f}',
        )
        .replace("__INITIAL_PERCENT__", f'{100 * replay["scenario"]["initial_soc"]:.0f}')
        .replace("__TARGET_PERCENT__", f'{100 * replay["scenario"]["target_soc"]:.0f}')
    )


__all__ = [
    "BatteryReplayError",
    "FALLBACK_ID",
    "RUN_ORDER",
    "render_battery_replay",
]
