"""Browser-native replay for the recorded camera-gimbal experiment.

The renderer accepts only completed Python trajectories.  Its embedded script
updates SVG geometry, plots trajectory prefixes, and controls playback; it does
not contain plant, estimator, or controller equations.
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


RUN_ORDER = ("accelerometer", "gyro", "complementary")
FALLBACK_ID = "fig-gimbal-observation-fallback"


class GimbalReplayError(ValueError):
    """Raised when a replay artifact does not satisfy the renderer contract."""


def _load_source(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"gimbal replay artifact is missing: {path}") from None
    except json.JSONDecodeError as error:
        raise GimbalReplayError(f"invalid gimbal replay JSON: {error}") from error
    if not isinstance(value, dict):
        raise GimbalReplayError("gimbal replay root must be an object")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise GimbalReplayError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise GimbalReplayError(f"{field} must be numeric") from None
    if not math.isfinite(number):
        raise GimbalReplayError(f"{field} must be finite")
    return number


def _normalise(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    artifact = _load_source(source)
    if artifact.get("schema_version") != 1:
        raise GimbalReplayError("unsupported gimbal replay schema version")
    raw_runs = artifact.get("runs")
    if not isinstance(raw_runs, Mapping):
        raise GimbalReplayError("gimbal replay must contain a runs object")

    fields = (
        "time_s",
        "true_angle_deg",
        "estimated_angle_deg",
        "true_bias_deg_s",
        "estimated_bias_deg_s",
        "torque_nm",
        "base_acceleration_mps2",
        "tap_torque_nm",
        "accelerometer_angle_deg",
    )
    metrics_fields = (
        "rms_angle_deg",
        "peak_acceleration_window_deg",
        "final_abs_angle_deg",
        "normalized_torque_effort",
        "saturation_fraction",
        "regulation_score",
        "estimator_rmse_deg",
        "final_bias_true_deg_s",
        "final_bias_estimate_deg_s",
        "peak_torque_nm",
    )
    runs: dict[str, Any] = {}
    frame_count: int | None = None
    reference_times: list[float] | None = None
    for name in RUN_ORDER:
        raw_run = raw_runs.get(name)
        if not isinstance(raw_run, Mapping):
            raise GimbalReplayError(f"missing run: {name}")
        raw_frames = raw_run.get("frames")
        if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, (str, bytes)):
            raise GimbalReplayError(f"{name}.frames must be an array")
        if len(raw_frames) < 2:
            raise GimbalReplayError(f"{name}.frames must contain at least two samples")
        frames: list[dict[str, float]] = []
        for index, raw_frame in enumerate(raw_frames):
            if not isinstance(raw_frame, Mapping):
                raise GimbalReplayError(f"{name}.frames[{index}] must be an object")
            frames.append(
                {
                    field: _finite_number(
                        raw_frame.get(field), f"{name}.frames[{index}].{field}"
                    )
                    for field in fields
                }
            )
        times = [frame["time_s"] for frame in frames]
        if any(next_time <= time for time, next_time in zip(times, times[1:])):
            raise GimbalReplayError(f"{name} replay times must be strictly increasing")
        if frame_count is None:
            frame_count = len(frames)
            reference_times = times
        elif frame_count != len(frames) or any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(reference_times or [], times)
        ):
            raise GimbalReplayError("all estimators must share the same replay time grid")

        raw_metrics = raw_run.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise GimbalReplayError(f"{name}.metrics must be an object")
        metrics = {
            field: _finite_number(raw_metrics.get(field), f"{name}.metrics.{field}")
            for field in metrics_fields
        }
        style = raw_run.get("style", {})
        if not isinstance(style, Mapping):
            style = {}
        runs[name] = {
            "label": str(raw_run.get("label") or name),
            "color": str(style.get("color") or "#1B2430"),
            "dash": str(style.get("dash") or "-"),
            "frames": frames,
            "metrics": metrics,
        }

    scenario = artifact.get("scenario")
    parameters = artifact.get("parameters")
    if not isinstance(scenario, Mapping) or not isinstance(parameters, Mapping):
        raise GimbalReplayError("gimbal replay must include scenario and parameters")
    duration_s = _finite_number(scenario.get("duration_s"), "scenario.duration_s")
    if duration_s <= 0.0:
        raise GimbalReplayError("scenario.duration_s must be positive")
    if not reference_times or not math.isclose(
        reference_times[-1], duration_s, rel_tol=0.0, abs_tol=1e-9
    ):
        raise GimbalReplayError("replay must end at scenario.duration_s")
    replay = {
        "title": str(artifact.get("title") or "Partial observation in camera stabilization"),
        "description": str(
            artifact.get("description")
            or "Recorded state-estimation and feedback trajectories."
        ),
        "fps": max(1.0, _finite_number(artifact.get("playback_fps", 25), "playback_fps")),
        "duration_s": duration_s,
        "torque_limit_nm": _finite_number(
            parameters.get("torque_limit_nm"), "parameters.torque_limit_nm"
        ),
        "tap_start_s": _finite_number(scenario.get("tap_start_s"), "scenario.tap_start_s"),
        "tap_end_s": _finite_number(scenario.get("tap_end_s"), "scenario.tap_end_s"),
        "acceleration_start_s": _finite_number(
            scenario.get("acceleration_start_s"), "scenario.acceleration_start_s"
        ),
        "acceleration_end_s": _finite_number(
            scenario.get("acceleration_end_s"), "scenario.acceleration_end_s"
        ),
        "runs": runs,
    }
    return replay


def _dom_id(prefix: str | None) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", prefix or "gimbal-replay").strip("-")
    if not stem:
        stem = "gimbal-replay"
    return f"{stem}-{uuid.uuid4().hex[:10]}"


def _safe_json(value: Any) -> str:
    return (
        json.dumps(value, separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_gimbal_replay(
    source: Path | str | Mapping[str, Any],
    *,
    replay_id: str | None = None,
    fallback_id: str = FALLBACK_ID,
) -> str:
    """Return a self-contained accessible HTML replay of recorded trajectories."""

    replay = _normalise(source)
    root = _dom_id(replay_id)
    data_id = f"{root}-data"
    title_id = f"{root}-title"
    description_id = f"{root}-description"
    marker_id = f"{root}-arrow"
    escaped_fallback = json.dumps(str(fallback_id))
    data_json = _safe_json(replay)

    card_markup = "".join(
        f"""
        <article class="gimbal-card" data-run="{name}">
          <h4><span class="method-mark" aria-hidden="true"></span>{html.escape(replay['runs'][name]['label'])}</h4>
          <svg viewBox="0 0 300 190" role="img" aria-label="Recorded camera orientation for {html.escape(replay['runs'][name]['label'])}">
            <line class="world-horizon" x1="18" y1="78" x2="282" y2="78"></line>
            <text class="axis-note" x="20" y="70">world horizon</text>
            <rect class="base" x="112" y="150" width="76" height="14" rx="3"></rect>
            <line class="mount" x1="150" y1="150" x2="150" y2="78"></line>
            <circle class="pivot" cx="150" cy="78" r="7"></circle>
            <g data-camera>
              <rect class="camera-body" x="97" y="63" width="86" height="30" rx="5"></rect>
              <path class="camera-lens" d="M183 68 L210 72 L210 84 L183 88 Z"></path>
              <circle class="camera-detail" cx="116" cy="71" r="3"></circle>
            </g>
            <line class="estimate-ray" data-estimate x1="150" y1="78" x2="220" y2="78"></line>
            <g class="acceleration" data-acceleration hidden>
              <line x1="86" y1="137" x2="214" y2="137" marker-end="url(#{marker_id})"></line>
              <text x="150" y="129">base acceleration</text>
            </g>
            <line class="torque-gauge" data-torque x1="150" y1="176" x2="150" y2="176"></line>
            <text class="axis-note" x="20" y="181">motor torque</text>
          </svg>
          <dl class="readouts">
            <div><dt>true angle</dt><dd data-value="true">+0.0°</dd></div>
            <div><dt>estimate</dt><dd data-value="estimate">+0.0°</dd></div>
            <div><dt>bias estimate</dt><dd data-value="bias">+0.00°/s</dd></div>
            <div><dt>torque</dt><dd data-value="torque">+0.000 N·m</dd></div>
          </dl>
        </article>
        """
        for name in RUN_ORDER
    )
    metric_rows = "".join(
        f"""
        <tr data-run="{name}">
          <th scope="row">{html.escape(replay['runs'][name]['label'])}</th>
          <td>{replay['runs'][name]['metrics']['rms_angle_deg']:.2f}°</td>
          <td>{replay['runs'][name]['metrics']['peak_acceleration_window_deg']:.2f}°</td>
          <td>{replay['runs'][name]['metrics']['final_abs_angle_deg']:.2f}°</td>
          <td>{replay['runs'][name]['metrics']['regulation_score']:.3f}</td>
        </tr>
        """
        for name in RUN_ORDER
    )

    template = r"""
<section id="__ROOT__" class="gimbal-replay" aria-labelledby="__TITLE_ID__" aria-describedby="__DESCRIPTION_ID__">
  <style>
    #__ROOT__ {
      --paper: #F6F7F4; --raised: #FFFFFF; --ink: #1B2430; --muted: #5C6874;
      --rule: #D2D9D7; --teal: #2F6F8F; --stands: #2E7D5B; --caveat: #B8860B;
      color: var(--ink); background: var(--paper); border: 1px solid var(--rule);
      border-radius: 8px; padding: clamp(0.75rem, 2vw, 1.15rem);
      font-family: "IBM Plex Sans", system-ui, sans-serif; box-sizing: border-box;
      width: 100%; max-width: 52rem; margin-inline: auto;
      box-shadow: 0 1px 2px rgba(20,30,40,.05);
    }
    #__ROOT__ *, #__ROOT__ *::before, #__ROOT__ *::after { box-sizing: border-box; }
    #__ROOT__ [hidden] { display: none !important; }
    #__ROOT__ h3 { margin: 0; font-family: Newsreader, Georgia, serif; font-size: 1.32rem; font-weight: 500; line-height: 1.18; }
    #__ROOT__ .lede { margin: .3rem 0 .8rem; color: var(--muted); font-size: .9rem; max-width: 72ch; }
    #__ROOT__ .controls { display: flex; flex-wrap: wrap; align-items: center; gap: .45rem; padding: .55rem 0 .75rem; border-top: 1px solid var(--rule); }
    #__ROOT__ button, #__ROOT__ select { appearance: none; border: 1px solid var(--rule); border-radius: 5px; background: var(--raised); color: var(--ink); font: inherit; font-size: .82rem; min-height: 2rem; padding: .3rem .62rem; }
    #__ROOT__ button:hover, #__ROOT__ button:focus-visible, #__ROOT__ select:focus-visible { border-color: var(--teal); outline: 2px solid color-mix(in srgb, var(--teal) 22%, transparent); outline-offset: 1px; }
    #__ROOT__ .scrubber { flex: 1 1 14rem; accent-color: var(--teal); min-width: 9rem; }
    #__ROOT__ output { font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; min-width: 4.8rem; font-size: .82rem; }
    #__ROOT__ .event { margin-left: auto; border-radius: 999px; padding: .25rem .55rem; background: color-mix(in srgb, var(--teal) 10%, var(--paper)); color: var(--teal); font-size: .77rem; font-weight: 500; }
    #__ROOT__ .event[data-verdict="caveat"] { color: var(--caveat); background: color-mix(in srgb, var(--caveat) 12%, var(--paper)); }
    #__ROOT__ .event[data-verdict="stands"] { color: var(--stands); background: color-mix(in srgb, var(--stands) 12%, var(--paper)); }
    #__ROOT__ .mobile-method { display: none; }
    #__ROOT__ .cards { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: .55rem; }
    #__ROOT__ .gimbal-card { min-width: 0; background: color-mix(in srgb, var(--raised) 60%, var(--paper)); border: 1px solid var(--rule); border-radius: 6px; padding: .5rem; }
    #__ROOT__ .gimbal-card h4 { display: flex; gap: .4rem; align-items: center; margin: 0 0 .15rem; color: var(--ink); font-size: .79rem; font-weight: 600; }
    #__ROOT__ .method-mark { width: 1.2rem; height: 3px; background: var(--method-color); border-radius: 3px; }
    #__ROOT__ .gimbal-card[data-run="accelerometer"] { --method-color: var(--caveat); --method-dash: 7 5; }
    #__ROOT__ .gimbal-card[data-run="gyro"] { --method-color: var(--caveat); --method-dash: 2 4; }
    #__ROOT__ .gimbal-card[data-run="complementary"] { --method-color: var(--stands); --method-dash: none; }
    #__ROOT__ .gimbal-card svg { display: block; width: 100%; height: auto; overflow: visible; }
    #__ROOT__ .world-horizon { stroke: var(--teal); stroke-width: 1.25; opacity: .75; }
    #__ROOT__ .axis-note { fill: var(--muted); font: 10px "IBM Plex Sans", sans-serif; }
    #__ROOT__ .base { fill: #E8ECEB; stroke: var(--rule); }
    #__ROOT__ .mount { stroke: var(--muted); stroke-width: 5; }
    #__ROOT__ .pivot { fill: var(--paper); stroke: var(--ink); stroke-width: 2; }
    #__ROOT__ .camera-body { fill: var(--ink); }
    #__ROOT__ .camera-lens { fill: var(--teal); stroke: var(--ink); stroke-width: 1; }
    #__ROOT__ .camera-detail { fill: var(--paper); opacity: .7; }
    #__ROOT__ .estimate-ray { stroke: var(--method-color); stroke-width: 2.2; stroke-dasharray: var(--method-dash); }
    #__ROOT__ .acceleration line { stroke: var(--caveat); stroke-width: 2; }
    #__ROOT__ .acceleration text { fill: var(--caveat); font: 10px "IBM Plex Sans", sans-serif; text-anchor: middle; }
    #__ROOT__ .torque-gauge { stroke: var(--teal); stroke-width: 4; stroke-linecap: round; }
    #__ROOT__ .readouts { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: .15rem .45rem; margin: .1rem 0 0; }
    #__ROOT__ .readouts div { min-width: 0; }
    #__ROOT__ .readouts dt { color: var(--muted); font-size: .65rem; }
    #__ROOT__ .readouts dd { margin: 0; font: .72rem "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; white-space: nowrap; }
    #__ROOT__ .chart-wrap { margin-top: .65rem; border-top: 1px solid var(--rule); padding-top: .45rem; }
    #__ROOT__ .chart-title { margin: 0 0 .1rem; font-size: .78rem; font-weight: 600; }
    #__ROOT__ .chart { width: 100%; height: auto; display: block; }
    #__ROOT__ .chart .grid { stroke: var(--rule); stroke-width: 1; }
    #__ROOT__ .chart .zero { stroke: var(--teal); stroke-width: 1.2; }
    #__ROOT__ .chart .event-line { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 4; }
    #__ROOT__ .chart .acceleration-band { fill: var(--caveat); opacity: .11; }
    #__ROOT__ .chart .trace { fill: none; stroke-width: 2.25; vector-effect: non-scaling-stroke; }
    #__ROOT__ .chart text { fill: var(--muted); font: 11px "IBM Plex Sans", sans-serif; }
    #__ROOT__ .chart .numeric { font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }
    #__ROOT__ .chart .trace-label { font-weight: 500; paint-order: stroke; stroke: var(--paper); stroke-width: 4px; stroke-linejoin: round; }
    #__ROOT__ .metrics-wrap { overflow-x: auto; margin-top: .6rem; }
    #__ROOT__ table { border-collapse: collapse; width: 100%; min-width: 32rem; font-size: .75rem; }
    #__ROOT__ th, #__ROOT__ td { border-top: 1px solid var(--rule); padding: .35rem .45rem; text-align: right; }
    #__ROOT__ th:first-child { text-align: left; }
    #__ROOT__ thead th { color: var(--muted); font-weight: 500; }
    #__ROOT__ tbody td { font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }
    #__ROOT__ tbody tr[data-run="complementary"] th { color: var(--stands); }
    @media (max-width: 720px) {
      #__ROOT__ .mobile-method { display: flex; align-items: center; gap: .4rem; width: 100%; color: var(--muted); font-size: .76rem; }
      #__ROOT__ .cards { grid-template-columns: 1fr; }
      #__ROOT__ .gimbal-card { display: none; }
      #__ROOT__ .gimbal-card[data-active="true"] { display: block; }
      #__ROOT__ .event { margin-left: 0; }
    }
    @media (prefers-reduced-motion: reduce) {
      #__ROOT__ *, #__ROOT__ *::before, #__ROOT__ *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; }
    }
    @media (prefers-color-scheme: dark) {
      #__ROOT__ { --paper:#121920; --raised:#1B2430; --ink:#EDF1F1; --muted:#A8B2B9; --rule:#34434D; --teal:#72A8C2; --stands:#69B18F; --caveat:#D5B452; }
      #__ROOT__ .base { fill:#263139; }
    }
  </style>
  <svg width="0" height="0" aria-hidden="true" style="position:absolute">
    <defs><marker id="__MARKER__" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#B8860B"></path></marker></defs>
  </svg>
  <header>
    <h3 id="__TITLE_ID__">__TITLE__</h3>
    <p class="lede" id="__DESCRIPTION_ID__">__DESCRIPTION__</p>
  </header>
  <div class="controls">
    <button type="button" data-action="play">Play</button>
    <button type="button" data-action="step">Step</button>
    <button type="button" data-action="reset">Reset</button>
    <input class="scrubber" data-scrubber type="range" min="0" max="0" step="1" value="0" aria-label="Recorded gimbal replay time">
    <output data-time aria-live="polite">0.00 s</output>
    <span class="event" data-event data-verdict="structure">initial recovery</span>
    <label class="mobile-method">visible gimbal
      <select data-method aria-label="Visible estimator on a narrow screen">
        <option value="accelerometer">Accelerometer as state</option>
        <option value="gyro">Integrated gyroscope</option>
        <option value="complementary" selected>Complementary observer</option>
      </select>
    </label>
  </div>
  <div class="cards">__CARDS__</div>
  <div class="chart-wrap">
    <p class="chart-title">True camera angle, revealed to the current replay time</p>
    <svg class="chart" viewBox="0 0 820 270" role="img" aria-label="True camera angle through time for the three state estimators">
      <rect class="acceleration-band" data-band x="0" y="20" width="0" height="210"></rect>
      <line class="grid" x1="62" y1="20" x2="62" y2="230"></line>
      <line class="grid" x1="62" y1="230" x2="790" y2="230"></line>
      <line class="grid" x1="62" y1="62" x2="790" y2="62"></line>
      <line class="zero" x1="62" y1="104" x2="790" y2="104"></line>
      <line class="grid" x1="62" y1="146" x2="790" y2="146"></line>
      <line class="grid" x1="62" y1="188" x2="790" y2="188"></line>
      <line class="event-line" data-tap-line x1="0" y1="20" x2="0" y2="230"></line>
      <text class="numeric" x="54" y="66" text-anchor="end">+10</text>
      <text class="numeric" x="54" y="108" text-anchor="end">0</text>
      <text class="numeric" x="54" y="150" text-anchor="end">−10</text>
      <text class="numeric" x="54" y="192" text-anchor="end">−20</text>
      <text class="numeric" x="62" y="250" text-anchor="middle">0</text>
      <text class="numeric" data-duration-label x="790" y="250" text-anchor="middle">10 s</text>
      <text transform="translate(16 134) rotate(-90)" text-anchor="middle">true angle (degrees)</text>
      <polyline class="trace" data-trace="accelerometer" stroke="#B8860B" stroke-dasharray="8 5"></polyline>
      <polyline class="trace" data-trace="gyro" stroke="#B8860B" stroke-dasharray="2 4"></polyline>
      <polyline class="trace" data-trace="complementary" stroke="#2E7D5B"></polyline>
      <text class="trace-label" data-label="accelerometer" fill="#B8860B"></text>
      <text class="trace-label" data-label="gyro" fill="#B8860B"></text>
      <text class="trace-label" data-label="complementary" fill="#2E7D5B"></text>
    </svg>
  </div>
  <div class="metrics-wrap">
    <table>
      <thead><tr><th scope="col">state estimator</th><th scope="col">RMS angle</th><th scope="col">peak, 4.0–5.5 s</th><th scope="col">final error</th><th scope="col">regulation score</th></tr></thead>
      <tbody>__METRICS__</tbody>
    </table>
  </div>
  <script type="application/json" id="__DATA_ID__">__DATA__</script>
  <script>
  (() => {
    const root = document.getElementById("__ROOT__");
    const replay = JSON.parse(document.getElementById("__DATA_ID__").textContent);
    const names = ["accelerometer", "gyro", "complementary"];
    const frames = replay.runs.complementary.frames;
    const scrubber = root.querySelector("[data-scrubber]");
    const timeOutput = root.querySelector("[data-time]");
    const eventOutput = root.querySelector("[data-event]");
    const playButton = root.querySelector('[data-action="play"]');
    const methodSelect = root.querySelector("[data-method]");
    const chart = {left:62, right:790, top:20, bottom:230, yMin:-30, yMax:20};
    const xScale = value => chart.left + value / replay.duration_s * (chart.right - chart.left);
    const yScale = value => chart.bottom - (value-chart.yMin)/(chart.yMax-chart.yMin)*(chart.bottom-chart.top);
    let frameIndex = 0;
    let playing = false;
    let lastTick = 0;
    scrubber.max = String(frames.length - 1);
    root.querySelector("[data-duration-label]").textContent = `${replay.duration_s.toFixed(0)} s`;
    const band = root.querySelector("[data-band]");
    band.setAttribute("x", String(xScale(replay.acceleration_start_s)));
    band.setAttribute("width", String(xScale(replay.acceleration_end_s)-xScale(replay.acceleration_start_s)));
    const tapLine = root.querySelector("[data-tap-line]");
    tapLine.setAttribute("x1", String(xScale(replay.tap_start_s)));
    tapLine.setAttribute("x2", String(xScale(replay.tap_start_s)));

    const formatSigned = (value, digits) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
    const eventAt = time => {
      if (time >= replay.acceleration_start_s && time <= replay.acceleration_end_s) return ["accelerometer also sees translation", "caveat"];
      if (time >= replay.tap_start_s && time <= replay.tap_end_s) return ["mechanical tap", "structure"];
      if (time > replay.acceleration_end_s && time < 7.0) return ["translation ended; gyro bias remains", "caveat"];
      if (time >= 7.0) return ["history corrects accumulated bias", "stands"];
      return ["initial recovery", "structure"];
    };
    const pointsPrefix = (run, index) => run.frames.slice(0, index + 1).map(frame => `${xScale(frame.time_s).toFixed(1)},${yScale(frame.true_angle_deg).toFixed(1)}`).join(" ");

    const render = () => {
      const time = frames[frameIndex].time_s;
      scrubber.value = String(frameIndex);
      timeOutput.value = `${time.toFixed(2)} s`;
      const [eventText, verdict] = eventAt(time);
      eventOutput.textContent = eventText;
      eventOutput.dataset.verdict = verdict;
      names.forEach(name => {
        const run = replay.runs[name];
        const frame = run.frames[frameIndex];
        const card = root.querySelector(`.gimbal-card[data-run="${name}"]`);
        card.querySelector("[data-camera]").setAttribute("transform", `rotate(${-frame.true_angle_deg.toFixed(3)} 150 78)`);
        const estimateRadians = frame.estimated_angle_deg * Math.PI / 180;
        const estimate = card.querySelector("[data-estimate]");
        estimate.setAttribute("x2", String(150 + 70 * Math.cos(estimateRadians)));
        estimate.setAttribute("y2", String(78 - 70 * Math.sin(estimateRadians)));
        card.querySelector("[data-acceleration]").toggleAttribute(
          "hidden", Math.abs(frame.base_acceleration_mps2) < 0.1
        );
        const gauge = card.querySelector("[data-torque]");
        gauge.setAttribute("x2", String(150 + 58 * frame.torque_nm / replay.torque_limit_nm));
        card.querySelector('[data-value="true"]').textContent = `${formatSigned(frame.true_angle_deg,1)}°`;
        card.querySelector('[data-value="estimate"]').textContent = `${formatSigned(frame.estimated_angle_deg,1)}°`;
        card.querySelector('[data-value="bias"]').textContent = `${formatSigned(frame.estimated_bias_deg_s,2)}°/s`;
        card.querySelector('[data-value="torque"]').textContent = `${formatSigned(frame.torque_nm,3)} N·m`;
        root.querySelector(`[data-trace="${name}"]`).setAttribute("points", pointsPrefix(run, frameIndex));
        const label = root.querySelector(`[data-label="${name}"]`);
        const offsets = {accelerometer:-11, gyro:12, complementary:0};
        label.setAttribute("x", String(Math.min(xScale(time)+7, 700)));
        label.setAttribute("y", String(yScale(frame.true_angle_deg)+offsets[name]));
        label.textContent = time < 0.8
          ? (name === "complementary" ? "all three controllers" : "")
          : run.label;
      });
    };
    const updateVisibleCard = () => root.querySelectorAll(".gimbal-card").forEach(card => { card.dataset.active = String(card.dataset.run === methodSelect.value); });
    const stop = () => { playing=false; playButton.textContent="Play"; };
    const tick = timestamp => {
      if (!playing) return;
      if (!lastTick || timestamp-lastTick >= 1000/replay.fps) {
        if (frameIndex >= frames.length-1) { stop(); return; }
        frameIndex += 1; lastTick=timestamp; render();
      }
      requestAnimationFrame(tick);
    };
    playButton.addEventListener("click", () => {
      if (playing) { stop(); return; }
      if (frameIndex >= frames.length-1) frameIndex=0;
      playing=true; lastTick=0; playButton.textContent="Pause"; requestAnimationFrame(tick);
    });
    root.querySelector('[data-action="step"]').addEventListener("click", () => { stop(); frameIndex=Math.min(frameIndex+1,frames.length-1); render(); });
    root.querySelector('[data-action="reset"]').addEventListener("click", () => { stop(); frameIndex=0; render(); });
    scrubber.addEventListener("input", () => { stop(); frameIndex=Number(scrubber.value); render(); });
    methodSelect.addEventListener("change", updateVisibleCard);
    updateVisibleCard(); render();

    const fallbackId = __FALLBACK_JSON__;
    const hideFallback = doc => {
      if (!doc) return false;
      const fallback = doc.getElementById(fallbackId);
      if (!fallback) return false;
      fallback.hidden = true;
      fallback.setAttribute("aria-hidden", "true");
      return true;
    };
    hideFallback(document);
    try { if (window.parent && window.parent !== window) hideFallback(window.parent.document); } catch (_) {}
    const fallbackObserver = new MutationObserver(() => {
      if (hideFallback(document)) fallbackObserver.disconnect();
    });
    fallbackObserver.observe(document.documentElement, {childList:true, subtree:true});
  })();
  </script>
</section>
"""
    return (
        template.replace("__ROOT__", root)
        .replace("__TITLE_ID__", title_id)
        .replace("__DESCRIPTION_ID__", description_id)
        .replace("__MARKER__", marker_id)
        .replace("__TITLE__", html.escape(replay["title"]))
        .replace("__DESCRIPTION__", html.escape(replay["description"]))
        .replace("__CARDS__", card_markup)
        .replace("__METRICS__", metric_rows)
        .replace("__DATA_ID__", data_id)
        .replace("__DATA__", data_json)
        .replace("__FALLBACK_JSON__", escaped_fallback)
    )


__all__ = [
    "FALLBACK_ID",
    "GimbalReplayError",
    "render_gimbal_replay",
]
