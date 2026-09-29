"""Recorded replay and static figures for the three-station BIXI example.

Python creates every trajectory and all numerical summaries. The small script
in :func:`render_bixi_replay` only selects a recorded controller and advances a
playhead; it contains no dynamics or controller logic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import secrets
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PAPER = "#F6F7F4"
INK = "#1B2430"
TEAL = "#2F6F8F"
STANDS = "#2E7D5B"
CAVEAT = "#B8860B"
WITHDRAWN = "#A83A32"
MUTED = "#66727D"
CONTROLLER_COLORS = {"none": MUTED, "open_loop": CAVEAT, "feedback": STANDS}
CONTROLLER_LABELS = {
    "none": "No relocation",
    "open_loop": "Frozen open loop",
    "feedback": "Inventory feedback",
}

_STYLE = {
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
}


class BixiReplayDataError(ValueError):
    """Raised when a committed BIXI artifact is stale or malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_artifact(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, (Path, str)):
        payload = dict(source)
    else:
        path = Path(source).resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise BixiReplayDataError(f"invalid BIXI artifact JSON: {error}") from error
        metadata = payload.get("metadata", {})
        inputs = metadata.get("input_files", {}) if isinstance(metadata, Mapping) else {}
        if not isinstance(inputs, Mapping):
            raise BixiReplayDataError("metadata.input_files must be a mapping")
        for relative, expected in inputs.items():
            if re.fullmatch(r"[0-9a-f]{64}", str(expected)) is None:
                raise BixiReplayDataError(f"invalid checksum for {relative}")
            relative_path = Path(str(relative))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise BixiReplayDataError("input paths must be repository-relative")
            candidates = [parent / relative_path for parent in path.parents]
            actual_path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if actual_path is None or _sha256(actual_path) != expected:
                raise BixiReplayDataError(
                    f"BIXI artifact is stale or its input is missing: {relative}"
                )
    if payload.get("schema_version") != 1:
        raise BixiReplayDataError("unsupported BIXI artifact schema")
    try:
        controllers = payload["showcase"]["controllers"]
        if set(controllers) != {"none", "open_loop", "feedback"}:
            raise KeyError("controllers")
    except (KeyError, TypeError) as error:
        raise BixiReplayDataError("BIXI artifact lacks the three showcase runs") from error
    return payload


def _step_value(times: np.ndarray, values: np.ndarray, at: float) -> np.ndarray:
    index = int(np.searchsorted(times, at, side="right") - 1)
    return values[max(index, 0)]


def _recorded_frames(artifact: Mapping[str, Any], controller: str) -> list[dict[str, Any]]:
    trajectory = artifact["showcase"]["controllers"][controller]
    stations = artifact["scenario"]["stations"]
    checkpoint_locations = np.asarray(trajectory["truck_station"], dtype=int)
    destinations = np.asarray(trajectory["destination"], dtype=int)
    history_times = np.asarray(trajectory["history_time_minutes"], dtype=float)
    history_inventory = np.asarray(trajectory["history_station_inventory"], dtype=float)
    event_times = np.asarray(trajectory["event_time_minutes"], dtype=float)
    accepted = np.asarray(trajectory["event_accepted"], dtype=bool)
    kinds = np.asarray(trajectory["event_kind"], dtype=str)
    frame_times = np.arange(0.0, 181.0, 1.0)
    frames: list[dict[str, Any]] = []
    for minute in frame_times:
        inventory = np.asarray(_step_value(history_times, history_inventory, minute))
        event_mask = event_times <= minute
        failures = int(np.sum(event_mask & ~accepted))
        lost = int(np.sum(event_mask & ~accepted & (kinds == "rental")))
        rejected = failures - lost
        decision = min(int(minute // 15), len(destinations) - 1)
        origin = int(checkpoint_locations[decision])
        destination = int(destinations[decision])
        phase = min((minute - 15.0 * decision) / 11.0, 1.0)
        origin_lat = float(stations[origin]["lat"])
        origin_lon = float(stations[origin]["lon"])
        destination_lat = float(stations[destination]["lat"])
        destination_lon = float(stations[destination]["lon"])
        frames.append(
            {
                "minute": int(minute),
                "inventory": inventory.astype(int).tolist(),
                "failures": failures,
                "lost_rentals": lost,
                "rejected_returns": rejected,
                "truck_lat": origin_lat + phase * (destination_lat - origin_lat),
                "truck_lon": origin_lon + phase * (destination_lon - origin_lon),
            }
        )
    return frames


def render_bixi_replay(
    source: Path | str | Mapping[str, Any],
    *,
    replay_id: str | None = None,
    fallback_id: str = "fig-bixi-replay-fallback",
) -> str:
    """Return an inline, dependency-free player for immutable Python results.

    ``fallback_id`` names a separate MyST figure rendered from the same
    artifact. It is hidden only after the player initializes successfully, so
    failed JavaScript and print/PDF builds retain the static evidence.
    """

    artifact = _load_artifact(source)
    identifier = replay_id or f"bixi-{secrets.token_hex(4)}"
    stations = artifact["scenario"]["stations"]
    capacities = [int(station["capacity"]) for station in stations]
    data = {
        "stations": stations,
        "capacities": capacities,
        "frames": {
            name: _recorded_frames(artifact, name)
            for name in ("none", "open_loop", "feedback")
        },
        "metrics": {
            name: artifact["showcase"]["controllers"][name]["metrics"]
            for name in ("none", "open_loop", "feedback")
        },
    }
    encoded = (
        json.dumps(data, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    options = "".join(
        f'<option value="{name}">{label}</option>'
        for name, label in CONTROLLER_LABELS.items()
    )
    return f"""
<div id="{identifier}" class="bixi-replay">
  <style>
    #{identifier} {{ --paper:{PAPER}; --ink:{INK}; --teal:{TEAL}; --good:{STANDS};
      --warn:{CAVEAT}; --bad:{WITHDRAWN}; color:var(--ink); background:var(--paper);
      border:1px solid #cbd2d3; border-radius:10px; padding:1rem;
      font:14px/1.35 'IBM Plex Sans',system-ui,sans-serif; max-width:920px; }}
    #{identifier} .head, #{identifier} .controls {{ display:flex; flex-wrap:wrap;
      align-items:center; justify-content:space-between; gap:.65rem; }}
    #{identifier} h4 {{ font:600 20px/1.1 Newsreader,Georgia,serif; margin:0; }}
    #{identifier} .badge {{ color:var(--teal); font:600 12px 'IBM Plex Mono',monospace; }}
    #{identifier} .stage {{ display:grid; grid-template-columns:minmax(320px,1.2fr) minmax(250px,.8fr);
      gap:1rem; margin:.8rem 0; }}
    #{identifier} svg {{ width:100%; height:auto; display:block; }}
    #{identifier} .side {{ display:grid; gap:.5rem; align-content:start; }}
    #{identifier} .card {{ border-top:2px solid var(--teal); padding:.4rem 0; }}
    #{identifier} .number {{ font:600 21px 'IBM Plex Mono',monospace; }}
    #{identifier} button, #{identifier} select {{ color:var(--ink); background:white;
      border:1px solid #9aa6aa; border-radius:5px; padding:.38rem .6rem; }}
    #{identifier} button:hover {{ border-color:var(--teal); }}
    #{identifier} input[type=range] {{ flex:1; min-width:180px; accent-color:var(--teal); }}
    #{identifier} .note {{ color:#66727D; font-size:12px; margin:.5rem 0 0; }}
    @media (prefers-reduced-motion: reduce) {{ #{identifier} * {{ scroll-behavior:auto !important; }} }}
    @media (max-width:680px) {{ #{identifier} .stage {{ grid-template-columns:1fr; }} }}
  </style>
  <div class="head"><h4>Recorded BIXI morning</h4><span class="badge">07:00–10:00 · 4 July 2024</span></div>
  <div class="stage">
    <svg viewBox="0 0 520 410" role="img" aria-label="Three BIXI station inventories and relocation truck">
      <path d="M453 56 L138 184 L359 324 Z" fill="none" stroke="#c7ced0" stroke-width="4"/>
      <g class="station-layer"></g>
      <g class="truck" transform="translate(0 0)">
        <rect x="-15" y="-10" width="30" height="18" rx="3" fill="{TEAL}"/>
        <circle cx="-9" cy="10" r="4" fill="{INK}"/><circle cx="10" cy="10" r="4" fill="{INK}"/>
      </g>
    </svg>
    <div class="side">
      <div class="card"><div>Local time</div><div class="number clock">07:00</div></div>
      <div class="card" aria-live="polite"><div>Service failures so far</div><div class="number failures">0</div>
        <div class="failure-detail">0 lost rentals · 0 rejected returns</div></div>
      <svg class="trace" viewBox="0 0 340 155" role="img" aria-label="Recorded station inventory through the current time"></svg>
    </div>
  </div>
  <div class="controls">
    <select aria-label="Recorded controller">{options}</select>
    <button type="button" class="play">Play</button><button type="button" class="step">+1 min</button>
    <input type="range" min="0" max="180" step="1" value="0" aria-label="Minutes after 07:00"/>
  </div>
  <p class="note">The browser only replays committed Python trajectories. Actions occur every 15 minutes; the truck icon interpolates one recorded interstation travel step.</p>
  <script type="application/json" class="data">{encoded}</script>
  <script>
  (() => {{
    const root=document.getElementById({json.dumps(identifier)}), d=JSON.parse(root.querySelector('.data').textContent);
    const svg=root.querySelector('svg'), layer=root.querySelector('.station-layer'), truck=root.querySelector('.truck');
    const select=root.querySelector('select'), slider=root.querySelector('input'), play=root.querySelector('.play');
    let timer=null; const lon=x=>55+(x+73.575)*70000, lat=y=>365-(y-45.503)*19000;
    d.stations.forEach((s,i)=>{{ const g=document.createElementNS('http://www.w3.org/2000/svg','g');
      g.innerHTML=`<circle cx="${{lon(s.lon)}}" cy="${{lat(s.lat)}}" r="37" fill="white" stroke="{TEAL}" stroke-width="3"/>
      <circle class="fill" cx="${{lon(s.lon)}}" cy="${{lat(s.lat)}}" r="29" fill="none" stroke="{STANDS}" stroke-width="9" transform="rotate(-90 ${{lon(s.lon)}} ${{lat(s.lat)}})"/>
      <text x="${{lon(s.lon)}}" y="${{lat(s.lat)+5}}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="17" font-weight="600" class="count"></text>
      <text x="${{lon(s.lon)}}" y="${{lat(s.lat)+57}}" text-anchor="middle" font-family="IBM Plex Sans,sans-serif" font-size="13">${{s.name.split(' / ')[0]}}</text>`;
      layer.appendChild(g); }});
    function render() {{ const name=select.value, t=+slider.value, f=d.frames[name][t];
      root.querySelector('.clock').textContent=String(7+Math.floor(t/60)).padStart(2,'0')+':'+String(t%60).padStart(2,'0');
      root.querySelector('.failures').textContent=f.failures; root.querySelector('.failure-detail').textContent=`${{f.lost_rentals}} lost rentals · ${{f.rejected_returns}} rejected returns`;
      [...layer.children].forEach((g,i)=>{{ const ratio=f.inventory[i]/d.capacities[i], c=2*Math.PI*29; g.querySelector('.count').textContent=f.inventory[i]+'/'+d.capacities[i]; const ring=g.querySelector('.fill'); ring.setAttribute('stroke-dasharray',`${{c*ratio}} ${{c}}`); ring.setAttribute('stroke',ratio<.15||ratio>.9?'{WITHDRAWN}':'{STANDS}'); }});
      truck.setAttribute('transform',`translate(${{lon(f.truck_lon)}} ${{lat(f.truck_lat)}})`);
      const chart=root.querySelector('.trace'), frames=d.frames[name], colors=['{TEAL}','{STANDS}','{CAVEAT}'];
      let body='<path d="M28 8V130H330" fill="none" stroke="#aab4b7"/><text x="2" y="14" font-size="10">bikes</text>';
      for(let i=0;i<3;i++){{ let pts=''; for(let k=0;k<=t;k++) pts+=`${{28+302*k/180}},${{130-112*frames[k].inventory[i]/d.capacities[i]}} `; body+=`<polyline points="${{pts}}" fill="none" stroke="${{colors[i]}}" stroke-width="2.5"/>`; }}
      body+=`<line x1="${{28+302*t/180}}" y1="8" x2="${{28+302*t/180}}" y2="130" stroke="{INK}" stroke-dasharray="3 3"/>`;
      body+='<text x="28" y="148" font-size="10">07:00</text><text x="300" y="148" font-size="10">10:00</text>'; chart.innerHTML=body;
    }}
    function stop(){{if(timer)clearInterval(timer);timer=null;play.textContent='Play';}}
    play.onclick=()=>{{if(timer){{stop();return;}} play.textContent='Pause';timer=setInterval(()=>{{if(+slider.value>=180){{stop();return;}}slider.value=+slider.value+1;render();}},85);}};
    root.querySelector('.step').onclick=()=>{{stop();slider.value=Math.min(180,+slider.value+1);render();}};
    slider.oninput=()=>{{stop();render();}}; select.onchange=()=>{{stop();render();}}; render();
    const fallbackId={json.dumps(fallback_id)};
    let fallbackDocument=document;
    try {{ if(window.parent && window.parent!==window && window.parent.document) fallbackDocument=window.parent.document; }}
    catch(error) {{ /* Cross-origin embedding keeps the local document. */ }}
    const hideFallback=()=>{{
      if(!fallbackId) return true;
      const fallback=fallbackDocument.getElementById(fallbackId);
      if(!fallback) return false;
      fallback.hidden=true;
      fallback.setAttribute('aria-hidden','true');
      return true;
    }};
    if(!hideFallback() && typeof MutationObserver!=="undefined") {{
      const fallbackObserver=new MutationObserver(()=>{{if(hideFallback()) fallbackObserver.disconnect();}});
      fallbackObserver.observe(fallbackDocument.documentElement,{{childList:true,subtree:true}});
    }}
  }})();
  </script>
</div>
"""


def _station_xy(stations: list[Mapping[str, Any]]) -> np.ndarray:
    lon = np.asarray([station["lon"] for station in stations], dtype=float)
    lat = np.asarray([station["lat"] for station in stations], dtype=float)
    x = (lon - np.mean(lon)) * 75.0 * np.cos(np.radians(np.mean(lat)))
    y = (lat - np.mean(lat)) * 111.0
    return np.column_stack([x, y])


def make_model_interface_figure(source: Path | str | Mapping[str, Any]) -> plt.Figure:
    """Map the selected stations and the actual showcase completed events."""

    artifact = _load_artifact(source)
    stations = artifact["scenario"]["stations"]
    counts = artifact["showcase"]["completed_counts"]
    xy = _station_xy(stations)
    with mpl.rc_context(_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(9.1, 3.4), gridspec_kw={"width_ratios": [1.05, 1]})
        map_axis, balance_axis = axes
        for i in range(3):
            for j in range(i + 1, 3):
                map_axis.plot(xy[[i, j], 0], xy[[i, j], 1], color="#c8d0d2", lw=2, zorder=0)
        for index, station in enumerate(stations):
            map_axis.scatter(*xy[index], s=950, facecolor="white", edgecolor=TEAL, lw=2.2, zorder=2)
            map_axis.text(*xy[index], f"{station['capacity']}\ndocks", ha="center", va="center", fontsize=8, weight="bold")
            offset = (0, 34) if index != 2 else (0, -43)
            map_axis.annotate(station["name"].split(" / ")[0], xy[index], xytext=offset, textcoords="offset points", ha="center", fontsize=8)
        map_axis.set_title("A deliberately small control boundary", loc="left", weight="bold")
        map_axis.text(0.0, -0.04, "One truck · 15-minute decisions · one travel step", transform=map_axis.transAxes, color=MUTED, fontsize=8)
        map_axis.set_aspect("equal")
        map_axis.axis("off")

        names = [station["name"].split(" / ")[0] for station in stations]
        rentals = np.asarray([counts[station["name"]]["rentals"] for station in stations])
        returns = np.asarray([counts[station["name"]]["returns"] for station in stations])
        y = np.arange(3)
        balance_axis.barh(y + 0.17, rentals, height=0.28, color=TEAL, label="completed rentals")
        balance_axis.barh(y - 0.17, returns, height=0.28, color=STANDS, label="completed returns")
        for row, (start_count, end_count) in enumerate(zip(rentals, returns)):
            balance_axis.text(start_count + 1.5, row + 0.17, str(start_count), va="center", fontsize=8)
            balance_axis.text(end_count + 1.5, row - 0.17, str(end_count), va="center", fontsize=8)
        balance_axis.set_yticks(y, names)
        balance_axis.invert_yaxis()
        balance_axis.set_xlabel("completed events, 07:00–10:00")
        balance_axis.set_title("One recorded morning is strongly unbalanced", loc="left", weight="bold")
        balance_axis.legend(loc="lower right", fontsize=7.5)
        balance_axis.grid(axis="x", color="#dce1df", lw=.7)
        balance_axis.set_axisbelow(True)
        figure.tight_layout(w_pad=2.0)
    return figure


def make_feedback_evidence_figure(source: Path | str | Mapping[str, Any]) -> plt.Figure:
    """Compare recorded inventory trajectories and paired Monte Carlo failures."""

    artifact = _load_artifact(source)
    stations = artifact["scenario"]["stations"]
    showcase = artifact["showcase"]["controllers"]
    summaries = artifact["monte_carlo"]["scenarios"]
    with mpl.rc_context(_STYLE):
        figure = plt.figure(figsize=(9.2, 5.25))
        grid = figure.add_gridspec(3, 2, width_ratios=[1.45, 1], hspace=.28)
        left_axes: list[plt.Axes] = []
        for station_index, station in enumerate(stations):
            axis = figure.add_subplot(grid[station_index, 0])
            left_axes.append(axis)
            for controller in ("open_loop", "feedback"):
                run = showcase[controller]
                axis.step(
                    run["history_time_minutes"],
                    np.asarray(run["history_station_inventory"])[:, station_index],
                    where="post",
                    color=CONTROLLER_COLORS[controller],
                    lw=1.8,
                    label=CONTROLLER_LABELS[controller],
                )
            axis.axhline(station["capacity"], color=WITHDRAWN, lw=.8, ls="--")
            axis.set_ylim(-1, station["capacity"] + 3)
            axis.set_ylabel(station["name"].split(" / ")[0], rotation=0, ha="right", va="center")
            axis.grid(axis="y", color="#dce1df", lw=.6)
            if station_index < 2:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xticks([0, 60, 120, 180], ["07:00", "08:00", "09:00", "10:00"])
                axis.set_xlabel("local time")

        axis = figure.add_subplot(grid[:, 1])
        labels: list[str] = []
        positions: list[float] = []
        cursor = 0.0
        group_centers: list[tuple[float, str]] = []
        for scenario_key, scenario_label in (("nominal", "rate model"), ("paired_pulse", "paired pulse")):
            group_start = cursor
            for controller in ("none", "open_loop", "feedback"):
                row = summaries[scenario_key]["controllers"][controller]["service_failures"]
                low, median, high = row["q10"], row["median"], row["q90"]
                axis.plot([low, high], [cursor, cursor], color=CONTROLLER_COLORS[controller], lw=3, solid_capstyle="round")
                axis.scatter([median], [cursor], s=32, color=CONTROLLER_COLORS[controller], edgecolor=PAPER, zorder=3)
                labels.append(CONTROLLER_LABELS[controller])
                positions.append(cursor)
                cursor += 1.0
            group_centers.append((group_start + 1.0, scenario_label))
            cursor += .6
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("service failures\n10th—median—90th percentile")
        axis.grid(axis="x", color="#dce1df", lw=.7)
        axis.set_axisbelow(True)
        for center, label in group_centers:
            axis.text(
                1.02,
                center,
                label,
                transform=axis.get_yaxis_transform(),
                color=MUTED,
                fontsize=7.5,
                va="center",
                ha="left",
                rotation=90,
            )

        figure.subplots_adjust(left=.13, right=.94, bottom=.13, top=.79, wspace=.58)
        figure.text(.13, .945, "Recorded morning", fontsize=10, weight="bold", ha="left")
        figure.text(
            .13,
            .908,
            "Feedback reroutes after observing station inventory",
            fontsize=8,
            color=MUTED,
            ha="left",
        )
        handles, labels_for_legend = left_axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels_for_legend,
            ncol=2,
            fontsize=7.5,
            loc="upper left",
            bbox_to_anchor=(.125, .89),
        )
        figure.text(.635, .945, "Paired stochastic evidence", fontsize=10, weight="bold", ha="left")
        figure.text(
            .635,
            .908,
            "512 common event traces per scenario",
            fontsize=8,
            color=MUTED,
            ha="left",
        )
    return figure


def make_censoring_figure(source: Path | str | Mapping[str, Any]) -> plt.Figure:
    """Show two latent demand histories with one identical completed-trip log."""

    artifact = _load_artifact(source)
    data = artifact["censoring_counterexample"]
    time = np.asarray(data["time"])
    with mpl.rc_context(_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(8.9, 3.25), sharey=True)
        scenarios = (
            ("demand_stops", "World A: requests stop", STANDS),
            ("demand_continues", "World B: requests continue", WITHDRAWN),
        )
        for axis, (key, title, color) in zip(axes, scenarios):
            demand = np.asarray(data[key])
            completed = np.asarray(data["completed"])
            axis.step(time, demand, where="mid", color=color, lw=2.4, label="latent requests")
            axis.step(time, completed, where="mid", color=TEAL, lw=2.2, ls="--", label="completed-trip log")
            axis.fill_between(time, completed, demand, step="mid", color=WITHDRAWN, alpha=.13)
            axis.set_title(title, loc="left", weight="bold")
            axis.set_xlabel("time bin")
            axis.set_xticks(time)
            axis.set_ylim(-.1, 2.5)
            axis.grid(axis="y", color="#dce1df", lw=.7)
        axes[0].set_ylabel("rental requests")
        axes[0].legend(loc="upper right", fontsize=7.5)
        axes[1].text(.5, .12, "unobserved after stockout", transform=axes[1].transAxes, color=WITHDRAWN, ha="center", fontsize=8)
        figure.suptitle("Completed trips do not identify demand after a station empties", x=.06, ha="left", weight="bold", fontsize=11)
        figure.tight_layout(rect=(0, 0, 1, .93))
    return figure


def save_figure_pair(figure: plt.Figure, stem: Path | str) -> None:
    destination = Path(stem)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")


__all__ = [
    "BixiReplayDataError",
    "make_censoring_figure",
    "make_feedback_evidence_figure",
    "make_model_interface_figure",
    "render_bixi_replay",
    "save_figure_pair",
]
