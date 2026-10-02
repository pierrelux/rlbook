"""Recorded, browser-native replays for the inference-serving case study.

The browser is deliberately a player, not a simulator.  This module converts
Python trajectories into a compact sequence of immutable frames, embeds those
frames in the returned HTML fragment, and supplies only presentation controls
in JavaScript.  The same normalized trajectories feed the static SVG figure
used by print and JavaScript-free builds.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import html
import json
import math
from pathlib import Path
import re
import secrets
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


PAPER = "#F6F7F4"
INK = "#1B2430"
TEAL = "#2F6F8F"
STANDS = "#2E7D5B"
CAVEAT = "#B8860B"
WITHDRAWN = "#A83A32"
MUTED = "#66727D"

_ALIASES: dict[str, tuple[str, ...]] = {
    "time_s": ("time_s", "time", "t", "times"),
    "prefill_queue": (
        "prefill_queue",
        "queue_prefill",
        "prefill_jobs",
        "waiting_prefill",
    ),
    "decode_active": (
        "decode_active",
        "queue_decode",
        "decode_jobs",
        "active_decode",
    ),
    "completed_requests": (
        "completed_requests",
        "completed",
        "completions",
    ),
    "kv_tokens": ("kv_tokens", "kv_occupancy", "kv_used", "kv_cache_tokens"),
    "temperature_c": ("temperature_c", "temperature", "gpu_temperature_c"),
    "power_w": ("power_w", "power", "gpu_power_w"),
    "requested_clock_mhz": (
        "requested_clock_mhz",
        "requested_frequency_mhz",
        "frequency_mhz",
        "clock_mhz",
    ),
    "realized_clock_mhz": (
        "realized_clock_mhz",
        "actual_clock_mhz",
        "gpu_clock_mhz",
    ),
    "energy_j": ("energy_j", "cumulative_energy_j"),
}

_VIEW_TITLES = {
    "modeling": "Requests, work, and the GPU state",
    "open_loop": "Recorded open-loop frequency schedule",
    "mpc": "Recorded receding-horizon control",
    "scheduling": "Recorded prefill and decode scheduling",
    "fqi": "Recorded scheduling policy comparison",
}


def _profile_provenance(status: str, source_label: str = "") -> tuple[str, str]:
    normalized = status.strip().lower()
    if normalized == "measured_l4":
        return (source_label or "Measured NVIDIA L4 profile", "stands")
    if "proxy" in normalized or "surrogate" in normalized or "not_measured" in normalized:
        return ("Engineering surrogate, not L4 measurements", "caveat")
    if normalized and normalized != "unknown":
        return (source_label or normalized.replace("_", " "), "caveat")
    return (source_label or "Profile provenance not recorded", "caveat")

_FIGURE_STYLE = {
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "svg.fonttype": "none",
}


class ReplayDataError(ValueError):
    """Raised when a recorded replay artifact does not satisfy the contract."""


def _plain(value: Any) -> Any:
    """Convert dataclasses, NumPy objects, and light result wrappers to Python."""

    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    for method_name in ("to_dict", "as_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            return _plain(method())
    return value


def _load_source(source: Path | str | Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(source, (Path, str)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Inference replay artifact is missing: {path}")
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ReplayDataError(f"Invalid replay JSON in {path}: {error}") from error
    else:
        loaded = _plain(source)
    if not isinstance(loaded, Mapping):
        raise ReplayDataError("Replay input must be a mapping or a path to a JSON mapping.")
    return dict(loaded)


def _pick(mapping: Mapping[str, Any], canonical: str) -> Any | None:
    series = mapping.get("series")
    sources = (mapping, series) if isinstance(series, Mapping) else (mapping,)
    for source in sources:
        for name in _ALIASES[canonical]:
            if name in source:
                return source[name]
    return None


def _numeric_series(
    mapping: Mapping[str, Any], canonical: str, length: int, *, required: bool = False
) -> np.ndarray | None:
    value = _pick(mapping, canonical)
    if value is None:
        if required:
            accepted = ", ".join(_ALIASES[canonical])
            raise ReplayDataError(
                f"Replay trajectory is missing '{canonical}'. Accepted names: {accepted}."
            )
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ReplayDataError(f"'{canonical}' must be a one-dimensional numeric series.") from error
    if array.size != length:
        raise ReplayDataError(
            f"'{canonical}' has {array.size} values but time_s has {length}."
        )
    if not np.all(np.isfinite(array)):
        raise ReplayDataError(f"'{canonical}' contains a non-finite value.")
    return array


def _request_records(mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = mapping.get("requests", [])
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            continue
        arrival = raw.get("arrival_time_s", raw.get("arrival_s", raw.get("arrival")))
        if arrival is None:
            continue
        try:
            item = {
                "request_id": str(raw.get("request_id", raw.get("id", index))),
                "arrival_time_s": float(arrival),
                "prefill_start_s": float(
                    raw.get("prefill_start_s", raw.get("service_start_s", arrival))
                ),
                "first_token_time_s": float(
                    raw.get(
                        "first_token_time_s",
                        raw.get("ttft_time_s", raw.get("prefill_end_s", arrival)),
                    )
                ),
                "completion_time_s": float(
                    raw.get(
                        "completion_time_s",
                        raw.get("completed_time_s", raw.get("finish_time_s", np.inf)),
                    )
                ),
                "prompt_tokens": int(raw.get("prompt_tokens", 0)),
                "output_tokens": int(raw.get("output_tokens", 0)),
            }
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(item["arrival_time_s"]):
            continue
        normalized.append(item)
    return normalized


def _request_snapshot(records: Sequence[Mapping[str, Any]], time_s: float) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for request in records:
        arrival = float(request["arrival_time_s"])
        if time_s < arrival:
            continue
        prefill_start = max(arrival, float(request["prefill_start_s"]))
        first_token = max(prefill_start, float(request["first_token_time_s"]))
        completion = max(first_token, float(request["completion_time_s"]))
        if time_s < prefill_start:
            phase, start, end = "queue", arrival, prefill_start
        elif time_s < first_token:
            phase, start, end = "prefill", prefill_start, first_token
        elif time_s < completion:
            phase, start, end = "decode", first_token, completion
        elif time_s <= completion + 1.0:
            phase, start, end = "complete", completion, completion + 1.0
        else:
            continue
        duration = max(end - start, 1e-9)
        visible.append(
            {
                "id": str(request["request_id"]),
                "phase": phase,
                "progress": round(float(np.clip((time_s - start) / duration, 0.0, 1.0)), 4),
                "prompt_tokens": int(request["prompt_tokens"]),
                "output_tokens": int(request["output_tokens"]),
            }
        )
    # A dense trace is represented by counts; individual markers are kept legible.
    return visible[:28]


def _downsample_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=int)
    return np.unique(np.rint(np.linspace(0, length - 1, maximum)).astype(int))


def _controller_mapping(view_data: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    for key in ("controllers", "trajectories", "results"):
        possible = view_data.get(key)
        if isinstance(possible, Mapping) and possible:
            controllers = {str(name): value for name, value in possible.items()}
            default = str(view_data.get("default_controller", next(iter(controllers))))
            if default not in controllers:
                default = next(iter(controllers))
            return controllers, default

    if _pick(view_data, "time_s") is not None:
        name = str(
            view_data.get(
                "controller_name",
                view_data.get("controller", view_data.get("name", "recorded run")),
            )
        )
        return {name: view_data}, name

    candidates = {
        str(name): value
        for name, value in view_data.items()
        if isinstance(value, Mapping) and _pick(value, "time_s") is not None
    }
    if candidates:
        return candidates, next(iter(candidates))
    raise ReplayDataError(
        "A replay view must contain a trajectory with time_s, or a non-empty "
        "'controllers' mapping of such trajectories."
    )


def _policy_matrix(value: Any, *, coverage: bool) -> list[list[float | int]]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ReplayDataError("A policy or coverage slice must be a numeric 7 by 7 array.") from error
    if array.shape != (7, 7) or not np.all(np.isfinite(array)):
        raise ReplayDataError("A policy or coverage slice must be a finite 7 by 7 array.")
    if coverage:
        if np.any(array < 0.0):
            raise ReplayDataError("Coverage counts must be nonnegative.")
        return [[round(float(item), 4) for item in row] for row in array]
    rounded = np.rint(array).astype(int)
    if not np.allclose(array, rounded) or np.any((rounded < 0) | (rounded > 2)):
        raise ReplayDataError("Policy actions must be integers 0 (prefill), 1 (decode), or 2 (idle).")
    return rounded.tolist()


def _age_slices(value: Any, *, coverage: bool) -> dict[str, list[list[float | int]]]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, list[list[float | int]]] = {}
    for age, matrix in value.items():
        try:
            age_value = int(age)
        except (TypeError, ValueError):
            continue
        if 0 <= age_value <= 4:
            normalized[str(age_value)] = _policy_matrix(matrix, coverage=coverage)
    return dict(sorted(normalized.items(), key=lambda item: int(item[0])))


def _normalize_policy_data(
    view_data: Mapping[str, Any], normalized_view: str
) -> dict[str, Any]:
    actions: dict[str, dict[str, list[list[float | int]]]] = {}
    coverage: dict[str, dict[str, list[list[float | int]]]] = {}
    raw_actions = view_data.get("policy_slices")
    if normalized_view == "scheduling":
        exact = _age_slices(raw_actions, coverage=False)
        if not exact and "policy_slice_age_4" in view_data:
            exact = {"4": _policy_matrix(view_data["policy_slice_age_4"], coverage=False)}
        if exact:
            actions["exact_dp"] = exact
    elif normalized_view == "fqi" and isinstance(raw_actions, Mapping):
        for controller, slices in raw_actions.items():
            normalized = _age_slices(slices, coverage=False)
            if normalized:
                actions[str(controller)] = normalized
        raw_coverage = view_data.get("coverage_slices", {})
        if isinstance(raw_coverage, Mapping):
            for controller, slices in raw_coverage.items():
                normalized = _age_slices(slices, coverage=True)
                if normalized:
                    coverage[str(controller)] = normalized
    metrics = view_data.get("metrics", {})
    coverage_fraction: dict[str, float] = {}
    if isinstance(metrics, Mapping):
        for controller, values in metrics.items():
            if isinstance(values, Mapping) and "coverage_fraction" in values:
                fraction = float(values["coverage_fraction"])
                if np.isfinite(fraction):
                    coverage_fraction[str(controller)] = fraction
    return {
        "actions": actions,
        "coverage": coverage,
        "coverage_fraction": coverage_fraction,
        "action_labels": {"0": "prefill", "1": "decode", "2": "idle"},
    }


def _normalize_controller(
    value: Any, *, maximum_frames: int, include_request_snapshots: bool
) -> dict[str, Any]:
    mapping = _plain(value)
    if not isinstance(mapping, Mapping):
        raise ReplayDataError("Each controller trajectory must be a mapping.")
    if isinstance(mapping.get("trajectory"), Mapping):
        merged = dict(mapping["trajectory"])
        for key in ("requests", "metrics", "plan_dt_s", "limits"):
            if key in mapping and key not in merged:
                merged[key] = mapping[key]
        mapping = merged

    raw_time = _pick(mapping, "time_s")
    if raw_time is None:
        accepted = ", ".join(_ALIASES["time_s"])
        raise ReplayDataError(
            f"Replay trajectory is missing 'time_s'. Accepted names: {accepted}."
        )
    try:
        time_s = np.asarray(raw_time, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ReplayDataError("'time_s' must be a one-dimensional numeric series.") from error
    if time_s.size == 0:
        raise ReplayDataError("'time_s' must contain at least one recorded frame.")
    if not np.all(np.isfinite(time_s)):
        raise ReplayDataError("'time_s' contains a non-finite value.")
    if time_s.size > 1 and not np.all(np.diff(time_s) > 0.0):
        raise ReplayDataError("'time_s' must be strictly increasing.")

    arrays = {
        name: _numeric_series(mapping, name, time_s.size)
        for name in _ALIASES
        if name != "time_s"
    }
    raw_phase = mapping.get("phase")
    if raw_phase is None:
        phases = ["idle"] * time_s.size
    elif not isinstance(raw_phase, Sequence) or isinstance(raw_phase, (str, bytes)):
        raise ReplayDataError("'phase' must be one recorded string per time point.")
    elif len(raw_phase) != time_s.size:
        raise ReplayDataError(
            f"'phase' has {len(raw_phase)} values but time_s has {time_s.size}."
        )
    else:
        phases = [str(item) for item in raw_phase]
    if arrays["requested_clock_mhz"] is None and arrays["realized_clock_mhz"] is not None:
        arrays["requested_clock_mhz"] = arrays["realized_clock_mhz"].copy()
    if arrays["realized_clock_mhz"] is None and arrays["requested_clock_mhz"] is not None:
        arrays["realized_clock_mhz"] = arrays["requested_clock_mhz"].copy()

    plan = mapping.get("planned_clock_mhz", mapping.get("clock_plan_mhz"))
    if plan is not None:
        if not isinstance(plan, Sequence) or len(plan) != time_s.size:
            raise ReplayDataError(
                "'planned_clock_mhz' must contain one recorded plan per time point."
            )
        normalized_plan: list[list[float]] = []
        for row in plan:
            try:
                values = np.asarray(row, dtype=float).reshape(-1)
            except (TypeError, ValueError) as error:
                raise ReplayDataError("Every recorded clock plan must be numeric.") from error
            if not np.all(np.isfinite(values)):
                raise ReplayDataError("A recorded clock plan contains a non-finite value.")
            normalized_plan.append([round(float(item), 3) for item in values])
    else:
        normalized_plan = [[] for _ in range(time_s.size)]

    requests = _request_records(mapping)
    indices = _downsample_indices(time_s.size, maximum_frames)
    frames: list[dict[str, Any]] = []
    for source_index in indices:
        frame: dict[str, Any] = {"time_s": round(float(time_s[source_index]), 4)}
        frame["phase"] = phases[source_index]
        for name, array in arrays.items():
            if array is not None:
                frame[name] = round(float(array[source_index]), 4)
        frame["backlog"] = round(
            float(
                (arrays["prefill_queue"][source_index] if arrays["prefill_queue"] is not None else 0.0)
                + (arrays["decode_active"][source_index] if arrays["decode_active"] is not None else 0.0)
            ),
            4,
        )
        frame["requests"] = (
            _request_snapshot(requests, float(time_s[source_index]))
            if include_request_snapshots
            else []
        )
        frame["planned_clock_mhz"] = normalized_plan[source_index]
        frames.append(frame)

    metrics = mapping.get("metrics", {})
    return {
        "frames": frames,
        "metrics": _plain(metrics) if isinstance(metrics, Mapping) else {},
        "plan_dt_s": float(
            mapping.get(
                "plan_dt_s",
                np.median(np.diff(time_s)) if time_s.size > 1 else 1.0,
            )
        ),
        "limits": _plain(mapping.get("limits", {})),
    }


def _normalise_replay(
    source: Path | str | Mapping[str, Any] | Any,
    *,
    view: str,
    maximum_frames: int = 600,
) -> dict[str, Any]:
    if maximum_frames < 2:
        raise ValueError("maximum_frames must be at least 2.")
    normalized_view = view.replace("-", "_")
    root = _load_source(source)
    root_metadata = root.get("metadata", {})
    if not isinstance(root_metadata, Mapping):
        root_metadata = {}
    if normalized_view in root:
        view_data = root[normalized_view]
    elif _pick(root, "time_s") is not None or any(
        key in root for key in ("controllers", "trajectories", "results")
    ):
        view_data = root
    else:
        available = ", ".join(sorted(str(key) for key in root)) or "none"
        raise ReplayDataError(
            f"Replay view '{normalized_view}' is absent. Available top-level keys: {available}."
        )
    view_data = _plain(view_data)
    if not isinstance(view_data, Mapping):
        raise ReplayDataError(f"Replay view '{normalized_view}' must be a mapping.")
    controllers, default = _controller_mapping(view_data)
    policy = _normalize_policy_data(view_data, normalized_view)
    if "exact_dp" in controllers and "exact_dp" in policy["actions"]:
        default = "exact_dp"
    first_controller = next(iter(controllers.values()))
    first_controller = _plain(first_controller)
    controller_status = (
        first_controller.get("profile_status", "unknown")
        if isinstance(first_controller, Mapping)
        else "unknown"
    )
    profile_status = str(
        root_metadata.get(
            "profile_status",
            view_data.get("profile_status", controller_status),
        )
    )
    provenance_text, provenance_verdict = _profile_provenance(
        profile_status,
        str(root_metadata.get("profile_source", root_metadata.get("source_label", ""))),
    )
    normalized = {
        name: _normalize_controller(
            value,
            maximum_frames=maximum_frames,
            include_request_snapshots=normalized_view == "modeling",
        )
        for name, value in controllers.items()
    }
    return {
        "view": normalized_view,
        "title": str(view_data.get("title", _VIEW_TITLES.get(normalized_view, "Recorded inference serving"))),
        "default_controller": default,
        "controllers": normalized,
        "profile_status": profile_status,
        "provenance_text": provenance_text,
        "provenance_verdict": provenance_verdict,
        "policy": policy,
    }


def _safe_json(value: Any) -> str:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ReplayDataError(f"Replay data is not JSON serializable: {error}") from error
    return (
        serialized.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _dom_id(prefix: str | None) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", prefix or "inference-replay").strip("-")
    if not stem:
        stem = "inference-replay"
    if stem[0].isdigit():
        stem = f"replay-{stem}"
    return f"{stem}-{secrets.token_hex(4)}"


def render_serving_replay(
    source: Path | str | Mapping[str, Any] | Any,
    *,
    view: str = "modeling",
    replay_id: str | None = None,
    maximum_frames: int = 600,
) -> str:
    """Return an accessible HTML/SVG player for recorded Python trajectories.

    Parameters
    ----------
    source:
        A mapping, a result object exposing ``to_dict``/``as_dict``, or the path
        to ``textbook_results.json``.
    view:
        One of ``modeling``, ``open_loop``, ``mpc``, ``scheduling``, or ``fqi``.
        A custom name also works when the source is already one view.
    replay_id:
        Optional readable prefix.  A random suffix is always added so several
        instances can safely appear on the same page.
    maximum_frames:
        Upper bound on embedded frames.  Long recordings are evenly sampled by
        Python before they enter the page.
    """

    data = _normalise_replay(source, view=view, maximum_frames=maximum_frames)
    root_id = _dom_id(replay_id)
    data_id = f"{root_id}-data"
    title_id = f"{root_id}-title"
    description_id = f"{root_id}-description"
    controller_id = f"{root_id}-controller"
    scrubber_id = f"{root_id}-scrubber"
    options = "".join(
        f'<option value="{html.escape(name, quote=True)}"'
        + (" selected" if name == data["default_controller"] else "")
        + f">{html.escape(name.replace('_', ' '))}</option>"
        for name in data["controllers"]
    )
    disabled = " disabled" if len(data["controllers"]) == 1 else ""
    serialized = _safe_json(data)
    title = html.escape(data["title"])
    provenance = html.escape(data["provenance_text"])
    provenance_verdict = html.escape(data["provenance_verdict"], quote=True)

    fragment = r'''
<section id="__ROOT__" class="inference-replay" aria-labelledby="__TITLE__" aria-describedby="__DESCRIPTION__">
  <style>
    #__ROOT__ {
      --paper: #F6F7F4; --ink: #1B2430; --teal: #2F6F8F;
      --stands: #2E7D5B; --caveat: #B8860B; --withdrawn: #A83A32;
      --muted: #66727D; --line: #CBD2D6; --soft: #E8ECEB;
      color: var(--ink); background: var(--paper); font-family: "IBM Plex Sans", Inter, system-ui, sans-serif;
      max-width: 980px; margin: 0 auto; padding: 1rem; box-sizing: border-box;
      color-scheme: light;
    }
    #__ROOT__[data-theme="dark"] {
      --paper: #151C24; --ink: #EEF2F3; --teal: #70AFC8;
      --stands: #68B894; --caveat: #D6AD43; --withdrawn: #D87168;
      --muted: #A9B3BA; --line: #46515A; --soft: #202A33;
      color-scheme: dark;
    }
    #__ROOT__ *, #__ROOT__ *::before, #__ROOT__ *::after { box-sizing: border-box; }
    #__ROOT__ [hidden] { display: none !important; }
    #__ROOT__ h3 { margin: 0; font-family: Newsreader, Georgia, serif; font-weight: 500; font-size: 1.35rem; line-height: 1.2; }
    #__ROOT__ .description { margin: .3rem 0 .85rem; color: var(--muted); font-size: .91rem; }
    #__ROOT__ .provenance { margin: -.5rem 0 .85rem; font-size: .82rem; font-weight: 500; }
    #__ROOT__ .provenance[data-verdict="stands"] { color: var(--stands); }
    #__ROOT__ .provenance[data-verdict="caveat"] { color: var(--caveat); }
    #__ROOT__ .controls { display: flex; align-items: end; flex-wrap: wrap; gap: .55rem .75rem; margin-bottom: .8rem; }
    #__ROOT__ .field { display: grid; gap: .22rem; color: var(--muted); font-size: .78rem; }
    #__ROOT__ select, #__ROOT__ button, #__ROOT__ input { font: inherit; color: var(--ink); }
    #__ROOT__ select, #__ROOT__ button { min-height: 2.25rem; border: 1px solid var(--line); background: transparent; border-radius: .28rem; padding: .38rem .68rem; }
    #__ROOT__ button { cursor: pointer; }
    #__ROOT__ button[aria-pressed="true"] { color: var(--paper); background: var(--ink); border-color: var(--ink); }
    #__ROOT__ button:focus-visible, #__ROOT__ select:focus-visible, #__ROOT__ input:focus-visible { outline: 3px solid color-mix(in srgb, var(--teal) 42%, transparent); outline-offset: 2px; }
    #__ROOT__ .scrub { flex: 1 1 18rem; }
    #__ROOT__ .scrub-head { display: flex; justify-content: space-between; gap: .6rem; }
    #__ROOT__ output, #__ROOT__ .numeric { font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; color: var(--ink); }
    #__ROOT__ input[type="range"] { width: 100%; accent-color: var(--teal); min-height: 1.5rem; }
    #__ROOT__ .visuals { display: grid; grid-template-columns: minmax(0, 1.08fr) minmax(0, .92fr); gap: .8rem; }
    #__ROOT__ svg { display: block; width: 100%; height: auto; overflow: visible; }
    #__ROOT__ .frame { fill: none; stroke: var(--line); stroke-width: 1; }
    #__ROOT__ .structure { fill: var(--soft); stroke: var(--teal); stroke-width: 1.5; }
    #__ROOT__ .structure-label { fill: var(--ink); font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 12px; font-weight: 500; }
    #__ROOT__ .annotation { fill: var(--muted); font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 11px; }
    #__ROOT__ .number { fill: var(--ink); font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; font-variant-numeric: tabular-nums; }
    #__ROOT__ .arrow { fill: none; stroke: var(--teal); stroke-width: 1.6; }
    #__ROOT__ .request { fill: var(--teal); stroke: var(--paper); stroke-width: 1.1; }
    #__ROOT__ .request[data-phase="complete"] { fill: var(--stands); }
    #__ROOT__ .policy-cell { stroke: var(--paper); stroke-width: 1.2; }
    #__ROOT__ .policy-cell[data-action="0"] { fill: var(--teal); }
    #__ROOT__ .policy-cell[data-action="1"] { fill: var(--ink); }
    #__ROOT__ .policy-cell[data-action="2"] { fill: var(--muted); }
    #__ROOT__ .policy-letter { fill: var(--paper); font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 10px; font-weight: 500; pointer-events: none; }
    #__ROOT__ .coverage-dot { fill: var(--paper); stroke: var(--ink); stroke-width: .7; pointer-events: none; }
    #__ROOT__ .meter-track { fill: var(--soft); }
    #__ROOT__ .meter { fill: var(--teal); }
    #__ROOT__ .trace { fill: none; stroke: var(--teal); stroke-width: 2; vector-effect: non-scaling-stroke; }
    #__ROOT__ .trace.secondary { stroke: var(--ink); stroke-dasharray: 5 4; }
    #__ROOT__ .trace.plan { stroke: var(--caveat); stroke-dasharray: 4 4; }
    #__ROOT__ .now { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 3; }
    #__ROOT__ .status { margin: .65rem 0 0; color: var(--muted); font-size: .84rem; min-height: 1.3em; }
    #__ROOT__ .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
    @media (max-width: 690px) {
      #__ROOT__ { padding: .75rem .3rem; }
      #__ROOT__ .visuals { grid-template-columns: 1fr; }
      #__ROOT__ .controls { align-items: stretch; }
      #__ROOT__ .field.controller { flex: 1 1 100%; }
      #__ROOT__ select { width: 100%; }
      #__ROOT__ button { min-width: 4.1rem; min-height: 2.75rem; }
    }
    @media (prefers-reduced-motion: reduce) {
      #__ROOT__ *, #__ROOT__ *::before, #__ROOT__ *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; }
    }
  </style>
  <header>
    <h3 id="__TITLE__">__VISIBLE_TITLE__</h3>
    <p id="__DESCRIPTION__" class="description">Every mark is read from the recorded Python trajectory. The controls change only which recorded frame is shown.</p>
    <p class="provenance" data-verdict="__PROVENANCE_VERDICT__">Profile: __PROVENANCE__</p>
  </header>
  <div class="controls" aria-label="Replay controls">
    <label class="field controller" for="__CONTROLLER__">Controller
      <select id="__CONTROLLER__" data-controller__DISABLED__>__OPTIONS__</select>
    </label>
    <label class="field" data-age-field hidden>Oldest prefill-age bin
      <select data-age aria-label="Oldest prefill-age bin"></select>
    </label>
    <button type="button" data-action="play" aria-pressed="false">Play</button>
    <button type="button" data-action="step">Step</button>
    <button type="button" data-action="reset">Reset</button>
    <label class="field scrub" for="__SCRUBBER__">
      <span class="scrub-head"><span>Recorded time</span><output data-time>0.00 s</output></span>
      <input id="__SCRUBBER__" data-scrubber type="range" min="0" max="0" value="0" step="1" aria-label="Recorded replay time">
    </label>
  </div>
  <div class="visuals">
    <svg data-policy-view viewBox="0 0 520 270" role="img" hidden aria-labelledby="__ROOT__-policy-title __ROOT__-policy-description">
      <title id="__ROOT__-policy-title">Scheduling action over the reduced queue state</title>
      <desc id="__ROOT__-policy-description">A seven by seven grid shows the selected action for each prefill and decode queue length. Where logged coverage is available, white dots grow with the number of observed transitions.</desc>
      <rect class="frame" x=".5" y=".5" width="519" height="269"></rect>
      <text class="structure-label" data-policy-heading x="260" y="23" text-anchor="middle">policy slice</text>
      <g data-policy-grid></g>
      <text class="annotation" x="273" y="256" text-anchor="middle">decode jobs d</text>
      <text class="annotation" transform="translate(20 137) rotate(-90)" text-anchor="middle">prefill jobs p</text>
      <g aria-hidden="true">
        <rect class="policy-cell" data-action="0" x="382" y="55" width="15" height="15"></rect><text class="annotation" x="402" y="67">prefill (P)</text>
        <rect class="policy-cell" data-action="1" x="382" y="82" width="15" height="15"></rect><text class="annotation" x="402" y="94">decode (D)</text>
        <rect class="policy-cell" data-action="2" x="382" y="109" width="15" height="15"></rect><text class="annotation" x="402" y="121">idle (I)</text>
        <circle class="coverage-dot" cx="390" cy="143" r="5"></circle><text class="annotation" x="402" y="147">logged visits</text>
      </g>
      <text class="annotation" data-coverage-label x="382" y="177">coverage not recorded</text>
      <text class="annotation" data-policy-empty x="260" y="138" text-anchor="middle" hidden>No stored policy slice for this controller.</text>
    </svg>
    <svg data-scene viewBox="0 0 520 270" role="img" aria-labelledby="__ROOT__-scene-title __ROOT__-scene-description">
      <title id="__ROOT__-scene-title">Recorded request flow through the inference service</title>
      <desc id="__ROOT__-scene-description">Requests move from the queue through prompt prefill and token decode to completion. The GPU clock, power, temperature, and key-value cache are reported below.</desc>
      <defs><marker id="__ROOT__-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="var(--teal)"></path></marker></defs>
      <rect class="frame" x=".5" y=".5" width="519" height="269"></rect>
      <g aria-hidden="true">
        <rect class="structure" x="24" y="48" width="94" height="94" rx="5"></rect>
        <rect class="structure" x="151" y="48" width="94" height="94" rx="5"></rect>
        <rect class="structure" x="278" y="48" width="94" height="94" rx="5"></rect>
        <rect class="structure" x="405" y="48" width="90" height="94" rx="5"></rect>
        <path class="arrow" d="M119 95 H146" marker-end="url(#__ROOT__-arrow)"></path>
        <path class="arrow" d="M246 95 H273" marker-end="url(#__ROOT__-arrow)"></path>
        <path class="arrow" d="M373 95 H400" marker-end="url(#__ROOT__-arrow)"></path>
        <text class="structure-label" x="71" y="35" text-anchor="middle">queue</text>
        <text class="structure-label" x="198" y="35" text-anchor="middle">prefill</text>
        <text class="structure-label" x="325" y="35" text-anchor="middle">decode</text>
        <text class="structure-label" x="450" y="35" text-anchor="middle">complete</text>
        <text class="number" data-count="queue" x="71" y="130" text-anchor="middle">0</text>
        <text class="number" data-count="prefill" x="198" y="130" text-anchor="middle">0</text>
        <text class="number" data-count="decode" x="325" y="130" text-anchor="middle">0</text>
        <text class="number" data-count="complete" x="450" y="130" text-anchor="middle">0</text>
      </g>
      <g data-requests aria-hidden="true"></g>
      <text class="annotation" x="24" y="177">KV cache</text>
      <rect class="meter-track" x="88" y="166" width="407" height="13" rx="2"></rect>
      <rect class="meter" data-kv-meter x="88" y="166" width="0" height="13" rx="2"></rect>
      <text class="number" data-kv-value x="495" y="198" text-anchor="end">not recorded</text>
      <text class="annotation" x="24" y="222">GPU</text>
      <text class="number" data-gpu x="88" y="222">clock not recorded</text>
      <text class="annotation" x="24" y="248">thermal state</text>
      <text class="number" data-thermal x="118" y="248">not recorded</text>
    </svg>
    <svg data-trace viewBox="0 0 420 270" role="img" aria-labelledby="__ROOT__-trace-title __ROOT__-trace-description">
      <title id="__ROOT__-trace-title">Causal trace of the recorded controller</title>
      <desc id="__ROOT__-trace-description">Only measurements up to the selected recorded time are drawn. For MPC, the dashed amber curve is the plan available at that time rather than realized future data.</desc>
      <rect class="frame" x=".5" y=".5" width="419" height="269"></rect>
      <line class="frame" x1="54" y1="126" x2="398" y2="126"></line>
      <line class="frame" x1="54" y1="236" x2="398" y2="236"></line>
      <text class="structure-label" data-lane-label="0" x="54" y="24">backlog</text>
      <text class="structure-label" data-lane-label="1" x="54" y="145">GPU clock</text>
      <text class="annotation" x="226" y="260" text-anchor="middle">time (s)</text>
      <text class="annotation" data-x-start x="54" y="253" text-anchor="middle">0</text>
      <text class="annotation" data-x-end x="398" y="253" text-anchor="middle">0</text>
      <path class="trace" data-trace-path="0" d=""></path>
      <path class="trace secondary" data-trace-path="1" d=""></path>
      <path class="trace plan" data-plan-path d=""></path>
      <line class="now" data-now x1="54" x2="54" y1="28" y2="236"></line>
      <text class="number" data-lane-value="0" x="398" y="24" text-anchor="end"></text>
      <text class="number" data-lane-value="1" x="398" y="145" text-anchor="end"></text>
    </svg>
  </div>
  <p class="status" data-status aria-live="off"></p>
  <span class="sr-only" data-announcer aria-live="polite"></span>
</section>
<script type="application/json" id="__DATA_ID__">__DATA__</script>
<script>
(() => {
  "use strict";
  const root = document.getElementById("__ROOT__");
  const dataNode = document.getElementById("__DATA_ID__");
  if (!root || !dataNode) return;
  const replay = JSON.parse(dataNode.textContent);
  const controller = root.querySelector("[data-controller]");
  const scrubber = root.querySelector("[data-scrubber]");
  const playButton = root.querySelector('[data-action="play"]');
  const stepButton = root.querySelector('[data-action="step"]');
  const resetButton = root.querySelector('[data-action="reset"]');
  const timeOutput = root.querySelector("[data-time]");
  const status = root.querySelector("[data-status]");
  const announcer = root.querySelector("[data-announcer]");
  const requestLayer = root.querySelector("[data-requests]");
  const sceneView = root.querySelector("[data-scene]");
  const policyView = root.querySelector("[data-policy-view]");
  const policyGrid = root.querySelector("[data-policy-grid]");
  const policyHeading = root.querySelector("[data-policy-heading]");
  const policyEmpty = root.querySelector("[data-policy-empty]");
  const coverageLabel = root.querySelector("[data-coverage-label]");
  const ageField = root.querySelector("[data-age-field]");
  const ageSelect = root.querySelector("[data-age]");
  const svgNS = "http://www.w3.org/2000/svg";
  let frameIndex = 0;
  let timer = null;

  const currentRun = () => replay.controllers[controller.value];
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const format = (value, digits) => finite(value) ? value.toFixed(digits) : "not recorded";
  const hasPolicies = replay.policy && Object.keys(replay.policy.actions || {}).length > 0;
  const updateAgeOptions = () => {
    if (!hasPolicies) return;
    const slices = replay.policy.actions[controller.value] || {};
    const previous = ageSelect.value;
    ageSelect.replaceChildren();
    Object.keys(slices).sort((a, b) => Number(a) - Number(b)).forEach((age) => {
      const option = document.createElement("option");
      option.value = age; option.textContent = age;
      ageSelect.appendChild(option);
    });
    if ([...ageSelect.options].some((option) => option.value === previous)) ageSelect.value = previous;
    else if ([...ageSelect.options].some((option) => option.value === "4")) ageSelect.value = "4";
    ageSelect.disabled = ageSelect.options.length === 0;
  };
  const renderPolicy = () => {
    if (!hasPolicies) return;
    policyGrid.replaceChildren();
    const slices = replay.policy.actions[controller.value] || {};
    const matrix = slices[ageSelect.value];
    policyEmpty.toggleAttribute("hidden", Boolean(matrix));
    coverageLabel.textContent = "coverage not recorded";
    policyHeading.textContent = controller.options[controller.selectedIndex].text + " · age bin " + (ageSelect.value || "not stored");
    if (!matrix) return;
    const coverageSlices = (replay.policy.coverage || {})[controller.value] || {};
    const coverage = coverageSlices[ageSelect.value];
    const maximumCoverage = coverage ? Math.max(1, ...coverage.flat()) : 1;
    const letters = ["P", "D", "I"];
    for (let p = 0; p < 7; p += 1) {
      for (let d = 0; d < 7; d += 1) {
        const x = 54 + d * 43;
        const y = 39 + (6 - p) * 27;
        const action = Number(matrix[p][d]);
        const cell = document.createElementNS(svgNS, "rect");
        cell.setAttribute("class", "policy-cell");
        cell.setAttribute("data-action", String(action));
        cell.setAttribute("x", String(x)); cell.setAttribute("y", String(y));
        cell.setAttribute("width", "40"); cell.setAttribute("height", "24");
        policyGrid.appendChild(cell);
        const letter = document.createElementNS(svgNS, "text");
        letter.setAttribute("class", "policy-letter");
        letter.setAttribute("x", String(x + 20)); letter.setAttribute("y", String(y + 16));
        letter.setAttribute("text-anchor", "middle"); letter.textContent = letters[action] || "?";
        policyGrid.appendChild(letter);
        if (coverage && Number(coverage[p][d]) > 0) {
          const dot = document.createElementNS(svgNS, "circle");
          dot.setAttribute("class", "coverage-dot");
          dot.setAttribute("cx", String(x + 31)); dot.setAttribute("cy", String(y + 7));
          dot.setAttribute("r", String(1.8 + 5 * Math.sqrt(Number(coverage[p][d]) / maximumCoverage)));
          policyGrid.appendChild(dot);
        }
      }
    }
    for (let value = 0; value < 7; value += 1) {
      const xTick = document.createElementNS(svgNS, "text");
      xTick.setAttribute("class", "annotation"); xTick.setAttribute("text-anchor", "middle");
      xTick.setAttribute("x", String(74 + value * 43)); xTick.setAttribute("y", "237"); xTick.textContent = String(value);
      policyGrid.appendChild(xTick);
      const yTick = document.createElementNS(svgNS, "text");
      yTick.setAttribute("class", "annotation"); yTick.setAttribute("text-anchor", "end");
      yTick.setAttribute("x", "48"); yTick.setAttribute("y", String(56 + (6 - value) * 27)); yTick.textContent = String(value);
      policyGrid.appendChild(yTick);
    }
    if (coverage) {
      const fraction = replay.policy.coverage_fraction[controller.value];
      const observed = coverage.flat().filter((value) => Number(value) > 0).length / 49;
      coverageLabel.textContent = "state coverage " + (100 * (finite(fraction) ? fraction : observed)).toFixed(1) + "%";
    }
  };
  const laneDefinitions = () => {
    const frames = currentRun().frames;
    const has = (key) => frames.some((frame) => finite(frame[key]));
    if (replay.view === "open_loop" || replay.view === "mpc") {
      return [
        {key: "backlog", label: "backlog", unit: "requests"},
        {key: has("realized_clock_mhz") ? "realized_clock_mhz" : "requested_clock_mhz", label: "GPU clock", unit: "MHz"}
      ];
    }
    if (replay.view === "scheduling" || replay.view === "fqi") {
      return [
        {key: has("prefill_queue") ? "prefill_queue" : "backlog", label: "prefill queue", unit: "requests"},
        {key: has("decode_active") ? "decode_active" : "completed_requests", label: has("decode_active") ? "active decode" : "completed", unit: "requests"}
      ];
    }
    return [
      {key: "backlog", label: "unfinished requests", unit: "requests"},
      {key: has("temperature_c") ? "temperature_c" : (has("realized_clock_mhz") ? "realized_clock_mhz" : "power_w"), label: has("temperature_c") ? "GPU temperature" : (has("realized_clock_mhz") ? "GPU clock" : "GPU power"), unit: has("temperature_c") ? "°C" : (has("realized_clock_mhz") ? "MHz" : "W")}
    ];
  };
  const extent = (frames, key) => {
    const values = frames.map((frame) => frame[key]).filter(finite);
    if (!values.length) return [0, 1];
    let low = Math.min(...values), high = Math.max(...values);
    if (low === high) { low -= Math.max(1, Math.abs(low) * .05); high += Math.max(1, Math.abs(high) * .05); }
    const pad = .08 * (high - low);
    return [low - pad, high + pad];
  };
  const pathFor = (frames, key, lane, fullFrames) => {
    const domain = extent(fullFrames, key);
    const t0 = fullFrames[0].time_s;
    const t1 = fullFrames[fullFrames.length - 1].time_s;
    const yTop = lane === 0 ? 32 : 151;
    const yBottom = lane === 0 ? 116 : 230;
    const points = [];
    frames.forEach((frame) => {
      if (!finite(frame[key])) return;
      const x = 54 + 344 * ((frame.time_s - t0) / Math.max(t1 - t0, 1e-9));
      const y = yBottom - (yBottom - yTop) * ((frame[key] - domain[0]) / (domain[1] - domain[0]));
      points.push((points.length ? "L" : "M") + x.toFixed(2) + " " + y.toFixed(2));
    });
    return points.join(" ");
  };
  const renderRequests = (frame) => {
    requestLayer.replaceChildren();
    const phases = {queue: 71, prefill: 198, decode: 325, complete: 450};
    const requests = Array.isArray(frame.requests) ? frame.requests : [];
    if (requests.length) {
      const counts = {queue: 0, prefill: 0, decode: 0, complete: 0};
      requests.forEach((request) => {
        if (!(request.phase in phases)) return;
        const rank = counts[request.phase]++;
        const circle = document.createElementNS(svgNS, "circle");
        circle.setAttribute("class", "request");
        circle.setAttribute("data-phase", request.phase);
        circle.setAttribute("cx", String(phases[request.phase] - 24 + (rank % 5) * 12));
        circle.setAttribute("cy", String(62 + Math.floor(rank / 5) * 13));
        circle.setAttribute("r", "4.5");
        requestLayer.appendChild(circle);
      });
      Object.keys(counts).forEach((phase) => {
        root.querySelector('[data-count="' + phase + '"]').textContent = String(counts[phase]);
      });
    } else {
      const counts = {
        queue: finite(frame.prefill_queue) ? Math.max(0, Math.round(frame.prefill_queue) - (frame.phase === "prefill" ? 1 : 0)) : 0,
        prefill: frame.phase === "prefill" ? 1 : 0,
        decode: finite(frame.decode_active) ? Math.round(frame.decode_active) : 0,
        complete: finite(frame.completed_requests) ? Math.round(frame.completed_requests) : 0
      };
      Object.keys(counts).forEach((phase) => {
        root.querySelector('[data-count="' + phase + '"]').textContent = String(counts[phase]);
      });
    }
  };
  const renderFrame = (announce) => {
    const run = currentRun();
    const frames = run.frames;
    frameIndex = Math.max(0, Math.min(frameIndex, frames.length - 1));
    const frame = frames[frameIndex];
    scrubber.max = String(frames.length - 1);
    scrubber.value = String(frameIndex);
    timeOutput.value = format(frame.time_s, 2) + " s";
    renderRequests(frame);
    renderPolicy();

    const kvValues = frames.map((item) => item.kv_tokens).filter(finite);
    const kvMaximum = kvValues.length ? Math.max(...kvValues, 1) : 1;
    const kvWidth = finite(frame.kv_tokens) ? 407 * Math.max(0, frame.kv_tokens) / kvMaximum : 0;
    root.querySelector("[data-kv-meter]").setAttribute("width", String(Math.min(407, kvWidth)));
    root.querySelector("[data-kv-value]").textContent = finite(frame.kv_tokens) ? format(frame.kv_tokens, 0) + " recorded units" : "not recorded";
    const clock = finite(frame.realized_clock_mhz) ? frame.realized_clock_mhz : frame.requested_clock_mhz;
    root.querySelector("[data-gpu]").textContent = (finite(clock) ? format(clock, 0) + " MHz" : "clock not recorded") + (finite(frame.power_w) ? " · " + format(frame.power_w, 1) + " W" : "");
    root.querySelector("[data-thermal]").textContent = finite(frame.temperature_c) ? format(frame.temperature_c, 1) + " °C" : "not recorded";

    const lanes = laneDefinitions();
    const visibleFrames = frames.slice(0, frameIndex + 1);
    lanes.forEach((lane, index) => {
      root.querySelector('[data-lane-label="' + index + '"]').textContent = lane.label;
      root.querySelector('[data-lane-value="' + index + '"]').textContent = finite(frame[lane.key]) ? format(frame[lane.key], lane.unit === "MHz" ? 0 : 1) + " " + lane.unit : "not recorded";
      root.querySelector('[data-trace-path="' + index + '"]').setAttribute("d", pathFor(visibleFrames, lane.key, index, frames));
    });
    const t0 = frames[0].time_s, t1 = frames[frames.length - 1].time_s;
    const nowX = 54 + 344 * ((frame.time_s - t0) / Math.max(t1 - t0, 1e-9));
    const now = root.querySelector("[data-now]");
    now.setAttribute("x1", String(nowX)); now.setAttribute("x2", String(nowX));
    root.querySelector("[data-x-start]").textContent = format(t0, 0);
    root.querySelector("[data-x-end]").textContent = format(t1, 0);

    const planPath = root.querySelector("[data-plan-path]");
    const plan = frame.planned_clock_mhz;
    if (Array.isArray(plan) && plan.length && lanes[1].key.includes("clock")) {
      const domain = extent(frames, lanes[1].key);
      const points = plan.map((value, index) => {
        const planTime = frame.time_s + index * run.plan_dt_s;
        const x = 54 + 344 * ((planTime - t0) / Math.max(t1 - t0, 1e-9));
        const y = 230 - 79 * ((value - domain[0]) / (domain[1] - domain[0]));
        return (index ? "L" : "M") + Math.min(398, x).toFixed(2) + " " + Math.max(151, Math.min(230, y)).toFixed(2);
      });
      planPath.setAttribute("d", points.join(" "));
    } else {
      planPath.setAttribute("d", "");
    }

    const pieces = ["frame " + (frameIndex + 1) + " of " + frames.length];
    if (finite(frame.backlog)) pieces.push(format(frame.backlog, 0) + " unfinished requests");
    if (finite(frame.energy_j)) pieces.push(format(frame.energy_j, 0) + " J cumulative energy");
    status.textContent = pieces.join(" · ");
    if (announce) announcer.textContent = controller.value + ", " + format(frame.time_s, 2) + " seconds, " + pieces.slice(1).join(", ");
  };
  const stop = () => {
    if (timer !== null) window.clearTimeout(timer);
    timer = null;
    playButton.setAttribute("aria-pressed", "false");
    playButton.textContent = "Play";
  };
  const tick = () => {
    const frames = currentRun().frames;
    if (frameIndex >= frames.length - 1) { stop(); return; }
    frameIndex += 1;
    renderFrame(false);
    const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    timer = window.setTimeout(tick, reduced ? 900 : 180);
  };
  playButton.addEventListener("click", () => {
    if (timer !== null) { stop(); announcer.textContent = "Replay paused."; return; }
    if (frameIndex >= currentRun().frames.length - 1) frameIndex = 0;
    playButton.setAttribute("aria-pressed", "true");
    playButton.textContent = "Pause";
    announcer.textContent = "Replay started.";
    timer = window.setTimeout(tick, 0);
  });
  stepButton.addEventListener("click", () => { stop(); frameIndex = Math.min(frameIndex + 1, currentRun().frames.length - 1); renderFrame(true); });
  resetButton.addEventListener("click", () => { stop(); frameIndex = 0; renderFrame(true); });
  scrubber.addEventListener("input", () => { stop(); frameIndex = Number(scrubber.value); renderFrame(false); });
  scrubber.addEventListener("change", () => renderFrame(true));
  controller.addEventListener("change", () => { stop(); frameIndex = 0; updateAgeOptions(); renderFrame(true); });
  ageSelect.addEventListener("change", () => { renderPolicy(); announcer.textContent = "Oldest prefill-age bin " + ageSelect.value + "."; });

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
      new MutationObserver(applyTheme).observe(themeRoot, {attributes: true, attributeFilter: ["class", "data-theme", "data-mode", "style"]});
    }
  } catch (error) { /* Cross-origin embedding falls back to the media query. */ }
  if (window.matchMedia) {
    const themeQuery = window.matchMedia("(prefers-color-scheme: dark)");
    if (typeof themeQuery.addEventListener === "function") themeQuery.addEventListener("change", applyTheme);
  }
  if (hasPolicies) {
    sceneView.setAttribute("hidden", "");
    policyView.removeAttribute("hidden");
    ageField.hidden = false;
    updateAgeOptions();
  }
  renderFrame(false);
})();
</script>
'''
    replacements = {
        "__ROOT__": root_id,
        "__TITLE__": title_id,
        "__DESCRIPTION__": description_id,
        "__VISIBLE_TITLE__": title,
        "__PROVENANCE__": provenance,
        "__PROVENANCE_VERDICT__": provenance_verdict,
        "__CONTROLLER__": controller_id,
        "__SCRUBBER__": scrubber_id,
        "__OPTIONS__": options,
        "__DISABLED__": disabled,
        "__DATA_ID__": data_id,
        "__DATA__": serialized,
    }
    for marker, replacement in replacements.items():
        fragment = fragment.replace(marker, replacement)
    fragment = fragment.strip()
    if len(fragment.encode("utf-8")) > 1_000_000:
        raise ReplayDataError(
            "The replay exceeds 1 MB after downsampling. Reduce maximum_frames or request detail."
        )
    return fragment


def render_static_figure(
    source: Path | str | Mapping[str, Any] | Any,
    output: Path | str | None = None,
    *,
    view: str = "modeling",
) -> plt.Figure:
    """Render a print-safe SVG fallback from the same recorded trajectories."""

    replay = _normalise_replay(source, view=view, maximum_frames=10_000)
    if replay["view"] in {"scheduling", "fqi"} and replay["policy"]["actions"]:
        return _render_policy_static(replay, output)
    styles = ["-", "--", "-.", ":", (0, (5, 2, 1, 2))]
    colors = [TEAL, INK, MUTED, CAVEAT, STANDS]
    with mpl.rc_context(_FIGURE_STYLE):
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(7.2, 4.5),
            sharex=True,
            constrained_layout=True,
        )
        scheduling_view = replay["view"] in {"scheduling", "fqi"}
        for index, (name, run) in enumerate(replay["controllers"].items()):
            frames = run["frames"]
            time_s = np.asarray([frame["time_s"] for frame in frames], dtype=float)
            prefill = np.asarray(
                [frame.get("prefill_queue", 0.0) for frame in frames], dtype=float
            )
            decode = np.asarray(
                [frame.get("decode_active", 0.0) for frame in frames], dtype=float
            )
            first_series = prefill if scheduling_view else prefill + decode
            clock = np.asarray(
                [
                    frame.get(
                        "realized_clock_mhz",
                        frame.get("requested_clock_mhz", np.nan),
                    )
                    for frame in frames
                ],
                dtype=float,
            )
            line_style = styles[index % len(styles)]
            color = colors[index % len(colors)]
            display_name = name.replace("_", " ")
            axes[0].plot(time_s, first_series, line_style, color=color, label=display_name)
            if scheduling_view:
                axes[1].plot(time_s, decode, line_style, color=color, label=display_name)
            elif np.isfinite(clock).any():
                axes[1].plot(time_s, clock, line_style, color=color, label=display_name)

        axes[0].set_ylabel("prefill queue" if scheduling_view else "unfinished requests")
        axes[1].set_ylabel("active decode" if scheduling_view else "realized clock (MHz)")
        axes[1].set_xlabel("time (s)")
        axes[0].grid(axis="y", color="#DCE1E2", linewidth=0.6)
        axes[1].grid(axis="y", color="#DCE1E2", linewidth=0.6)
        axes[0].legend(loc="upper right")
        if not axes[1].lines:
            axes[1].text(
                0.5,
                0.5,
                "GPU clock was not recorded for this view",
                transform=axes[1].transAxes,
                ha="center",
                va="center",
                color=MUTED,
            )
        figure.suptitle(replay["title"], fontfamily="serif", fontweight="normal")
        provenance_color = (
            STANDS if replay["provenance_verdict"] == "stands" else CAVEAT
        )
        axes[0].text(
            0.01,
            0.97,
            "Profile: " + replay["provenance_text"],
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=provenance_color,
            bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1.5, "alpha": 0.9},
        )

        if output is not None:
            path = Path(output)
            if path.suffix.lower() != ".svg":
                raise ValueError("Static inference fallbacks must be written as SVG files.")
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, format="svg", bbox_inches="tight")
    return figure


def _render_policy_static(
    replay: Mapping[str, Any], output: Path | str | None
) -> plt.Figure:
    """Render policy slices, and logged coverage when available, as small multiples."""

    policy = replay["policy"]
    if replay["view"] == "scheduling":
        slices = policy["actions"]["exact_dp"]
        selected_ages = [age for age in ("0", "2", "4") if age in slices]
        if not selected_ages:
            selected_ages = list(slices)[:3]
        panels = [
            (f"oldest age bin {age}", slices[age], None) for age in selected_ages
        ]
    else:
        preferred = ["exact_dp", "broad_fqi", "narrow_fqi"]
        controllers = [name for name in preferred if name in policy["actions"]]
        controllers.extend(
            name for name in policy["actions"] if name not in controllers
        )
        panels = []
        for controller in controllers[:3]:
            slices = policy["actions"][controller]
            age = "4" if "4" in slices else next(iter(slices))
            coverage = policy["coverage"].get(controller, {}).get(age)
            panels.append((controller.replace("_", " "), slices[age], coverage))

    color_map = ListedColormap([TEAL, INK, MUTED])
    with mpl.rc_context(_FIGURE_STYLE):
        figure, axes = plt.subplots(
            1,
            len(panels),
            figsize=(7.2, 3.05),
            squeeze=False,
        )
        axes_row = list(axes[0])
        for axis, (title, matrix, coverage) in zip(axes_row, panels):
            actions = np.asarray(matrix, dtype=int)
            axis.imshow(
                actions,
                origin="lower",
                vmin=-0.5,
                vmax=2.5,
                cmap=color_map,
                interpolation="nearest",
                aspect="equal",
            )
            for p in range(7):
                for d in range(7):
                    axis.text(
                        d,
                        p,
                        ("P", "D", "I")[int(actions[p, d])],
                        ha="center",
                        va="center",
                        color=PAPER,
                        fontsize=7.5,
                        fontfamily="monospace",
                        fontweight="bold",
                    )
            if coverage is not None:
                counts = np.asarray(coverage, dtype=float)
                maximum = max(float(np.max(counts)), 1.0)
                for p, d in np.argwhere(counts > 0.0):
                    axis.scatter(
                        [d + 0.27],
                        [p + 0.27],
                        s=7.0 + 34.0 * math.sqrt(float(counts[p, d]) / maximum),
                        facecolor=PAPER,
                        edgecolor=INK,
                        linewidth=0.35,
                        zorder=4,
                    )
            axis.set_title(title)
            axis.set_xticks(range(7))
            axis.set_yticks(range(7))
            axis.set_xlabel("decode jobs $d$")
            axis.set_ylabel("prefill jobs $p$")
            axis.set_xticks(np.arange(-0.5, 7, 1), minor=True)
            axis.set_yticks(np.arange(-0.5, 7, 1), minor=True)
            axis.grid(which="minor", color=PAPER, linewidth=0.7)
            axis.tick_params(which="minor", length=0)
            axis.spines.top.set_visible(True)
            axis.spines.right.set_visible(True)
            axis.spines.top.set_linewidth(0.6)
            axis.spines.right.set_linewidth(0.6)

        handles = [
            Patch(facecolor=TEAL, edgecolor="none", label="prefill (P)"),
            Patch(facecolor=INK, edgecolor="none", label="decode (D)"),
            Patch(facecolor=MUTED, edgecolor="none", label="idle (I)"),
        ]
        if any(coverage is not None for _, _, coverage in panels):
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=PAPER,
                    markeredgecolor=INK,
                    markersize=5,
                    label="logged visits",
                )
            )
        figure.suptitle(replay["title"], y=0.985, fontfamily="serif", fontweight="normal")
        provenance_color = (
            STANDS if replay["provenance_verdict"] == "stands" else CAVEAT
        )
        figure.text(
            0.5,
            0.91,
            "Profile: " + replay["provenance_text"],
            ha="center",
            va="center",
            fontsize=8,
            color=provenance_color,
        )
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=len(handles),
            fontsize=8,
            frameon=False,
        )
        figure.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.82, wspace=0.28)

        if output is not None:
            path = Path(output)
            if path.suffix.lower() != ".svg":
                raise ValueError("Static inference fallbacks must be written as SVG files.")
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, format="svg", bbox_inches="tight")
    return figure


__all__ = [
    "ReplayDataError",
    "render_serving_replay",
    "render_static_figure",
]
