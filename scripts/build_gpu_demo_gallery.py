#!/usr/bin/env python3
"""Build the standalone gallery of recorded GPU inference-control demos."""

from __future__ import annotations

import argparse
import base64
import hashlib
from html import escape
import json
import math
from pathlib import Path
from statistics import mean
import sys


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from inference_replay import render_serving_replay  # noqa: E402


ARTIFACT = ROOT / "artifacts" / "inference_serving" / "textbook_results.json"
DEFAULT_OUTPUT = ROOT / "interactive" / "gpu-demos.html"
LIVE_DIRECTORY = ROOT / "artifacts/inference_serving/live_control_development"
MATHJAX_BUNDLE = ROOT / "scripts" / "vendor" / "mathjax-3.2.2-tex-svg-full.js"
MATHJAX_CONFIG = r'''window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]]
  },
  svg: {fontCache: "local"},
  options: {enableMenu: false}
};'''

DEMOS = (
    {
        "view": "modeling",
        "label": "System state",
        "stage": "01 · Frame the system",
        "question": "What must a controller observe about requests, work, and the GPU?",
        "approach": (
            "The boundary contains one GPU, its request queue, scheduler, model execution, "
            "key-value cache, and hardware state. Token and cache balances provide the known "
            "structure. Measurements supply the service-rate and power maps."
        ),
        "equation": (
            r"\(x_t=(p_t,d_t,m_t,T_t,f_t),\qquad "
            r"u_t=(f_t^{\mathrm{req}},\sigma_t)\)"
        ),
        "watch": (
            "Follow a request from arrival through prefill, decode, and completion. The "
            "eventual output length stays hidden until completion, so the controller never "
            "gets to schedule with information that would be unavailable online."
        ),
        "expect": (
            "All 20 replayed requests finish, both work queues return to zero, and modeled "
            "cache is released. That verifies conservation and software closure. It does not "
            "prove that the reduced state is fully Markov or that every calibrated map is accurate."
        ),
        "try": (
            "Pause while several requests overlap. Inspect the active phase and queue state, "
            "then predict whether the next interval will process prefill or decode before stepping forward."
        ),
    },
    {
        "view": "open_loop",
        "label": "Open-loop planning",
        "stage": "02 · Plan once",
        "question": "What clock schedule works when the request trace is known in advance?",
        "approach": (
            "A linear program chooses all 60 one-second clock decisions before execution. "
            "It trades normalized power against backlog in an aggregate fluid model. The "
            "request-level replay then audits that fixed plan under nominal and shifted arrivals."
        ),
        "equation": (
            r"\(\displaystyle \min_{\nu,B}\ "
            r"\sum_{k=0}^{59}[\alpha_P\nu_k+20B_{k+1}]+20B_{60}\)"
        ),
        "watch": (
            "The dashed clock plan never changes. Switch between nominal and shifted arrivals: "
            "the same 31-request burst moves twenty seconds earlier and reaches the server while "
            "the fixed plan still expects light work."
        ),
        "expect": (
            "The timing error leaves 29 requests queued at 30 seconds. Mean time to first token "
            "rises from 16.35 to 23.23 seconds even though energy falls slightly. The plan is "
            "efficient for its forecast and brittle to this explicit forecast error."
        ),
        "try": (
            "Play the nominal trace first and note when the clock rises. Then select the shifted "
            "trace and scrub to 30 seconds. The mismatch between available work and planned clock "
            "is the failure that motivates feedback."
        ),
    },
    {
        "view": "mpc",
        "label": "Receding-horizon control",
        "stage": "03 · Close the loop",
        "question": "How does repeated replanning respond as the measured backlog changes?",
        "approach": (
            "Every second, model predictive control observes the new aggregate state, solves a "
            "ten-second problem, applies only its first clock request, and repeats. The objective "
            "prices energy, delay overruns, clock movement, constraint slacks, and terminal backlog."
        ),
        "equation": (
            r"\(\displaystyle u_t=\operatorname{first}\!\left("
            r"\arg\min_{u_{t:t+9}}J_t(x_t,u_{t:t+9})\right),\quad "
            r"\text{then re-solve at }t+1\)"
        ),
        "watch": (
            "The dashed horizon changes after each observation while the solid line records only "
            "actions already applied. Compare MPC with maximum clock, the reactive governor, and "
            "the unchanged offline plan under identical shifted arrivals."
        ),
        "expect": (
            "MPC records 11.53 seconds mean time to first token at 3,341.8 J. Maximum clock is "
            "0.60 seconds faster but consumes 164.2 J more; the offline plan reaches 23.23 seconds "
            "and 3,646.8 J. These are model-based trajectories calibrated by measured L4 curves."
        ),
        "try": (
            "Select the offline plan, then MPC, and replay the burst. Watch when each controller "
            "raises the clock and how quickly the queue turns. That timing difference is the "
            "operational value of closing the loop."
        ),
    },
    {
        "view": "scheduling",
        "label": "Exact scheduling",
        "stage": "04 · Compute a policy",
        "question": "Can prefill and decode decisions be computed for every reduced queue state?",
        "approach": (
            "The clock is fixed and scheduling becomes the action. Queue lengths and the oldest "
            "prefill-age bin form a 245-state Markov decision process. A measured L4 profile "
            "calibrates completion probabilities and energy; value iteration solves the finite model."
        ),
        "equation": (
            r"\(\displaystyle V_{n+1}(s)=\min_{u\in\mathcal A(s)}\left["
            r"c(s,u)+0.99\sum_{s'}P^u_{ss'}V_n(s')\right]\)"
        ),
        "watch": (
            "Move through the five age slices and inspect the action map over prefill and decode "
            "queue lengths. In this reduced model every slice yields the same rule: serve decode "
            "when active, otherwise prefill, and idle only when empty."
        ),
        "expect": (
            "Value iteration takes 2,454 sweeps and finishes with a Bellman residual of 9.82×10⁻¹¹. "
            "That is a numerical certificate for the 245-state abstraction, not for the original "
            "request-level server whose identities, lengths, and exact ages were compressed away."
        ),
        "try": (
            "Choose a crowded state and change the oldest-age slice. If the selected action does "
            "not change, ask whether that invariance comes from the problem itself or from the "
            "state and cost chosen for this abstraction."
        ),
    },
    {
        "view": "fqi",
        "label": "Fitted Q scheduling",
        "stage": "05 · Learn from transitions",
        "question": "What changes when the scheduling policy must be learned from sampled transitions?",
        "approach": (
            "Fitted Q-iteration replaces the known transition matrix with a fixed table of 50,000 "
            "sampled transitions. Fifty Bellman-regression sweeps fit a 200-tree Extra-Trees model. "
            "Broad and narrow buffers have equal size; only their state-action coverage changes."
        ),
        "equation": (
            r"\(\displaystyle y_i^{(n)}=c_i+0.99"
            r"\min_{u'\in\mathcal A(s_i')}q_n(s_i',u')\)"
        ),
        "watch": (
            "Compare each learned action map with exact dynamic programming. Coverage markers show "
            "whether the buffer contains the action chosen in a displayed state. Missing markers "
            "identify decisions supported mainly by the regressor's extrapolation."
        ),
        "expect": (
            "Broad sampling covers every feasible state-action pair but still disagrees with the "
            "exact policy on 13.88% of states. Narrow sampling covers 13.86%, disagrees on 68.16%, "
            "and raises mean waiting time from 3.02 to 21.20 seconds in the reduced-model evaluation."
        ),
        "try": (
            "Switch between broad and narrow buffers at the same age slice. Find a state where the "
            "policy changes and inspect its coverage marker. This turns an abstract distribution-shift "
            "warning into a concrete scheduling decision."
        ),
    },
)


def _tab(view: str, label: str, index: int) -> str:
    selected = "true" if index == 0 else "false"
    tab_index = "0" if index == 0 else "-1"
    return f'''<button class="demo-tab" id="tab-{view}" type="button" role="tab"
        aria-selected="{selected}" aria-controls="panel-{view}"
        tabindex="{tab_index}" data-demo-tab="{view}">
      <span class="tab-index">{index + 1:02d}</span>
      <span>{label}</span>
    </button>'''


def _panel(demo: dict[str, str], index: int) -> str:
    view = demo["view"]
    fragment = render_serving_replay(
        ARTIFACT,
        view=view,
        replay_id=f"gpu-demo-{view.replace('_', '-')}",
        maximum_frames=360,
        stable_id=True,
    )
    hidden = "" if index == 0 else " hidden"
    return f'''<section class="demo-panel" id="panel-{view}" role="tabpanel"
      aria-labelledby="tab-{view}" data-demo-panel="{view}"{hidden}>
      <div class="demo-intro">
        <p class="demo-stage">{demo["stage"]}</p>
        <h2>{demo["question"]}</h2>
        <div class="demo-method">
          <div>
            <h3>Approach</h3>
            <p>{demo["approach"]}</p>
          </div>
          <div>
            <h3>Mathematical object</h3>
            <p class="equation">{demo["equation"]}</p>
          </div>
        </div>
        <div class="demo-guides">
          <div><h3>What to watch</h3><p>{demo["watch"]}</p></div>
          <div><h3>What to expect</h3><p>{demo["expect"]}</p></div>
          <div class="try-card"><h3>Try it</h3><p>{demo["try"]}</p></div>
        </div>
      </div>
      {fragment}
    </section>'''


def _json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _live_method_label(method: str, protocol: dict) -> str:
    if method.startswith("fixed_") and method[6:].isdigit():
        clock = int(method[6:])
        label = f"Fixed {clock:,} MHz"
        if clock == protocol.get("selected_fixed_clock_mhz"):
            label += " (selected on development)"
        return label
    return {"governor": "Governor", "mpc": "MPC including fallback"}.get(method, method)


def _method_measurements(rows: list[dict], targets: dict) -> dict:
    metrics = ("energy_per_request_j", "p95_ttft_s", "p95_mean_itl_s")
    valid = [row for row in rows if row.get("valid") is True
             and all(_positive_number(row.get(key)) for key in metrics)]
    complete = (len(rows) == len(valid) == 5
                and sorted(row.get("repetition", -1) for row in rows) == list(range(5)))
    feasible = complete and all(row["p95_ttft_s"] <= targets["ttft_s"]
                                and row["p95_mean_itl_s"] <= targets["mean_itl_s"] for row in valid)
    return {"rows": valid, "count": len(valid), "retained": len(rows),
            "invalid": len(rows) - len(valid), "complete": complete, "feasible": feasible,
            "means": {key: mean(row[key] for row in valid) if valid else None for key in metrics}}


def _matched_methods(summaries: dict, methods: list[str]) -> bool:
    for repetition in range(5):
        rows = [next(row for row in summaries[method]["rows"] if row["repetition"] == repetition)
                for method in methods]
        keys = [(row.get("workload_sha256"), row.get("session_path"), row.get("power_field")) for row in rows]
        if any(not all(key) for key in keys) or len(set(keys)) != 1:
            return False
    return bool(methods)


def _live_schedule_summary(rows: list[dict], protocol: dict, comparisons: list[dict], schedule: str) -> str:
    targets = protocol["latency_targets"]
    methods = protocol["methods"]
    summaries = {method: _method_measurements([r for r in rows if r.get("method") == method
                                              and r.get("schedule") == schedule], targets) for method in methods}
    feasible = [method for method in methods if summaries[method]["feasible"]]
    matched = _matched_methods(summaries, feasible)
    if matched:
        ordered = sorted(feasible, key=lambda method: (summaries[method]["means"]["energy_per_request_j"], method))
        lowest = ordered[0]
        minimum_energy = summaries[lowest]["means"]["energy_per_request_j"]
        leaders = [method for method in ordered if summaries[method]["means"]["energy_per_request_j"] == minimum_energy]
        lead = " and ".join(_live_method_label(method, protocol) for method in leaders)
        description = (f"Among {len(feasible)} methods with five matched repetitions meeting both frozen latency targets, "
                       f"{lead} {'has' if len(leaders) == 1 else 'tie for'} the lowest observed mean GPU energy: "
                       f"{minimum_energy:.1f} J/request. "
                       "This is a descriptive ordering; supported differences are reported separately below.")
    elif feasible:
        description = "Methods meet both latency targets individually, but their workload, session, or power channel is unmatched. No cross-method energy ordering is reported."
    else:
        description = "No method has five complete valid repetitions meeting both frozen latency targets for this schedule."
    table_rows = []
    # A measured simple baseline may lead; incomplete methods remain in the table.
    order = sorted(methods, key=lambda method: (not summaries[method]["feasible"],
                   summaries[method]["means"]["energy_per_request_j"] or math.inf, method)) if matched else methods
    for method in order:
        summary = summaries[method]
        values = summary["means"]
        status = "Meets both targets in 5/5" if summary["feasible"] else (
            "Misses a frozen target" if summary["complete"] else "Incomplete; feasibility undetermined")
        energy = f"{values['energy_per_request_j']:.1f}" if values["energy_per_request_j"] is not None else "—"
        ttft = f"{values['p95_ttft_s']:.3f}" if values["p95_ttft_s"] is not None else "—"
        itl = f"{1000 * values['p95_mean_itl_s']:.1f}" if values["p95_mean_itl_s"] is not None else "—"
        table_rows.append(f"<tr><td>{escape(_live_method_label(method, protocol))}</td><td>{energy}</td><td>{ttft}</td><td>{itl}</td>"
                          f"<td>{summary['count']}/5 valid; {summary['invalid']} invalid; {summary['retained']} retained</td><td>{status}</td></tr>")
    supported = []
    for comparison in comparisons:
        candidate, reference = comparison.get("candidate"), comparison.get("reference")
        metric = comparison.get("metrics", {}).get("energy_j", {})
        interval = metric.get("reduction_percent_ci95", [])
        if (comparison.get("schedule") != schedule or candidate not in summaries or reference not in summaries
                or comparison.get("positive_energy_saving_supported") is not True
                or comparison.get("eligible_for_efficiency_claim") is not True
                or comparison.get("pair_count") != 5 or comparison.get("paired_repetitions") != list(range(5))
                or not all(summaries[m]["feasible"] for m in (candidate, reference))
                or not _matched_methods(summaries, [candidate, reference])
                or len(interval) != 2 or not all(_positive_number(v) for v in interval)
                or interval[0] > interval[1] or not _positive_number(metric.get("reduction_percent"))):
            continue
        supported.append(f"{escape(_live_method_label(candidate, protocol))} versus {escape(_live_method_label(reference, protocol))}: "
                         f"{metric['reduction_percent']:.1f}% lower total GPU energy "
                         f"(95% paired bootstrap interval {interval[0]:.1f}% to {interval[1]:.1f}%).")
    significance = ("<ul>" + "".join(f"<li>{claim}</li>" for claim in supported) + "</ul>" if supported else
                    "<p>No positive energy-saving difference passes the audited five-pair, both-latency-target, and 95% interval gates.</p>")
    return (f"<h3>{escape(schedule.capitalize())} arrivals</h3><p>{escape(description)}</p>"
            '<div class="comparison-wrap"><table class="comparison"><thead><tr><th>Method</th><th>Mean J/request</th>'
            '<th>Mean p95 TTFT (s)</th><th>Mean p95 mean ITL (ms)</th><th>Repetitions</th><th>Frozen-target feasibility</th></tr></thead>'
            '<tbody>' + "".join(table_rows) + '</tbody></table></div>' + significance)


def _live_figure(entry: dict, directory: Path, stem: str, alt: str) -> str:
    candidates = [Path(path) for path in entry.get("figure_paths", []) if Path(path).name == stem + ".svg"]
    if len(candidates) != 1:
        return f"<p>{escape(alt)} is unavailable: the figure manifest has no unique SVG.</p>"
    path = candidates[0]
    if not path.is_absolute():
        path = directory / path
    if not path.is_file():
        # Generated manifests also work after moving the repository checkout.
        path = ROOT / "_static/inference_serving" / path.name
    if not path.is_file():
        return f"<p>{escape(alt)} is unavailable: its recorded SVG file is missing.</p>"
    image = base64.b64encode(path.read_bytes()).decode("ascii")
    caption_path = directory / (stem + ".md")
    caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.is_file() else alt
    return (f'<figure class="measured-figure"><img src="data:image/svg+xml;base64,{image}" alt="{escape(alt, quote=True)}">'
            f'<figcaption>{escape(caption).replace(chr(10), "<br>")}</figcaption></figure>')


def live_results_section(directory: Path = LIVE_DIRECTORY) -> str:
    pending = ('<p class="section-lede">The live measurement pipeline is implemented, but a validated controller comparison is not yet available. '
               'Confirmation requires qualified baselines and a protocol frozen on development data. '
               'Its 40 W power cap requires fresh calibration; the historical batch curves shown here were measured at 64.8 W. '
               'The planned comparison evaluates maximum clock, a fixed clock selected on development requests, the governor, and MPC using the same fresh requests and latency targets. '
               'Energy includes the time needed to finish every request. The measured winner can be a simple fixed setting.</p>')
    results_path, figures_path = directory / "live_results.json", directory / "live_figures_manifest.json"
    if not results_path.is_file() or not figures_path.is_file():
        return pending
    try:
        results, manifest = _json_object(results_path), _json_object(figures_path)
    except (OSError, ValueError) as error:
        return pending + f"<p>Measured artifacts could not be read: {escape(str(error))}</p>"
    if (results.get("source") != "direct_l4_measurement" or results.get("synthetic_substitution") is not False
            or manifest.get("source") != "supplied_audited_confirmation_sessions" or manifest.get("synthetic_substitution") is not False):
        return pending + "<p>The supplied artifacts do not establish audited direct L4 confirmation provenance.</p>"
    confirmation_rows = [row for row in results.get("trials", []) if row.get("role") == "validation"]
    if confirmation_rows and not any(row.get("valid") is True for row in confirmation_rows):
        return pending + (f"<p>All {len(confirmation_rows)} supplied confirmation observations are invalid and remain in the audited source data. "
                          "They do not establish a measured controller comparison.</p>")
    sections = []
    for digest, protocol in sorted(results.get("protocols", {}).items()):
        rows = [r for r in results.get("trials", []) if r.get("protocol_sha256") == digest and r.get("role") == "validation"]
        targets = protocol.get("latency_targets", {})
        if not rows or not all(_positive_number(targets.get(k)) for k in ("ttft_s", "mean_itl_s")):
            continue
        entries = [entry for entry in manifest.get("figures", []) if entry.get("protocol_sha256") == digest]
        comparisons = [c for c in results.get("comparisons", []) if c.get("protocol_sha256") == digest]
        invalid = sum(r.get("valid") is not True for r in rows)
        section = (f'<article class="live-protocol"><h3>Frozen protocol {escape(digest[:12])}</h3>'
                   f'<p>{len(rows)} confirmation observations retained, including {invalid} invalid observations. '
                   f'Frozen targets: p95 time to first token ≤ {targets["ttft_s"]:.3f} s and p95 of per-request mean inter-token latency ≤ {targets["mean_itl_s"] * 1000:.1f} ms. '
                   'Energy includes request drain. Table entries are means across valid repetitions; partial means do not establish feasibility. '
                   'Invalid and incomplete records remain in the audited source data.</p>')
        for schedule in ("steady", "bursty"):
            section += _live_schedule_summary(rows, protocol, comparisons, schedule)
        mpc = [row for row in rows if row.get("method") == "mpc"]
        if mpc and all(row.get("decision_trace_available") is True and isinstance(row.get("fallback_count"), int) for row in mpc):
            fallbacks = sum(row["fallback_count"] for row in mpc)
            decisions = sum(row.get("decision_count", 0) for row in mpc)
            optimizations = sum(row.get("optimization_decision_count", 0) for row in mpc)
            section += f"<p>MPC used fallback on {fallbacks}/{decisions} decisions across all retained MPC observations; {optimizations} optimization decisions were exercised. Comparisons describe the complete controller including fallback."
            if not optimizations:
                section += " These measurements cannot demonstrate an MPC optimizer advantage."
            section += "</p>"
        elif "mpc" in protocol.get("methods", []):
            section += "<p>MPC fallback coverage is incomplete or unavailable; an optimizer advantage cannot be attributed from these records.</p>"
        if len(entries) == 1 and entries[0].get("rendered") is True:
            entry = entries[0]
            section += _live_figure(entry, directory, f"live-controller-outcomes-{digest[:12]}", "Measured energy and latency outcomes under steady and bursty arrivals")
            if entry.get("trace_rendered") is True:
                section += _live_figure(entry, directory, f"live-controller-rep2-bursty-{digest[:12]}", "Prespecified repetition 2 bursty trace of power, clocks, and outstanding requests")
            else:
                reasons = "; ".join(map(str, entry.get("trace_unavailable_reasons", []))) or "incomplete trace coverage"
                section += f"<p>The prespecified repetition 2 bursty trace is unavailable: {escape(reasons)}. No other repetition is substituted.</p>"
        else:
            section += "<p>Measured figures are unavailable for this protocol; incomplete or invalid observations are retained above.</p>"
        sections.append(section + "</article>")
    if not sections:
        return pending
    return ('<div id="live-control-results"><h2 id="live-control-title">Measured live controller comparison</h2>'
            '<p>These are fresh hardware measurements, separate from the five model-based replays that follow. '
            'Each arrival schedule is assessed separately at its frozen latency targets; a fixed clock or governor can lead.</p>'
            + "".join(sections) + '</div>')


def load_adaptive_report(directory: Path) -> tuple[dict, bytes]:
    """Accept a closed, hash-verified report without changing confirmation gates."""
    directory = Path(directory).resolve()
    summary_bytes = (directory / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    manifest = _json_object(directory / "manifest.json")
    page = (directory / "index.html").read_bytes()
    for name, data in (("summary.json", summary_bytes), ("index.html", page)):
        if manifest.get("output_sha256", {}).get(name) != hashlib.sha256(data).hexdigest():
            raise ValueError("Adaptive report hash mismatch: " + name)
    count = summary.get("audited_valid_trial_count")
    retained = summary.get("recorded_outcome_count")
    acquisitions = summary.get("acquisition_outcomes")
    if (summary.get("schema_version") != 1 or manifest.get("schema_version") != 1
            or summary.get("all_acquisitions_closed") is not True
            or not isinstance(acquisitions, list) or not acquisitions
            or any(row.get("status") not in ("complete", "failed") or row.get("pending") for row in acquisitions)
            or not isinstance(count, int) or isinstance(count, bool) or count <= 0
            or not isinstance(retained, int) or isinstance(retained, bool) or retained < count
            or not summary.get("input_sources") or manifest.get("inputs") != summary["input_sources"]
            or summary.get("unmeasured_is_not_zero") is not True):
        raise ValueError("Adaptive gallery entry requires a closed report with independently valid trials")
    return summary, page


def adaptive_results_section(directory: Path | None = None) -> str:
    if directory is None:
        return ""
    summary, _ = load_adaptive_report(directory)
    context = summary.get("study_context") or {}
    revision_note = ""
    if context.get("revision"):
        variants = context.get("source_variants") or {}
        timing = "; ".join(source.capitalize() + " " + str(variant) for source, variant in sorted(variants.items()))
        revision_note = '<p>Primary workload timing: ' + escape(timing) + '. '
        count = context.get("expected_global_pair_count")
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            revision_note += f'The protocol retains {count} global paired groups per comparator, including when acquisitions are separated by source. '
        revision_note += 'Completed trials that fail a post-trial readiness gate remain descriptive evidence and cannot support selection.</p>'
    return (
        '<section class="explanation" id="adaptive-study-results" aria-labelledby="adaptive-study-title">'
        '<p class="section-kicker">Exploratory hardware evidence</p>'
        '<h2 id="adaptive-study-title">Exploratory broader-workload comparison</h2>'
        f'<p class="section-lede">The final independent audit retains {summary["audited_valid_trial_count"]} valid completed trials '
        f'and {summary["recorded_outcome_count"]} total outcomes; the report separates diagnostic probes, sustained workload screening, '
        'and any completed adaptive comparisons, preserving invalid and unstarted outcomes.</p>'
        + revision_note +
        '<p>This study is separate from the original governor/MPC confirmation protocol and the five model-based replays.</p>'
        '<a class="action" href="gpu-adaptive-study.html">Open the audited exploratory study</a>'
        '</section>')


def build_gallery(live_directory: Path = LIVE_DIRECTORY, adaptive_report: Path | None = None) -> str:
    if not MATHJAX_BUNDLE.is_file():
        raise FileNotFoundError(
            "The vendored MathJax bundle is missing: " + str(MATHJAX_BUNDLE)
        )
    mathjax_bundle = MATHJAX_BUNDLE.read_text(encoding="utf-8").replace(
        "</script", "<\\/script"
    )
    tabs = "\n".join(
        _tab(demo["view"], demo["label"], index)
        for index, demo in enumerate(DEMOS)
    )
    panels = "\n".join(_panel(demo, index) for index, demo in enumerate(DEMOS))
    batch_figure = base64.b64encode(
        (ROOT / "_static/inference_serving/batch-calibration.svg").read_bytes()
    ).decode("ascii")
    live_evidence = live_results_section(live_directory)
    has_measured_results = 'id="live-control-results"' in live_evidence
    measured_lead = (f'<section class="explanation" aria-labelledby="live-control-title">{live_evidence}</section>'
                     if has_measured_results else "")
    calibration_status = "" if has_measured_results else live_evidence
    adaptive_evidence = adaptive_results_section(adaptive_report)
    return rf'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <meta name="description" content="Five recorded GPU inference-control demos: modeling, open-loop planning, MPC, dynamic programming, and fitted Q iteration.">
  <title>GPU Inference Control Demos</title>
  <script>{MATHJAX_CONFIG}</script>
  <script data-mathjax-version="3.2.2">{mathjax_bundle}</script>
  <style>
    :root {{
      --page: #f2f7fa;
      --surface: #f9fcfd;
      --surface-raised: #ffffff;
      --ink: #142b3d;
      --muted: #536b7a;
      --line: #c9dae4;
      --accent: #087da2;
      --accent-soft: #e1f2f7;
      --warning: #a76509;
      --shadow: 0 18px 48px rgb(29 67 88 / 0.10);
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      min-width: 19rem;
      background: var(--page);
      color: var(--ink);
      font-size: 1rem;
      line-height: 1.55;
    }}
    button, a {{ font: inherit; }}
    button:focus-visible, a:focus-visible {{
      outline: 3px solid color-mix(in srgb, var(--accent) 58%, transparent);
      outline-offset: 3px;
    }}
    .page-shell {{ width: min(100% - 2rem, 76rem); margin: 0 auto; padding: 2rem 0 4rem; }}
    .masthead {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 1.5rem;
      align-items: start;
      border-top: 3px solid var(--accent);
      padding-top: 1.4rem;
    }}
    .eyebrow {{
      margin: 0 0 .55rem;
      color: var(--accent);
      font-size: .79rem;
      font-weight: 750;
      letter-spacing: .13em;
      text-transform: uppercase;
    }}
    h1 {{ margin: 0; max-width: 15ch; font-size: clamp(2.1rem, 5vw, 4.25rem); line-height: .98; letter-spacing: -.045em; }}
    .lede {{ max-width: 46rem; margin: 1.15rem 0 0; color: var(--muted); font-size: 1.05rem; }}
    .actions {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: .55rem; }}
    .action {{
      display: inline-flex;
      min-height: 2.65rem;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--line);
      border-radius: .35rem;
      padding: .55rem .8rem;
      color: var(--ink);
      background: var(--surface);
      text-decoration: none;
      cursor: pointer;
    }}
    .action:hover {{ border-color: var(--accent); }}
    .facts {{ display: flex; flex-wrap: wrap; gap: .45rem; margin: 1.4rem 0 0; padding: 0; list-style: none; }}
    .facts li {{ border: 1px solid var(--line); border-radius: 999px; padding: .32rem .65rem; color: var(--muted); font-size: .84rem; }}
    .caveat {{
      margin: 1.35rem 0 0;
      border-left: 3px solid var(--warning);
      padding: .15rem 0 .15rem .8rem;
      color: var(--muted);
      font-size: .9rem;
    }}
    .case-map {{
      display: grid;
      grid-template-columns: 1.15fr 1fr 1fr;
      gap: 1px;
      margin-top: 1.5rem;
      border: 1px solid var(--line);
      border-radius: .55rem;
      overflow: hidden;
      background: var(--line);
    }}
    .case-map article {{ padding: 1rem; background: var(--surface); }}
    .case-map h2, .case-map h3 {{ margin: 0 0 .35rem; font-size: .88rem; color: var(--accent); text-transform: uppercase; letter-spacing: .08em; }}
    .case-map p {{ margin: 0; color: var(--muted); font-size: .92rem; }}
    .workspace {{ display: grid; grid-template-columns: 14.5rem minmax(0, 1fr); gap: 1rem; margin-top: 2.2rem; align-items: start; }}
    .demo-tabs {{
      position: sticky;
      top: 1rem;
      display: grid;
      gap: .4rem;
      border: 1px solid var(--line);
      border-radius: .55rem;
      padding: .5rem;
      background: var(--surface);
    }}
    .demo-tab {{
      display: grid;
      grid-template-columns: 2.1rem 1fr;
      gap: .5rem;
      align-items: center;
      width: 100%;
      min-height: 3.25rem;
      border: 1px solid transparent;
      border-radius: .35rem;
      padding: .45rem .55rem;
      color: var(--muted);
      background: transparent;
      text-align: left;
      cursor: pointer;
    }}
    .demo-tab:hover {{ color: var(--ink); background: var(--surface-raised); }}
    .demo-tab[aria-selected="true"] {{ color: var(--ink); border-color: var(--accent); background: var(--accent-soft); }}
    .tab-index {{ color: var(--accent); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .78rem; }}
    .demo-panel {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: .6rem;
      padding: clamp(.75rem, 2vw, 1.25rem);
      background: var(--surface-raised);
      box-shadow: var(--shadow);
    }}
    .demo-panel[hidden] {{ display: none; }}
    .demo-intro {{ margin-bottom: 1.25rem; }}
    .demo-stage {{ margin: 0 0 .3rem; color: var(--accent); font-size: .78rem; font-weight: 750; letter-spacing: .1em; text-transform: uppercase; }}
    .demo-intro h2 {{ margin: 0; max-width: 34ch; font-size: clamp(1.35rem, 3vw, 2rem); line-height: 1.1; letter-spacing: -.025em; }}
    .demo-intro h3 {{ margin: 0 0 .35rem; font-size: .78rem; color: var(--accent); letter-spacing: .08em; text-transform: uppercase; }}
    .demo-intro p {{ margin: 0; }}
    .demo-method {{
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 1rem;
      margin-top: 1rem;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: .9rem 0;
    }}
    .demo-method p {{ color: var(--muted); font-size: .94rem; }}
    .demo-method .equation {{
      display: flex;
      min-height: 4.25rem;
      align-items: center;
      overflow-x: auto;
      border-left: 2px solid var(--warning);
      padding: .55rem .7rem;
      color: var(--ink);
      background: var(--surface);
      font-size: 1.02rem;
    }}
    .equation mjx-container {{ margin: 0 !important; min-width: max-content; }}
    .equation mjx-container[jax="SVG"] > svg {{ overflow: visible; }}
    .demo-guides {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .65rem; margin-top: .75rem; }}
    .demo-guides > div {{ border: 1px solid var(--line); border-radius: .35rem; padding: .75rem; background: var(--surface); }}
    .demo-guides p {{ color: var(--muted); font-size: .89rem; }}
    .demo-guides .try-card {{ border-color: color-mix(in srgb, var(--accent) 58%, var(--line)); background: var(--accent-soft); }}
    .demo-panel .inference-replay {{ max-width: none !important; border-radius: .35rem; }}
    .explanation {{ width: min(100%, 60rem); margin: 4rem auto 0; }}
    .section-kicker {{ margin: 0 0 .45rem; color: var(--accent); font-size: .78rem; font-weight: 750; letter-spacing: .11em; text-transform: uppercase; }}
    .explanation h2 {{ margin: 0; font-size: clamp(1.7rem, 4vw, 2.7rem); line-height: 1.05; letter-spacing: -.035em; }}
    .explanation .section-lede {{ max-width: 49rem; margin: .8rem 0 0; color: var(--muted); font-size: 1.03rem; }}
    .measured-figure {{ margin: 1.4rem 0 0; padding: .8rem; background: #fff; border: 1px solid var(--line); border-radius: .4rem; }}
    .measured-figure img {{ display: block; width: 100%; height: auto; }}
    .measured-figure figcaption {{ margin-top: .6rem; color: var(--muted); font-size: .88rem; }}
    .problem-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.4rem; }}
    .problem-card {{ border-top: 2px solid var(--accent); padding-top: .8rem; }}
    .problem-card h3 {{ margin: 0 0 .4rem; font-size: 1.05rem; }}
    .problem-card p {{ margin: 0; color: var(--muted); }}
    .math-stack {{ display: grid; gap: .7rem; margin-top: 1.4rem; }}
    .math-row {{ display: grid; grid-template-columns: 2.1rem minmax(14rem, .85fr) 1.4fr; gap: .8rem; align-items: start; border-bottom: 1px solid var(--line); padding: 0 0 .8rem; }}
    .math-number {{ color: var(--accent); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8rem; }}
    .math-row .equation {{ margin: 0; overflow-x: auto; color: var(--ink); font-size: 1.08rem; }}
    .math-row .meaning {{ margin: 0; color: var(--muted); font-size: .94rem; }}
    .comparison-wrap {{ margin-top: 1.4rem; overflow-x: auto; border: 1px solid var(--line); border-radius: .5rem; }}
    .comparison {{ width: 100%; border-collapse: collapse; min-width: 46rem; }}
    .comparison th, .comparison td {{ padding: .75rem; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    .comparison th {{ color: var(--accent); background: var(--surface); font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; }}
    .comparison td {{ color: var(--muted); font-size: .91rem; }}
    .comparison td:first-child {{ color: var(--ink); font-weight: 700; }}
    .comparison tr:last-child td {{ border-bottom: 0; }}
    .result-strip {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem; margin-top: 1.4rem; }}
    .result {{ border: 1px solid var(--line); border-radius: .4rem; padding: .9rem; background: var(--surface); }}
    .result strong {{ display: block; color: var(--ink); font-size: 1.18rem; line-height: 1.15; }}
    .result span {{ display: block; margin-top: .35rem; color: var(--muted); font-size: .86rem; }}
    .pilot {{ display: grid; grid-template-columns: 1.1fr 1fr; gap: 1.4rem; align-items: start; }}
    .pilot-steps {{ margin: 1.2rem 0 0; padding: 0; list-style: none; counter-reset: pilot; }}
    .pilot-steps li {{ position: relative; margin-bottom: .75rem; padding: 0 0 .75rem 2.7rem; border-bottom: 1px solid var(--line); color: var(--muted); counter-increment: pilot; }}
    .pilot-steps li::before {{ content: counter(pilot, decimal-leading-zero); position: absolute; left: 0; top: 0; color: var(--accent); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8rem; }}
    .pilot-steps strong {{ color: var(--ink); }}
    .guardrail {{ margin-top: 1.2rem; border: 1px solid var(--line); border-top: 3px solid var(--warning); border-radius: .4rem; padding: 1rem; background: var(--surface); }}
    .guardrail h3 {{ margin: 0 0 .5rem; font-size: 1rem; }}
    .guardrail p {{ margin: 0; color: var(--muted); font-size: .92rem; }}
    .closing-action {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1rem; margin-top: 1.3rem; border: 1px solid var(--accent); border-radius: .45rem; padding: 1rem; background: var(--accent-soft); }}
    .closing-action p {{ margin: 0; max-width: 40rem; }}
    .page-footer {{ margin-top: 1.6rem; color: var(--muted); font-size: .82rem; text-align: right; }}
    @media (max-width: 800px) {{
      .masthead {{ grid-template-columns: 1fr; }}
      .actions {{ justify-content: flex-start; }}
      .case-map {{ grid-template-columns: 1fr; }}
      .workspace {{ grid-template-columns: 1fr; }}
      .demo-tabs {{ position: static; grid-template-columns: repeat(5, minmax(9rem, 1fr)); overflow-x: auto; }}
      .demo-method, .pilot {{ grid-template-columns: 1fr; }}
      .demo-guides, .result-strip {{ grid-template-columns: 1fr; }}
      .math-row {{ grid-template-columns: 2rem 1fr; }}
      .math-row .meaning {{ grid-column: 2; }}
    }}
    @media (max-width: 520px) {{
      .page-shell {{ width: min(100% - 1rem, 76rem); padding-top: 1rem; }}
      h1 {{ font-size: 2.5rem; }}
      .action {{ flex: 1 1 auto; }}
    }}
    @media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto; }} }}
    @media print {{
      :root {{ --page: #fff; --surface: #fff; --surface-raised: #fff; --ink: #111; --muted: #444; --line: #bbb; --shadow: none; }}
      .actions, .demo-tabs {{ display: none; }}
      .workspace {{ display: block; }}
      .demo-panel, .demo-panel[hidden] {{ display: block; break-inside: avoid; margin-bottom: 1rem; box-shadow: none; }}
    }}
  </style>
</head>
<body>
  <main class="page-shell">
    <header class="masthead">
      <div>
        <p class="eyebrow">RL &amp; Control · GPU case study</p>
        <h1>GPU inference control demos</h1>
        <p class="lede">A language-model server has to move irregular request bursts through prefill and decode without wasting energy or violating operating limits. These five replays show how the same system becomes a trajectory-optimization problem, a feedback controller, a finite Markov decision process, and an offline reinforcement-learning experiment.</p>
        <ul class="facts" aria-label="Demo provenance">
          <li>NVIDIA L4 profile</li>
          <li>Qwen2.5-7B-Instruct</li>
          <li>Azure request trace</li>
          <li>Recorded trajectories</li>
          <li>Five control views</li>
        </ul>
        <p class="caveat">Measured clock and power data calibrate the serving model. The five replayed controller trajectories are model-based, and the one-state thermal fit used by those replays remains weak; the demos do not establish hardware safety guarantees.</p>
      </div>
      <div class="actions" aria-label="Page actions">
        <a class="action" href="#method">Method &amp; evidence</a>
        <a class="action" href="gpu-demos.html" download="gpu-inference-control-demos.html">Download HTML</a>
      </div>
    </header>

    <section class="case-map" aria-label="Case study in brief">
      <article>
        <h2>The problem</h2>
        <p>Changing the GPU clock affects service speed and board power. We compare energy for completed work while requiring each method to meet the same latency targets.</p>
      </article>
      <article>
        <h3>What we changed</h3>
        <p>The GPU clock or the prefill/decode scheduling decision, depending on the experiment.</p>
      </article>
      <article>
        <h3>What we observed</h3>
        <p>Queues, cache, completed work, power, temperature, and realized clock at the information available online.</p>
      </article>
    </section>

{adaptive_evidence}{measured_lead}

    <div class="workspace">
      <nav class="demo-tabs" role="tablist" aria-label="GPU inference demos">
        {tabs}
      </nav>
      <div class="demo-panels">
        {panels}
      </div>
    </div>

    <section class="explanation" id="method">
      <p class="section-kicker">Problem formulation</p>
      <h2>Why inference serving is a control problem</h2>
      <p class="section-lede">A decoder-only model does not process every token in the same way. Prefill processes prompt tokens and creates the key-value cache; decode then generates output one token at a time. A long prefill can delay the short decode iterations of requests already in flight. Meanwhile, the GPU clock changes both the rate and energy cost of that work.</p>
      <div class="problem-grid">
        <article class="problem-card">
          <h3>The disturbance is timing</h3>
          <p>Requests arrive irregularly, prompt sizes vary, and the eventual output length is unknown until completion. The controller must act before the whole workload is revealed.</p>
        </article>
        <article class="problem-card">
          <h3>The tradeoff is dynamic</h3>
          <p>A low clock can be sensible while the system is empty and harmful during a burst. The useful decision depends on the current queue, cache pressure, temperature, and work likely to arrive next.</p>
        </article>
        <article class="problem-card">
          <h3>The model mixes laws and data</h3>
          <p>Token and cache conservation determine how work moves. An NVIDIA L4 profile supplies the phase-specific service rates, board power, and requested-versus-realized clock behavior that conservation alone cannot predict.</p>
        </article>
        <article class="problem-card">
          <h3>The comparison holds the plant fixed</h3>
          <p>The same requests and reduced serving dynamics appear across the controllers. Changing the decision method, rather than the workload, exposes why their trajectories differ.</p>
        </article>
      </div>
    </section>

    <section class="explanation" aria-labelledby="math-title">
      <p class="section-kicker">Mathematical spine</p>
      <h2 id="math-title">Four objects connect the five demos</h2>
      <p class="section-lede">The notation changes with the decision interface, but the logic stays compact: represent what matters now, state what can be changed, predict what follows, and attach a cost to the consequences.</p>
      <div class="math-stack">
        <div class="math-row">
          <span class="math-number">01</span>
          <p class="equation">\(x_{{t+1}}=F(x_t,u_t,w_t)\)</p>
          <p class="meaning"><strong>Dynamics.</strong> The next queue, cache, and thermal state follows from the current state, the chosen clock or schedule, and new requests. The disturbance \(w_t\) keeps arrivals separate from actions.</p>
        </div>
        <div class="math-row">
          <span class="math-number">02</span>
          <p class="equation">\(\displaystyle \min_{{u_{{0:T-1}}}}\sum_{{t=0}}^{{T-1}}c(x_t,u_t)+c_T(x_T)\)</p>
          <p class="meaning"><strong>Finite-horizon planning.</strong> Running cost prices energy, waiting, clock movement, and limit violations. Terminal cost prevents a planner from appearing efficient by leaving work just beyond the horizon.</p>
        </div>
        <div class="math-row">
          <span class="math-number">03</span>
          <p class="equation">\(u_t\leftarrow\operatorname{{first}}(\operatorname{{plan}}(x_t))\)</p>
          <p class="meaning"><strong>Feedback.</strong> Open loop computes the sequence once. MPC measures \(x_t\), recomputes a short plan, applies the first action, and repeats. The new measurement changes later decisions.</p>
        </div>
        <div class="math-row">
          <span class="math-number">04</span>
          <p class="equation">\(\displaystyle V(s)=\min_u\left[c(s,u)+\gamma\,\mathbb{{E}}\,V(S')\right]\)</p>
          <p class="meaning"><strong>State-contingent decisions.</strong> Dynamic programming computes an action for every finite reduced state. Fitted Q-iteration approximates the same backup from sampled transitions when the transition matrix is not directly available.</p>
        </div>
      </div>
    </section>

    <section class="explanation" aria-labelledby="comparison-title">
      <p class="section-kicker">Approach</p>
      <h2 id="comparison-title">Each method pays for different information</h2>
      <p class="section-lede">The methods are not a ladder with one universal winner. They move computation, modeling assumptions, and data requirements to different places.</p>
      <div class="comparison-wrap">
        <table class="comparison">
          <thead><tr><th>Method</th><th>Information used</th><th>Computation</th><th>What it buys</th><th>Main failure mode</th></tr></thead>
          <tbody>
            <tr><td>Open loop</td><td>Initial state and a complete arrival forecast</td><td>One optimization before execution</td><td>A globally coordinated clock schedule for the assumed trace</td><td>The plan cannot react when arrival timing changes</td></tr>
            <tr><td>MPC</td><td>Current measured state plus a short forecast</td><td>One constrained solve per control interval</td><td>Feedback without discarding the predictive model</td><td>Model error, solve deadlines, and infeasibility require fallbacks</td></tr>
            <tr><td>Exact DP</td><td>A finite state, cost, and full transition matrix</td><td>Value iteration offline</td><td>A scheduling action available immediately for every modeled state</td><td>Exactness applies only to the reduced state and transition model</td></tr>
            <tr><td>Fitted Q</td><td>A fixed table of sampled state-action transitions</td><td>Repeated Bellman target construction and regression</td><td>A policy without enumerating a known transition matrix</td><td>Unsupported regions inherit approximation and coverage error</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="explanation" aria-labelledby="evidence-title">
      <p class="section-kicker">Evidence in the replays</p>
      <h2 id="evidence-title">What the controlled comparisons reveal</h2>
      <p class="section-lede">Three results carry the experimental argument. Each isolates one change while holding the surrounding model and evaluation protocol fixed.</p>
      <div class="result-strip">
        <div class="result"><strong>+6.88 s mean TTFT</strong><span>Moving one 31-request burst earlier raises the fixed open-loop plan from 16.35 to 23.23 seconds.</span></div>
        <div class="result"><strong>11.53 s at 3,341.8 J</strong><span>MPC on the shifted trace, compared with 10.93 s at 3,506.0 J for maximum clock.</span></div>
        <div class="result"><strong>68.16% disagreement</strong><span>Narrow-buffer FQI covers 13.86% of feasible pairs and differs from exact DP on more than two thirds of states.</span></div>
      </div>
      <div class="problem-grid">
        <article class="problem-card">
          <h3>Forecast error changes the right action</h3>
          <p>The open-loop plan lowers its clock when the nominal trace contains little work. Moving a burst into that interval creates a queue that the fixed plan cannot see. Feedback sees the queue and moves the clock response earlier.</p>
        </article>
        <article class="problem-card">
          <h3>Latency and energy must remain separate</h3>
          <p>Maximum clock gives the lowest mean time to first token on this trace. MPC gives up 0.60 seconds and consumes 164.2 J less. A single score would hide the operating choice between these outcomes.</p>
        </article>
        <article class="problem-card">
          <h3>A large buffer can still be narrow</h3>
          <p>Both FQI buffers contain 50,000 transitions. Their difference comes from where those rows lie in state-action space. Equal row counts do not imply equal support for counterfactual decisions.</p>
        </article>
        <article class="problem-card">
          <h3>Exact optimization can certify the wrong abstraction</h3>
          <p>The DP residual is below 10⁻¹⁰, so the finite Bellman equation is solved accurately. Request identities, exact lengths, and richer thermal dynamics remain absent from that finite equation.</p>
        </article>
      </div>
    </section>

    <section class="explanation" id="measured-evidence" aria-labelledby="batch-title">
      <p class="section-kicker">Measured calibration</p>
      <h2 id="batch-title">Batch size changes the useful clock decision</h2>
      <p class="section-lede">At a requested 2,040 MHz, eight concurrent requests generate 105.6 tokens per second, compared with 17.5 for one request at the same prompt length. A single pooled throughput curve misses that difference. The high-clock plateau also makes a carefully selected constant clock a necessary baseline.</p>
      <figure class="measured-figure">
        <img src="data:image/svg+xml;base64,{batch_figure}" alt="Measured prefill and decode throughput at five requested clocks, with separate curves for one, four, and eight concurrent requests. Decode capacity increases with batch size and plateaus above 1,125 MHz.">
        <figcaption>Recorded L4 profile from 2 September 2026. Each point is the median of five concurrent batches; whiskers show their minimum and maximum. Context is fixed within each panel. These are calibration measurements, not live controller results.</figcaption>
      </figure>
      {calibration_status}
    </section>

    <section class="explanation pilot" aria-labelledby="pilot-title">
      <div>
        <p class="section-kicker">From replay to practice</p>
        <h2 id="pilot-title">A small pilot can test whether the idea survives contact with your stack</h2>
        <p class="section-lede">The practical next step is not an autonomous production controller. It is a shadow-mode experiment that turns your existing telemetry into matched, replayable comparisons.</p>
        <ol class="pilot-steps">
          <li><strong>Define the boundary.</strong> Choose one model, one serving process, one accelerator type, and the actions the controller may request.</li>
          <li><strong>Record the plant.</strong> Join request timestamps and token counts with queue depth, phase, requested and realized clock, board power, temperature, and cache use.</li>
          <li><strong>Fit only what conservation cannot supply.</strong> Estimate phase-specific service and power maps, then validate each map on held-out workloads.</li>
          <li><strong>Replay matched baselines.</strong> Compare maximum clock, an efficient fixed clock, the current heuristic, a forecast plan, and MPC on identical traces. Report latency, energy, queueing, and violations separately.</li>
          <li><strong>Run in shadow mode.</strong> Compute proposed actions without applying them. Log solve time, infeasibility, fallback triggers, and disagreement with the deployed governor.</li>
          <li><strong>Constrain the first live test.</strong> Use allow-listed clocks, hard vendor limits, a tested fallback, a kill switch, and a workload slice with low blast radius.</li>
        </ol>
      </div>
      <aside class="guardrail">
        <h3>What would count as a useful first result?</h3>
        <p>A useful pilot identifies the lowest-energy method that meets common latency targets, including the possibility that a fixed clock wins. It also measures where the model fails, how often the optimizer misses its deadline, and which telemetry is missing. Fresh repetitions test whether the measured advantage persists.</p>
      </aside>
    </section>

    <section class="explanation" aria-labelledby="scope-title">
      <p class="section-kicker">Scope and limits</p>
      <h2 id="scope-title">What is measured, and what remains modeled?</h2>
      <p class="section-lede">The L4 clock, service-rate, power, and phase-confirmation data are hardware measurements for Qwen2.5-7B-Instruct served by vLLM 0.28.0. The five replayed controller trajectories, their queue transitions, DP episodes, and FQI buffers are generated by the reduced serving model. The original one-state thermal models missed held-out pulse temperatures. A two-state candidate selected from six complete acquisition histories awaits evaluation using power forecasts available when the controller acts. Neither result establishes a hardware safety guarantee. Network transfer, host tokenization, multi-GPU communication, model-quality effects, and production scheduler details remain outside the boundary.</p>
      <div class="closing-action">
        <p><strong>Start with a replay.</strong> Download this single HTML file, choose one method, and ask which logged signals and actions would be available in your own serving stack.</p>
        <a class="action" href="gpu-demos.html" download="gpu-inference-control-demos.html">Download the case study</a>
      </div>
    </section>
    <footer class="page-footer">Recorded inference-serving artifact · Pierre-Luc Bacon</footer>
  </main>
  <script>
    (() => {{
      "use strict";
      const tabs = [...document.querySelectorAll("[data-demo-tab]")];
      const panels = [...document.querySelectorAll("[data-demo-panel]")];
      const views = new Set(tabs.map((tab) => tab.dataset.demoTab));
      const activate = (view, updateHash = true) => {{
        if (!views.has(view)) view = tabs[0].dataset.demoTab;
        tabs.forEach((tab) => {{
          const active = tab.dataset.demoTab === view;
          tab.setAttribute("aria-selected", String(active));
          tab.tabIndex = active ? 0 : -1;
        }});
        panels.forEach((panel) => {{ panel.hidden = panel.dataset.demoPanel !== view; }});
        if (updateHash) history.replaceState(null, "", "#" + view.replaceAll("_", "-"));
      }};
      tabs.forEach((tab, index) => {{
        tab.addEventListener("click", () => activate(tab.dataset.demoTab));
        tab.addEventListener("keydown", (event) => {{
          if (!["ArrowDown", "ArrowUp", "ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          let next = index;
          if (event.key === "Home") next = 0;
          else if (event.key === "End") next = tabs.length - 1;
          else if (["ArrowDown", "ArrowRight"].includes(event.key)) next = (index + 1) % tabs.length;
          else next = (index - 1 + tabs.length) % tabs.length;
          tabs[next].focus();
          activate(tabs[next].dataset.demoTab);
        }});
      }});
      const hashView = location.hash.slice(1).replaceAll("-", "_");
      activate(hashView || tabs[0].dataset.demoTab, false);

      document.querySelectorAll(".inference-replay").forEach((replay) => {{
        replay.dataset.theme = "light";
      }});

    }})();
  </script>
</body>
</html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--live-directory", type=Path, default=LIVE_DIRECTORY)
    parser.add_argument("--adaptive-report", type=Path, help="Optional final audited shareable report directory; copies its self-contained HTML beside the gallery")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    gallery = build_gallery(arguments.live_directory, arguments.adaptive_report)
    adaptive_page = None
    if arguments.adaptive_report is not None:
        _, adaptive_page = load_adaptive_report(arguments.adaptive_report)
        destination = output.parent / "gpu-adaptive-study.html"
        if output == destination or output.is_relative_to(arguments.adaptive_report.resolve()):
            raise ValueError("Gallery output must be separate from the final report")
        if destination.exists() and destination.read_bytes() != adaptive_page:
            raise ValueError("A different linked adaptive report already exists: " + str(destination))
    output.parent.mkdir(parents=True, exist_ok=True)
    if adaptive_page is not None:
        destination.write_bytes(adaptive_page)
    output.write_text(gallery, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
