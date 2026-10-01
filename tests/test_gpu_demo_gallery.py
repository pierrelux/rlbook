"""Structural checks for the standalone GPU demo gallery."""

from html.parser import HTMLParser
import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "interactive" / "gpu-demos.html"
SPEC = importlib.util.spec_from_file_location("gpu_gallery_builder", ROOT / "scripts/build_gpu_demo_gallery.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class _ExternalResourceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.external_resources: list[tuple[str, str]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        resources = {"script": "src", "link": "href", "img": "src", "source": "src"}
        attribute = resources.get(tag)
        if attribute is None:
            return
        value = dict(attrs).get(attribute) or ""
        if value.startswith(("http://", "https://", "//")):
            self.external_resources.append((tag, value))


def test_gpu_gallery_is_single_file_and_contains_every_view() -> None:
    page = GALLERY.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert "GPU inference control demos" in page
    parser = _ExternalResourceParser()
    parser.feed(page)
    assert parser.external_resources == []
    for view in ("modeling", "open_loop", "mpc", "scheduling", "fqi"):
        assert f'data-demo-tab="{view}"' in page
        assert f'data-demo-panel="{view}"' in page
        assert f'id="gpu-demo-{view.replace("_", "-")}"' in page


def test_gpu_gallery_keeps_replays_distinct() -> None:
    page = GALLERY.read_text(encoding="utf-8")
    assert page.count('class="inference-replay"') == 5
    assert page.count('type="application/json"') == 5
    assert page.count('aria-label="Replay controls"') == 5


def test_gpu_gallery_explains_the_problem_math_evidence_and_next_step() -> None:
    page = GALLERY.read_text(encoding="utf-8")
    for heading in (
        "Why inference serving is a control problem",
        "Four objects connect the five demos",
        "Each method pays for different information",
        "What the controlled comparisons reveal",
        "A small pilot can test whether the idea survives contact with your stack",
        "What is measured, and what remains modeled?",
    ):
        assert heading in page
    assert page.count("Mathematical object") == 5
    assert page.count("What to watch") == 5
    assert page.count("What to expect") == 5
    assert page.count("Try it") == 5


def test_gpu_gallery_embeds_mathjax_and_uses_a_light_page_theme() -> None:
    page = GALLERY.read_text(encoding="utf-8")
    assert 'data-mathjax-version="3.2.2"' in page
    assert "window.MathJax" in page
    assert r"\(x_{t+1}=F(x_t,u_t,w_t)\)" in page
    assert '<html lang="en" data-theme="dark">' not in page
    assert "--page: #f2f7fa" in page
    assert 'replay.dataset.theme = "light"' in page


def test_gpu_gallery_distinguishes_batch_measurements_from_pending_live_trials(tmp_path: Path) -> None:
    page = GALLERY.read_text(encoding="utf-8")
    assert 'id="measured-evidence"' in page
    assert 'src="data:image/svg+xml;base64,' in page
    assert "These are calibration measurements, not live controller results." in page
    assert "a validated controller comparison is not yet available" in BUILDER.live_results_section(tmp_path)
    assert "including the possibility that a fixed clock wins" in page
    pending_page = BUILDER.build_gallery(tmp_path)
    assert pending_page.count("a validated controller comparison is not yet available") == 1
    assert 'id="live-control-results"' not in pending_page
    assert pending_page.index('<div class="workspace">') < pending_page.index('id="measured-evidence"') < pending_page.index("a validated controller comparison is not yet available")


def _adaptive_report_fixture(directory: Path, *, closed=True, valid_count=24, status="failed") -> bytes:
    directory.mkdir()
    summary = {"schema_version": 1, "all_acquisitions_closed": closed, "audited_valid_trial_count": valid_count,
               "recorded_outcome_count": 38, "unmeasured_is_not_zero": True,
               "acquisition_outcomes": [{"status": status, "pending": []}],
               "input_sources": [{"path": "audit.json", "sha256": "a" * 64}]}
    summary_bytes = json.dumps(summary).encode()
    page = b'<!doctype html><html><body>Explicit synthetic test fixture; not measured evidence.</body></html>'
    (directory / "summary.json").write_bytes(summary_bytes)
    (directory / "index.html").write_bytes(page)
    (directory / "manifest.json").write_text(json.dumps({"schema_version": 1, "inputs": summary["input_sources"],
        "output_sha256": {"summary.json": hashlib.sha256(summary_bytes).hexdigest(), "index.html": hashlib.sha256(page).hexdigest()}}))
    return page


def test_revised_adaptive_gallery_preserves_source_timing_and_global_count(tmp_path: Path) -> None:
    folder = tmp_path / "report"
    _adaptive_report_fixture(folder)
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["study_context"] = {"revision": "conversation-rate4x-v3",
                                 "source_variants": {"code": "rate_2x", "conversation": "rate_4x"},
                                 "expected_global_pair_count": 8}
    summary_path.write_text(json.dumps(summary))
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"]["summary.json"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    page = BUILDER.adaptive_results_section(folder)
    assert "Code rate_2x; Conversation rate_4x" in page
    assert "8 global paired groups per comparator" in page
    assert "remain descriptive evidence and cannot support selection" in page


def test_optional_adaptive_entry_keeps_confirmation_gate_and_five_replays(tmp_path: Path) -> None:
    report = tmp_path / "report"
    _adaptive_report_fixture(report)
    page = BUILDER.build_gallery(tmp_path, adaptive_report=report)
    section = BUILDER.adaptive_results_section(report)
    assert "24 valid completed trials" in section and "38 total outcomes" in section
    assert "Exploratory broader-workload comparison" in section
    assert 'href="gpu-adaptive-study.html"' in section
    assert page.index('id="adaptive-study-results"') < page.index('<div class="workspace">')
    assert page.count('class="inference-replay"') == 5
    assert "a validated controller comparison is not yet available" in page
    assert 'id="live-control-results"' not in page
    assert page.replace(section, "", 1) == BUILDER.build_gallery(tmp_path)


@pytest.mark.parametrize("options", [{"closed": False}, {"valid_count": 0}, {"status": "running"}])
def test_adaptive_entry_rejects_pending_or_unmeasured_report(tmp_path: Path, options: dict) -> None:
    folder = tmp_path / "report"
    _adaptive_report_fixture(folder, **options)
    with pytest.raises(ValueError, match="closed report"):
        BUILDER.adaptive_results_section(folder)


def test_adaptive_entry_rejects_modified_report_bytes(tmp_path: Path) -> None:
    folder = tmp_path / "report"
    _adaptive_report_fixture(folder)
    with (folder / "index.html").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        BUILDER.adaptive_results_section(folder)


def test_optional_report_cli_copies_exact_html_without_original_writes(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "report"
    page = _adaptive_report_fixture(folder)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    output = tmp_path / "interactive" / "gpu-demos.html"
    monkeypatch.setattr(BUILDER.sys, "argv", ["build_gpu_demo_gallery.py", "--live-directory", str(tmp_path),
                                           "--adaptive-report", str(folder), "--output", str(output)])
    BUILDER.main()
    assert (output.parent / "gpu-adaptive-study.html").read_bytes() == page
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    assert 'id="adaptive-study-results"' in output.read_text()
    (output.parent / "gpu-adaptive-study.html").write_text("different report")
    gallery_before = output.read_bytes()
    with pytest.raises(ValueError, match="different linked adaptive report"):
        BUILDER.main()
    assert output.read_bytes() == gallery_before


def _confirmation_fixture(directory: Path) -> tuple[dict, dict]:
    digest = "0123456789abcdef" * 4
    methods = ["fixed_2040", "fixed_1125", "governor", "mpc"]
    protocol = {"methods": methods, "selected_fixed_clock_mhz": 1125,
                "latency_targets": {"ttft_s": 1.5, "mean_itl_s": .04}}
    rows = []
    for schedule in ("steady", "bursty"):
        for repetition in range(5):
            for method, energy in zip(methods, (100., 60., 70., 80.)):
                rows.append({"protocol_sha256": digest, "role": "validation", "schedule": schedule,
                             "method": method, "repetition": repetition, "valid": True,
                             "energy_per_request_j": energy, "p95_ttft_s": .5, "p95_mean_itl_s": .025,
                             "workload_sha256": f"work-{schedule}-{repetition}", "session_path": "fixture-session",
                             "power_field": "power_instant_w", "decision_trace_available": True,
                             "decision_count": 10, "fallback_count": 10 if method == "mpc" else 0,
                             "optimization_decision_count": 0})
    comparisons = [{"protocol_sha256": digest, "candidate": "fixed_1125", "reference": "fixed_2040",
                    "schedule": schedule, "eligible_for_efficiency_claim": True,
                    "positive_energy_saving_supported": True, "pair_count": 5, "paired_repetitions": list(range(5)),
                    "metrics": {"energy_j": {"reduction_percent": 40., "reduction_percent_ci95": [38., 42.]}}}
                   for schedule in ("steady", "bursty")]
    results = {"source": "direct_l4_measurement", "synthetic_substitution": False,
               "protocols": {digest: protocol}, "trials": rows, "comparisons": comparisons}
    figures = []
    for name in ("live-controller-outcomes", "live-controller-rep2-bursty"):
        path = directory / f"{name}-{digest[:12]}.svg"
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="80"><text x="10" y="40">Test fixture: {name}</text></svg>')
        path.with_suffix(".md").write_text("Temporary fixture caption; no measured result.")
        figures.append(str(path))
    manifest = {"source": "supplied_audited_confirmation_sessions", "synthetic_substitution": False,
                "figures": [{"protocol_sha256": digest, "rendered": True, "trace_rendered": True,
                             "figure_paths": figures, "trace_unavailable_reasons": []}]}
    _write_confirmation(directory, results, manifest)
    return results, manifest


def _write_confirmation(directory: Path, results: dict, manifest: dict) -> None:
    (directory / "live_results.json").write_text(json.dumps(results))
    (directory / "live_figures_manifest.json").write_text(json.dumps(manifest))


def test_live_gallery_embeds_measured_figures_and_all_replays_in_single_file(tmp_path: Path) -> None:
    results, manifest = _confirmation_fixture(tmp_path)
    page = BUILDER.build_gallery(tmp_path)
    assert page.count('class="inference-replay"') == 5
    assert 'data-mathjax-version="3.2.2"' in page
    assert page.count('src="data:image/svg+xml;base64,') == 3
    for path in manifest["figures"][0]["figure_paths"]:
        assert base64.b64encode(Path(path).read_bytes()).decode() in page
    parser = _ExternalResourceParser()
    parser.feed(page)
    assert parser.external_resources == []
    assert "a validated controller comparison is not yet available" not in page
    assert page.count('id="live-control-results"') == 1
    assert '<section class="explanation" aria-labelledby="live-control-title">' in page
    assert page.index('<section class="case-map"') < page.index('id="live-control-results"') < page.index('<div class="workspace">')
    assert "five model-based replays that follow" in page
    assert "five model-based replays above" not in page
    assert "Fixed 1,125 MHz (selected on development) has the lowest observed mean GPU energy: 60.0 J/request" in page
    assert "40.0% lower total GPU energy (95% paired bootstrap interval 38.0% to 42.0%)" in page
    assert "MPC used fallback on 100/100 decisions" in page
    assert "cannot demonstrate an MPC optimizer advantage" in page
    assert "p95 time to first token ≤ 1.500 s" in page


def test_governor_can_lead_without_an_unsupported_significance_claim(tmp_path: Path) -> None:
    results, manifest = _confirmation_fixture(tmp_path)
    for row in results["trials"]:
        if row["method"] == "governor":
            row["energy_per_request_j"] = 50.
    for comparison in results["comparisons"]:
        comparison["positive_energy_saving_supported"] = False
    _write_confirmation(tmp_path, results, manifest)
    page = BUILDER.live_results_section(tmp_path)
    assert "Governor has the lowest observed mean GPU energy: 50.0 J/request" in page
    assert "This is a descriptive ordering" in page
    assert "No positive energy-saving difference passes" in page
    assert "40.0% lower total GPU energy" not in page


def test_incomplete_and_latency_failing_methods_cannot_lead(tmp_path: Path) -> None:
    results, manifest = _confirmation_fixture(tmp_path)
    for row in results["trials"]:
        if row["method"] == "fixed_1125" and row["repetition"] == 0:
            row["valid"] = False
        if row["method"] == "governor":
            row["p95_ttft_s"] = 2.
    _write_confirmation(tmp_path, results, manifest)
    page = BUILDER.live_results_section(tmp_path)
    assert "MPC including fallback has the lowest observed mean GPU energy: 80.0 J/request" in page
    assert "4/5 valid; 1 invalid; 5 retained" in page
    assert "Incomplete; feasibility undetermined" in page
    assert "Misses a frozen target" in page
    assert "40.0% lower total GPU energy" not in page


def test_unmatched_workloads_block_descriptive_ordering_and_paired_claims(tmp_path: Path) -> None:
    results, manifest = _confirmation_fixture(tmp_path)
    for row in results["trials"]:
        if row["method"] == "fixed_1125":
            row["workload_sha256"] = "different-workload"
    _write_confirmation(tmp_path, results, manifest)
    page = BUILDER.live_results_section(tmp_path)
    assert "No cross-method energy ordering is reported" in page
    assert "has the lowest observed" not in page
    assert "40.0% lower total GPU energy" not in page


def test_trace_unavailable_is_disclosed_without_substitution(tmp_path: Path) -> None:
    results, manifest = _confirmation_fixture(tmp_path)
    manifest["figures"][0].update(trace_rendered=False, trace_unavailable_reasons=["rep2/mpc: missing requests"])
    _write_confirmation(tmp_path, results, manifest)
    page = BUILDER.live_results_section(tmp_path)
    assert page.count('src="data:image/svg+xml;base64,') == 1
    assert "rep2/mpc: missing requests" in page
    assert "No other repetition is substituted" in page


def test_development_only_or_unverified_artifacts_keep_pending_message(tmp_path: Path) -> None:
    results, manifest = _confirmation_fixture(tmp_path)
    for row in results["trials"]:
        row["role"] = "development"
    _write_confirmation(tmp_path, results, manifest)
    assert "a validated controller comparison is not yet available" in BUILDER.live_results_section(tmp_path)
    results["synthetic_substitution"] = True
    _write_confirmation(tmp_path, results, manifest)
    assert "do not establish audited direct L4 confirmation provenance" in BUILDER.live_results_section(tmp_path)


def test_missing_svg_and_html_text_are_handled_explicitly(tmp_path: Path) -> None:
    results, manifest = _confirmation_fixture(tmp_path)
    Path(manifest["figures"][0]["figure_paths"][0]).unlink()
    manifest["figures"][0].update(trace_rendered=False, trace_unavailable_reasons=['<script>alert("bad")</script>'])
    _write_confirmation(tmp_path, results, manifest)
    page = BUILDER.live_results_section(tmp_path)
    assert "recorded SVG file is missing" in page
    assert "&lt;script&gt;" in page
    assert '<script>alert("bad")</script>' not in page


def test_all_invalid_confirmation_records_are_pending_and_counted(tmp_path: Path) -> None:
    results, manifest = _confirmation_fixture(tmp_path)
    for row in results["trials"]:
        row["valid"] = False
    _write_confirmation(tmp_path, results, manifest)
    page = BUILDER.live_results_section(tmp_path)
    assert "a validated controller comparison is not yet available" in page
    assert "All 40 supplied confirmation observations are invalid" in page
    assert "has the lowest observed" not in page
    assert 'src="data:image/svg+xml;base64,' not in page


def test_equal_energy_methods_are_reported_as_a_tie(tmp_path: Path) -> None:
    results, manifest = _confirmation_fixture(tmp_path)
    for row in results["trials"]:
        if row["method"] == "governor":
            row["energy_per_request_j"] = 60.
    _write_confirmation(tmp_path, results, manifest)
    page = BUILDER.live_results_section(tmp_path)
    assert "Fixed 1,125 MHz (selected on development) and Governor tie for the lowest observed mean GPU energy" in page
