"""Checks for the tile coding subsection of Fitted Value Methods."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import tile_coding as tc  # noqa: E402


def test_chapters_embed_the_widget():
    for chapter in ("approximate-bellman-equations.md", "fr/approximate-bellman-equations.md"):
        text = (ROOT / chapter).read_text()
        for view in ("kernel", "plane", "learn"):
            assert f"../interactive/tile-coding.html#view={view}" in text
            assert f":class: tile-coding-{view}" in text
        assert "(sec-abe-tile-coding)=" in text
    assert (ROOT / "interactive" / "tile-coding.html").exists()


def test_iframe_has_a_height_rule():
    css = (ROOT / "_static" / "custom.css").read_text()
    for view in ("kernel", "plane", "learn"):
        assert f".tile-coding-{view} > div > div" in css


def test_even_offsets_give_a_staircase_within_one_over_k_of_the_hat():
    for m in (2, 4, 8):
        r = tc.kernel_report_1d(m=m, h=0.25, trials=50)
        assert r["uniform: max |k - hat|"] < 1.0 / m


def test_one_dimension_spans_the_fine_cells_and_two_do_not():
    r1 = tc.projection_report(m=4, h=0.25, n=64, d=1)
    assert r1["rank"] == r1["fine cells"]
    assert r1["joint LS: min entry"] > -1e-9
    assert abs(r1["joint LS: sup norm"] - 1) < 1e-9
    r2 = tc.projection_report(m=4, h=0.25, n=32, d=2)
    assert r2["rank"] < r2["fine cells"]
    assert r2["joint LS: min entry"] < -0.01
    assert r2["joint LS: sup norm"] > 4
    for r in (r1, r2):
        assert r["averaged: min entry"] >= 0
        assert abs(r["averaged: sup norm"] - 1) < 1e-9


def test_averaged_tilings_are_the_shared_tile_kernel_smoother():
    assert tc.smoother_matches_kernel() < 1e-12
