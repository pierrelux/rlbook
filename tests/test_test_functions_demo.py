"""Checks for the test-function lab in the weighted-residual chapter."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from test_functions_demo import table  # noqa: E402


def test_chapters_embed_the_lab():
    for chapter in ("weighted-residual-methods.md", "fr/weighted-residual-methods.md"):
        text = (ROOT / chapter).read_text()
        assert "../interactive/test-functions.html" in text
        assert ":class: test-functions-replay" in text
    assert (ROOT / "interactive" / "test-functions.html").exists()


def test_iframe_has_a_height_rule():
    css = (ROOT / "_static" / "custom.css").read_text()
    assert ".test-functions-replay > div > div" in css


def test_each_method_zeroes_its_own_tests_only():
    for kind in ("hat", "poly"):
        for n in (4, 6, 8):
            rows = table(kind=kind, n=n)
            for method, row in rows.items():
                assert row[method] < 1e-8
                others = [row[f] for f in rows if f != method]
                assert min(others) > 5e-4
