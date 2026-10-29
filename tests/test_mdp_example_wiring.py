"""Checks that the MDP example replays are embedded in their chapters with static placeholders."""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [
    ("stochastic-dp.md", "storm-surge-barrier"),
    ("infinite-horizon-mdps.md", "sailing-wind-shifts"),
    ("infinite-horizon-mdps.md", "bridge-deck-maintenance"),
    ("infinite-horizon-mdps.md", "tamarisk-river"),
]


@pytest.mark.parametrize("chapter,slug", EXAMPLES)
def test_example_is_embedded_with_placeholder_and_citations(chapter, slug):
    text = (ROOT / chapter).read_text()
    folder = slug.replace("-", "_")
    assert f"({'sec-' + slug})=" in text
    assert f"../interactive/{slug}.html" in text
    assert (ROOT / "interactive" / f"{slug}.html").exists()
    placeholder = re.search(rf":placeholder: (_static/{folder}/\S+\.svg)", text)
    assert placeholder and (ROOT / placeholder.group(1)).exists()
    assert f"artifacts/{folder}/results.md" in text
    bib = (ROOT / "references.bib").read_text()
    for key in re.findall(r"\{cite:[tp]\}`([^`]+)`", text):
        for k in key.split(","):
            assert f"{{{k.strip()}," in bib, k


@pytest.mark.parametrize("chapter,slug", EXAMPLES + [("stochastic-dp.md", "antibiotic-steering")])
def test_replay_iframe_has_a_height_rule(chapter, slug):
    text = (ROOT / chapter).read_text()
    css = (ROOT / "_static" / "custom.css").read_text()
    assert f":class: {slug}-replay" in text
    assert f".{slug}-replay > div > div {{" in css
