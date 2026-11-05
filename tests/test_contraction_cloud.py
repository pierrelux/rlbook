"""Checks for the contraction cloud animation in the infinite-horizon chapter."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_chapter_embeds_contraction_cloud():
    chapter = (ROOT / "infinite-horizon-mdps.md").read_text()
    assert "../interactive/contraction-cloud.html" in chapter
    assert ":class: contraction-cloud-replay" in chapter
    assert (ROOT / "interactive" / "contraction-cloud.html").exists()


def test_replay_iframe_has_a_height_rule():
    css = (ROOT / "_static" / "custom.css").read_text()
    assert ".contraction-cloud-replay > div > div" in css
