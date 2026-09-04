import json
import tempfile
import unittest
from pathlib import Path

from tools.presentation_cues import DeckError, embed_maps, import_deck, validate_deck


def sample_deck():
    return {
        "version": 2,
        "kind": "recorded-spotlight-deck",
        "slug": "modeling-controlled-systems",
        "source": "/modeling-controlled-systems/",
        "source_sha256": "a" * 64,
        "recorded_at": "2026-09-01T12:00:00Z",
        "cues": [
            {
                "id": "cue-1",
                "title": "System boundary",
                "anchors": [
                    {
                        "stable_id": "system-boundaries",
                        "heading_id": None,
                        "type": "heading",
                        "tag": "h2",
                        "ordinal": 0,
                        "text_hash": "12ab34cd",
                        "text_start": "System boundaries",
                    }
                ],
                "captured_at": "2026-09-01T12:00:01Z",
                "viewport": {"width": 1440, "height": 900},
                "rectangle": {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.2},
            }
        ],
    }


class PresentationCueTests(unittest.TestCase):
    def test_valid_deck(self):
        self.assertEqual(
            validate_deck(sample_deck())["slug"], "modeling-controlled-systems"
        )

    def test_rejects_duplicate_cue_ids(self):
        deck = sample_deck()
        deck["cues"].append(dict(deck["cues"][0]))
        with self.assertRaisesRegex(DeckError, "duplicated"):
            validate_deck(deck)

    def test_rejects_invalid_rectangle(self):
        deck = sample_deck()
        deck["cues"][0]["rectangle"]["width"] = 2
        with self.assertRaisesRegex(DeckError, "between 0 and 1"):
            validate_deck(deck)

    def test_embeds_without_script_breakout(self):
        presenter = "before\n<!-- RL_RECORDED_DECKS_START -->old<!-- RL_RECORDED_DECKS_END -->\nafter"
        deck = sample_deck()
        deck["cues"][0]["title"] = "</script>"
        result = embed_maps(presenter, {"modeling-controlled-systems": deck})
        self.assertIn("\\u003c/script>", result)
        self.assertEqual(result.count("RL_RECORDED_DECKS_START"), 1)

    def test_import_installs_and_bundles_deck(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "_static").mkdir()
            (root / "_static" / "presenter.html").write_text(
                "before\n<!-- RL_RECORDED_DECKS_START -->old<!-- RL_RECORDED_DECKS_END -->\nafter",
                encoding="utf-8",
            )
            source = root / "download.json"
            source.write_text(json.dumps(sample_deck()), encoding="utf-8")
            destination = import_deck(source, root)
            self.assertEqual(
                destination, root / "_present" / "modeling-controlled-systems.json"
            )
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8"))["slug"],
                "modeling-controlled-systems",
            )
            bundled = (root / "_static" / "presenter.html").read_text(encoding="utf-8")
            self.assertIn('"modeling-controlled-systems"', bundled)


if __name__ == "__main__":
    unittest.main()
