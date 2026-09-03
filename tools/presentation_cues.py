#!/usr/bin/env python3
"""Import and bundle authoritative recorded spotlight presentations."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAP_START = "<!-- RL_RECORDED_DECKS_START -->"
MAP_END = "<!-- RL_RECORDED_DECKS_END -->"
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
HASH = re.compile(r"^[0-9a-f]{8}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DeckError(ValueError):
    """Raised when a recorded deck does not match the public schema."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DeckError(message)


def validate_anchor(anchor: Any, cue_number: int, anchor_number: int) -> None:
    label = f"cue {cue_number}, anchor {anchor_number}"
    require(isinstance(anchor, dict), f"{label} must be an object")
    require(anchor.get("stable_id") is None or isinstance(anchor.get("stable_id"), str), f"{label}.stable_id must be a string or null")
    require(anchor.get("heading_id") is None or isinstance(anchor.get("heading_id"), str), f"{label}.heading_id must be a string or null")
    require(isinstance(anchor.get("type"), str) and bool(re.fullmatch(r"[a-z][a-z0-9_]*", anchor["type"])), f"{label}.type is invalid")
    require(anchor.get("tag") is None or (isinstance(anchor.get("tag"), str) and bool(re.fullmatch(r"[a-z][a-z0-9-]*", anchor["tag"]))), f"{label}.tag is invalid")
    require(type(anchor.get("ordinal")) is int and anchor["ordinal"] >= 0, f"{label}.ordinal must be a non-negative integer")
    require(isinstance(anchor.get("text_hash"), str) and bool(HASH.fullmatch(anchor["text_hash"])), f"{label}.text_hash must be an 8-character lowercase hex value")
    require(isinstance(anchor.get("text_start"), str), f"{label}.text_start must be a string")


def validate_rectangle(rectangle: Any, cue_number: int) -> None:
    require(isinstance(rectangle, dict), f"cue {cue_number}.rectangle must be an object")
    for field in ("x", "y", "width", "height"):
        value = rectangle.get(field)
        require(type(value) in (int, float) and 0 <= value <= 1, f"cue {cue_number}.rectangle.{field} must be between 0 and 1")


def validate_deck(deck: Any) -> dict[str, Any]:
    require(isinstance(deck, dict), "deck must be a JSON object")
    require(deck.get("version") == 2, "deck.version must be 2")
    require(deck.get("kind") == "recorded-spotlight-deck", "deck.kind must be recorded-spotlight-deck")
    require(isinstance(deck.get("slug"), str) and bool(SLUG.fullmatch(deck["slug"])), "deck.slug is invalid")
    require(isinstance(deck.get("source"), str) and deck["source"].startswith("/"), "deck.source must be an absolute site path")
    require(isinstance(deck.get("source_sha256"), str) and bool(SHA256.fullmatch(deck["source_sha256"])), "deck.source_sha256 must be a lowercase SHA-256 hash")
    require(isinstance(deck.get("recorded_at"), str) and bool(deck["recorded_at"]), "deck.recorded_at is required")
    require(isinstance(deck.get("cues"), list) and bool(deck["cues"]), "deck.cues must contain at least one cue")

    cue_ids: set[str] = set()
    for cue_number, cue in enumerate(deck["cues"], start=1):
        require(isinstance(cue, dict), f"cue {cue_number} must be an object")
        cue_id = cue.get("id")
        require(isinstance(cue_id, str) and bool(cue_id), f"cue {cue_number}.id is required")
        require(cue_id not in cue_ids, f"cue {cue_number}.id is duplicated")
        cue_ids.add(cue_id)
        require(isinstance(cue.get("title"), str) and bool(cue["title"].strip()), f"cue {cue_number}.title is required")
        require(isinstance(cue.get("anchors"), list) and bool(cue["anchors"]), f"cue {cue_number}.anchors must not be empty")
        for anchor_number, anchor in enumerate(cue["anchors"], start=1):
            validate_anchor(anchor, cue_number, anchor_number)
        require(isinstance(cue.get("captured_at"), str) and bool(cue["captured_at"]), f"cue {cue_number}.captured_at is required")
        viewport = cue.get("viewport")
        require(isinstance(viewport, dict), f"cue {cue_number}.viewport must be an object")
        require(type(viewport.get("width")) in (int, float) and viewport["width"] > 0, f"cue {cue_number}.viewport.width must be positive")
        require(type(viewport.get("height")) in (int, float) and viewport["height"] > 0, f"cue {cue_number}.viewport.height must be positive")
        validate_rectangle(cue.get("rectangle"), cue_number)
    return deck


def embed_maps(presenter: str, maps: dict[str, Any]) -> str:
    require(MAP_START in presenter and MAP_END in presenter, "recorded-deck markers are missing from presenter.html")
    payload = json.dumps(maps, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    replacement = f'{MAP_START}\n  <script id="rl-recorded-decks" type="application/json">{payload}</script>\n  {MAP_END}'
    return re.sub(re.escape(MAP_START) + r".*?" + re.escape(MAP_END), lambda _: replacement, presenter, flags=re.DOTALL)


def atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(contents)
        temporary = Path(handle.name)
    temporary.replace(path)


def bundle_maps(root: Path = ROOT) -> int:
    map_dir = root / "_present"
    maps: dict[str, Any] = {}
    if map_dir.exists():
        for path in sorted(map_dir.glob("*.json")):
            deck = validate_deck(json.loads(path.read_text(encoding="utf-8")))
            require(path.stem == deck["slug"], f"{path.name} must match deck slug {deck['slug']}")
            maps[deck["slug"]] = deck
    presenter_path = root / "_static" / "presenter.html"
    presenter = presenter_path.read_text(encoding="utf-8")
    atomic_write(presenter_path, embed_maps(presenter, maps))
    return len(maps)


def import_deck(source: Path, root: Path = ROOT) -> Path:
    deck = validate_deck(json.loads(source.read_text(encoding="utf-8")))
    destination = root / "_present" / f"{deck['slug']}.json"
    atomic_write(destination, json.dumps(deck, indent=2, ensure_ascii=False) + "\n")
    bundle_maps(root)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    importer = subparsers.add_parser("import", help="Validate and install a downloaded recorded deck")
    importer.add_argument("file", type=Path)
    subparsers.add_parser("bundle", help="Re-embed all installed recorded decks in the presenter")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "import":
            source = args.file.expanduser().resolve()
            if not source.is_file():
                raise DeckError(f"recorded deck not found: {source}")
            destination = import_deck(source)
            print(f"Imported {source.name} to {destination.relative_to(ROOT)} and updated the presenter")
        else:
            count = bundle_maps()
            print(f"Bundled {count} recorded presentation{'s' if count != 1 else ''} into _static/presenter.html")
    except (DeckError, json.JSONDecodeError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
