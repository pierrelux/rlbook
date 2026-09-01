#!/usr/bin/env python3
"""Generate semantic presentation beats from a rendered MyST chapter.

The script extracts a deterministic inventory from MyST's built HTML, asks a
non-interactive model to group adjacent blocks, validates the response, writes
the result to ``_present/``, and bundles all maps into the presenter overlay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRESENTER = ROOT / "_static" / "presenter.html"
MAP_DIR = ROOT / "_present"
MAP_START = "<!-- RL_CHUNK_MAPS_START -->"
MAP_END = "<!-- RL_CHUNK_MAPS_END -->"


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class ArticleParser(HTMLParser):
    """Collect direct children of ``article.content`` without dependencies."""

    VISUAL_TAGS = {"img", "svg", "canvas", "iframe", "pre", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, dict[str, str]]] = []
        self.article_depth: int | None = None
        self.current: dict[str, Any] | None = None
        self.blocks: list[dict[str, Any]] = []
        self.title_depth: int | None = None
        self.title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        self.stack.append((tag, attributes))
        depth = len(self.stack)
        classes = attributes.get("class", "").split()
        if tag == "article" and "content" in classes:
            self.article_depth = depth
        elif self.article_depth and depth == self.article_depth + 1:
            self.current = {
                "tag": tag,
                "attrs": attributes,
                "text": [],
                "visual": tag in self.VISUAL_TAGS,
            }
            self.blocks.append(self.current)
        elif self.current and tag in self.VISUAL_TAGS:
            self.current["visual"] = True
        if self.article_depth and tag == "h1" and self.title_depth is None:
            self.title_depth = depth

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        depth = len(self.stack)
        if self.title_depth == depth and tag == "h1":
            self.title_depth = None
        if self.article_depth and depth == self.article_depth + 1:
            self.current = None
        if self.article_depth == depth and tag == "article":
            self.article_depth = None
        if self.stack:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.current["text"].append(data)
        if self.title_depth is not None:
            self.title_parts.append(data)


def is_presentable(block: dict[str, Any]) -> bool:
    tag = block["tag"]
    classes = block["attrs"].get("class", "").split()
    if tag in {"script", "style", "link"}:
        return False
    if "hidden" in classes or "myst-fm-block" in classes:
        return False
    if "myst-footer-links" in classes:
        return False
    if {"block", "my-10", "lg:sticky"}.issubset(classes):
        return False
    if block["attrs"].get("id") == "skip-to-article":
        return False
    return bool(compact("".join(block["text"])) or block["visual"])


def block_kind(block: dict[str, Any]) -> str:
    tag = block["tag"]
    classes = block["attrs"].get("class", "")
    if re.fullmatch(r"h[1-6]", tag):
        return "heading"
    if tag == "p":
        return "paragraph"
    if tag in {"ul", "ol"}:
        return "list"
    if tag == "aside" or "admonition" in classes:
        return "admonition"
    if tag == "blockquote":
        return "quotation"
    if tag in {"pre", "code"} or "code" in classes:
        return "code"
    if block["visual"]:
        return "visual-or-computation"
    return tag


def extract_inventory(html_path: Path) -> list[dict[str, Any]]:
    parser = ArticleParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    title = compact("".join(parser.title_parts)).removesuffix("¶").strip()
    if not title:
        raise RuntimeError(f"Could not find the chapter title in {html_path}")
    inventory = [{"index": 0, "tag": "h1", "kind": "title", "text": title}]
    for block in filter(is_presentable, parser.blocks):
        text = compact("".join(block["text"])).removesuffix("¶").strip()
        if len(text) > 700:
            text = text[:697].rstrip() + "…"
        inventory.append(
            {
                "index": len(inventory),
                "tag": block["tag"],
                "kind": block_kind(block),
                "text": text or "[visual content]",
            }
        )
    return inventory


def rendered_chapter(source: Path) -> tuple[str, Path]:
    """Resolve MyST's URL slug, including filename-to-slug normalization."""
    output_root = ROOT / "_build" / "html"
    relative_source = source.relative_to(ROOT).as_posix()
    for metadata_path in output_root.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(metadata.get("location", "")).lstrip("/") != relative_source:
            continue
        slug = str(metadata.get("slug", "")).strip("/")
        html_path = output_root / slug / "index.html"
        if slug and html_path.is_file():
            return slug, html_path
    fallback_slug = source.stem.replace("_", "-")
    return fallback_slug, output_root / fallback_slug / "index.html"


def output_schema(last_index: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "chunks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "integer", "minimum": 0, "maximum": last_index},
                        "end": {"type": "integer", "minimum": 0, "maximum": last_index},
                        "title": {"type": "string"},
                    },
                    "required": ["start", "end", "title"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["chunks"],
        "additionalProperties": False,
    }


def make_prompt(source: Path, inventory: list[dict[str, Any]]) -> str:
    return f"""You are editing presentation beats for a university lecture chapter.

Group the numbered rendered blocks below into semantic presentation chunks. A
chunk is the amount the lecturer should focus or enlarge at once. You are only
choosing boundaries: do not rewrite, omit, reorder, or duplicate any block.

Rules:
- Cover every index from 0 through {len(inventory) - 1}, exactly once, in order.
- Chunks must be contiguous ranges with no gaps or overlaps.
- Keep the chapter title (index 0) alone.
- Aim for 2–5 blocks per chunk, but use one block for a large equation, figure,
  table, code output, admonition, exercise, or other visually dense object.
- Keep a claim with its explanation, an equation with its introduction or
  interpretation, and a figure with nearby framing text when that fits.
- A heading may begin a chunk with the material it introduces. Never place a
  heading at the end of the preceding chunk.
- Do not cross an h2 section boundary. Prefer not to cross h3 boundaries.
- Give each chunk a short navigation title (roughly 2–7 words).
- Use only the inventory below. Do not inspect files or run commands.

Source: {source.name}
Inventory:
{json.dumps(inventory, ensure_ascii=False, separators=(",", ":"))}
"""


def run_codex(prompt: str, schema: dict[str, Any], model: str | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rl-chunks-") as temp:
        schema_path = Path(temp) / "schema.json"
        output_path = Path(temp) / "result.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        command = [
            "codex", "exec", "--ephemeral", "--sandbox", "read-only",
            "--output-schema", str(schema_path), "--output-last-message", str(output_path),
            "--cd", str(ROOT),
        ]
        if model:
            command.extend(["--model", model])
        command.append("-")
        result = subprocess.run(command, input=prompt, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"Codex failed ({result.returncode}):\n{result.stderr.strip()}")
        return json.loads(output_path.read_text(encoding="utf-8"))


def run_claude(prompt: str, schema: dict[str, Any], model: str | None) -> dict[str, Any]:
    command = [
        "claude", "--print", "--output-format", "json", "--json-schema",
        json.dumps(schema, separators=(",", ":")), "--permission-mode", "dontAsk",
        "--tools", "", "--no-session-persistence",
    ]
    if model:
        command.extend(["--model", model])
    result = subprocess.run(command, input=prompt, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Claude failed ({result.returncode}):\n{result.stderr.strip()}")
    envelope = json.loads(result.stdout)
    structured = envelope.get("structured_output")
    if structured is not None:
        return structured
    raw = envelope.get("result", envelope)
    return json.loads(raw) if isinstance(raw, str) else raw


def validate(chunks: Any, inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("The model returned no chunks")
    expected = 0
    headings = {item["index"] for item in inventory if item["kind"] == "heading"}
    for number, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {number} is not an object")
        start, end = chunk.get("start"), chunk.get("end")
        if start != expected or not isinstance(end, int) or end < start:
            raise ValueError(f"Chunk {number} does not continue at index {expected}: {chunk}")
        if end >= len(inventory):
            raise ValueError(f"Chunk {number} ends outside the inventory: {chunk}")
        if end in headings and start != end:
            raise ValueError(f"Chunk {number} leaves a heading at its end: {chunk}")
        internal_h2 = [
            item["index"] for item in inventory[start + 1 : end + 1]
            if item["tag"] == "h2"
        ]
        if internal_h2:
            raise ValueError(f"Chunk {number} crosses an h2 boundary at {internal_h2[0]}")
        if not compact(str(chunk.get("title", ""))):
            raise ValueError(f"Chunk {number} has no navigation title")
        expected = end + 1
    if chunks[0]["start"] != 0 or chunks[0]["end"] != 0:
        raise ValueError("The chapter title at index 0 must be its own chunk")
    if expected != len(inventory):
        raise ValueError(f"Chunks stop at index {expected - 1}; expected {len(inventory) - 1}")
    return chunks


def bundle_maps() -> None:
    maps: dict[str, Any] = {}
    if MAP_DIR.exists():
        for path in sorted(MAP_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            maps[data["slug"]] = data
    presenter = PRESENTER.read_text(encoding="utf-8")
    if MAP_START not in presenter or MAP_END not in presenter:
        raise RuntimeError(f"Chunk-map markers are missing from {PRESENTER}")
    payload = json.dumps(maps, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    replacement = (
        f'{MAP_START}\n<script id="rl-presentation-chunks" type="application/json">'
        f"{payload}</script>\n{MAP_END}"
    )
    updated = re.sub(
        re.escape(MAP_START) + r".*?" + re.escape(MAP_END),
        lambda _: replacement,
        presenter,
        flags=re.DOTALL,
    )
    PRESENTER.write_text(updated, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, help="Chapter Markdown file, e.g. dynamics.md")
    parser.add_argument("--provider", choices=("auto", "codex", "claude"), default="auto")
    parser.add_argument("--model", help="Optional provider-specific model override")
    parser.add_argument("--bundle-only", action="store_true", help="Re-embed existing maps without calling a model")
    parser.add_argument("--print-inventory", action="store_true", help="Print extracted blocks and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bundle_only:
        bundle_maps()
        print(f"Bundled presentation maps into {PRESENTER.relative_to(ROOT)}")
        return 0
    if args.source is None:
        raise SystemExit("A chapter Markdown file is required unless --bundle-only is used")
    source = (ROOT / args.source).resolve() if not args.source.is_absolute() else args.source.resolve()
    if ROOT not in source.parents or not source.is_file():
        raise SystemExit(f"Source must be an existing file inside {ROOT}")
    slug, html_path = rendered_chapter(source)
    if not html_path.is_file():
        raise SystemExit(f"Missing {html_path.relative_to(ROOT)}; build the book first")
    if html_path.stat().st_mtime < source.stat().st_mtime:
        print("warning: built HTML is older than the source; consider rebuilding first", file=sys.stderr)

    inventory = extract_inventory(html_path)
    if args.print_inventory:
        print(json.dumps(inventory, indent=2, ensure_ascii=False))
        return 0
    provider = args.provider
    if provider == "auto":
        provider = "codex" if shutil.which("codex") else "claude" if shutil.which("claude") else ""
    if not provider or not shutil.which(provider):
        raise SystemExit("Neither Codex nor Claude CLI is available")

    schema = output_schema(len(inventory) - 1)
    prompt = make_prompt(source, inventory)
    print(f"Asking {provider} to group {len(inventory)} rendered blocks…")
    response = run_codex(prompt, schema, args.model) if provider == "codex" else run_claude(prompt, schema, args.model)
    chunks = validate(response.get("chunks"), inventory)

    MAP_DIR.mkdir(exist_ok=True)
    output = MAP_DIR / f"{slug}.json"
    manifest = {
        "version": 1,
        "slug": slug,
        "source": source.relative_to(ROOT).as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "provider": provider,
        "block_count": len(inventory),
        "block_tags": [item["tag"] for item in inventory],
        "chunks": chunks,
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    bundle_maps()
    print(f"Wrote {len(chunks)} semantic chunks to {output.relative_to(ROOT)} and updated the presenter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
