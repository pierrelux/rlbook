#!/usr/bin/env python3
"""Build and maintain the French edition of the book.

French chapters live in fr/ under the same file names as the English ones.
fr/translations.yml records, for each translated chapter, the English commit
it was last brought in line with, so English edits can be carried over as
diffs instead of retranslating whole chapters.

    python3 tools/french.py status            # translated, stale, missing
    python3 tools/french.py diff stochastic-dp.md
    python3 tools/french.py mark stochastic-dp.md
    BASE_URL=/rlbook python3 tools/french.py build

`build` assembles _build/fr-src, a copy of the project in which every
directory is a symlink to the English one and each chapter is the French file
when it exists, or else the English file under a short notice. Code cells are
kept identical to the English ones, so the French build reuses the English
execution cache in _build/execute. The site lands in _build/html/fr.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
FR = ROOT / "fr"
MANIFEST = FR / "translations.yml"
BOOK = FR / "book.yml"
SRC = ROOT / "_build" / "fr-src"
SKIP = {".git", "_build", "fr", "myst.yml", ".jupyter-book-marimo", ".pytest_cache"}

UNTRANSLATED = """:::{note} Chapitre non traduit
Ce chapitre n'est pas encore traduit. Le texte ci-dessous est la version anglaise.
:::
"""
STALE = """:::{note} Traduction à mettre à jour
La version anglaise de ce chapitre a été modifiée depuis cette traduction.
Le menu « English » mène à la version à jour.
:::
"""


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def toc_files(entries: list) -> list[str]:
    files = []
    for entry in entries:
        if "file" in entry:
            files.append(entry["file"])
        files.extend(toc_files(entry.get("children", [])))
    return files


def chapters() -> list[str]:
    config = load_yaml(ROOT / "myst.yml")
    return [f for f in toc_files(config["project"]["toc"]) if f.endswith(".md")]


def is_stale(chapter: str, manifest: dict) -> bool:
    """True when the English chapter differs from the recorded commit."""
    sha = manifest.get(chapter)
    if not sha:
        return False
    return subprocess.run(["git", "diff", "--quiet", sha, "--", chapter], cwd=ROOT).returncode != 0


def insert_notice(text: str, notice: str) -> str:
    """Put a notice right after the chapter's first top-level heading."""
    lines = text.splitlines(keepends=True)
    start = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                start = i + 1
                break
    fence = None
    for i in range(start, len(lines)):
        stripped = lines[i].lstrip()
        match = re.match(r"(`{3,}|:{3,})", stripped)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and stripped.strip() == marker:
                fence = None
            continue
        if fence is None and lines[i].startswith("# "):
            return "".join(lines[: i + 1]) + "\n" + notice + "".join(lines[i + 1 :])
    return "".join(lines[:start]) + notice + "\n" + "".join(lines[start:])


def code_cells(text: str) -> list[str]:
    """Bodies of code cells, without their option lines."""
    cells = []
    for match in re.finditer(r"^(`{3,})\{code-cell\}[^\n]*\n(.*?)^\1\s*$", text, flags=re.M | re.S):
        body = [line for line in match.group(2).splitlines() if not line.startswith(":")]
        cells.append("\n".join(body).strip())
    return cells


def code_mismatches() -> list[str]:
    """Translated chapters whose code cells differ from the English ones."""
    bad = []
    for chapter in chapters():
        french = FR / chapter
        if french.exists() and code_cells(french.read_text(encoding="utf-8")) != code_cells(
            (ROOT / chapter).read_text(encoding="utf-8")
        ):
            bad.append(chapter)
    return bad


def translate_toc(entries: list, parts: dict[str, str]) -> None:
    for entry in entries:
        if "title" in entry:
            entry["title"] = parts.get(entry["title"], entry["title"])
        translate_toc(entry.get("children", []), parts)


def french_config() -> dict:
    config = load_yaml(ROOT / "myst.yml")
    book = load_yaml(BOOK)
    project, site = config["project"], config["site"]
    project.update(book.get("project", {}))
    project.pop("id", None)
    translate_toc(project["toc"], book.get("parts", {}))
    project.pop("exports", None)
    site["options"].update(book.get("site", {}))
    names = book.get("actions", {})
    actions = []
    for action in site.get("actions", []):
        if action.get("url", "").endswith("francais.html"):
            actions.append({"title": "English", "url": "_static/english.html"})
        else:
            actions.append({**action, "title": names.get(action["title"], action["title"])})
    site["actions"] = actions
    return config


def cmd_build(args: argparse.Namespace) -> int:
    manifest = load_yaml(MANIFEST)
    for chapter in code_mismatches():
        print(f"warning: code cells in fr/{chapter} differ from the English; "
              "the French build will re-execute them", file=sys.stderr)
    if SRC.exists():
        shutil.rmtree(SRC)
    SRC.mkdir(parents=True)
    for entry in ROOT.iterdir():
        if entry.name not in SKIP:
            (SRC / entry.name).symlink_to(entry)
    cache = ROOT / "_build" / "execute"
    cache.mkdir(parents=True, exist_ok=True)
    (SRC / "_build").mkdir()
    (SRC / "_build" / "execute").symlink_to(cache)
    # Reuse the English build's downloaded site theme and citation cache.
    for name in ("templates", "cache"):
        if (ROOT / "_build" / name).exists():
            (SRC / "_build" / name).symlink_to(ROOT / "_build" / name)

    for chapter in chapters():
        target = SRC / chapter
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        french = FR / chapter
        if french.exists():
            text = french.read_text(encoding="utf-8")
            if is_stale(chapter, manifest):
                text = insert_notice(text, STALE)
        else:
            text = insert_notice((ROOT / chapter).read_text(encoding="utf-8"), UNTRANSLATED)
        target.write_text(text, encoding="utf-8")

    (SRC / "myst.yml").write_text(
        "# Generated by tools/french.py from myst.yml and fr/book.yml. Do not edit.\n"
        + yaml.safe_dump(french_config(), allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["BASE_URL"] = env.get("BASE_URL", "/rlbook").rstrip("/") + "/fr"
    command = args.command or ["jupyter-book", "build", "--html", "--execute", "--strict"]
    result = subprocess.run(command, cwd=SRC, env=env)
    if result.returncode != 0:
        return result.returncode
    output = ROOT / "_build" / "html" / "fr"
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(SRC / "_build" / "html", output)
    print(f"French site written to {output.relative_to(ROOT)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    manifest = load_yaml(MANIFEST)
    translated, stale, missing = [], [], []
    for chapter in chapters():
        if not (FR / chapter).exists():
            missing.append(chapter)
        elif chapter not in manifest:
            stale.append((chapter, "no entry in fr/translations.yml"))
        elif is_stale(chapter, manifest):
            changed = git("diff", "--numstat", manifest[chapter], "--", chapter).split()
            stale.append((chapter, f"+{changed[0]} -{changed[1]} English lines since {manifest[chapter][:7]}"))
        else:
            translated.append(chapter)
    print(f"Up to date ({len(translated)}): " + (", ".join(translated) or "none"))
    print(f"Stale ({len(stale)}):")
    for chapter, why in stale:
        print(f"  {chapter}: {why}")
    print(f"Not translated ({len(missing)}): " + (", ".join(missing) or "none"))
    parts = load_yaml(BOOK).get("parts", {})
    titles: list[str] = []

    def walk(entries: list) -> None:
        for entry in entries:
            if "title" in entry:
                titles.append(entry["title"])
            walk(entry.get("children", []))

    walk(load_yaml(ROOT / "myst.yml")["project"]["toc"])
    for title in titles:
        if title not in parts:
            print(f"Part title without a French name in fr/book.yml: {title}")
    mismatched = code_mismatches()
    for chapter in mismatched:
        print(f"Code cells differ from the English in fr/{chapter}; copy them over unchanged")
    return 1 if args.strict and (stale or mismatched) else 0


def cmd_diff(args: argparse.Namespace) -> int:
    sha = load_yaml(MANIFEST).get(args.chapter)
    if not sha:
        print(f"{args.chapter} has no entry in fr/translations.yml", file=sys.stderr)
        return 1
    print(git("diff", sha, "--", args.chapter), end="")
    return 0


def cmd_mark(args: argparse.Namespace) -> int:
    manifest = load_yaml(MANIFEST)
    for chapter in args.chapters:
        if not (FR / chapter).exists():
            print(f"fr/{chapter} does not exist", file=sys.stderr)
            return 1
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", chapter], cwd=ROOT).returncode != 0:
            print(f"{chapter} has uncommitted English changes; commit them first", file=sys.stderr)
            return 1
        manifest[chapter] = git("log", "-1", "--format=%H", "--", chapter).strip()
    header = "".join(line for line in MANIFEST.read_text(encoding="utf-8").splitlines(keepends=True) if line.startswith("#"))
    body = yaml.safe_dump(dict(sorted(manifest.items())), sort_keys=False) if manifest else "{}\n"
    MANIFEST.write_text(header + body, encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="build the French site into _build/html/fr")
    build.add_argument("command", nargs=argparse.REMAINDER, help="override the build command")
    build.set_defaults(func=cmd_build)
    status = sub.add_parser("status", help="list translated, stale and untranslated chapters")
    status.add_argument("--strict", action="store_true", help="exit 1 if any translation is stale")
    status.set_defaults(func=cmd_status)
    diff = sub.add_parser("diff", help="English changes since a chapter was last translated")
    diff.add_argument("chapter")
    diff.set_defaults(func=cmd_diff)
    mark = sub.add_parser("mark", help="record that French chapters match the current English")
    mark.add_argument("chapters", nargs="+")
    mark.set_defaults(func=cmd_mark)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
