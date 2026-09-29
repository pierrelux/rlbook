"""Build the data behind interactive/lectures.html (lecture recordings page).

Inputs (teaching/recordings/):
  lecture_<YYYY-MM-DD>.json  Caption bundle for one recording, saved from the
                             Google Drive viewer: {fileId, title, date,
                             en: [[t_ms, dur_ms, text], ...], fr: [...],
                             thumb: "data:image/jpeg;base64,..."}.
                             "fr" is Google's auto-translation of the English
                             auto-captions.
  <YYYY-MM-DD>.jpg           Optional thumbnail that replaces the Drive one.
  clean/<YYYY-MM-DD>.json    Optional edited transcript:
                             {tldr: [{notion, detail, notion_fr, detail_fr, t, exam}],
                              outline: [{t, title, title_fr}],
                              paras: [{t, text, fr}, ...]}, one entry per caption
                             paragraph (same t as produced here); "text" is the
                             edited English, "fr" its French translation.
                             "[Break]" or "[No speech]" marks paragraphs that
                             are not lecture content; their raw captions are
                             dropped.
  lectures.json              Titles (en/fr), book chapters and names to redact
                             from the raw captions per lecture, word corrections
                             for recurring caption errors, and course-level
                             settings.

Outputs (interactive/lectures-data/):
  index.json                 One entry per lecture, newest first.
  <YYYY-MM-DD>.json          {paras: [[t_ms, en_raw, fr_machine, en_edited, fr_edited], ...],
                              tldr, outline} with "p" = paragraph index.
  <YYYY-MM-DD>.jpg           Thumbnail.

Usage:  python scripts/build_lecture_recordings.py
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "teaching" / "recordings"
OUT = ROOT / "interactive" / "lectures-data"

# A paragraph closes at the first sentence end after MIN_MS, and always by MAX_MS.
MIN_MS = 30_000
MAX_MS = 60_000
GAP_MARKERS = {"[Break]", "[No speech]"}


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,])", r"\1", text)
    # ">>" marks a change of speaker in the captions: start a new line there.
    text = re.sub(r"\s*>>\s*", "\n", text).strip()
    return text


def corrector(table: dict[str, str]):
    """Exact, case-sensitive, whole-word replacements (longest keys first)."""
    if not table:
        return lambda text: text
    keys = sorted(table, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")
    return lambda text: pattern.sub(lambda m: table[m.group(1)], text)


def paragraphs(en: list, fr: list, fix_en=lambda t: t, fix_fr=lambda t: t) -> list[list]:
    """Group English caption events into paragraphs and align French by time."""
    starts: list[int] = []
    buckets: list[list[str]] = []
    for t, _dur, text in en:
        if not buckets:
            starts.append(t)
            buckets.append([text])
            continue
        elapsed = t - starts[-1]
        prev = buckets[-1][-1]
        if elapsed >= MAX_MS or (elapsed >= MIN_MS and re.search(r"[.?!]$", prev)):
            starts.append(t)
            buckets.append([text])
        else:
            buckets[-1].append(text)

    fr_buckets: list[list[str]] = [[] for _ in starts]
    k = 0
    for t, _dur, text in fr:
        while k + 1 < len(starts) and t >= starts[k + 1]:
            k += 1
        fr_buckets[k].append(text)

    fr_text = [clean(" ".join(f)) for f in fr_buckets]
    # The translation is timed per sentence, so a paragraph can start with the
    # punctuation that closes the previous one. Move it back.
    for i in range(1, len(fr_text)):
        m = re.match(r"^([.,;:!?\u2026]+)\s*", fr_text[i])
        if m:
            fr_text[i - 1] += m.group(1)
            fr_text[i] = fr_text[i][m.end():]

    return [
        [s, fix_en(clean(" ".join(b))), fix_fr(f)]
        for s, b, f in zip(starts, buckets, fr_text)
    ]


def teaching_week(day: dt.date, term_start: dt.date, breaks: list[dt.date]) -> int:
    week = (day - term_start).days // 7 + 1
    return week - sum(1 for b in breaks if b <= day)


def chapter_title(slug: str) -> str:
    md = ROOT / f"{slug}.md"
    if md.exists():
        for line in md.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    return slug.replace("-", " ").capitalize()


def main() -> None:
    manifest = json.loads((SRC / "lectures.json").read_text(encoding="utf-8"))
    meta = manifest.get("lectures", {})
    term_start = dt.date.fromisoformat(manifest["term_start"])
    breaks = [dt.date.fromisoformat(d) for d in manifest.get("break_weeks", [])]
    fixes = manifest.get("corrections", {})
    fix_en, fix_fr = corrector(fixes.get("en", {})), corrector(fixes.get("fr", {}))

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    entries = []
    for bundle_path in sorted(SRC.glob("lecture_*.json")):
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        date = bundle["date"]
        info = meta.get(date, {})
        paras = paragraphs(bundle["en"], bundle.get("fr", []), fix_en, fix_fr)
        hide = corrector({w: "[name]" for w in info.get("redact", [])})
        for p in paras:
            p[1], p[2] = hide(p[1]), hide(p[2])
        last = bundle["en"][-1]
        duration_ms = last[0] + last[1]

        starts = [p[0] for p in paras]

        def para_index(t: int) -> int:
            """Index of the paragraph that contains time t."""
            return max(0, sum(1 for s in starts if s <= t) - 1)

        clean_path = SRC / "clean" / f"{date}.json"
        tldr, outline = [], []
        if clean_path.exists():
            clean = json.loads(clean_path.read_text(encoding="utf-8"))
            by_t = {c["t"]: c for c in clean.get("paras", [])}
            missing = [t for t in starts if t not in by_t]
            if missing:
                raise SystemExit(
                    f"{clean_path.name}: no cleaned text for {len(missing)} paragraphs "
                    f"(first t={missing[0]}); was the paragraphing changed?"
                )
            for p in paras:
                c = by_t[p[0]]
                if c["text"] in GAP_MARKERS:
                    p[1] = p[2] = ""  # break chatter and caption noise are not published
                p.extend([c["text"], c.get("fr", "")])
            tldr = [
                {"notion": {"en": x["notion"], "fr": x.get("notion_fr", "")},
                 "detail": {"en": x["detail"], "fr": x.get("detail_fr", "")},
                 "p": para_index(x["t"]), "exam": bool(x.get("exam"))}
                for x in clean.get("tldr", [])
            ]
            outline = [
                {"title": {"en": x["title"], "fr": x.get("title_fr", "")}, "p": para_index(x["t"])}
                for x in clean.get("outline", [])
            ]
        else:
            for p in paras:
                p.extend(["", ""])

        (OUT / f"{date}.json").write_text(
            json.dumps({"date": date, "paras": paras, "tldr": tldr, "outline": outline},
                       ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

        override = SRC / f"{date}.jpg"
        if override.exists():
            shutil.copy(override, OUT / f"{date}.jpg")
        else:
            data = bundle["thumb"].split(",", 1)[1]
            (OUT / f"{date}.jpg").write_bytes(base64.b64decode(data))

        day = dt.date.fromisoformat(date)
        entries.append(
            {
                "date": date,
                "week": teaching_week(day, term_start, breaks),
                "title": info.get("title", {"en": "", "fr": ""}),
                "chapters": [
                    {"slug": s, "title": chapter_title(s)} for s in info.get("chapters", [])
                ],
                "drive": f"https://drive.google.com/file/d/{bundle['fileId']}/view",
                "duration_ms": duration_ms,
                "thumb": f"{date}.jpg",
                "paragraphs": len(paras),
                "cleaned": clean_path.exists(),
                "tldr": [{"notion": x["notion"], "p": x["p"], "exam": x["exam"]} for x in tldr],
            }
        )
        print(f"{date}: {len(paras)} paragraphs, {duration_ms / 60000:.0f} min"
              f"{', cleaned' if clean_path.exists() else ''}")

    entries.sort(key=lambda e: e["date"], reverse=True)
    index = {
        "course": manifest.get("course", ""),
        "term": manifest.get("term", {}),
        "folder_url": manifest.get("folder_url", ""),
        "updated": dt.date.today().isoformat(),
        "lectures": entries,
    }
    (OUT / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"Wrote {len(entries)} lectures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
