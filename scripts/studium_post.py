"""Post an announcement to a StudiUM (Moodle) forum from a Markdown draft.

The draft's first line gives the subject, as "**Subject:** ..." (or
"Subject: ...", "**Sujet :** ..."). The rest is converted from Markdown to
HTML and becomes the message.

The Moodle web-service key is read from the macOS Keychain item created with

    security add-generic-password -a studium -s studium-token -w

It is sent in the POST body only and never printed. Without --send the script
is a dry run and makes no request at all.

Usage:
    python scripts/studium_post.py draft.md            # dry run (default)
    python scripts/studium_post.py draft.md --send     # post, print the URL
    python scripts/studium_post.py --list              # latest discussions
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://studium.umontreal.ca/webservice/rest/server.php"
NOUVELLES = 498079  # IFT6162-A-A26 > Nouvelles
DISCUSSION_URL = "https://studium.umontreal.ca/mod/forum/discuss.php?d={}"
SUBJECT_RE = re.compile(r"^\s*\**\s*(?:Subject|Sujet)\s*:?\s*\**\s*:?\s*(.+?)\s*\**\s*$", re.I)


def read_token() -> str:
    r = subprocess.run(
        ["security", "find-generic-password", "-a", "studium", "-s", "studium-token", "-w"],
        capture_output=True, text=True,
    )
    if r.returncode or not r.stdout.strip():
        sys.exit("No StudiUM key in the Keychain. Store it with:\n"
                 "  security add-generic-password -a studium -s studium-token -w")
    return r.stdout.strip()


def call(function: str, **params) -> dict:
    body = {"wstoken": read_token(), "wsfunction": function, "moodlewsrestformat": "json", **params}
    data = urllib.parse.urlencode(body).encode()
    with urllib.request.urlopen(ENDPOINT, data, timeout=60) as resp:
        out = json.load(resp)
    if isinstance(out, dict) and "exception" in out:
        if out.get("errorcode") == "invalidtoken":
            sys.exit("StudiUM rejected the key (Jeton non valide). Reset the 'Moodle mobile web service' "
                     "key at https://studium.umontreal.ca/user/managetoken.php and store the new one with\n"
                     "  security add-generic-password -U -a studium -s studium-token -w")
        sys.exit(f"{function}: {out.get('errorcode')}: {out.get('message')}")
    return out


def inline(text: str) -> str:
    """Minimal inline Markdown: links, bold, italics, bare URLs."""
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"(?<![\"'>=])(https?://[^\s<)]+)", r'<a href="\1">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<em>\1</em>", text)
    text = re.sub(r"(?<![\w])_(?!\s)(.+?)(?<!\s)_(?![\w])", r"<em>\1</em>", text)
    return text


def markdown_to_html(md: str) -> str:
    """Paragraphs, '-' lists, '---' rules and inline markup; enough for announcements.

    Bare URLs become links, which the `markdown` package does not do.
    """
    out, para, items = [], [], []

    def flush():
        if para:
            out.append("<p>" + "<br>".join(inline(l) for l in para) + "</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for line in md.splitlines():
        s = line.strip()
        if not s:
            flush()
        elif re.fullmatch(r"-{3,}|\*{3,}", s):
            flush()
            out.append("<hr>")
        elif re.match(r"^[-*] ", s):
            if para:
                flush()
            items.append(s[2:])
        else:
            if items:
                flush()
            para.append(s)
    flush()
    return "\n".join(out)


def parse_draft(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    m = SUBJECT_RE.match(lines[0]) if lines else None
    if not m:
        sys.exit(f"{path}: the first line must be '**Subject:** ...'")
    subject = m.group(1).strip().strip("*").strip()
    return subject, markdown_to_html("\n".join(lines[1:]).strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("draft", nargs="?", type=Path)
    ap.add_argument("--forum", type=int, default=NOUVELLES)
    ap.add_argument("--send", action="store_true", help="post it (default is a dry run)")
    ap.add_argument("--dry-run", action="store_true", help="the default; kept for explicitness")
    ap.add_argument("--list", action="store_true", help="list the latest discussions of the forum")
    a = ap.parse_args()

    if a.list:
        r = call("mod_forum_get_forum_discussions", forumid=a.forum, sortorder=-1, perpage=10)
        for d in r.get("discussions", []):
            print(DISCUSSION_URL.format(d["discussion"]), d["name"])
        return
    if a.draft is None:
        ap.error("give a draft file, or --list")

    subject, body = parse_draft(a.draft)
    if not a.send:
        print(f"DRY RUN (no request made), forum {a.forum}")
        print(f"Subject: {subject}")
        print(f"HTML: {len(body)} characters, {body.count('<a ')} links")
        print("\n".join(body.splitlines()[:8]))
        print("Add --send to post.")
        return

    r = call("mod_forum_add_discussion", forumid=a.forum, subject=subject, message=body)
    print(DISCUSSION_URL.format(r["discussionid"]))


if __name__ == "__main__":
    main()
