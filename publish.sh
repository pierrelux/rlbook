#!/usr/bin/env bash
set -euo pipefail

# Build the book and browser labs for subdirectory deployment.
BOOK_BASE_URL="${BASE_URL:-/rlbook}"
if ! uv run micromamba --version >/dev/null 2>&1; then
  echo "micromamba is required to build the xeus-python browser lab." >&2
  exit 1
fi

BASE_URL="$BOOK_BASE_URL" uv run jupyter-book build --html --execute --strict
# French edition into _build/html/fr. It reuses the English execution cache.
BASE_URL="$BOOK_BASE_URL" uv run python tools/french.py build
uv run python tools/french.py status
uv run jupyter lite build \
  --contents notebooks \
  --output-dir _build/html/lab \
  --lite-dir lab

# Deploy the assembled site to GitHub Pages.
ghp-import -n -p -f _build/html
