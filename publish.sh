#!/bin/bash
# Build with BASE_URL=/rlbook for subdirectory deployment
BASE_URL=/rlbook uv run jupyter-book build --html
# Deploy to GitHub Pages
ghp-import -n -p -f _build/html
