"""Matplotlib settings that make data figures match the book's page theme.

Figures are saved with a transparent background, drawn in the ink colors of
_static/custom.css, and each SVG carries a small stylesheet that switches the
ink and data colors to their dark-theme variants under
prefers-color-scheme: dark, as the hand-drawn SVG figures do.

Usage in a builder script:

    from book_figure_theme import PALETTE, apply_style, save
    apply_style("my-figure-salt")
    fig, ax = plt.subplots()
    ...
    save(fig, STATIC, "name")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

INK = "#1b2430"
MUTED = "#5c6874"
RULE = "#d2d9d7"
SOFT = "#eaeeec"

# Okabe-Ito colors, as in the other example builders.
PALETTE = {
    "blue": "#0072b2",
    "orange": "#e69f00",
    "green": "#009e73",
    "vermilion": "#d55e00",
    "sky": "#56b4e9",
    "purple": "#cc79a7",
    "gray": "#7b858a",
}

# Light color -> color used when the reader's system is in dark mode.
DARK = {
    INK: "#edf1f1",
    MUTED: "#a8b2b9",
    RULE: "#34434d",
    SOFT: "#263139",
    PALETTE["blue"]: "#5aa9e6",
    PALETTE["vermilion"]: "#f0884a",
    PALETTE["green"]: "#3fc29b",
    PALETTE["orange"]: "#f0b840",
    PALETTE["purple"]: "#e3a3c7",
    PALETTE["gray"]: "#9ca7ae",
}


def apply_style(salt: str) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9,
        "legend.fontsize": 7.5, "figure.dpi": 160, "savefig.dpi": 180,
        "svg.fonttype": "none", "svg.hashsalt": salt, "pdf.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.6, "grid.alpha": 1.0,
        "legend.frameon": False, "axes.titlelocation": "left",
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
    })


def _dark_stylesheet() -> str:
    rules = []
    for light, dark in DARK.items():
        for prop in ("fill", "stroke"):
            rules.append(f'[style*="{prop}: {light}"]{{{prop}:{dark}!important}}')
    return ("<style>@media (prefers-color-scheme: dark){" + "".join(rules) + "}</style>")


def save(fig, directory: Path, name: str) -> None:
    """Write name.svg, name.pdf and name.png with reproducible metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        options = {"metadata": {"Date": None}} if suffix == "svg" else {}
        if suffix == "pdf":
            options = {"metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(directory / f"{name}.{suffix}", bbox_inches="tight", pad_inches=.04,
                    transparent=True, **options)
    svg_path = directory / f"{name}.svg"
    text = "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    head, sep, rest = text.partition("<defs>")
    if sep:
        text = head + _dark_stylesheet() + "\n" + sep + rest
    svg_path.write_text(text)
    plt.close(fig)
