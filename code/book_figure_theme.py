"""Matplotlib settings that make data figures match the book's page theme.

Figures are drawn in the ink colors of _static/custom.css, on a transparent
background, with Computer Modern text and math, so that labels look like the
equations on the page. Text is written as outlines, so every reader sees the
same glyphs whatever fonts their system has. Each SVG carries a small
stylesheet that switches the ink and data colors to their dark-theme variants
under prefers-color-scheme: dark, as the hand-drawn SVG figures do.

Usage in a builder script:

    from book_figure_theme import PALETTE, apply_style, save
    apply_style("my-figure-salt")
    fig, ax = plt.subplots()
    ...
    save(fig, STATIC, "name")

Usage in a notebook cell, where the figure is the output of the cell:

    from book_figure_theme import figure_style, show
    with figure_style():
        fig, ax = plt.subplots()
        ...
    show(fig)
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import io

import matplotlib as mpl
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
    PALETTE["sky"]: "#7cc6f0",
    PALETTE["purple"]: "#e3a3c7",
    PALETTE["gray"]: "#9ca7ae",
}

TEXT_WIDTH = 7.0   # inches; the reading column at the book's default zoom


def rc(salt: str = "rlbook") -> dict:
    """The rcParams of the book's data figures."""
    return {
        "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": True,
        "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "figure.dpi": 160, "savefig.dpi": 200,
        "svg.fonttype": "path", "svg.hashsalt": salt, "pdf.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
        "axes.linewidth": 0.7, "axes.titlepad": 7, "axes.labelpad": 3,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": RULE, "grid.linewidth": 0.6,
        "lines.linewidth": 1.5, "lines.markersize": 4,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "axes.titlelocation": "left",
        "figure.facecolor": "none", "axes.facecolor": "none",
        "savefig.transparent": True,
    }


def apply_style(salt: str) -> None:
    plt.rcParams.update(rc(salt))


@contextmanager
def figure_style(salt: str = "rlbook"):
    """The book style for figures drawn inside a notebook cell."""
    with mpl.rc_context(rc(salt)):
        yield


def label(ax, x, y, text, color=INK, **kwargs) -> None:
    """A direct label next to the curve it names, in the curve's color."""
    kwargs.setdefault("fontsize", 8.5)
    kwargs.setdefault("va", "center")
    ax.text(x, y, text, color=color, **kwargs)


def _dark_stylesheet() -> str:
    rules = []
    for light, dark in DARK.items():
        for prop in ("fill", "stroke"):
            rules.append(f'[style*="{prop}: {light}"]{{{prop}:{dark}!important}}')
    return ("<style>@media (prefers-color-scheme: dark){" + "".join(rules) + "}</style>")


def _themed_svg(text: str) -> str:
    text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    head, sep, rest = text.partition("<defs>")
    if sep:
        return head + _dark_stylesheet() + "\n" + sep + rest
    head, sep, rest = text.partition("<g id=")
    return head + _dark_stylesheet() + "\n" + sep + rest


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
    svg_path.write_text(_themed_svg(svg_path.read_text()))
    plt.close(fig)


def show(fig, alt: str = "") -> None:
    """Display the figure as a themed SVG image, the only representation of the cell's output.

    The book theme drops image/svg+xml outputs and prefers text/plain to them, so the
    SVG is embedded in an HTML image, where its dark-theme stylesheet still applies.
    """
    import base64
    import html

    from IPython.display import display

    buffer = io.StringIO()
    with mpl.rc_context(rc()):           # fonts and text outlines are resolved when drawing
        fig.savefig(buffer, format="svg", bbox_inches="tight", pad_inches=.04, transparent=True,
                    metadata={"Date": None})
    plt.close(fig)
    svg = _themed_svg(buffer.getvalue())
    data = base64.b64encode(svg.encode()).decode()
    display({"text/html": f'<img src="data:image/svg+xml;base64,{data}" alt="{html.escape(alt)}" '
                          'style="display:block;width:100%;height:auto;margin:0 auto">'}, raw=True)
