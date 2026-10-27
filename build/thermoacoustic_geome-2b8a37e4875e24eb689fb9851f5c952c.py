"""Render the geometry schematic for the thermoacoustic pull-down example.

The top panel draws the physical device: a sealed resonator with a loudspeaker
at one end, a stack of thin plates, and two heat exchangers. The bottom panel
shows how that device collapses into the two-state lumped model used by iLQR
and DDP in the chapter.

Run from the repository root with:
    uv run python code/thermoacoustic_geometry.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np

STYLE = {
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "svg.fonttype": "none",
    "svg.hashsalt": "thermoacoustic-geometry",
    "pdf.fonttype": 42,
}
BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#6B7480"
INK = "#303840"


def arrow(ax, start, end, color=INK, lw=1.4, style="-|>", ms=11, **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, color=color,
                                 lw=lw, mutation_scale=ms, **kw))


def device_panel(ax):
    """Sealed tube: driver, hot exchanger, stack, cold exchanger, standing wave."""
    ax.set(xlim=(-1.9, 11.4), ylim=(-2.9, 2.35))
    ax.axis("off")
    x0, x1, yb, yt = 0.0, 10.0, -0.7, 0.7

    # Tube wall and gas.
    ax.add_patch(Rectangle((x0, yb), x1 - x0, yt - yb, fc="#F3F5F7", ec=INK, lw=1.6))

    # Loudspeaker driver: piston face inside the tube, cone drawn at the left.
    ax.add_patch(Polygon([(-0.95, -0.55), (-0.95, 0.55), (-0.15, 0.62), (-0.15, -0.62)],
                         closed=True, fc="#D9DEE3", ec=INK, lw=1.2))
    ax.add_patch(Rectangle((-0.15, -0.62), 0.3, 1.24, fc=GRAY, ec=INK, lw=1.0))
    arrow(ax, (0.35, 1.0), (1.05, 1.0), color=INK, style="<|-|>", ms=9, lw=1.2)
    ax.text(-0.55, 1.35, "loudspeaker,\namplitude $u\\in[0,1]$", ha="center",
            va="bottom", color=INK, fontsize=9.5)
    ax.text(0.7, 1.12, "piston", ha="center", va="bottom", fontsize=8, color=INK)

    # Hot exchanger (near the driver, pressure antinode) and cold exchanger.
    for x, color in ((2.0, ORANGE), (4.9, BLUE)):
        ax.add_patch(Rectangle((x, yb), 0.45, yt - yb, fc=color, ec=INK, lw=1.0, alpha=.85))
        for y in np.linspace(yb + .15, yt - .15, 5):
            ax.plot([x + .05, x + .4], [y, y], color="white", lw=.9)

    # Stack of thin parallel plates between the exchangers.
    xs, xe = 2.55, 4.85
    for y in np.linspace(yb + .12, yt - .12, 7):
        ax.plot([xs, xe], [y, y], color=INK, lw=1.1)
    ax.text((xs + xe) / 2, yb - .12, "stack of thin plates", ha="center",
            va="top", fontsize=9.5, color=INK)

    # One gas parcel shuttling along a plate: compress near the driver, expand away.
    yp = 0.0
    ax.add_patch(FancyBboxPatch((3.05, yp - .07), .35, .14,
                                boxstyle="round,pad=0.02", fc="white", ec=INK, lw=.9))
    ax.add_patch(FancyBboxPatch((3.95, yp - .07), .45, .14,
                                boxstyle="round,pad=0.02", fc="white", ec=INK, lw=.9, ls=":"))
    arrow(ax, (3.45, yp), (3.9, yp), color=INK, style="<|-|>", ms=8, lw=1.0)
    ax.annotate("a gas parcel oscillates along a plate,\nhanding heat toward the hot end",
                xy=(3.9, yp - .12), xytext=(7.9, -1.05), ha="center", va="top",
                fontsize=8.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": GRAY, "lw": .8})

    # Net heat flow through the stack: cold end to hot end.
    arrow(ax, (4.75, yt + .5), (2.65, yt + .5), color=ORANGE, lw=1.8, ms=12)
    ax.text(3.7, yt + .58, "net heat flow", ha="center", va="bottom",
            fontsize=8.5, color=ORANGE)

    # External couplings of the two exchangers.
    arrow(ax, (2.22, yb - .05), (2.22, yb - 1.15), color=ORANGE, lw=1.8, ms=12)
    ax.text(2.22, yb - 1.25, "reject $UA\\,(T_h-T_{\\mathrm{amb}})$\nto ambient air",
            ha="center", va="top", fontsize=9, color=ORANGE)
    ax.text(2.22, yt + .12, "hot exchanger, $T_h$", ha="center", va="bottom",
            fontsize=9.5, color=ORANGE, fontweight="bold")

    arrow(ax, (5.12, yb - 1.15), (5.12, yb - .05), color=BLUE, lw=1.8, ms=12)
    ax.text(5.12, yb - 1.25, "payload leak $Q_{\\mathrm{load}}$\ninto the cold side",
            ha="center", va="top", fontsize=9, color=BLUE)
    ax.text(5.12, yt + .12, "cold exchanger, $T_c$", ha="center", va="bottom",
            fontsize=9.5, color=BLUE, fontweight="bold")

    # Standing-wave pressure amplitude along the tube (half-wave resonator).
    x = np.linspace(x0, x1, 200)
    env = 0.22 * np.abs(np.cos(np.pi * (x - x0) / (x1 - x0)))
    ax.plot(x, env, color=GREEN, lw=1.1, ls="--")
    ax.plot(x, -env, color=GREEN, lw=1.1, ls="--")
    ax.text(8.0, .28, "standing-wave pressure envelope", ha="center", va="bottom",
            fontsize=8.5, color=GREEN)
    ax.text(7.7, -.28, "antinodes at the piston and the sealed end, node mid-tube", ha="center",
            va="top", fontsize=7.5, color=GREEN)
    ax.text(9.9, -.85, "sealed end", ha="right", va="top", fontsize=8.5, color=INK)
    ax.text(10.0, .85, "resonator (gas-filled tube)", ha="right", va="bottom",
            fontsize=9.5, color=INK)


def model_panel(ax):
    """Two lumped thermal masses: the state, the action, and the energy flows."""
    ax.set(xlim=(-0.4, 11.4), ylim=(-2.1, 1.55))
    ax.axis("off")

    def node(x, color, title, sub):
        ax.add_patch(FancyBboxPatch((x - .85, -.5), 1.7, 1.0,
                                    boxstyle="round,pad=0.05", fc="white", ec=color, lw=1.8))
        ax.text(x, .0, title, ha="center", va="center", fontsize=12, color=color)
        ax.text(x, .62, sub, ha="center", va="bottom", fontsize=8.5, color=color)

    xu, xh, xc = 0.55, 3.7, 9.4
    node(xh, ORANGE, "$T_h$", "hot exchanger, $C_h=50$ J/K")
    node(xc, BLUE, "$T_c$", "cold exchanger + payload, $C_c=20$ J/K")

    # Action: driver amplitude feeds acoustic work into the hot side.
    ax.add_patch(FancyBboxPatch((xu - .55, -.35), 1.1, .7, boxstyle="round,pad=0.05",
                                fc="#F3F5F7", ec=INK, lw=1.2))
    ax.text(xu, 0, "$u_t$", ha="center", va="center", fontsize=12, color=INK)
    ax.text(xu, .62, "action", ha="center", va="bottom", fontsize=8.5, color=INK)
    arrow(ax, (xu + .6, 0), (xh - .9, 0), color=INK, lw=1.6)
    xm = (xu + .6 + xh - .9) / 2
    ax.text(xm, .12, "work $\\dot W$", ha="center", va="bottom",
            fontsize=9, color=INK)
    ax.text(xm, -.14, "stage cost $h\\,w_E\\dot W$", ha="center", va="top",
            fontsize=8, color=GRAY)

    # Heat pumped from cold to hot, weakened by the temperature span.
    arrow(ax, (xc - .9, 0), (xh + .9, 0), color=BLUE, lw=1.8)
    xm = (xh + xc) / 2
    ax.text(xm, .12, "pumped heat $\\dot Q_c=k_q u^2\\,\\eta(\\Delta T)$",
            ha="center", va="bottom", fontsize=9, color=BLUE)
    ax.text(xm, -.14, "$\\eta=1-\\Delta T/\\Delta T_0$ weakens with the span",
            ha="center", va="top", fontsize=8, color=GRAY)

    # External couplings.
    arrow(ax, (xh, -.57), (xh, -1.3), color=ORANGE, lw=1.6)
    ax.text(xh + .15, -.95, "$UA\\,(T_h-T_{\\mathrm{amb}})$ to ambient", ha="left",
            va="center", fontsize=9, color=ORANGE)
    arrow(ax, (xc, -1.3), (xc, -.57), color=BLUE, lw=1.6)
    ax.text(xc + .15, -.95, "$Q_{\\mathrm{load}}$", ha="left", va="center",
            fontsize=9, color=BLUE)
    ax.text(xc, -1.45, "terminal cost $w_T(T_{c,T}-T^\\star)^2$",
            ha="center", va="top", fontsize=8.5, color=BLUE)

    ax.text(xu, -1.0, "state $\\mathbf{x}_t=(T_{c,t},\\,T_{h,t})$\nRK4 step, $h=1$ s",
            ha="left", va="top", fontsize=9, color=INK, linespacing=1.4,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GRAY, "lw": .8})


def geometry():
    with plt.rc_context(STYLE):
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(7.2, 5.9), gridspec_kw={"height_ratios": [1.45, 1]})
        fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.98, hspace=0)
        device_panel(top)
        model_panel(bottom)
        for ax, tag in ((top, "(a) device"), (bottom, "(b) lumped model")):
            ax.text(0.0, 1.0, tag, transform=ax.transAxes, ha="left", va="top",
                    fontsize=10, color=INK, fontweight="bold")
    return fig


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "_static" / "thermoacoustic_pulldown"
    output.mkdir(parents=True, exist_ok=True)
    figure = geometry()
    for extension in ("svg", "pdf", "png"):
        figure.savefig(output / f"geometry.{extension}", dpi=150,
                       metadata={"Date": None} if extension == "svg" else None)
    plt.close(figure)
