"""Draw a two-state MDP as a graph, and the same MDP with a cemetery state.

Used in infinite-horizon-mdps.md (random-horizon section).
Output: _static/mdp-graph-absorbing.svg
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parents[1] / "_static" / "mdp-graph-absorbing.svg"
R = 0.32  # node radius
ACTION_COLORS = {"a_1": "#1f6fb4", "a_2": "#c4532a"}


def node(ax, xy, label, dashed=False):
    ax.add_patch(
        Circle(xy, R, facecolor="white", edgecolor="black", lw=1.4,
               ls="--" if dashed else "-", zorder=3)
    )
    ax.text(*xy, label, ha="center", va="center", fontsize=15, zorder=4)


def edge(ax, a, b, label, color, rad=0.0, label_offset=(0, 0)):
    # start and end on the node boundaries, rotated toward the arc's bulge
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    tilt = math.atan(2 * rad)
    c, s = math.cos(tilt), math.sin(tilt)
    sa = (a[0] + R * (c * ux + s * uy), a[1] + R * (-s * ux + c * uy))
    eb = (b[0] - R * (c * ux - s * uy), b[1] - R * (s * ux + c * uy))
    arrow = FancyArrowPatch(
        sa, eb, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
        mutation_scale=16, lw=1.5, color=color, shrinkA=0, shrinkB=0, zorder=2,
    )
    ax.add_patch(arrow)
    mx = (a[0] + b[0]) / 2 + label_offset[0]
    my = (a[1] + b[1]) / 2 + label_offset[1]
    ax.text(mx, my, label, ha="center", va="center", fontsize=11.5, color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=5)


def loop(ax, xy, label, color, direction=180.0, label_dist=1.25, fontsize=11.5, label_xy=None):
    """Self-loop drawn as most of a small circle sitting outside the node."""
    rl = 0.30
    phi = math.radians(direction)
    cx, cy = xy[0] + (R + rl * 0.15) * math.cos(phi), xy[1] + (R + rl * 0.15) * math.sin(phi)
    # walk around the far side of the small circle, from one contact point to the other
    a0 = phi + math.radians(180 - 75)
    a1 = phi + math.radians(180 + 75) - 2 * math.pi
    pts = [(cx + rl * math.cos(a0 + (a1 - a0) * k / 40), cy + rl * math.sin(a0 + (a1 - a0) * k / 40))
           for k in range(41)]
    path = MplPath(pts)
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=14, lw=1.5,
                                 color=color, zorder=2))
    lx, ly = xy[0] + label_dist * math.cos(phi), xy[1] + label_dist * math.sin(phi)
    if label_xy is not None:
        lx, ly = label_xy
    ax.text(lx, ly, label, ha="center", va="center", fontsize=fontsize, color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=5)


def panel(ax, title, discounted):
    s1, s2, dead = (0.0, 0.0), (3.0, 0.0), (1.5, -2.4)
    ax.set_xlim(-1.4, 4.3)
    ax.set_ylim(-3.1, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=4)

    g = r"\gamma\," if discounted else ""
    node(ax, s1, r"$s_1$")
    node(ax, s2, r"$s_2$")
    loop(ax, s1, rf"$a_1$: ${g}p(s_1\mid s_1,a_1)$", ACTION_COLORS["a_1"], 150.0, label_xy=(-0.5, 0.92))
    edge(ax, s1, s2, rf"$a_2$: ${g}p(s_2\mid s_1,a_2)$", ACTION_COLORS["a_2"],
         rad=-0.35, label_offset=(0.55, 0.72))
    edge(ax, s2, s1, rf"$a_1$: ${g}p(s_1\mid s_2,a_1)$", ACTION_COLORS["a_1"],
         rad=-0.35, label_offset=(0, 0.0))
    if discounted:
        node(ax, dead, r"$\Delta$", dashed=True)
        edge(ax, s1, dead, r"$1-\gamma$", "0.35", rad=0.0, label_offset=(-0.55, -0.1))
        edge(ax, s2, dead, r"$1-\gamma$", "0.35", rad=0.0, label_offset=(0.55, -0.1))
        loop(ax, dead, "prob. 1, reward 0", "0.35", 0.0, 1.85, fontsize=10.5)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    panel(axes[0], "(a) Two-state MDP, drawn as a graph", discounted=False)
    panel(axes[1], r"(b) Same MDP with a cemetery state $\Delta$", discounted=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    if "--png" in sys.argv:
        fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=110)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
