"""Draw the original MDP and the Gumbel-augmented MDP as graphical models.

Used in regularized-dp.md (section on Gumbel noise on the rewards).
Output: _static/gumbel-augmented-pgm.svg

Panel (a) is the original MDP under its softmax policy: the action is a random
child of the state. Panel (b) is the augmented MDP: a fresh shock vector
eps_t is drawn every period, the augmented state is (s_t, eps_t), and the
action is a deterministic function of it (the hard argmax). The shock has no
edge into s_{t+1} or eps_{t+1}; that missing edge is Rust's conditional
independence assumption, and it is what lets eps_t be integrated out slice by
slice, which turns panel (b) back into panel (a).

The SVG is written by hand rather than with matplotlib so that it matches the
book's look: transparent background, the page palette from _static/custom.css,
and a dark variant through prefers-color-scheme. Same conventions as
scripts/build_mdp_graph_figure.py.
"""

import math
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "_static" / "gumbel-augmented-pgm.svg"

W, H = 870, 350
R = 22  # node radius
PANEL_DX = 430  # x offset of panel (b)
Y_EPS, Y_A, Y_S = 92, 172, 252
COLS = (40, 200, 360)  # x of s_t, s_{t+1}, s_{t+2} within a panel

STYLE = """
  .title { font: 600 11.5px 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
           letter-spacing: .07em; text-transform: uppercase; fill: #5c6874; }
  .node { fill: #ffffff; stroke: #1b2430; stroke-width: 1.6; }
  .node.inner { fill: none; stroke-width: 1.2; }
  .node.eps { fill: #ffffff; stroke: #2f6f8f; stroke-dasharray: 4 3; }
  .nlab { font: italic 18px Newsreader, Georgia, 'Times New Roman', serif; fill: #1b2430; }
  .nlab.eps { fill: #2f6f8f; }
  .up { font-style: normal; }
  .lab { font: italic 14px Newsreader, Georgia, 'Times New Roman', serif; fill: #5c6874; }
  .eq { font: italic 14.5px Newsreader, Georgia, 'Times New Roman', serif; fill: #1b2430; }
  .note { font: 12px 'IBM Plex Sans', system-ui, -apple-system, sans-serif; fill: #6e7982; }
  .box { fill: none; stroke: #9ca7ae; stroke-width: 1.1; stroke-dasharray: 2 3; }
  .e { fill: none; stroke-width: 1.6; }
  .k { stroke: #1b2430; } .kh { fill: #1b2430; }
  .b { stroke: #2f6f8f; } .bh { fill: #2f6f8f; }
  .dots { font: 20px Georgia, serif; fill: #6e7982; }
  @media (prefers-color-scheme: dark) {
    .title { fill: #a8b2b9; }
    .node { fill: #1b2430; stroke: #edf1f1; }
    .node.inner { fill: none; }
    .node.eps { fill: #1b2430; stroke: #72a8c2; }
    .nlab { fill: #edf1f1; }
    .nlab.eps { fill: #72a8c2; }
    .lab { fill: #a8b2b9; }
    .eq { fill: #edf1f1; }
    .note { fill: #9ca7ae; }
    .box { stroke: #6e7982; }
    .k { stroke: #edf1f1; } .kh { fill: #edf1f1; }
    .b { stroke: #72a8c2; } .bh { fill: #72a8c2; }
    .dots { fill: #9ca7ae; }
  }
"""


def f(x):
    return f"{x:.1f}"


def mtext(src, size=11):
    """Render 'a_{t} ~ π_{β}' style markup as tspans.

    Each run carries its own dy, so the baseline comes back after a sub- or
    superscript (an empty tspan used only to reset dy is ignored by browsers).
    """
    out, base = [], 0
    for m in re.finditer(r"([_^])\{([^}]*)\}|([^_^]+)", src):
        if m.group(3) is not None:
            target, text, extra = 0, m.group(3), ""
        else:
            target = 4 if m.group(1) == "_" else -6
            text, extra = m.group(2), f' class="up" font-size="{size}"'
        dy = target - base
        base = target
        if target == 0:  # operator names and words are set upright
            pieces = re.split(r"(action|reward|argmax|exp)", text)
            out.append(f'<tspan dy="{dy}">{pieces[0]}</tspan>')
            for k, piece in enumerate(pieces[1:]):
                cls = ' class="up"' if k % 2 == 0 else ""
                out.append(f'<tspan{cls}>{piece}</tspan>')
        else:
            out.append(f'<tspan{extra} dy="{dy}">{text}</tspan>')
    return "".join(out)


def on_circle(c, toward, r=R):
    dx, dy = toward[0] - c[0], toward[1] - c[1]
    n = math.hypot(dx, dy)
    return (c[0] + r * dx / n, c[1] + r * dy / n)


def edge(a, b, cls="k"):
    p0 = on_circle(a, b)
    p1 = on_circle(b, a, R + 1.5)
    return (f'<path class="e {cls}" d="M{f(p0[0])},{f(p0[1])} L{f(p1[0])},{f(p1[1])}" '
            f'marker-end="url(#h-{cls})"/>')


def node(c, name, t, kind="plain"):
    cls = "node eps" if kind == "eps" else "node"
    out = f'<circle class="{cls}" cx="{f(c[0])}" cy="{f(c[1])}" r="{R}"/>'
    if kind == "det":  # double circle: a deterministic function of its parents
        out += f'<circle class="node inner" cx="{f(c[0])}" cy="{f(c[1])}" r="{R - 4}"/>'
    lab = "nlab eps" if kind == "eps" else "nlab"
    out += (f'<text class="{lab}" x="{f(c[0] - 3)}" y="{f(c[1] + 5)}" text-anchor="middle">'
            f'{mtext(name + "_{" + t + "}")}</text>')
    return out


def panel(dx, augmented):
    ts = ("t", "t+1", "t+2")
    s = [(dx + x, Y_S) for x in COLS]
    a = [(dx + (COLS[i] + COLS[i + 1]) / 2, Y_A) for i in range(2)]
    e = [(dx + x, Y_EPS) for x in COLS]
    title = "(b) augmented MDP, Gumbel shocks" if augmented else "(a) original MDP, softmax policy"
    parts = [f'<text class="title" x="{dx + 6}" y="22">{title}</text>']
    if augmented:
        for x in COLS:
            parts.append(f'<rect class="box" x="{f(dx + x - 31)}" y="{f(Y_EPS - 33)}" width="62" '
                         f'height="{f(Y_S - Y_EPS + 66)}" rx="14"/>')
    for i in range(2):
        parts.append(edge(s[i], s[i + 1]))
        parts.append(edge(s[i], a[i]))
        parts.append(edge(a[i], s[i + 1]))
        if augmented:
            parts.append(edge(e[i], a[i], "b"))
    # edge labels on the first slice
    if not augmented:
        parts.append(f'<text class="lab" x="{f((s[0][0] + a[0][0]) / 2 - 12)}" '
                     f'y="{f((s[0][1] + a[0][1]) / 2 - 6)}" text-anchor="end">{mtext("π_{β}")}</text>')
        parts.append(f'<text class="lab" x="{f((a[0][0] + s[1][0]) / 2 + 12)}" '
                     f'y="{f((a[0][1] + s[1][1]) / 2 - 6)}">p</text>')
    for i in range(3):
        parts.append(node(s[i], "s", ts[i]))
        if augmented:
            parts.append(node(e[i], "ε", ts[i], kind="eps"))
    for i in range(2):
        parts.append(node(a[i], "a", ts[i], kind="det" if augmented else "plain"))
    parts.append(f'<text class="dots" x="{f(dx + COLS[2] + 40)}" y="{f(Y_S + 6)}">⋯</text>')
    # the action rule and the per-step reward, written under the graph
    y0 = 306
    if augmented:
        rule = "action  a_{t} = argmax_{a} { q(s_{t}, a) + ε_{t}(a) }"
        rew = "reward  r(s_{t}, a_{t}) + ε_{t}(a_{t})"
    else:
        rule = "action  a_{t} ~ π_{β}(· | s_{t}) ∝ exp(β q(s_{t}, ·))"
        rew = "reward  r(s_{t}, a_{t}) + β^{−1} H(π_{β}(· | s_{t}))"
    parts.append(f'<text class="eq" x="{dx + 6}" y="{y0}">{mtext(rule)}</text>')
    parts.append(f'<text class="eq" x="{dx + 6}" y="{y0 + 24}">{mtext(rew)}</text>')
    return "\n  ".join(parts)


def markers():
    out = []
    for cls in ("k", "b"):
        out.append(
            f'<marker id="h-{cls}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse">'
            f'<path class="{cls}h" d="M0,0.8 L10,5 L0,9.2 z"/></marker>'
        )
    return "".join(out)


def main():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
     aria-label="Graphical models of the original MDP under a softmax policy and of the augmented MDP with Gumbel shocks">
<style>{STYLE}</style>
<defs>{markers()}</defs>
<g>
  {panel(0, augmented=False)}
</g>
<g>
  {panel(PANEL_DX, augmented=True)}
</g>
</svg>
"""
    OUT.write_text(svg)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
