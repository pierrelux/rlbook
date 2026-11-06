"""Draw a two-state MDP as a graph, and the same MDP with a cemetery state.

Used in infinite-horizon-mdps.md (random-horizon section).
Output: _static/mdp-graph-absorbing.svg

The SVG is written by hand rather than with matplotlib so that it matches the
book's look: transparent background, the page palette from _static/custom.css,
and a dark variant through prefers-color-scheme.
"""

import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "_static" / "mdp-graph-absorbing.svg"

W, H = 800, 330
R = 23  # node radius
PANEL_DX = 410  # x offset of panel (b)

STYLE = """
  .title { font: 600 11.5px 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
           letter-spacing: .07em; text-transform: uppercase; fill: #5c6874; }
  .node { fill: #ffffff; stroke: #1b2430; stroke-width: 1.6; }
  .node.dead { fill: none; stroke: #6e7982; stroke-dasharray: 4 3; }
  .nlab { font: italic 19px Newsreader, Georgia, 'Times New Roman', serif; fill: #1b2430; }
  .nlab.dead { font-style: normal; fill: #6e7982; }
  .lab { font: italic 14.5px Newsreader, Georgia, 'Times New Roman', serif; }
  .up { font-style: normal; }
  .note { font: 12px 'IBM Plex Sans', system-ui, -apple-system, sans-serif; fill: #6e7982; }
  .e { fill: none; stroke-width: 1.7; }
  .a1 { stroke: #2f6f8f; } .a1t { fill: #2f6f8f; } .a1h { fill: #2f6f8f; }
  .a2 { stroke: #b0642a; } .a2t { fill: #b0642a; } .a2h { fill: #b0642a; }
  .k  { stroke: #6e7982; } .kt  { fill: #6e7982; } .kh  { fill: #6e7982; }
  @media (prefers-color-scheme: dark) {
    .title { fill: #a8b2b9; }
    .node { fill: #1b2430; stroke: #edf1f1; }
    .node.dead { fill: none; stroke: #9ca7ae; }
    .nlab { fill: #edf1f1; }
    .nlab.dead { fill: #9ca7ae; }
    .note { fill: #9ca7ae; }
    .a1 { stroke: #72a8c2; } .a1t, .a1h { fill: #72a8c2; }
    .a2 { stroke: #e0a06a; } .a2t, .a2h { fill: #e0a06a; }
    .k { stroke: #9ca7ae; } .kt, .kh { fill: #9ca7ae; }
  }
"""


def f(x):
    return f"{x:.1f}"


def on_circle(c, toward, r=R):
    dx, dy = toward[0] - c[0], toward[1] - c[1]
    n = math.hypot(dx, dy)
    return (c[0] + r * dx / n, c[1] + r * dy / n)


def curve(a, b, ctrl, cls):
    """Quadratic edge from node a to node b bending through ctrl."""
    p0 = on_circle(a, ctrl)
    p1 = on_circle(b, ctrl, R + 1)
    return (f'<path class="e {cls}" d="M{f(p0[0])},{f(p0[1])} Q{f(ctrl[0])},{f(ctrl[1])} '
            f'{f(p1[0])},{f(p1[1])}" marker-end="url(#h-{cls})"/>')


def line(a, b, cls):
    p0 = on_circle(a, b)
    p1 = on_circle(b, a, R + 1)
    return (f'<path class="e {cls}" d="M{f(p0[0])},{f(p0[1])} L{f(p1[0])},{f(p1[1])}" '
            f'marker-end="url(#h-{cls})"/>')


def self_loop(c, angle_deg, cls, size=34):
    """Loop leaving and re-entering node c around direction angle_deg."""
    th = math.radians(angle_deg)
    spread = math.radians(28)
    p0 = (c[0] + R * math.cos(th - spread), c[1] + R * math.sin(th - spread))
    p1 = (c[0] + (R + 1) * math.cos(th + spread), c[1] + (R + 1) * math.sin(th + spread))
    c0 = (c[0] + (R + size * 2.6) * math.cos(th - spread * 1.05),
          c[1] + (R + size * 2.6) * math.sin(th - spread * 1.05))
    c1 = (c[0] + (R + size * 2.6) * math.cos(th + spread * 1.05),
          c[1] + (R + size * 2.6) * math.sin(th + spread * 1.05))
    return (f'<path class="e {cls}" d="M{f(p0[0])},{f(p0[1])} C{f(c0[0])},{f(c0[1])} '
            f'{f(c1[0])},{f(c1[1])} {f(p1[0])},{f(p1[1])}" marker-end="url(#h-{cls})"/>')


def node(c, name, sub=None, dead=False):
    k = " dead" if dead else ""
    out = f'<circle class="node{k}" cx="{f(c[0])}" cy="{f(c[1])}" r="{R}"/>'
    if sub is None:
        out += (f'<text class="nlab{k}" x="{f(c[0])}" y="{f(c[1] + 6.5)}" '
                f'text-anchor="middle">{name}</text>')
    else:
        out += (f'<text class="nlab" x="{f(c[0] - 2)}" y="{f(c[1] + 5.5)}" text-anchor="middle">'
                f'{name}<tspan class="up" font-size="12" dy="5">{sub}</tspan></text>')
    return out


def sub(t):
    return f'<tspan class="up" font-size="10.5" dy="3.5">{t}</tspan><tspan dy="-3.5"></tspan>'


def prob(j, s, a, scaled):
    g = "γ " if scaled else ""
    return (f'{g}p(s{sub(j)} | s{sub(s)}, a{sub(a)})')


def label(x, y, cls, action, text, anchor="middle"):
    return (f'<text class="lab {cls}t" x="{f(x)}" y="{f(y)}" text-anchor="{anchor}">'
            f'a{sub(action)}<tspan class="up">: </tspan>{text}</text>')


def panel(dx, scaled):
    y = 150
    s1, s2, d = (dx + 62, y), (dx + 328, y), (dx + 195, 278)
    parts = [
        f'<text class="title" x="{dx + 4}" y="22">'
        + ("(b) with a cemetery state" if scaled else "(a) a two-state MDP") + "</text>",
        self_loop(s1, -90, "a1", size=24),
        label(s1[0] - 30, 52, "a1", "1", prob("1", "1", "1", scaled), anchor="start"),
        curve(s1, s2, (dx + 195, y - 92), "a2"),
        label(dx + 195, y - 20, "a2", "2", prob("2", "1", "2", scaled)),
        curve(s2, s1, (dx + 195, y + 92), "a1"),
        label(dx + 195, y + 26, "a1", "1", prob("1", "2", "1", scaled)),
    ]
    if scaled:
        parts += [
            curve(s1, d, (dx + 70, d[1] - 10), "k"),
            curve(s2, d, (dx + 320, d[1] - 10), "k"),
            f'<text class="lab kt" x="{f(dx + 60)}" y="{f(250)}" text-anchor="end">'
            f'<tspan class="up">1 − </tspan>γ</text>',
            f'<text class="lab kt" x="{f(dx + 330)}" y="{f(250)}" text-anchor="start">'
            f'<tspan class="up">1 − </tspan>γ</text>',
            self_loop(d, 90, "k", size=12),
            f'<text class="note" x="{f(d[0] + 40)}" y="{f(d[1] + 44)}">probability 1, reward 0</text>',
        ]
    parts += [node(s1, "s", "1"), node(s2, "s", "2")]
    if scaled:
        parts.append(node(d, "Δ", dead=True))
    return "\n  ".join(parts)


def markers():
    out = []
    for cls in ("a1", "a2", "k"):
        out.append(
            f'<marker id="h-{cls}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7.5" '
            f'markerHeight="7.5" orient="auto-start-reverse">'
            f'<path class="{cls}h" d="M0,0.8 L10,5 L0,9.2 z"/></marker>'
        )
    return "".join(out)


def main():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
     aria-label="A two-state MDP drawn as a graph, and the same MDP with a cemetery state">
<style>{STYLE}</style>
<defs>{markers()}</defs>
<g>
  {panel(0, scaled=False)}
</g>
<g>
  {panel(PANEL_DX, scaled=True)}
</g>
</svg>
"""
    OUT.write_text(svg)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
