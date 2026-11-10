"""Draw behavioral cloning and Rust's inverse RL estimator as computation graphs.

Used in regularized-dp.md (section on Rust's bus engines and inverse RL).
Output: _static/irl-computation-graph.svg

Panel (a) is behavioral cloning: the state enters a generic network that
outputs a distribution over actions, and the cross-entropy loss on the
observed action is minimized by backpropagation. Panel (b) is Rust's
estimator: the same data and the same loss, but the network is
theta -> reward -> Bellman operator applied until its fixed point -> softmax.
Under panel (b), two ways to carry the gradient from v back to theta:
unrolling the K Bellman layers, or one implicit-function step that is valid
only at the fixed point.

Hand-written SVG with the same conventions as scripts/build_gumbel_pgm_figure.py:
transparent background, page palette from _static/custom.css, dark variant
through prefers-color-scheme.
"""

import math
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "_static" / "irl-computation-graph.svg"

W, H = 900, 405
R = 21  # radius of data / variable nodes
BH = 42  # box height
YA = 92  # row of panel (a)
YB = 268  # row of panel (b)

STYLE = """
  .title { font: 600 11.5px 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
           letter-spacing: .07em; text-transform: uppercase; fill: #5c6874; }
  .node { fill: #ffffff; stroke: #1b2430; stroke-width: 1.6; }
  .data { fill: #eef2f3; stroke: #1b2430; stroke-width: 1.6; }
  .op { fill: #ffffff; stroke: #1b2430; stroke-width: 1.4; }
  .bell { fill: #ffffff; stroke: #2f6f8f; stroke-width: 1.4; }
  .fix { fill: none; stroke: #2f6f8f; stroke-width: 1.1; stroke-dasharray: 2 3; }
  .nlab { font: italic 17px Newsreader, Georgia, 'Times New Roman', serif; fill: #1b2430; }
  .blab { font: italic 15px Newsreader, Georgia, 'Times New Roman', serif; fill: #1b2430; }
  .blab.bl { fill: #2f6f8f; }
  .up { font-style: normal; }
  .lab { font: italic 14px Newsreader, Georgia, 'Times New Roman', serif; fill: #5c6874; }
  .note { font: 12px 'IBM Plex Sans', system-ui, -apple-system, sans-serif; fill: #6e7982; }
  .gnote { font: 12.5px 'IBM Plex Sans', system-ui, -apple-system, sans-serif; fill: #b0543a; }
  .geq { font: italic 14.5px Newsreader, Georgia, 'Times New Roman', serif; fill: #b0543a; }
  .e { fill: none; stroke-width: 1.6; }
  .k { stroke: #1b2430; } .kh { fill: #1b2430; }
  .g { stroke: #b0543a; } .gh { fill: #b0543a; }
  .gd { stroke: #b0543a; stroke-dasharray: 5 4; }
  .rule { stroke: #d5dcdf; stroke-width: 1; }
  .dots { font: 20px Georgia, serif; fill: #2f6f8f; }
  @media (prefers-color-scheme: dark) {
    .title { fill: #a8b2b9; }
    .node, .op { fill: #1b2430; stroke: #edf1f1; }
    .data { fill: #26313d; stroke: #edf1f1; }
    .bell { fill: #1b2430; stroke: #72a8c2; }
    .fix { stroke: #72a8c2; }
    .nlab, .blab { fill: #edf1f1; }
    .blab.bl { fill: #72a8c2; }
    .lab { fill: #a8b2b9; }
    .note { fill: #9ca7ae; }
    .gnote, .geq { fill: #e08a6d; }
    .k { stroke: #edf1f1; } .kh { fill: #edf1f1; }
    .g, .gd { stroke: #e08a6d; } .gh { fill: #e08a6d; }
    .rule { stroke: #3a4652; }
    .dots { fill: #72a8c2; }
  }
"""


def f(x):
    return f"{x:.1f}"


def mtext(src, size=11):
    """Render 'π_{θ}(· | s_{t})' style markup as tspans (same as the PGM figure)."""
    out, base = [], 0
    for m in re.finditer(r"([_^])\{([^}]*)\}|([^_^]+)", src):
        if m.group(3) is not None:
            target, text, extra = 0, m.group(3), ""
        else:
            target = 4 if m.group(1) == "_" else -6
            text, extra = m.group(2), f' class="up" font-size="{size}"'
        dy = target - base
        base = target
        text = text.replace(" ", "\u00a0")  # keep spaces between tspans
        if target == 0:
            pieces = re.split(r"(softmax|reward|network|loss|log|as in)", text)
            out.append(f'<tspan dy="{dy}">{pieces[0]}</tspan>')
            for k, piece in enumerate(pieces[1:]):
                cls = ' class="up"' if k % 2 == 0 else ""
                lead = len(piece) - len(piece.lstrip("\u00a0"))
                gap = f' dx="{4 * lead}"' if lead else ""  # leading spaces are dropped by some renderers
                out.append(f'<tspan{cls}{gap}>{piece.lstrip(chr(160))}</tspan>')
        else:
            out.append(f'<tspan{extra} dy="{dy}">{text}</tspan>')
    return "".join(out)


class Circle:
    def __init__(self, x, y, label, cls="node"):
        self.x, self.y, self.label, self.cls = x, y, label, cls

    def anchor(self, toward, pad=0.0):
        dx, dy = toward[0] - self.x, toward[1] - self.y
        n = math.hypot(dx, dy)
        return (self.x + (R + pad) * dx / n, self.y + (R + pad) * dy / n)

    def svg(self):
        return (f'<circle class="{self.cls}" cx="{f(self.x)}" cy="{f(self.y)}" r="{R}"/>'
                f'<text class="nlab" x="{f(self.x)}" y="{f(self.y + 5)}" text-anchor="middle">'
                f'{mtext(self.label)}</text>')


class Box:
    def __init__(self, x, y, w, label, cls="op", lcls="blab"):
        self.x, self.y, self.w, self.label, self.cls, self.lcls = x, y, w, label, cls, lcls

    def anchor(self, toward, pad=0.0):
        # point where the segment from the centre to `toward` leaves the box
        dx, dy = toward[0] - self.x, toward[1] - self.y
        hx, hy = self.w / 2 + pad, BH / 2 + pad
        t = min(hx / abs(dx) if dx else math.inf, hy / abs(dy) if dy else math.inf)
        return (self.x + t * dx, self.y + t * dy)

    def svg(self):
        return (f'<rect class="{self.cls}" x="{f(self.x - self.w / 2)}" y="{f(self.y - BH / 2)}" '
                f'width="{f(self.w)}" height="{BH}" rx="7"/>'
                f'<text class="{self.lcls}" x="{f(self.x)}" y="{f(self.y + 5)}" text-anchor="middle">'
                f'{mtext(self.label)}</text>')


def arrow(a, b, cls="k"):
    p0 = a.anchor((b.x, b.y))
    p1 = b.anchor((a.x, a.y), 1.5)
    return (f'<path class="e {cls}" d="M{f(p0[0])},{f(p0[1])} L{f(p1[0])},{f(p1[1])}" '
            f'marker-end="url(#h-{cls[0]})"/>')


def markers():
    return "".join(
        f'<marker id="h-{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path class="{c}h" d="M0,0.8 L10,5 L0,9.2 z"/></marker>'
        for c in ("k", "g")
    )


def panel_a():
    s = Circle(48, YA, "s_{t}", "data")
    net = Box(205, YA, 150, "network π_{w}")
    pi = Box(445, YA, 120, "π_{w}(· | s_{t})")
    loss = Box(640, YA, 170, "loss −log π_{w}(a_{t} | s_{t})")
    a = Circle(830, YA, "a_{t}", "data")
    parts = ['<text class="title" x="6" y="24">(a) behavioral cloning</text>']
    parts += [arrow(s, net), arrow(net, pi), arrow(pi, loss), arrow(a, loss)]
    parts += [x.svg() for x in (s, net, pi, loss, a)]
    parts.append(f'<text class="note" x="{f(net.x)}" y="{f(YA - 30)}" text-anchor="middle">'
                 f'any architecture: MLP, transformer, ...</text>')
    # backpropagation through the generic network
    y = YA + 44
    parts.append(f'<path class="e g" d="M{f(loss.x)},{f(YA + BH / 2 + 2)} L{f(loss.x)},{f(y)} '
                 f'L{f(net.x)},{f(y)} L{f(net.x)},{f(YA + BH / 2 + 3.5)}" marker-end="url(#h-g)"/>')
    parts.append(f'<text class="gnote" x="{f((loss.x + net.x) / 2)}" y="{f(y + 17)}" '
                 f'text-anchor="middle">backpropagation through a fixed number of layers</text>')
    return "\n  ".join(parts)


def panel_b():
    th = Circle(48, YB, "θ")
    rew = Box(150, YB, 96, "reward  r_{θ}")
    xs = (262, 342, 452)
    bell = [Box(x, YB, 58, "L_{β,θ}", "bell", "blab bl") for x in xs]
    v = Circle(548, YB, "v_{θ}")
    soft = Box(650, YB, 104, "softmax  π_{θ}")
    loss = Box(815, YB, 130, "loss, as in (a)")
    parts = ['<text class="title" x="6" y="200">(b) inverse RL as in Rust (1987)</text>']
    # dotted frame around the Bellman layers
    x0, x1 = xs[0] - 42, xs[-1] + 42
    parts.append(f'<rect class="fix" x="{f(x0)}" y="{f(YB - 40)}" width="{f(x1 - x0)}" height="80" rx="14"/>')
    parts.append(f'<text class="note" x="{f((x0 + x1) / 2)}" y="{f(YB - 48)}" text-anchor="middle">'
                 f'Bellman layers, repeated until v = L v</text>')
    parts += [arrow(th, rew), arrow(rew, bell[0]), arrow(bell[0], bell[1]), arrow(bell[-1], v),
              arrow(v, soft), arrow(soft, loss)]
    parts.append(f'<text class="dots" x="{f((xs[1] + xs[2]) / 2)}" y="{f(YB + 6)}" text-anchor="middle">⋯</text>')
    parts += [x.svg() for x in (th, rew, *bell, v, soft, loss)]
    parts.append(f'<text class="lab" x="{f(loss.x)}" y="{f(YB - 30)}" text-anchor="middle">'
                 f'{mtext("data  (s_{t}, a_{t})")}</text>')

    # gradient route 1: unrolled, back through every Bellman layer
    y1 = YB + 52
    parts.append(f'<path class="e gd" d="M{f(v.x)},{f(YB + R + 2)} L{f(v.x)},{f(y1)} '
                 f'L{f(rew.x)},{f(y1)} L{f(rew.x)},{f(YB + BH / 2 + 3.5)}" marker-end="url(#h-g)"/>')
    for x in xs:
        parts.append(f'<path class="e gd" d="M{f(x)},{f(y1)} L{f(x)},{f(YB + BH / 2 + 3)}"/>')
    parts.append(f'<text class="gnote" x="{f(v.x + 14)}" y="{f(y1 + 5)}">'
                 f'unrolled: backpropagate through all K layers;</text>')
    parts.append(f'<text class="gnote" x="{f(v.x + 14)}" y="{f(y1 + 21)}">'
                 f'memory and truncation error grow with depth</text>')

    # gradient route 2: one implicit step from the fixed point back to theta
    y2 = YB + 100
    parts.append(f'<path class="e g" d="M{f(v.x - 8)},{f(YB + R)} C{f(v.x - 40)},{f(y2 + 22)} '
                 f'{f(th.x + 30)},{f(y2 + 22)} {f(th.x + 6)},{f(YB + R + 3)}" marker-end="url(#h-g)"/>')
    parts.append(f'<text class="gnote" x="{f(v.x + 14)}" y="{f(y2 - 4)}">'
                 f'implicit: one linear solve, valid at the fixed point only</text>')
    parts.append(f'<text class="geq" x="{f(v.x + 14)}" y="{f(y2 + 20)}">'
                 f'{mtext("∂v_{θ}/∂θ = (I − γP_{π})^{−1} ∂r_{π}/∂θ")}</text>')
    return "\n  ".join(parts)


def main():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
     aria-label="Computation graphs of behavioral cloning and of Rust's inverse RL estimator">
<style>{STYLE}</style>
<defs>{markers()}</defs>
<g>
  {panel_a()}
</g>
<line class="rule" x1="0" y1="176" x2="{W}" y2="176"/>
<g>
  {panel_b()}
</g>
</svg>
"""
    OUT.write_text(svg)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
