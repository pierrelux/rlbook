"""LaTeX-typeset vector figures for stages and NLP quantities.

Run: uv run python code/collocation_nlp_figures.py
Requires a local LaTeX installation (including amsmath) and pdftocairo.
"""

from argparse import ArgumentParser
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

from collocation_transcription import cardinal_polynomials


INK = "#253440"
MUTED = "#5B6470"
RULE = "#CCD3DA"
BLUE = "#0072B2"
ORANGE = "#B64E00"
STYLE = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "svg.fonttype": "path",
    "svg.hashsalt": "collocation-nlp-figures",
}


def p(tau):
    return 1 + tau + 2 * tau**2


def u(tau):
    # With h_k=2 and xdot=u, p'(tau)=h_k*u(tau).
    return 0.5 + 2 * tau


def label(fig, x, y, text, **kwargs):
    return fig.text(x, y, text, **kwargs)


def arrow(ax, start, end, color, **kwargs):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4,
                    "shrinkA": 0, "shrinkB": 0, **kwargs},
    )


def stage_and_endpoint():
    """Keep the two integration spans, now with real LaTeX mathematics."""
    fig = plt.figure(figsize=(8.2, 6.0), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    canvas.axis("off")
    left, right = 0.15, 0.86
    middle = (left + right) / 2
    label(fig, .05, .95, r"One interval $k$, one state polynomial", fontsize=16)
    canvas.plot([left, left, right, right], [.87, .885, .885, .87],
                color=RULE, lw=1)
    label(fig, middle, .90, r"whole interval: $h_k=2$", ha="center", color=MUTED)
    state = fig.add_axes([left, .49, right-left, .35])
    tau = np.linspace(0, 1, 301)
    state.plot(tau, p(tau), color=INK, lw=2)
    state.scatter([0, 1], [1, 4], marker="s", c=[INK, ORANGE], s=34, zorder=4,
                  clip_on=False)
    state.scatter([.5], [2], color=BLUE, s=45, zorder=4)
    state.vlines([0, .5, 1], .55, [1, 2, 4], color=RULE, lw=.9, ls=":")
    state.set(xlim=(0, 1), ylim=(.55, 4.65), xticks=[], yticks=[])
    state.spines[["left", "right", "top"]].set_visible(False)
    state.spines["bottom"].set_color(RULE)
    state.text(.04, 3.95, r"$p_k(\tau)=1+\tau+2\tau^2$", fontsize=14)
    state.text(.04, 3.38, "one piece across the entire interval", color=MUTED)
    state.annotate(r"$x_k=1$", (0, 1), (9, 11), textcoords="offset points")
    state.annotate(r"$x_{k,i}=p_k(\tau_i)=2$", (.5, 2), (9, -20),
                   textcoords="offset points", color=BLUE, fontsize=13)
    state.annotate(r"stage $i$ is this point", (.5, 2), (9, -37),
                   textcoords="offset points", color=BLUE)
    state.annotate(r"$x_{k+1}=p_k(1)=4$", (1, 4), (-7, 12),
                   textcoords="offset points", ha="right", color=ORANGE, fontsize=13)
    label(fig, .035, .448, r"local $\tau$", color=MUTED)
    label(fig, .035, .405, r"time $t$", color=MUTED)
    for x, local, time, color in [
        (left, r"$0$", r"$t_k=2$", INK),
        (middle, r"$\tau_i=1/2$", r"$t_k+h_k\tau_i=3$", BLUE),
        (right, r"$1$", r"$t_{k+1}=4$", ORANGE),
    ]:
        label(fig, x, .448, local, ha="center", color=color, fontsize=13)
        label(fig, x, .405, time, ha="center", color=color, fontsize=13)
    arrow(canvas, (left, .345), (middle, .345), BLUE)
    label(fig, middle+.04, .337, r"integrate to stage $i$", color=BLUE)
    label(fig, left, .288,
          r"$x_{k,i}=x_k+h_k\sum_j A_{ij}f_{k,j}=1+1=2$", color=BLUE, fontsize=14)
    label(fig, left, .247, r"Row $i$ of $A$ integrates from $0$ to $\tau_i$.", color=MUTED)
    arrow(canvas, (left, .18), (right, .18), ORANGE)
    label(fig, right, .197, "integrate to the right endpoint", ha="right", color=ORANGE)
    label(fig, left, .122,
          r"$x_{k+1}=x_k+h_k\sum_j b_jf_{k,j}=1+3=4$", color=ORANGE, fontsize=14)
    label(fig, left, .077, r"The weights $b$ integrate from $0$ to $1$.", color=MUTED)
    return fig


def nlp_quantities():
    """Map every row of the NLP-quantity table to geometry and computation."""
    fig = plt.figure(figsize=(9.0, 7.1), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    canvas.axis("off")
    left, right = .09, .56
    width = right-left
    stage = 2/3
    stage_x = left + width*stage
    label(fig, .035, .954, r"Decision variables and evaluated quantities on interval $k$",
          fontsize=16)
    canvas.plot([left, left, right, right], [.875, .89, .89, .875], color=RULE, lw=1)
    label(fig, (left+right)/2, .903, r"$t_k\leq t\leq t_{k+1}$", ha="center")

    state = fig.add_axes([left, .59, width, .26], xlim=(0, 1), ylim=(.3, 4.95))
    control = fig.add_axes([left, .285, width, .22], xlim=(0, 1), ylim=(-.1, 3.0))
    tau = np.linspace(0, 1, 301)
    nodes = np.array([1/3, 2/3])
    for ax in (state, control):
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.spines["bottom"].set_color(RULE)
        ax.set(xticks=[], yticks=[])
        ax.axvline(stage, color=ORANGE, ls=(0, (2, 3)), lw=1, zorder=0)
    state.plot(tau, p(tau), color=INK, lw=1.8)
    state.scatter([0, 1], p(np.array([0, 1])), color=BLUE, marker="s", s=40,
                  zorder=4, clip_on=False)
    state.scatter(nodes, p(nodes), color=BLUE, s=43, zorder=4)
    state.annotate(r"$\mathbf{x}_k$", (0, 1), (5, 12), textcoords="offset points",
                   color=BLUE, fontsize=15)
    state.annotate(r"$\mathbf{x}_{k+1}$", (1, 4), (-3, 13), textcoords="offset points",
                   color=BLUE, fontsize=15, ha="right")
    for t, symbol in [(nodes[0], r"$\mathbf{x}_{k,1}$"),
                      (stage, r"$\mathbf{x}_{k,j}$")]:
        state.annotate(symbol, (t, p(t)), (0, -23), textcoords="offset points",
                       ha="center", color=BLUE, fontsize=15)
    state.text(.04, 4.48, r"state piece $\mathbf{p}_k(\tau)$", fontsize=13)
    # Both plotted curves obey xdot=u; the picture is a feasible scalar example.
    tangent = stage + np.array([-.10, .10])
    state.plot(tangent, p(stage)+(1+4*stage)*(tangent-stage),
               color=ORANGE, lw=1.2, ls="--")
    label(fig, left, .552, "Squares: endpoints shared with adjacent intervals", color=BLUE,
          fontsize=11)
    label(fig, left, .525, "Filled circles: stage state variables", color=BLUE, fontsize=11)

    control.plot(tau, u(tau), color=INK, lw=1.8)
    control.scatter([0, 1], u(np.array([0, 1])), color=BLUE, marker="D", s=42,
                    zorder=4, clip_on=False)
    control.scatter([stage], [u(stage)], facecolor="white", edgecolor=ORANGE,
                    linewidth=1.8, s=63, zorder=5)
    control.annotate(r"$\widehat{\mathbf{u}}_{k,0}$", (0, u(0)), (5, 14),
                     textcoords="offset points", color=BLUE, fontsize=15)
    control.annotate(r"$\widehat{\mathbf{u}}_{k,1}$", (1, u(1)), (-4, 12),
                     textcoords="offset points", color=BLUE, fontsize=15, ha="right")
    control.annotate(r"$\mathbf{u}_{k,j}$", (stage, u(stage)), (-9, 14),
                     textcoords="offset points", color=ORANGE, fontsize=15, ha="right")
    control.text(.04, 2.55, "linear control", fontsize=13)
    label(fig, left, .188, "Diamonds: control support variables", color=BLUE, fontsize=11)
    for t, text in [(0, r"$0=\rho_0$"), (nodes[0], r"$\tau_1$"),
                    (stage, r"$\tau_j$"), (1, r"$1=\rho_1$")]:
        label(fig, left+width*t, .252, text, ha="center", fontsize=13,
              color=ORANGE if t == stage else MUTED)
    label(fig, stage_x, .218, r"$t_k+h_k\tau_j$", ha="center", color=ORANGE)

    # Direct evaluation from the support variables to the open stage marker.
    label(fig, .65, .40, r"Evaluate the control using $B$", color=ORANGE, fontsize=13)
    label(fig, .65, .35,
          r"$\mathbf{u}_{k,j}=\sum_r B_{jr}\widehat{\mathbf{u}}_{k,r}$",
          color=ORANGE, fontsize=15)
    label(fig, .65, .294, r"Here $B_{j,:}=(1-\tau_j,\ \tau_j)$.", fontsize=12)
    label(fig, .65, .253, "The open circle adds no variable.", color=MUTED, fontsize=11)
    connector = ConnectionPatch(
        xyA=(.635, .37), coordsA=fig.transFigure,
        xyB=(stage+.018, u(stage)), coordsB=control.transData,
        arrowstyle="-|>", lw=1.1, color=ORANGE, shrinkA=0, shrinkB=4,
    )
    fig.add_artist(connector)

    label(fig, .65, .835, r"At the same stage $j$", fontsize=14)
    label(fig, .65, .792, r"state, control, and time", color=MUTED, fontsize=12)
    label(fig, .65, .739, r"$\mathbf{x}_{k,j},\quad\mathbf{u}_{k,j},\quad t_k+h_k\tau_j$",
          fontsize=14)
    arrow(canvas, (.80, .711), (.80, .674), ORANGE)
    label(fig, .65, .635, r"$\mathbf{f}_{k,j}=\mathbf{f}(\mathbf{x}_{k,j},\mathbf{u}_{k,j},t_k+h_k\tau_j)$",
          color=ORANGE, fontsize=12)
    label(fig, .65, .588, r"$c_{k,j}=c(\mathbf{x}_{k,j},\mathbf{u}_{k,j},t_k+h_k\tau_j)$",
          color=ORANGE, fontsize=12)
    label(fig, .65, .542, "Evaluated rates, not extra variables", color=ORANGE, fontsize=11)
    label(fig, .65, .496, r"$\mathbf{f}_{k,j}$ enters the dynamics constraints;",
          fontsize=11, color=MUTED)
    label(fig, .65, .465, r"$c_{k,j}$ enters the running-cost sum.",
          fontsize=11, color=MUTED)
    fig.add_artist(ConnectionPatch(
        xyA=(stage, p(stage)), coordsA=state.transData,
        xyB=(.633, .752), coordsB=fig.transFigure,
        arrowstyle="-|>", color=BLUE, lw=1.1, shrinkA=6, shrinkB=0,
    ))

    canvas.plot([.035, .965], [.145, .145], color=RULE, lw=.8)
    label(fig, .045, .103,
          r"The NLP chooses $\mathbf{x}_k,\ \mathbf{x}_{k+1},\ \mathbf{x}_{k,j},\ "
          r"\widehat{\mathbf{u}}_{k,r}$; the constraints place the state points on $\mathbf{p}_k$.",
          fontsize=12)
    label(fig, .045, .059,
          r"If $\tau_j=0$ or $1$, the stage state can share the corresponding mesh variable.",
          fontsize=12, color=MUTED)
    return fig


def evaluation_and_differentiation():
    """Evaluate and differentiate one interpolant at a non-support node."""
    fig = plt.figure(figsize=(9.0, 6.5), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    canvas.axis("off")
    label(fig, .04, .952, r"Same support values: a value from $E$, a slope from $D$",
          fontsize=16)

    plot = fig.add_axes([.08, .46, .43, .37],
                        xlim=(-.04, 1.04), ylim=(.35, 4.85))
    tau = np.linspace(0, 1, 301)
    support = np.array([0., .5, 1.])
    node = .25
    value = p(node)
    local_slope = 1 + 4*node
    plot.plot(tau, p(tau), color=INK, lw=1.8)
    plot.scatter(support, p(support), color=BLUE, marker="s", s=37, zorder=4)
    plot.vlines(node, .35, value, color=RULE, ls=":", lw=1)
    tangent = np.array([.01, .72])
    plot.plot(tangent, value+local_slope*(tangent-node),
              color=ORANGE, ls="--", lw=1.6)
    plot.scatter([node], [value], facecolor="white", edgecolor=BLUE,
                 lw=1.7, s=65, zorder=5)
    plot.text(.02, 4.5, r"$p_k(\tau)=1+\tau+2\tau^2$", fontsize=14)
    for x, symbol, dx, dy, ha in [
        (0, r"$\widehat{x}_{k,0}=1$", 3, 14, "left"),
        (.5, r"$\widehat{x}_{k,1}=2$", -2, 17, "left"),
        (1, r"$\widehat{x}_{k,2}=4$", -4, 14, "right"),
    ]:
        plot.annotate(symbol, (x, p(x)), (dx, dy),
                      textcoords="offset points", color=BLUE,
                      fontsize=13, ha=ha)
    plot.annotate(r"$p_k(\tau_i)=\frac{11}{8}$", (node, value),
                  (-11, 35), textcoords="offset points", color=BLUE,
                  fontsize=13, ha="left",
                  arrowprops={"arrowstyle": "-", "lw": .8, "color": BLUE,
                              "shrinkA": 3, "shrinkB": 5})
    plot.annotate(r"tangent: $p_k'(\tau_i)=2$",
                  (.66, value+local_slope*(.66-node)), (.46, .8),
                  color=ORANGE, fontsize=12,
                  arrowprops={"arrowstyle": "-", "lw": .8, "color": ORANGE})
    plot.set(yticks=[], xticks=[0, node, .5, 1],
             xticklabels=[r"$0$", r"$\tau_i=\frac14$", r"$\frac12$", r"$1$"])
    plot.tick_params(axis="x", length=3, color=RULE, pad=7, labelsize=12)
    plot.spines[["left", "right", "top"]].set_visible(False)
    plot.spines["bottom"].set_color(RULE)
    label(fig, .08, .378, r"Squares: stored states at $\sigma=(0,\frac12,1)$.",
          color=BLUE, fontsize=12)
    label(fig, .08, .343, r"Open circle: evaluation at $\tau_i=\frac14$.",
          color=BLUE, fontsize=12)

    label(fig, .60, .851, r"Evaluate with $E$", color=BLUE, fontsize=14)
    label(fig, .60, .793,
          r"$\sum_r E_{ir}\widehat{x}_{k,r}=p_k(\tau_i)=\frac{11}{8}$",
          color=BLUE, fontsize=15)
    label(fig, .60, .747, "The state supplied to the ODE", color=MUTED, fontsize=12)

    label(fig, .60, .663, r"Differentiate with $D$", color=ORANGE, fontsize=14)
    label(fig, .60, .603,
          r"$\sum_r D_{ir}\widehat{x}_{k,r}=p_k'(\tau_i)=2$",
          color=ORANGE, fontsize=15)
    arrow(canvas, (.675, .571), (.675, .507), ORANGE)
    label(fig, .715, .535, r"divide by $h_k=2$", color=ORANGE, fontsize=12)
    label(fig, .60, .454, r"$\frac{1}{h_k}\,p_k'(\tau_i)=1$",
          color=ORANGE, fontsize=16)
    label(fig, .60, .407, "The slope per unit of physical time", color=MUTED, fontsize=12)

    canvas.plot([.04, .96], [.297, .297], color=RULE, lw=.8)
    label(fig, .5, .256,
          r"Use $\dot{x}=u$, $t_k=2$, $h_k=2$, and $u_{k,i}=1$.",
          ha="center", fontsize=13)
    label(fig, .5, .175,
          r"$\underbrace{\frac{1}{h_k}\sum_r D_{ir}\widehat{x}_{k,r}}_{"
          r"\text{polynomial slope }=\,1}"
          r"\;=\;\underbrace{f\!\left(\sum_r E_{ir}\widehat{x}_{k,r},\,u_{k,i},\,"
          r"t_k+h_k\tau_i\right)}_{\text{ODE slope }=\,1}$",
          ha="center", fontsize=16)
    label(fig, .5, .034,
          "The collocation constraint equates these two physical-time rates.",
          ha="center", fontsize=12, color=MUTED)
    return fig


def destination_and_slope_stages():
    """Fix one destination row and integrate each slope basis over its span."""
    fig = plt.figure(figsize=(9.0, 5.7), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    canvas.axis("off")
    nodes = np.array([0., .5, 1.])
    basis = cardinal_polynomials(nodes)
    destination = nodes[1]
    tau = np.linspace(0, 1, 301)
    integrated_tau = np.linspace(0, destination, 151)

    label(fig, .055, .951,
          r"One destination $i=2$, three contributing slope stages $j$", fontsize=16)
    label(fig, .055, .895,
          r"Every shaded integral runs from $0$ to $\tau_2=\frac12$.",
          color=ORANGE, fontsize=13)

    for index, left in enumerate([.075, .38, .685]):
        j = index + 1
        polynomial = basis[index]
        ax = fig.add_axes([left, .455, .24, .335],
                          xlim=(-.035, 1.035), ylim=(-.25, 1.15))
        ax.axhline(0, color=MUTED, lw=.7)
        ax.axvline(destination, color=ORANGE, ls="--", lw=1.2)
        ax.plot(tau, polynomial(tau), color=BLUE, lw=1.7)
        ax.fill_between(integrated_tau, 0, polynomial(integrated_tau),
                        facecolor=BLUE, alpha=.2)
        ax.scatter([nodes[index]], [1.], s=35, color=BLUE,
                   edgecolor="white", lw=.7, zorder=4)
        ax.set(xticks=nodes, xticklabels=[r"$0$", r"$\frac12$", r"$1$"],
               yticks=[0, 1], yticklabels=[r"$0$", r"$1$"], xlabel=r"local $\tau$")
        ax.tick_params(length=3, color=RULE, pad=4, labelsize=11)
        ax.xaxis.labelpad = 3
        ax.get_xticklabels()[1].set_color(ORANGE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color(RULE)
        label(fig, left + .12, .823,
              rf"$j={j}:\quad\ell_{j}(\tau)$", ha="center", color=BLUE, fontsize=14)
        fraction = [r"\frac{5}{24}", r"\frac13", r"-\frac1{24}"][index]
        label(fig, left + .12, .322,
              rf"$A_{{2{j}}}=\int_0^{{1/2}}\ell_{j}(\tau)\,d\tau={fraction}$",
              ha="center", color=BLUE, fontsize=14)

    canvas.plot([.045, .955], [.272, .272], color=RULE, lw=.8)
    label(fig, .5, .208,
          r"$x_{k,2}=x_k+h_k\underbrace{\left("
          r"\frac{5}{24}f_{k,1}+\frac13f_{k,2}-\frac1{24}f_{k,3}"
          r"\right)}_{\sum_{j=1}^{3}A_{2j}f_{k,j}}$",
          ha="center", fontsize=17)
    label(fig, .5, .042,
          r"Even the slope at $\tau_3=1$ contributes to the state at $\tau_2=\frac12$.",
          ha="center", fontsize=12, color=MUTED)
    return fig


def main():
    figures = {
        "stage-and-endpoint": stage_and_endpoint,
        "nlp-quantities": nlp_quantities,
        "evaluation-and-differentiation": evaluation_and_differentiation,
        "destination-and-slope-stages": destination_and_slope_stages,
    }
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--figure", choices=figures,
                        help="Render only this figure (default: all).")
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "_static" / "collocation"
    with plt.rc_context(STYLE):
        selected = {args.figure: figures[args.figure]} if args.figure else figures
        for name, build in selected.items():
            fig = build()
            pdf = output / f"{name}.pdf"
            fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
            # The native SVG backend mis-scales CMEX glyphs (sums and wide hats)
            # with this TeX installation. Convert the typeset vector PDF so the
            # web and print versions retain identical mathematical glyphs.
            subprocess.run(
                ["pdftocairo", "-svg", str(pdf), str(output / f"{name}.svg")],
                check=True,
            )
            if args.preview_dir:
                args.preview_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.preview_dir / f"{name}.png",
                            dpi=800 / fig.get_figwidth())
            plt.close(fig)
            print(f"Rendered {name} with LaTeX to SVG/PDF")


if __name__ == "__main__":
    main()
