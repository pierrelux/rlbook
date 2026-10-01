"""Render introductory figures for polynomial pieces and local time.

Run from the repository root with:
    uv run python code/collocation_mesh_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "svg.hashsalt": "collocation-mesh-figures",
    "pdf.fonttype": 42,
}
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"


def style_axis(ax):
    ax.spines[["left", "bottom"]].set_color("#9BA2A9")
    ax.tick_params(length=3, color="#9BA2A9")


def switching_trajectory():
    """Compare the exact corner with the quadratic through the same samples."""
    t = np.linspace(0, 1, 401)
    with plt.rc_context(STYLE):
        fig, (state, slope) = plt.subplots(
            2, 1, figsize=(6.6, 4.8), sharex=True,
            gridspec_kw={"height_ratios": [1.4, 1]},
        )
        fig.subplots_adjust(left=.11, right=.98, bottom=.12, top=.84, hspace=.27)
        state.plot(t, np.minimum(t, 1-t), color=BLUE, lw=2.5,
                   label="Two linear pieces: exact trajectory")
        state.plot(t, 2*t*(1-t), color=ORANGE, ls="--", lw=2,
                   label="One quadratic through the same three points")
        state.scatter([0, .5, 1], [0, .5, 0], c="#303840", s=27, zorder=5)
        state.annotate("corner at the switch", xy=(.5, .5), xytext=(.6, .57),
                       fontsize=10, arrowprops={"arrowstyle": "->", "color": BLUE},
                       color=BLUE)
        state.text(.18, .075, r"$x(t)=t$", color=BLUE)
        state.text(.68, .075, r"$x(t)=1-t$", color=BLUE)
        state.text(.06, .51, r"$q(t)=2t(1-t)$", color=ORANGE)
        state.set_ylabel("state")
        state.set_ylim(-.03, .67)
        state.set_yticks([0, .25, .5])
        state.legend(loc="lower left", bbox_to_anchor=(-.01, 1.02),
                     borderaxespad=0, frameon=False, fontsize=10)
        for a, b, u in [(0, .5, 1), (.5, 1, -1)]:
            slope.plot([a, b], [u, u], c=BLUE, lw=2.5)
            slope.plot(.5, u, marker="o", mfc="white", mec=BLUE, ms=5, zorder=5)
        slope.plot(t, 2-4*t, color=ORANGE, ls="--", lw=2)
        slope.text(.04, .48, r"$\dot x=u=+1$", color=BLUE)
        slope.text(.54, -1.85, r"$\dot x=u=-1$", color=BLUE)
        slope.text(.67, 1.35, r"$q'(t)=2-4t$", color=ORANGE)
        slope.set_ylabel("physical-time slope")
        slope.set_xlabel(r"physical time $t$")
        slope.set_ylim(-2.25, 2.25)
        slope.set_yticks([-2, -1, 0, 1, 2])
        slope.set_xticks([0, .5, 1], ["0", "1/2", "1"])
        slope.set_xlim(-.025, 1.025)
        for ax in (state, slope):
            ax.axvline(.5, color=".8", ls=":", lw=.8, zorder=0)
            style_axis(ax)
    return fig


def local_time_pieces():
    """Show the same continuous curve in physical and per-piece coordinates."""
    mesh = np.array([0., 1., 3., 4.])
    coefficients = [[.4, .3, .4], [1.1, 1.2, -.6], [1.7, .2, -.6]]
    endpoints = [.4, 1.1, 1.7, 1.3]
    tau = np.linspace(0, 1, 201)
    colors = [BLUE, ORANGE, GREEN]
    styles = ["-", "--", "-."]
    evaluate = np.polynomial.polynomial.polyval
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(6.6, 5.2))
        grid = fig.add_gridspec(2, 3, height_ratios=[1.3, 1],
                               left=.10, right=.98, bottom=.13, top=.93,
                               hspace=.8, wspace=.35)
        physical = fig.add_subplot(grid[0, :])
        physical.set_title("One trajectory, three polynomial pieces", loc="left", pad=12)
        for k, (coef, color, ls) in enumerate(zip(coefficients, colors, styles)):
            physical.plot(mesh[k]+np.diff(mesh)[k]*tau, evaluate(tau, coef),
                          c=color, ls=ls, lw=2.4)
            physical.text((mesh[k]+mesh[k+1])/2, 2.04, rf"$p_{k}$", color=color,
                          ha="center")
        for k, (time, value) in enumerate(zip(mesh, endpoints)):
            physical.scatter(time, value, c="#303840", s=25, zorder=5)
            physical.annotate(rf"$x_{k}$", (time, value), xytext=(10, 12) if k == 0 else (0, -18),
                              textcoords="offset points", ha="center")
        for boundary in mesh[1:-1]:
            physical.axvline(boundary, c=".8", ls=":", lw=.8, zorder=0)
        physical.set_xticks(mesh, [r"$t_0=0$", r"$t_1=1$", r"$t_2=3$", r"$t_3=4$"])
        physical.set_xlabel(r"physical time $t$")
        physical.set_ylabel(r"state $x_h(t)$")
        physical.set_xlim(-.1, 4.1)
        physical.set_ylim(.1, 2.25)
        physical.set_yticks([.5, 1, 1.5, 2])
        style_axis(physical)
        for k, (coef, color, ls) in enumerate(zip(coefficients, colors, styles)):
            ax = fig.add_subplot(grid[1, k])
            ax.plot(tau, evaluate(tau, coef), c=color, ls=ls, lw=2.4)
            ax.scatter([0, 1], endpoints[k:k+2], c="#303840", s=25, zorder=5)
            ax.set_title([r"$p_0(\tau)=x_h(\tau)$", r"$p_1(\tau)=x_h(1+2\tau)$",
                          r"$p_2(\tau)=x_h(3+\tau)$"][k], color=color, fontsize=11, pad=10)
            ax.set_xlabel(r"local time $\tau$")
            ax.set_xlim(-.07, 1.07)
            ax.set_xticks([0, 1])
            ax.set_ylim(.1, 2.25)
            ax.set_yticks([.5, 1, 1.5, 2])
            if k == 0:
                ax.set_ylabel(r"state $p_k(\tau)$")
            else:
                ax.set_yticklabels([])
            style_axis(ax)
        fig.text(.54, .015, r"Shared joins: $p_0(1)=p_1(0)=x_1$ and $p_1(1)=p_2(0)=x_2$",
                 ha="center", fontsize=11)
    return fig


def collocation_stage():
    """A scalar stage combines time, state, control, and an ODE slope."""
    t = np.linspace(2, 4, 301)
    delta = t - 2
    state_values = 1 + .5*delta + .5*delta**2
    control_values = .5 + delta
    with plt.rc_context(STYLE):
        fig, (state, control) = plt.subplots(
            2, 1, figsize=(6.6, 5), sharex=True,
            gridspec_kw={"height_ratios": [1.6, 1]},
        )
        fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.9, hspace=.3)
        state.set_title(r"One stage on interval $k$: $\dot x=u$, $h_k=2$", loc="left", pad=16)
        state.plot(t, state_values, color=BLUE, lw=2.5)
        tangent_t = np.array([2.6, 3.4])
        state.plot(tangent_t, 2 + 1.5*(tangent_t-3), color=ORANGE, lw=2, ls="--")
        state.scatter([2, 4], [1, 4], color="#303840", marker="s", s=25, zorder=5)
        state.scatter(3, 2, color=ORANGE, s=40, zorder=6)
        state.annotate(r"$x_{k,j}=2$", (3, 2), xytext=(2.22, 2.3),
                       color=ORANGE, arrowprops={"arrowstyle":"->", "color":ORANGE})
        state.annotate("tangent slope\n" + r"$f_{k,j}=u_{k,j}=3/2$",
                       (3.35, 2.525), xytext=(3.28, 1.12), fontsize=10,
                       color=ORANGE, arrowprops={"arrowstyle":"->", "color":ORANGE})
        state.text(2.08, .85, r"$x_k=1$", color="#303840")
        state.text(3.59, 4.1, r"$x_{k+1}=4$", color="#303840")
        state.set_ylabel(r"state $x_h(t)$")
        state.set_ylim(.65, 4.6)
        state.set_yticks([1, 2, 3, 4])
        control.plot(t, control_values, color=BLUE, lw=2.5)
        control.scatter(3, 1.5, color=ORANGE, s=40, zorder=6)
        control.annotate(r"$u_{k,j}=3/2$", (3, 1.5), xytext=(2.14, 1.95),
                         color=ORANGE, arrowprops={"arrowstyle":"->", "color":ORANGE})
        control.text(3.2, .65, r"$\tau_j=1/2$", color=ORANGE)
        control.set_ylabel(r"control $u_h(t)$")
        control.set_ylim(.2, 2.9)
        control.set_yticks([.5, 1.5, 2.5])
        control.set_xticks([2, 3, 4], [r"$t_k=2$", r"$t_k+h_k\tau_j=3$", r"$t_{k+1}=4$"])
        control.set_xlabel(r"physical time $t$", labelpad=7)
        control.set_xlim(1.95, 4.05)
        for ax in (state, control):
            ax.axvline(3, c=ORANGE, ls=":", lw=1, alpha=.7, zorder=0)
            style_axis(ax)
    return fig


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "_static" / "collocation"
    output.mkdir(parents=True, exist_ok=True)
    for name, build in [("switching-trajectory", switching_trajectory),
                        ("local-time-pieces", local_time_pieces),
                        ("collocation-stage", collocation_stage)]:
        figure = build()
        for extension in ("svg", "pdf", "png"):
            figure.savefig(output / f"{name}.{extension}", dpi=150,
                           metadata={"Date": None} if extension == "svg" else None)
        plt.close(figure)
