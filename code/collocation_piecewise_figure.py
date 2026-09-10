"""Render the textbook's schematic of joined collocation polynomials."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_figure():
    # Each array contains coefficients in its own normalized time coordinate.
    mesh = np.array([0.0, 0.8, 2.4, 3.5])
    coefficients = [
        np.array([0.4, 0.3, 0.4]),
        np.array([1.1, 1.2, -0.6]),
        np.array([1.7, 0.2, -0.6]),
    ]
    tau = np.linspace(0.0, 1.0, 301)
    nodes = np.array([1.0 / 3.0, 2.0 / 3.0])
    evaluate = np.polynomial.polynomial.polyval
    middle = coefficients[1]
    stage_values = evaluate(nodes, middle)
    endpoint_values = np.array([0.4, 1.1, 1.7, 1.3])

    blue, orange, gray = "#0072B2", "#D55E00", "#626970"
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }):
        figure, (physical, reference) = plt.subplots(
            2, 1, figsize=(7.0, 5.3), sharey=True, layout="constrained"
        )
        physical.set_title("Three polynomial pieces in physical time", loc="left", pad=12)
        physical.axvspan(mesh[1], mesh[2], color=blue, alpha=0.06, linewidth=0)
        for k, coefficient in enumerate(coefficients):
            physical.plot(
                mesh[k] + np.diff(mesh)[k] * tau,
                evaluate(tau, coefficient),
                color=blue if k == 1 else gray,
                linewidth=2.2,
                linestyle="-" if k == 1 else "--",
            )
            physical.text(
                (mesh[k] + mesh[k + 1]) / 2, 2.0,
                [r"$p_{k-1}$", r"$p_k$", r"$p_{k+1}$"][k],
                ha="center", color=blue if k == 1 else gray,
            )
        for boundary in mesh[1:3]:
            physical.axvline(boundary, color="#CBD0D4", linewidth=0.7, zorder=0)
        physical.scatter(mesh, endpoint_values, marker="s", color=gray, s=30, zorder=4)
        physical.scatter(
            mesh[1] + (mesh[2] - mesh[1]) * nodes,
            stage_values, color=orange, edgecolor="white", linewidth=0.6,
            s=38, zorder=5,
        )
        for time, value, label, offset in zip(
            mesh, endpoint_values,
            [r"$x_{k-1}$", r"$x_k$", r"$x_{k+1}$", r"$x_{k+2}$"],
            [(6, 12), (-10, 12), (0, 12), (-10, -20)],
        ):
            physical.annotate(label, (time, value), xytext=offset,
                              textcoords="offset points", ha="center", color=gray)
        physical.annotate("", xy=(mesh[1], 0.62), xytext=(mesh[2], 0.62),
                          arrowprops={"arrowstyle": "<->", "color": blue, "lw": 0.9})
        physical.text((mesh[1] + mesh[2]) / 2, 0.39,
                      r"$h_k=t_{k+1}-t_k$", ha="center", color=blue)
        physical.set_xticks(mesh, [r"$t_{k-1}$", r"$t_k$", r"$t_{k+1}$", r"$t_{k+2}$"])
        physical.set_xlabel(r"physical time $t$", labelpad=3)
        physical.set_xlim(-0.12, mesh[-1] + 0.12)

        reference.set_title(
            r"The same middle piece: $t=t_k+h_k\tau$", loc="left", pad=12
        )
        reference.plot(tau, evaluate(tau, middle), color=blue, linewidth=2.2)
        reference.scatter([0, 1], endpoint_values[1:3], marker="s", color=gray, s=30, zorder=4)
        reference.scatter(nodes, stage_values, color=orange, edgecolor="white",
                          linewidth=0.6, s=38, zorder=5)
        for j, (node, value) in enumerate(zip(nodes, stage_values), start=1):
            # dp_k/dtau = h_k * dx_h/dt, so these are normalized-time slopes.
            tangent_tau = node + np.array([-0.065, 0.065])
            normalized_slope = middle[1] + 2 * middle[2] * node
            reference.plot(
                tangent_tau, value + normalized_slope * (tangent_tau - node),
                color=orange, linewidth=1.5, zorder=4,
            )
            reference.vlines(node, 0.2, value, color="#CBD0D4", linewidth=0.8,
                             linestyle=":", zorder=0)
            reference.annotate(
                rf"$x_{{k,{j}}}=p_k(\tau_{j})$", (node, value),
                xytext=(0, -28), textcoords="offset points", ha="center",
                color=orange,
            )
        reference.annotate(r"$p_k(0)=x_k$", (0, endpoint_values[1]),
                           xytext=(5, 14), textcoords="offset points", color=blue)
        reference.annotate(r"$p_k(1)=x_{k+1}$", (1, endpoint_values[2]),
                           xytext=(-5, 14), textcoords="offset points", ha="right", color=blue)
        reference.text(0.5, 0.45, "squares: shared endpoints    dots: stages",
                       ha="center", color=gray, fontsize=9)
        reference.set_xticks([0, *nodes, 1],
                             [r"$0$", r"$\tau_1=\frac{1}{3}$", r"$\tau_2=\frac{2}{3}$", r"$1$"])
        reference.set_xlim(-0.04, 1.04)
        reference.set_xlabel(r"normalized time $\tau$", labelpad=3)
        for axis in (physical, reference):
            axis.set_ylabel("state")
            axis.set_ylim(0.2, 2.13)
            axis.set_yticks([0.5, 1.0, 1.5, 2.0])
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#9BA2A9")
            axis.tick_params(length=3, color="#9BA2A9")

    return figure


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "_static" / "collocation"
    output.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    for extension in ("svg", "pdf", "png"):
        figure.savefig(output / f"piecewise-trajectory.{extension}", dpi=160)
    plt.close(figure)
