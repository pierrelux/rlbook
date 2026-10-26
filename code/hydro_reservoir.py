"""Figures for the seasonal hydro reservoir example in finite-horizon DP.

The chapter solves the problem in a visible code cell. This module only draws
the results and simulates the baseline operating rules used for comparison, so
the functions take the chapter's arrays as arguments. Each figure is closed
before it is returned, so a notebook shows it once, as the value of the cell.
"""

from __future__ import annotations

from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
}

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.dpi": 150,
    "svg.fonttype": "none",
}

MONTHS = ("Jan", "Mar", "May", "Jul", "Sep", "Nov")
MONTH_WEEKS = (0, 8.7, 17.3, 26, 34.7, 43.3)


def _month_axis(axis: plt.Axes, labels: bool = True) -> None:
    axis.set_xticks(MONTH_WEEKS, MONTHS if labels else [""] * len(MONTHS))
    axis.set_xlim(0, 52)


def simulate_rule(
    rule: Callable[[int, float], float],
    x0: float,
    w: np.ndarray,
    p: np.ndarray,
    head: Callable,
    terminal_value: Callable,
    x_min: float,
    x_max: float,
    u_min: float,
    u_max: float,
    kappa: float,
) -> dict:
    """Run an operating rule through the reservoir dynamics for one year.

    The release requested by ``rule`` is clipped to the turbine limits and to
    the water available above dead storage. Water above capacity spills.
    """
    storage = [float(x0)]
    releases, revenue, spill = [], 0.0, 0.0
    for t in range(len(w)):
        x = storage[-1]
        u = min(max(rule(t, x), u_min), u_max, x + w[t] - x_min)
        revenue += p[t] * kappa * head(x) * u
        spill += max(0.0, x + w[t] - u - x_max)
        storage.append(min(x + w[t] - u, x_max))
        releases.append(u)
    revenue += float(terminal_value(storage[-1]))
    return {
        "storage": np.array(storage),
        "release": np.array(releases),
        "revenue": revenue,
        "spill": spill,
    }


def format_comparison(results: dict[str, dict]) -> str:
    """Tabulate revenue and spill for several operating rules."""
    lines = [f"{'operating rule':<34}{'revenue (M$)':>14}{'spill (hm3)':>13}"]
    for name, result in results.items():
        lines.append(
            f"{name:<34}{result['revenue'] / 1e6:>14.1f}{result['spill']:>13.0f}"
        )
    return "\n".join(lines)


def make_policy_figure(
    w: np.ndarray,
    p: np.ndarray,
    X: np.ndarray,
    policy: np.ndarray,
    water_value: np.ndarray,
    rollouts: list[np.ndarray],
) -> plt.Figure:
    """Policy and water-value maps over (week, storage) with optimal rollouts.

    An inflow strip sits above the policy map and a price strip above the
    water-value map so that each map shares its time axis with the input that
    explains it.
    """
    T = len(w)
    weeks = np.arange(T)
    extent = (0, T, X[0], X[-1])
    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(7.4, 4.3))
        grid = figure.add_gridspec(
            2,
            2,
            height_ratios=(1, 3.2),
            left=0.085,
            right=0.985,
            bottom=0.16,
            top=0.93,
            hspace=0.08,
            wspace=0.2,
        )
        inflow_axis = figure.add_subplot(grid[0, 0])
        price_axis = figure.add_subplot(grid[0, 1])
        policy_axis = figure.add_subplot(grid[1, 0], sharex=inflow_axis)
        value_axis = figure.add_subplot(grid[1, 1], sharex=price_axis)

        inflow_axis.bar(weeks + 0.5, w, width=0.85, color=OI["skyblue"])
        inflow_axis.set_ylabel("inflow\n(hm³/wk)", fontsize=8)
        inflow_axis.set_title("Optimal release $u_t^\\star(x)$", loc="left",
                              fontweight="semibold", pad=4)
        price_axis.plot(weeks + 0.5, p, color=OI["vermilion"], linewidth=1.6)
        price_axis.set_ylabel("price\n($/MWh)", fontsize=8)
        price_axis.set_title("Water value $\\partial J_t^\\star/\\partial x$",
                             loc="left", fontweight="semibold", pad=4)
        for axis in (inflow_axis, price_axis):
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=7.5, labelbottom=False)

        policy_image = policy_axis.imshow(
            policy.T, origin="lower", aspect="auto", extent=extent, cmap="cividis",
            interpolation="nearest",
        )
        value_image = value_axis.imshow(
            water_value.T / 1e3, origin="lower", aspect="auto", extent=extent,
            cmap="viridis", interpolation="nearest",
        )
        for axis in (policy_axis, value_axis):
            for storage in rollouts:
                axis.plot(np.arange(T + 1), storage, color="white", linewidth=1.1)
            _month_axis(axis)
            axis.set_xlabel("week of the year")
        policy_axis.set_ylabel("storage $x$ (hm³)")
        value_axis.tick_params(labelleft=False)

        for axis, image, label in (
            (policy_axis, policy_image, "release (hm³/week)"),
            (value_axis, value_image, "k$ per hm³"),
        ):
            bar = figure.colorbar(
                image, ax=axis, orientation="horizontal", fraction=0.05, pad=0.2,
                aspect=40,
            )
            bar.set_label(label, fontsize=8)
            bar.ax.tick_params(labelsize=7.5)

        policy_axis.text(17.5, 1780, "full release", color="0.15", fontsize=8,
                         ha="center")
        policy_axis.text(33, 330, "minimum release", color="white", fontsize=8,
                         ha="center")
        policy_axis.annotate(
            "all starting levels\njoin one rule curve", xy=(14.5, 600),
            xytext=(27, 820), color="white", fontsize=7.5, ha="left",
            arrowprops={"arrowstyle": "->", "color": "white", "linewidth": 0.8},
        )
        value_axis.annotate(
            "full during the freshet:\nextra water would spill",
            xy=(21, 1950), xytext=(25.5, 1500), color="white", fontsize=7.5,
            arrowprops={"arrowstyle": "->", "color": "white", "linewidth": 0.8},
        )
        plt.close(figure)  # displayed once, as the cell's value
        return figure


def make_threshold_figure(
    p: np.ndarray,
    storage: np.ndarray,
    release: np.ndarray,
    marginal_release: np.ndarray,
    marginal_storage: np.ndarray,
    u_min: float,
    u_max: float,
) -> plt.Figure:
    """Compare the value of releasing water now with the value of keeping it."""
    T = len(p)
    weeks = np.arange(T) + 0.5
    with mpl.rc_context(STYLE):
        figure, (value_axis, release_axis) = plt.subplots(
            2, 1, figsize=(7.4, 3.6), sharex=True,
            gridspec_kw={"height_ratios": (2.2, 1), "hspace": 0.1, "left": 0.1,
                         "right": 0.98, "top": 0.93, "bottom": 0.13},
        )
        full = release >= u_max - 1e-9
        for t in np.flatnonzero(full):
            value_axis.axvspan(t, t + 1, color=OI["orange"], alpha=0.16, lw=0)
            release_axis.axvspan(t, t + 1, color=OI["orange"], alpha=0.16, lw=0)
        value_axis.plot(weeks, marginal_release / 1e3, color=OI["vermilion"],
                        linewidth=1.8, marker="o", markersize=2.6)
        value_axis.plot(weeks, marginal_storage / 1e3, color=OI["blue"],
                        linewidth=1.8, marker="o", markersize=2.6)
        value_axis.text(29, marginal_release[29] / 1e3 - 0.35,
                        "release one hm³ now: $p_t\\,\\kappa\\,h(x_t)$",
                        color=OI["vermilion"], fontsize=8.5, va="top",
                        ha="center")
        low = min(marginal_release.min(), marginal_storage.min()) / 1e3
        value_axis.set_ylim(low - 1.6, None)
        value_axis.text(23, marginal_storage[23] / 1e3 + 0.5,
                        "keep it for later: $\\partial J_{t+1}^\\star/\\partial x$ at $x_{t+1}$",
                        color=OI["blue"], fontsize=8.5, va="bottom")
        value_axis.set_ylabel("k$ per hm³")
        value_axis.set_title(
            "Turbines run at capacity (shaded) exactly when releasing is worth more",
            loc="left", fontweight="semibold", pad=6,
        )
        release_axis.step(np.arange(T + 1), np.append(release, release[-1]),
                          where="post", color="0.2", linewidth=1.4)
        release_axis.axhline(u_min, color="0.6", linewidth=0.8, linestyle=":")
        release_axis.axhline(u_max, color="0.6", linewidth=0.8, linestyle=":")
        release_axis.set_ylabel("release\n(hm³/wk)")
        release_axis.set_ylim(0, u_max * 1.15)
        release_axis.set_xlabel("week of the year")
        for axis in (value_axis, release_axis):
            _month_axis(axis)
            axis.spines[["top", "right"]].set_visible(False)
        plt.close(figure)  # displayed once, as the cell's value
        return figure


def make_interpolation_figure(
    T: int,
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    reference: str,
) -> plt.Figure:
    """Initial value function and storage path for several grid schemes.

    ``curves`` maps a label to (grid, J_0 on that grid, simulated storage).
    """
    colors = {0: "0.15", 1: OI["vermilion"], 2: OI["blue"]}
    styles = {0: "-", 1: "--", 2: "-"}
    with mpl.rc_context(STYLE):
        figure, (value_axis, path_axis) = plt.subplots(
            1, 2, figsize=(7.4, 3.0),
            gridspec_kw={"wspace": 0.28, "left": 0.09, "right": 0.98,
                         "top": 0.88, "bottom": 0.16},
        )
        for k, (label, (grid, J0, storage)) in enumerate(curves.items()):
            width = 2.6 if label == reference else 1.6
            alpha = 0.35 if label == reference else 1.0
            marker = "" if label == reference else "o"
            value_axis.plot(grid, J0 / 1e6, styles[k], color=colors[k],
                            linewidth=width, alpha=alpha, marker=marker,
                            markersize=3, label=label)
            path_axis.plot(np.arange(T + 1), storage, styles[k], color=colors[k],
                           linewidth=width, alpha=alpha, label=label)
        value_axis.set_xlabel("storage $x$ in week 0 (hm³)")
        value_axis.set_ylabel("$J_0^\\star(x)$ (M\\$)")
        value_axis.set_title("Value function", loc="left", fontweight="semibold")
        path_axis.set_xlabel("week of the year")
        path_axis.set_ylabel("storage (hm³)")
        path_axis.set_title("Storage under each policy", loc="left",
                            fontweight="semibold")
        _month_axis(path_axis)
        path_axis.legend(frameon=False, fontsize=7.5, loc="lower right")
        for axis in (value_axis, path_axis):
            axis.spines[["top", "right"]].set_visible(False)
        plt.close(figure)  # displayed once, as the cell's value
        return figure
