"""Figures for the stochastic hydro reservoir example in stochastic DP.

The chapter solves the problem in a visible code cell. This module only draws
the results, so the functions take the chapter's arrays as arguments. The
style, colors and month axis are shared with the deterministic example in
``hydro_reservoir``. Each figure is closed before it is returned, so a notebook
shows it once, as the value of the cell.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from hydro_reservoir import OI, STYLE, _month_axis


def make_stochastic_policy_figure(
    w_bar: np.ndarray,
    factors: np.ndarray,
    X: np.ndarray,
    policy: np.ndarray,
    storage_paths: np.ndarray,
    week: int,
    water_value_ce: np.ndarray,
    water_value_sto: np.ndarray,
) -> plt.Figure:
    """Stochastic policy map with simulated storage paths, and water values.

    The left panel repeats the layout of the deterministic policy figure: an
    inflow strip, now with the forecast and the range of the random inflow,
    above the release map. ``storage_paths`` has one column per simulated
    year. The right panel compares the water value of week ``week`` under the
    certainty-equivalent and the stochastic model.
    """
    T = len(w_bar)
    weeks = np.arange(T)
    extent = (0, T, X[0], X[-1])
    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(7.4, 4.3))
        grid = figure.add_gridspec(
            2,
            2,
            height_ratios=(1, 3.2),
            width_ratios=(1.35, 1),
            left=0.085,
            right=0.985,
            bottom=0.16,
            top=0.93,
            hspace=0.08,
            wspace=0.24,
        )
        inflow_axis = figure.add_subplot(grid[0, 0])
        policy_axis = figure.add_subplot(grid[1, 0], sharex=inflow_axis)
        value_axis = figure.add_subplot(grid[:, 1])

        low, high = factors.min() * w_bar, factors.max() * w_bar
        inflow_axis.bar(weeks + 0.5, high - low, bottom=low, width=0.85,
                        color=OI["skyblue"], alpha=0.45, linewidth=0)
        inflow_axis.bar(weeks + 0.5, 4, bottom=w_bar - 2, width=0.85,
                        color=OI["blue"], linewidth=0)
        inflow_axis.set_ylabel("inflow\n(hm³/wk)", fontsize=8)
        inflow_axis.set_title("Stochastic policy $u_t^\\star(x)$", loc="left",
                              fontweight="semibold", pad=4)
        inflow_axis.spines[["top", "right"]].set_visible(False)
        inflow_axis.tick_params(labelsize=7.5, labelbottom=False)

        image = policy_axis.imshow(
            policy.T, origin="lower", aspect="auto", extent=extent,
            cmap="cividis", interpolation="nearest",
        )
        for storage in storage_paths.T:
            policy_axis.plot(np.arange(T + 1), storage, color="white",
                             linewidth=0.6, alpha=0.35)
        _month_axis(policy_axis)
        policy_axis.set_xlabel("week of the year")
        policy_axis.set_ylabel("storage $x$ (hm³)")
        bar = figure.colorbar(
            image, ax=policy_axis, orientation="horizontal", fraction=0.05,
            pad=0.2, aspect=40,
        )
        bar.set_label("release (hm³/week)", fontsize=8)
        bar.ax.tick_params(labelsize=7.5)

        value_axis.plot(X, water_value_ce / 1e3, color=OI["vermilion"],
                        linewidth=1.6, label="certainty equivalent")
        value_axis.plot(X, water_value_sto / 1e3, color=OI["blue"],
                        linewidth=1.6, label="stochastic")
        value_axis.set_xlabel("storage $x$ (hm³)")
        value_axis.set_ylabel("k$ per hm³")
        value_axis.set_title(
            f"Water value $\\partial J_{{{week}}}^\\star/\\partial x$ in week {week}",
            loc="left", fontweight="semibold", pad=4,
        )
        value_axis.legend(frameon=False, fontsize=7.5, loc="lower left")
        value_axis.spines[["top", "right"]].set_visible(False)
        plt.close(figure)  # displayed once, as the cell's value
        return figure


def make_stochastic_comparison_figure(
    storage_ce: np.ndarray,
    storage_sto: np.ndarray,
    revenue_ce: np.ndarray,
    revenue_sto: np.ndarray,
    x_max: float,
    first_week: int = 30,
) -> plt.Figure:
    """Storage near capacity and per-year revenue gain on common inflows.

    ``storage_ce`` and ``storage_sto`` have one row per week and one column per
    simulated year, both driven by the same inflow draws.
    """
    T = storage_ce.shape[0] - 1
    weeks = np.arange(T + 1)
    shown = weeks >= first_week
    with mpl.rc_context(STYLE):
        figure, (storage_axis, gain_axis) = plt.subplots(
            1, 2, figsize=(7.4, 3.0),
            gridspec_kw={"width_ratios": (1.35, 1), "wspace": 0.3,
                         "left": 0.1, "right": 0.98, "top": 0.88,
                         "bottom": 0.16},
        )
        for storage, color, label in (
            (storage_ce, OI["vermilion"], "certainty equivalent"),
            (storage_sto, OI["blue"], "stochastic"),
        ):
            low, mid, high = np.quantile(storage[shown], (0.05, 0.5, 0.95),
                                         axis=1)
            for edge in (low, high):
                storage_axis.plot(weeks[shown], edge, color=color,
                                  linewidth=0.8, linestyle="--", alpha=0.9)
            storage_axis.plot(weeks[shown], mid, color=color, linewidth=1.6,
                              label=label)
        storage_axis.axhline(x_max, color="0.3", linewidth=0.8, linestyle=":")
        storage_axis.text(first_week + 0.5, x_max + 8, "capacity", fontsize=7.5,
                          color="0.3", va="bottom")
        storage_axis.set_ylim(1500, x_max + 60)
        _month_axis(storage_axis)
        storage_axis.set_xlim(first_week, T)
        storage_axis.set_xlabel("week of the year")
        storage_axis.set_ylabel("storage (hm³)")
        storage_axis.set_title("Storage: median and 5 to 95% range", loc="left",
                               fontweight="semibold")
        storage_axis.legend(frameon=False, fontsize=7.5, loc="lower left")

        gain = (revenue_sto - revenue_ce) / 1e6
        gain_axis.hist(gain, bins=40, color=OI["blue"], alpha=0.75)
        gain_axis.axvline(0, color="0.3", linewidth=0.8)
        gain_axis.axvline(gain.mean(), color=OI["black"], linewidth=1.2,
                          linestyle="--")
        gain_axis.text(
            gain.mean() + 0.08, 0.97, f"mean {gain.mean():.2f} M\\$\n"
            f"positive in {np.mean(gain > 0):.0%}\nof the years",
            transform=gain_axis.get_xaxis_transform(), fontsize=7.5, va="top",
        )
        gain_axis.set_xlabel("revenue difference (M\\$)")
        gain_axis.set_ylabel("simulated years")
        gain_axis.set_title("Revenue gain in each year", loc="left",
                            fontweight="semibold")
        for axis in (storage_axis, gain_axis):
            axis.spines[["top", "right"]].set_visible(False)
        plt.close(figure)  # displayed once, as the cell's value
        return figure
