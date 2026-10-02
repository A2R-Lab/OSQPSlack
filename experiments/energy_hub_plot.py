"""Energy hub schedule figure from data/energy_hub.csv.

figures/energy_hub.{pdf,png}: one panel per soft constraint group, showing the
constrained quantity, its band (gray), and its slack (tinted in the color of
the group's penalty).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from osqpslack import energy_hub as hub
from osqpslack.plotting import (
    BAND,
    COLUMN_WIDTH,
    INK,
    INK_SECONDARY,
    PENALTY_COLORS,
    PENALTY_LABELS,
    save,
    use_paper_style,
)

DATA = Path(__file__).parents[1] / "data" / "energy_hub.csv"
TEXT_SIZE = 8  # in-panel titles and legends
SLACK_ALPHA = 0.55
LINE_WIDTH = 1.5


def soft_band(ax, time, values, lower, upper, penalty, step=False, **line):
    """A soft-constrained quantity: its band in gray, its slack tinted.

    Array bounds change halfway between samples. `step` says whether `values`
    is piecewise constant (an input) or piecewise linear (a state).
    """
    color = PENALTY_COLORS[penalty]
    if np.isscalar(lower) and np.isscalar(upper):
        ax.axhspan(lower, upper, color=BAND, linewidth=0, zorder=0)
    else:
        ax.fill_between(time, lower, upper, step="mid", color=BAND, linewidth=0, zorder=0)
    lower = np.broadcast_to(lower, time.shape)
    upper = np.broadcast_to(upper, time.shape)
    if step:
        t, v, lo, hi = time, values, lower, upper
    else:
        # A linear line against a step band: evaluate both on a dense grid,
        # bracketing each step closely so the slack edge is vertical there.
        midpoints = 0.5 * (time[:-1] + time[1:])
        width = 1e-6 * (time[-1] - time[0])
        t = np.sort(
            np.concatenate((np.linspace(time[0], time[-1], 50 * time.size), midpoints - width, midpoints + width))
        )
        sample = np.searchsorted(midpoints, t)
        v, lo, hi = np.interp(t, time, values), lower[sample], upper[sample]
    ax.fill_between(t, np.clip(v, lo, hi), v, step="mid" if step else None, color=color, alpha=SLACK_ALPHA, linewidth=0)
    line.setdefault("linewidth", LINE_WIDTH)
    if step:
        ax.step(time, values, where="mid", color=color, **line)
    else:
        ax.plot(time, values, color=color, **line)


def title(ax, group, penalty):
    """Label the panel inside its top-left corner."""
    ax.text(
        0.015,
        0.95,
        f"{group}: {PENALTY_LABELS[penalty]}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TEXT_SIZE,
        zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
    )


def price_periods(ax, time, price):
    """A dotted line at each tariff change and the price on top of each period."""
    dt = hub.DT
    changes = np.flatnonzero(np.diff(price)) + 1
    starts, ends = np.concatenate(([0], changes)), np.concatenate((changes, [price.size]))
    for change in changes:
        ax.axvline(time[change] - 0.5 * dt, color=INK_SECONDARY, linestyle=":", linewidth=0.8, zorder=1)
    for start, end in zip(starts, ends):
        left, right = time[start] - 0.5 * dt, time[end - 1] + 0.5 * dt
        first = start == 0  # right-aligned, clear of the panel title
        ax.text(
            right - 0.4 if first else 0.5 * (left + right),
            0.96,
            f"{price[start]:.2f}/kWh",
            transform=ax.get_xaxis_transform(),
            ha="right" if first else "center",
            va="top",
            fontsize=7,
            color=INK_SECONDARY,
        )


def main():
    use_paper_style()
    df = pd.read_csv(DATA)
    t = df.time.to_numpy()
    t_input = t[:-1] + 0.5 * hub.DT  # inputs plotted at the middle of their step
    inputs = df.iloc[:-1]

    fig, axes = plt.subplots(4, 1, figsize=(COLUMN_WIDTH, 4.0), sharex=True, layout="constrained")

    ax = axes[0]
    for zone, style in ((1, "-"), (2, "--")):
        soft_band(
            ax,
            t,
            df[f"zone{zone}_temperature"].to_numpy(),
            df.comfort_lower.to_numpy(),
            df.comfort_upper.to_numpy(),
            "quadratic",
            linestyle=style,
            label=f"Zone {zone}",
        )
    temperatures = df[["zone1_temperature", "zone2_temperature"]].to_numpy()
    ax.set_ylim(np.floor(temperatures.min()) - 0.5, np.ceil(temperatures.max()) + 2.5)
    ax.set_ylabel("Temp. [°C]")
    title(ax, "Comfort", "quadratic")
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.0, 0.8), ncol=2, fontsize=TEXT_SIZE, handlelength=1.8, borderaxespad=0.3
    )

    ax = axes[1]
    grid = inputs.grid_import.to_numpy()
    soft_band(ax, t_input, grid, 0.0, hub.CONTRACT_LIMIT, "linf", step=True)
    ax.set_ylim(0.0, 1.35 * grid.max())
    price_periods(ax, t_input, inputs.price.to_numpy())
    ax.set_ylabel("Grid [kW]")
    title(ax, "Grid contract", "linf")

    ax = axes[2]
    soc = 100 * df.battery_charge.to_numpy() / hub.BATTERY_CAPACITY
    soft_band(ax, t, soc, 100 * hub.SOC_BAND[0], 100 * hub.SOC_BAND[1], "elastic-net")
    ax.set_ylim(0.0, 120.0)
    ax.set_yticks([0, 20, 50, 80, 100])
    ax.set_ylabel("SOC [%]")
    title(ax, "Battery health", "elastic-net")

    # HVAC power on the left axis; its ramp rate, the soft-constrained
    # quantity, on the right axis. The same headroom keeps the zeros aligned.
    ax = axes[3]
    headroom = 1.6
    power = inputs.hvac_power.to_numpy()
    power_line = ax.step(t_input, power, where="mid", color=INK, label="Power")[0]
    ax.axhline(hub.HVAC_POWER_LIMIT, color=INK_SECONDARY, linestyle="--", linewidth=0.8)
    extent = 1.15 * max(np.abs(power).max(), hub.HVAC_POWER_LIMIT)
    ax.set_ylim(-extent, headroom * extent)
    ax.set_yticks([0, round(hub.HVAC_POWER_LIMIT / 2), round(hub.HVAC_POWER_LIMIT)])
    ax.set_ylabel("HVAC [kW]")
    title(ax, "HVAC ramp", "huber")

    ramp_ax = ax.twinx()
    ramp_ax.set_zorder(ax.get_zorder() - 1)  # ramp behind the power line
    ax.patch.set_visible(False)
    ramps = np.diff(power) / hub.DT
    soft_band(ramp_ax, t[1:-1], ramps, -hub.RAMP_LIMIT, hub.RAMP_LIMIT, "huber", step=True)
    extent = 1.15 * max(np.abs(ramps).max(), hub.RAMP_LIMIT)
    ramp_ax.set_ylim(-extent, headroom * extent)
    tick = np.round(hub.RAMP_LIMIT)
    ramp_ax.set_yticks([-tick, 0, tick])
    ramp_ax.set_ylabel("Ramp [kW/h]", color=PENALTY_COLORS["huber"])
    ramp_ax.tick_params(axis="y", colors=PENALTY_COLORS["huber"])
    ramp_ax.grid(False)
    ramp_line = plt.Line2D([], [], color=PENALTY_COLORS["huber"], linewidth=LINE_WIDTH, label="Ramp")
    ax.legend(handles=[power_line, ramp_line], loc="lower left", ncol=2, handlelength=1.5, fontsize=TEXT_SIZE)

    ax.set_xlim(0.0, t[-1])
    ax.set_xticks(np.arange(0.0, t[-1] + 1e-9, 6.0))
    ax.set_xlabel("Time [h]")
    fig.align_ylabels(axes)
    save(fig, "energy_hub")


if __name__ == "__main__":
    main()
