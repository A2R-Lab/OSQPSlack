"""Scaling figure and timing table from data/scaling.csv.

figures/scaling.{pdf,png}: per penalty, the speedup (lifted over native
setup-plus-solve time), ADMM iterations, and the runtime breakdown normalized
to the lifted runtime. figures/scaling_times.tex: median runtimes, a bare
tabular for the paper's table.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator

from osqpslack.plotting import (
    FIGURES,
    INK,
    INK_SECONDARY,
    NEUTRAL,
    PALETTE,
    PENALTY_LABELS,
    TEXT_WIDTH,
    save,
    use_paper_style,
)

DATA = Path(__file__).parents[1] / "data" / "scaling.csv"
PENALTIES = ("quadratic", "l1", "elastic-net", "huber", "linf")
TABLE_N = (25, 50, 100, 400, 800)

# Stacked runtime components, bottom to top.
COMPONENTS = {
    "Setup": PALETTE[0],
    "Linear solve": PALETTE[1],
    "Projection": PALETTE[2],
    r"$\rho$ update": PALETTE[3],
    "Other": NEUTRAL,
}
HATCHES = {"native": "////", "lifted": ""}
LINES = {"native": "-", "lifted": "--"}
FACES = {"native": INK, "lifted": "white"}
BAR_WIDTH, BAR_GAP = 0.36, 0.04


def load():
    """One row per instance, with native and lifted columns side by side."""
    df = pd.read_csv(DATA)
    df["total"] = df.setup_time + df.solve_time
    df["Setup"] = df.setup_time
    df["Linear solve"] = df.linsys_time
    df["Projection"] = df.projection_time
    df[r"$\rho$ update"] = df.rho_update_time
    df["Other"] = (df.solve_time - df.linsys_time - df.projection_time - df.rho_update_time).clip(lower=0)
    pairs = df.pivot(index=["penalty", "n", "seed"], columns="formulation")
    solved = (pairs["status"] == "solved").all(axis=1)
    if not solved.all():
        print(f"dropping {(~solved).sum()} instances that did not solve")
    return pairs[solved]


def quartiles(series):
    """Median and interquartile range per n."""
    grouped = series.groupby("n")
    return grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)


def plot(pairs):
    n_values = sorted(pairs.index.get_level_values("n").unique())
    x = np.arange(len(n_values), dtype=float)
    fig, axes = plt.subplots(
        3,
        len(PENALTIES),
        figsize=(TEXT_WIDTH, 3.4),
        sharex=True,
        sharey="row",
        height_ratios=(1, 1, 2),
        layout="constrained",
    )
    for col, penalty in enumerate(PENALTIES):
        data = pairs.loc[penalty]

        ax = axes[0, col]
        median, low, high = quartiles(data["total", "lifted"] / data["total", "native"])
        ax.fill_between(x, low, high, color=INK, alpha=0.12, linewidth=0)
        ax.plot(x, median, color=INK, marker="o")
        ax.axhline(1.0, color=INK_SECONDARY, linewidth=0.6, linestyle=":")
        ax.set_title(PENALTY_LABELS[penalty])

        ax = axes[1, col]
        for side in ("native", "lifted"):
            median, low, high = quartiles(data["iter", side])
            ax.fill_between(x, low, high, color=INK, alpha=0.12, linewidth=0)
            ax.plot(x, median, color=INK, linestyle=LINES[side], marker="o", markerfacecolor=FACES[side])

        ax = axes[2, col]
        medians = data[list(COMPONENTS)].groupby("n").median()
        reference = sum(medians[c, "lifted"] for c in COMPONENTS)
        for side, offset in (("native", -1), ("lifted", 1)):
            bottom = np.zeros(len(n_values))
            for component, color in COMPONENTS.items():
                share = 100 * medians[component, side] / reference
                ax.bar(
                    x + offset * (BAR_WIDTH + BAR_GAP) / 2,
                    share,
                    BAR_WIDTH,
                    bottom=bottom,
                    color=color,
                    edgecolor=INK,
                    linewidth=0.3,
                    hatch=HATCHES[side],
                )
                bottom += share
        ax.axhline(100.0, color=INK_SECONDARY, linewidth=0.6, linestyle=":")
        ax.set_xticks(x, [str(n) for n in n_values])
        ax.set_xlim(-0.55, len(n_values) - 0.45)

    for ax in axes.flat:
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
    for ax, label in zip(axes[:, 0], ("Speedup\n", "Iterations", "Time [%]")):
        ax.set_ylabel(label)
    axes[0, 0].annotate(
        "(Lifted / Native)",
        xy=(1.0, 0.5),
        xycoords=axes[0, 0].yaxis.label,
        rotation=90,
        ha="right",
        va="center",
        fontsize=7,
    )
    axes[0, 0].set_ylim(0.0, 4.0)
    axes[0, 0].set_yticks([1, 2, 3, 4], [rf"{k}$\times$" for k in (1, 2, 3, 4)])
    iteration_peak = pairs["iter"].groupby(["penalty", "n"]).quantile(0.75).max().max()
    axes[1, 0].set_ylim(0.0, 1.1 * iteration_peak)
    axes[1, 0].yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
    axes[2, 0].set_ylim(0.0, 110.0)
    axes[2, 0].set_yticks([0, 25, 50, 75, 100])
    fig.align_ylabels(axes[:, 0])
    fig.supxlabel(r"Variables $n$", fontsize=9)
    fig.get_layout_engine().set(w_pad=0.01, wspace=0.0)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=INK, linewidth=0.3) for c in COMPONENTS.values()]
    handles += [
        (
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, linewidth=0.3, hatch=HATCHES[side]),
            plt.Line2D([], [], color=INK, marker="o", markerfacecolor=FACES[side], linestyle=LINES[side]),
        )
        for side in ("native", "lifted")
    ]
    labels = list(COMPONENTS) + ["Native", "Lifted"]
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(labels),
        handlelength=2.2,
        handletextpad=0.4,
        columnspacing=1.0,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
    )
    save(fig, "scaling")


def ms(value):
    """Two significant digits below 10 ms, whole milliseconds above."""
    return f"{value:#.2g}".rstrip(".") if value < 10.0 else f"{value:.0f}"


def table(pairs):
    medians = 1e3 * pairs["total"].groupby(["penalty", "n"]).median()
    k = 2 * len(TABLE_N)
    lines = [
        r"\begin{tabular}{l" + r"r@{\,/\,}l" * len(TABLE_N) + "}",
        r"\toprule",
        rf"& \multicolumn{{{k}}}{{c}}{{Native / Lifted Runtime [ms]}} \\",
        rf"\cmidrule(lr){{2-{k + 1}}}",
        "Penalty & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$n={n}$}}" for n in TABLE_N) + r" \\",
        r"\midrule",
    ]
    for penalty in PENALTIES:
        cells = [
            f"{ms(medians.loc[(penalty, n), 'native'])} & {ms(medians.loc[(penalty, n), 'lifted'])}" for n in TABLE_N
        ]
        lines.append(f"{PENALTY_LABELS[penalty]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (FIGURES / "scaling_times.tex").write_text("\n".join(lines))
    print(f"wrote {FIGURES / 'scaling_times.tex'}")


if __name__ == "__main__":
    use_paper_style()
    pairs = load()
    plot(pairs)
    table(pairs)
