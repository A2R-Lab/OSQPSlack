"""Shared figure style of the paper."""

from pathlib import Path

import matplotlib.pyplot as plt

FIGURES = Path(__file__).parents[1] / "figures"

# IEEEtran widths in inches. Figures are drawn at their final size, so the
# fonts print at their nominal 9 pt.
COLUMN_WIDTH = 3.5
TEXT_WIDTH = 7.16

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
NEUTRAL = "#c3c2b7"
BAND = "#e4e3df"

PENALTY_COLORS = {
    "quadratic": PALETTE[0],
    "l1": PALETTE[1],
    "huber": PALETTE[2],
    "elastic-net": PALETTE[3],
    "linf": PALETTE[4],
}
PENALTY_LABELS = {
    "quadratic": "Quadratic",
    "l1": r"$\ell_1$",
    "elastic-net": "Elastic net",
    "huber": "Huber",
    "linf": r"$\ell_\infty$",
}


def use_paper_style():
    """9 pt Times text with Computer Modern math, without a LaTeX install."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.5,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.columnspacing": 1.0,
            "hatch.linewidth": 0.4,
            "pdf.fonttype": 42,  # TrueType fonts, as IEEE PDF checks require
        }
    )


def save(fig, name):
    """Save `fig` as figures/<name>.pdf (paper) and figures/<name>.png (README)."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {FIGURES / name}.{{pdf,png}}")
