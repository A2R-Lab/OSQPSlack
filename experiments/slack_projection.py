"""Figure: soft slack penalties and the generalized slack projection.

Top:    the effective penalty g(z) = Phi(R(z)) over one coordinate. It is zero on
        the box [l,u]; the hard constraint has infinite walls at l and u, the
        soft penalties grow outside the box.
Bottom: where points v outside the box [l,u] in 2D are mapped by
            Proj^Phi(v) = Proj(v) + prox_{Phi/rho}(R(v)),  R(v) = v - Proj(v).
        The shaded region is mapped exactly onto the box (prox = 0).

Writes figures/slack_projection.{pdf,png}. No solver involved.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from osqpslack.plotting import BAND, COLUMN_WIDTH, INK, PALETTE, save

# Problem data
L = np.array([-1.0, -1.0])
U = np.array([1.0, 1.0])
RHO = 1.0
LIM = 2.6  # axis limits [-LIM, LIM]
V_EXAMPLE = np.array([2.25, 1.5])  # highlighted grid point

# Top panel penalty parameters
ALPHA = 1.0
DELTA = 0.5  # Huber threshold

C_BLUE, C_ORANGE, C_AQUA = PALETTE[:3]
C_INK = INK
C_FRAME = "#9a9994"
C_BOX_FILL = BAND


def proj_box(v):
    return np.clip(v, L, U)


def residual(v):
    return v - proj_box(v)


def prox_hard(r):
    return np.zeros_like(r)


def prox_quadratic(r, alpha=1.0):
    return RHO / (RHO + alpha) * r


def prox_l2(r, alpha=0.6):
    norm = np.linalg.norm(r, axis=-1, keepdims=True)
    return np.maximum(1.0 - alpha / (RHO * np.maximum(norm, 1e-15)), 0.0) * r


# Bottom panels, left to right: (title, prox of Phi/rho, color). Each penalty
# keeps its color from the top panel.
PANELS = [
    (r"hard: $\Phi=I_{\{0\}}$", prox_hard, C_INK),
    (r"quadratic: $\Phi=\frac{\alpha}{2}\|\xi\|_2^2$", prox_quadratic, C_BLUE),
    (r"$\ell_2$: $\Phi=\alpha\|\xi\|_2$", prox_l2, C_ORANGE),
]


def slack_projection(v, prox):
    v = np.atleast_2d(v)
    return proj_box(v) + prox(residual(v))


def huber(r):
    a = np.abs(r)
    return np.where(a <= DELTA, 0.5 * r**2, DELTA * a - 0.5 * DELTA**2)


def draw_landscape(ax):
    z = np.linspace(-LIM, LIM, 1001)
    r = z - np.clip(z, L[0], U[0])
    ymax = 1.8  # headroom above the curves (max 1.6) for the axis arrow

    ax.axvspan(L[0], U[0], color=C_BOX_FILL, zorder=0, lw=0)

    # hard constraint: zero on the box, infinite walls at its edges
    ax.plot([L[0], U[0]], [0, 0], color=C_INK, lw=1.5, zorder=3, clip_on=False, label="hard")
    for edge in (L[0], U[0]):
        ax.plot([edge, edge], [0, ymax], color=C_INK, lw=1.5, ls=(0, (3, 2)), zorder=3)

    curves = [
        (0.5 * ALPHA * r**2, C_BLUE, "-", "quadratic"),
        (ALPHA * np.abs(r), C_ORANGE, "--", r"$\ell_1$, $\ell_2$"),
        (ALPHA * huber(r), C_AQUA, "-.", "Huber"),
    ]
    for y, color, ls, label in curves:
        ax.plot(z, y, color=color, ls=ls, lw=1.5, zorder=2, label=label, clip_on=False)
    ax.legend(loc="upper center", frameon=False, handlelength=2.2, borderaxespad=0.2, labelspacing=0.25)

    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(0, ymax)  # x-axis sits at g = 0
    ax.set_xticks([L[0], U[0]], [r"$\ell$", r"$u$"])
    ax.set_yticks([])
    ax.set_xlabel(r"$z$", ha="right", va="top")
    ax.xaxis.set_label_coords(1.0, -0.03)  # at the right end, below the axis
    ax.set_ylabel(r"$g(z)$", labelpad=2)
    # arrowheads at the ends of the axes
    ax.plot(0, 1, marker="^", markersize=3.5, color=C_INK, transform=ax.transAxes, clip_on=False, zorder=4)
    ax.plot(1, 0, marker=">", markersize=3.5, color=C_INK, transform=ax.transAxes, clip_on=False, zorder=4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)


def grid_points():
    ticks = np.arange(-2.25, 2.26, 0.75)
    pts = np.array([(x, y) for x in ticks for y in ticks])
    return pts[np.any((pts < L) | (pts > U), axis=1)]


def draw_projection(ax, title, prox, color):
    # Region mapped exactly onto the box. For the hard constraint this is the
    # whole plane, so it is not shaded.
    if prox is not prox_hard:
        g = np.linspace(-LIM, LIM, 700)
        X, Y = np.meshgrid(g, g)
        s = prox(residual(np.stack([X.ravel(), Y.ravel()], axis=1)))
        captured = (np.linalg.norm(s, axis=1) < 1e-9).reshape(X.shape).astype(float)
        ax.contourf(X, Y, captured, levels=[0.5, 1.5], colors=[color], alpha=0.35, zorder=0)
        ax.contour(X, Y, captured, levels=[0.5], colors=[color], linewidths=1.0, zorder=1)

    ax.add_patch(Rectangle(L, *(U - L), facecolor=C_BOX_FILL, edgecolor=C_INK, linewidth=0.8, zorder=2))

    V = grid_points()
    Z = slack_projection(V, prox)
    ax.scatter(V[:, 0], V[:, 1], s=2.5, color=color, zorder=3, linewidths=0)
    for v, z in zip(V, Z):
        if np.linalg.norm(z - v) < 1e-9 or np.allclose(v, V_EXAMPLE):
            continue
        ax.annotate(
            "",
            xy=z,
            xytext=v,
            zorder=3,
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.25,head_width=0.12", color=color, lw=0.5, shrinkA=0, shrinkB=0
            ),
        )

    z = slack_projection(V_EXAMPLE, prox)[0]
    ax.annotate(
        "",
        xy=z,
        xytext=V_EXAMPLE,
        zorder=4,
        arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.2", color=color, lw=1.5, shrinkA=0, shrinkB=0),
    )
    ax.scatter(*V_EXAMPLE, s=10, color=color, zorder=5, linewidths=0)
    ax.annotate(r"$v$", V_EXAMPLE, xytext=(0, -3), textcoords="offset points", ha="center", va="top", zorder=6)

    ax.set_title(title, pad=3)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color(C_FRAME)


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "cm",  # match the paper's Computer Modern math
            "font.size": 9,
            "axes.titlesize": 9,
            "pdf.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(COLUMN_WIDTH, 2.55))

    # top: leave room on the left for the y-axis label
    top = fig.add_gridspec(1, 1, left=0.07, right=0.99, top=0.965, bottom=0.6)
    draw_landscape(fig.add_subplot(top[0]))

    bottom = fig.add_gridspec(1, len(PANELS), left=0.005, right=0.995, top=0.43, bottom=0.005, wspace=0.04)
    for i, (title, prox, color) in enumerate(PANELS):
        draw_projection(fig.add_subplot(bottom[0, i]), title, prox, color)

    save(fig, "slack_projection")


if __name__ == "__main__":
    main()
