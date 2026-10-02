"""Slack penalties: native OSQP setup and the equivalent lifted standard QP.

A soft QP is `min 1/2 x'Px + q'x + Phi(xi)  s.t.  l <= Ax + xi <= u`. Its
penalty `Phi` is stored row by row, in exactly the form of the arguments to
`OSQP.setup_penalty`:

    type    OSQP penalty type (NONE = hard row)
    alpha1  l1 weight (L1L2), weight (HUBER, NORMINF)
    alpha2  quadratic weight (L1L2)
    delta   Huber threshold
    group   NORMINF group label, -1 for rows that pool with nothing

so `Penalty.setup` passes the arrays straight to the solver and `lift` builds
the standard QP reformulation of the paper's appendix from the same data.
"""

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

NONE = int(osqp.PenaltyType.OSQP_PENALTY_NONE)
L1L2 = int(osqp.PenaltyType.OSQP_PENALTY_L1L2)
HUBER = int(osqp.PenaltyType.OSQP_PENALTY_HUBER)
NORMINF = int(osqp.PenaltyType.OSQP_PENALTY_NORMINF)


@dataclass
class Penalty:
    type: np.ndarray
    alpha1: np.ndarray
    alpha2: np.ndarray
    delta: np.ndarray
    group: np.ndarray

    @classmethod
    def hard(cls, m):
        """All `m` rows hard."""
        zeros = np.zeros(m)
        return cls(np.full(m, NONE), zeros, zeros.copy(), zeros.copy(), np.full(m, -1))

    def soften(self, rows, type, alpha1=0.0, alpha2=0.0, delta=0.0, group=-1):
        """Give `rows` one penalty. NORMINF rows sharing `group` pool together."""
        self.type[rows] = type
        self.alpha1[rows] = alpha1
        self.alpha2[rows] = alpha2
        self.delta[rows] = delta
        self.group[rows] = group
        return self

    def setup(self, solver):
        """Soften the rows of an already set up OSQP solver."""
        solver.setup_penalty(
            self.type,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            delta=self.delta,
            group_id=self.group if np.any(self.group >= 0) else None,
        )


def lift(P, q, A, l, u, penalty):
    """Reformulate a soft QP as a standard QP (paper appendix).

    The lifted variables are `(x, xi, w, s)`, so `x` is the first `n` entries:

    - quadratic rows (L1L2 with alpha1 = 0) get a slack `xi_i` with cost
      `alpha2/2 xi_i^2`, entering the row as `l_i <= a_i'x + xi_i <= u_i`;
    - every other soft row is relaxed to `l_i - r_i <= a_i'x <= u_i + r_i`,
      split into one row per finite side, where `r_i = s_i` (l1, elastic net,
      shared by the group for NORMINF) or `r_i = w_i + s_i` (Huber);
    - each `s` is nonnegative and costs `alpha1 s + alpha2/2 s^2`
      (`alpha1 delta s` for Huber), each Huber `w_i` costs `alpha1/2 w_i^2`.
    """
    A = sparse.csr_matrix(A)
    m, n = A.shape
    t, a1, a2, delta = penalty.type, penalty.alpha1, penalty.alpha2, penalty.delta

    quadratic = (t == L1L2) & (a1 == 0.0)
    relaxed = (t != NONE) & ~quadratic
    slack_rows, huber_rows, relaxed_rows = (np.flatnonzero(rows) for rows in (quadratic, t == HUBER, relaxed))

    # One s per relaxed row, except that a NORMINF group shares one.
    s_index, s_linear, s_quadratic = {}, [], []
    s_of_row = []
    for i in relaxed_rows:
        key = ("group", penalty.group[i]) if t[i] == NORMINF else ("row", i)
        if key not in s_index:
            s_index[key] = len(s_index)
            s_linear.append(a1[i] * delta[i] if t[i] == HUBER else a1[i])
            s_quadratic.append(a2[i] if t[i] == L1L2 else 0.0)
        s_of_row.append(s_index[key])
    n_xi, n_w, n_s = len(slack_rows), len(huber_rows), len(s_index)

    def selector(rows, cols, n_cols):
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, n_cols))

    Xi = selector(slack_rows, np.arange(n_xi), n_xi)
    W = selector(huber_rows, np.arange(n_w), n_w)
    S = selector(relaxed_rows, s_of_row, n_s)
    R = sparse.hstack((W, S), format="csr")  # row i of R @ (w, s) is r_i

    P_lift = sparse.block_diag(
        (P, sparse.diags(np.concatenate((a2[slack_rows], a1[huber_rows], s_quadratic)))),
        format="csc",
    )
    q_lift = np.concatenate((q, np.zeros(n_xi + n_w), s_linear))

    # Each relaxed row keeps its upper side in place (its lower side if the
    # upper is infinite); a second finite side becomes an extra row.
    upper_in_place = relaxed & (np.isfinite(u) | ~np.isfinite(l))
    lower_in_place = relaxed & ~upper_in_place
    sign = np.where(upper_in_place, -1.0, 1.0)
    l_top = np.where(upper_in_place, -np.inf, l)
    u_top = np.where(lower_in_place, np.inf, u)
    extra = np.flatnonzero(upper_in_place & np.isfinite(l))

    A_lift = sparse.vstack(
        (
            sparse.hstack((A, Xi, sparse.diags(sign) @ R)),
            sparse.hstack((A[extra], sparse.csr_matrix((len(extra), n_xi)), R[extra])),
            sparse.hstack((sparse.csr_matrix((n_s, n + n_xi + n_w)), sparse.eye(n_s))),
        ),
        format="csc",
    )
    l_lift = np.concatenate((l_top, l[extra], np.zeros(n_s)))
    u_lift = np.concatenate((u_top, np.full(len(extra) + n_s, np.inf)))
    return P_lift, q_lift, A_lift, l_lift, u_lift
