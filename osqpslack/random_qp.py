"""Random sparse QPs of the paper's scaling benchmark."""

import math

import numpy as np
from scipy import sparse


def random_qp(n, seed, p_density=0.08, a_density=0.05, regularization=0.1, width=0.5):
    """Random sparse QP with `n` variables and `n` constraints, all of them soft.

    `P = L'L + regularization I` with density `p_density`, and `A` has density
    `a_density`, at least one entry per row, and unit-norm rows. For an anchor
    `x_a` and a target `x_t`, `q = -P x_t` makes `x_t` the unconstrained
    minimizer, and each row's band is centered on the anchor with half-width
    `width |a_i'(x_t - x_a)|`, so the problem is feasible but `x_t` violates
    every row.
    """
    rng = np.random.default_rng(seed)

    def normal(size):
        values = rng.standard_normal(size)
        while np.any(values == 0.0):
            values[values == 0.0] = rng.standard_normal(np.count_nonzero(values == 0.0))
        return values

    # Factor density such that L'L, together with the dense diagonal, reaches
    # p_density on average.
    off_diagonal = (p_density * n - 1.0) / (n - 1.0)
    factor_density = math.sqrt(-math.expm1(math.log1p(-off_diagonal) / n))
    L = sparse.random(n, n, density=factor_density, format="csc", rng=rng, data_rvs=normal)
    P = sparse.csc_matrix(L.T @ L + regularization * sparse.eye(n))

    A = sparse.random(n, n, density=a_density, format="csr", rng=rng, data_rvs=normal)
    empty = np.flatnonzero(np.diff(A.indptr) == 0)
    if empty.size:
        columns = rng.integers(0, n, size=empty.size)
        A = sparse.csr_matrix(A + sparse.coo_matrix((normal(empty.size), (empty, columns)), shape=(n, n)))
    A = sparse.csc_matrix(sparse.diags(1.0 / sparse.linalg.norm(A, axis=1)) @ A)

    anchor = rng.standard_normal(n)
    target = rng.standard_normal(n)
    q = -P @ target
    center = A @ anchor
    half_width = width * np.abs(A @ target - center)
    return P, q, A, center - half_width, center + half_width
