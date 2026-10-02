"""Scaling benchmark: native slack penalties against explicit lifting.

For every size n, penalty, and seed, solve one random QP (all rows soft) both
natively and lifted to a standard QP, and record times and iterations in
data/scaling.csv. Times are machine dependent; run single-threaded on an
otherwise idle machine.
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import csv
from pathlib import Path
from time import perf_counter

import osqp
from scipy import sparse

from osqpslack.penalties import HUBER, L1L2, NORMINF, Penalty, lift
from osqpslack.random_qp import random_qp

N = (25, 50, 100, 200, 400, 800)
SEEDS = range(1, 101)
PENALTIES = {
    "quadratic": dict(type=L1L2, alpha2=0.8),
    "l1": dict(type=L1L2, alpha1=0.35),
    "elastic-net": dict(type=L1L2, alpha1=0.2, alpha2=0.7),
    "huber": dict(type=HUBER, alpha1=1.1, delta=0.15),
    "linf": dict(type=NORMINF, alpha1=1.0, group=0),  # one group: all rows
}
SETTINGS = dict(
    eps_abs=1e-6,
    eps_rel=1e-6,
    rho=0.1,
    adaptive_rho=True,
    adaptive_rho_interval=50,
    scaling=10,
    max_iter=10_000,
    check_termination=25,
    polishing=False,
    verbose=False,
)
OUTPUT = Path(__file__).parents[1] / "data" / "scaling.csv"


def solve(P, q, A, l, u, penalty=None):
    """Set up and solve from scratch; only the OSQP calls are timed."""
    P, A = sparse.triu(P, format="csc"), sparse.csc_matrix(A)
    start = perf_counter()
    solver = osqp.OSQP(algebra="builtin")
    solver.setup(P=P, q=q, A=A, l=l, u=u, **SETTINGS)
    if penalty is not None:
        penalty.setup(solver)
    setup_time = perf_counter() - start
    start = perf_counter()
    result = solver.solve(raise_error=False)
    solve_time = perf_counter() - start
    info = result.info
    return {
        "status": info.status,
        "iter": info.iter,
        "obj_val": info.obj_val,
        "setup_time": setup_time,
        "solve_time": solve_time,
        "linsys_time": info.linear_system_time,
        "projection_time": info.projection_time,
        "rho_update_time": info.rho_update_time,
    }


def main():
    OUTPUT.parent.mkdir(exist_ok=True)
    rows = []
    for n in N:
        for name, spec in PENALTIES.items():
            for seed in SEEDS:
                P, q, A, l, u = random_qp(n, seed)
                penalty = Penalty.hard(n).soften(slice(None), **spec)
                problems = {"native": (P, q, A, l, u, penalty), "lifted": lift(P, q, A, l, u, penalty)}

                # One discarded warm-up run each, then one measured run each;
                # alternate which formulation goes first to cancel order effects.
                order = ["native", "lifted"] if seed % 2 == 0 else ["lifted", "native"]
                for formulation in order:
                    solve(*problems[formulation])
                results = {formulation: solve(*problems[formulation]) for formulation in order}

                native, lifted = results["native"], results["lifted"]
                if abs(native["obj_val"] - lifted["obj_val"]) > 1e-3 * max(1.0, abs(lifted["obj_val"])):
                    print(f"warning: objectives differ for n={n} {name} seed={seed}")
                for formulation, result in results.items():
                    rows.append({"n": n, "penalty": name, "seed": seed, "formulation": formulation, **result})
            print(f"n={n} {name} done", flush=True)

    with open(OUTPUT, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()})
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
