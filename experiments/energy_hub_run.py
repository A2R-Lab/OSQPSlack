"""Energy hub: the one-day schedule, and native against lifted solve time.

Writes the trajectories to data/energy_hub.csv and the timing comparison to
data/energy_hub.json. Times are machine dependent; run single-threaded on an
otherwise idle machine.
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import osqp
from scipy import sparse

from osqpslack import energy_hub as hub
from osqpslack.penalties import lift

SETTINGS = dict(
    eps_abs=1e-6,
    eps_rel=1e-6,
    rho=0.1,
    adaptive_rho=True,
    adaptive_rho_interval=50,
    scaling=10,
    max_iter=100_000,
    check_termination=25,
    polishing=False,
    verbose=False,
)
REPETITIONS = 10
DATA = Path(__file__).parents[1] / "data"


def solve(P, q, A, l, u, penalty=None, **settings):
    """Set up and solve from scratch; only the OSQP calls are timed."""
    P, A = sparse.triu(P, format="csc"), sparse.csc_matrix(A)
    start = perf_counter()
    solver = osqp.OSQP(algebra="builtin")
    solver.setup(P=P, q=q, A=A, l=l, u=u, **{**SETTINGS, **settings})
    if penalty is not None:
        penalty.setup(solver)
    result = solver.solve(raise_error=False)
    assert result.info.status == "solved", result.info.status
    return result, perf_counter() - start


def main():
    DATA.mkdir(exist_ok=True)
    P, q, A, l, u, penalty = hub.problem()

    # Schedule. Checking termination every iteration stops at the first
    # iterate within tolerance.
    result, _ = solve(P, q, A, l, u, penalty, check_termination=1)
    x, inputs = hub.states(result.x), hub.inputs(result.x)
    lower, upper = hub.comfort_band()
    columns = {
        "time": np.arange(hub.N + 1) * hub.DT,
        "zone1_temperature": x[:, hub.TEMP1],
        "zone2_temperature": x[:, hub.TEMP2],
        "comfort_lower": lower,
        "comfort_upper": upper,
        "battery_charge": x[:, hub.CHARGE],
        # Inputs hold over [t_k, t_k+1); the final state has none.
        "hvac_power": np.append(inputs[:, hub.HVAC], np.nan),
        "grid_import": np.append(inputs[:, hub.GRID], np.nan),
        "price": np.append(hub.price(), np.nan),
    }
    np.savetxt(
        DATA / "energy_hub.csv",
        np.column_stack(list(columns.values())),
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt="%.6g",
    )
    print(f"wrote {DATA / 'energy_hub.csv'}")

    # Timing: one discarded warm-up each, then alternating repetitions.
    problems = {"native": (P, q, A, l, u, penalty), "lifted": lift(P, q, A, l, u, penalty)}
    times, iterations = {"native": [], "lifted": []}, {}
    for name in problems:
        solve(*problems[name])
    for repetition in range(REPETITIONS):
        for name in ("native", "lifted") if repetition % 2 == 0 else ("lifted", "native"):
            result, time = solve(*problems[name])
            times[name].append(time)
            iterations[name] = result.info.iter
    timing = {
        name: {
            "variables": problems[name][0].shape[0],
            "constraints": problems[name][2].shape[0],
            "iterations": iterations[name],
            "time_ms": round(1e3 * float(np.median(times[name])), 2),
        }
        for name in problems
    }
    timing["speedup"] = round(timing["lifted"]["time_ms"] / timing["native"]["time_ms"], 2)
    (DATA / "energy_hub.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
