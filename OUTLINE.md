# OSQPSlack — Implementation Outline

Code to reproduce the results of the paper *ADMM with general slack penalties*
(sources: `~/Papers/admm_general_slack`, kept separate on Overleaf).

Guiding principle: **as simple as possible**. Only what is needed for the paper
figures/tables plus one tutorial notebook. No framework, no config system, no
generic result infrastructure.

---

## 1. Repositories

Three repos, all under the `A2R-Lab` GitHub organization.

| Repo | Role | Branch / pin |
|---|---|---|
| `A2R-Lab/OSQPSlack` | **Entry point.** Experiments, data, figures, notebook. | `main` |
| `A2R-Lab/osqp` | Fork of `osqp/osqp` with the C implementation (penalties, generalized slack projection, polishing, codegen). Full upstream history. | branch `general_slack` (commit pinned) |
| `A2R-Lab/osqp-python` | Fork of `osqp/osqp-python` with the Python bindings (`setup_penalty`, `update_penalty_*`, `slack`, timing info). | branch `general_slack` (commit pinned) |

How they connect:

```
OSQPSlack
├── external/osqp          ── git submodule → A2R-Lab/osqp        @ <commit>
└── external/osqp-python   ── git submodule → A2R-Lab/osqp-python @ <commit>
                                   │
                                   └── built with OSQP_SOURCE_DIR=../osqp
                                       (C code comes from the submodule,
                                        not from a network fetch)
```

- The forks keep their full git history. Commits are not squashed. Uncommitted
  local work in `~/Projects/osqp` (docs, `PROPOSAL*.md`) stays out.
- Submodules pin exact commits on `general_slack`; no tags. `general_slack`
  must not be force-pushed or rebased after pinning, or the pinned
  commits may become unreachable.
- **No changes needed in the forks.** `osqp-python` already fetches
  `A2R-Lab/osqp @ general_slack` when built on its own. Inside OSQPSlack that
  fetch is bypassed via `OSQP_SOURCE_DIR`, so the pinned `external/osqp` commit
  is what gets compiled.
- Keep the two submodule pins in sync with each other: when `osqp` moves, bump
  both `external/osqp` and `external/osqp-python` together.

## 2. Layout of `OSQPSlack`

```
OSQPSlack/
├── README.md                 # paper ref, install, reproduce, figure → script map, embedded PNGs
├── LICENSE                   # MIT
├── requirements.txt          # pinned numpy/scipy/matplotlib/pandas/jupyter, ./external/osqp-python, -e .
├── pyproject.toml            # minimal: only declares the osqpslack package
├── Makefile                  # make install | data | figures | all
├── external/
│   ├── osqp/                 # submodule
│   └── osqp-python/          # submodule
├── osqpslack/                # small shared package (~400–600 lines total)
│   ├── penalties.py          # the 5 penalties: OSQP setup args + explicit lifting
│   ├── random_qp.py          # random sparse QP generator (Sec. "Random Sparse QPs")
│   ├── energy_hub.py         # building energy management model (Sec. "Building …")
│   └── plotting.py           # shared matplotlib style (IEEE column widths, fonts, colors)
├── experiments/
│   ├── slack_projection.py   # Fig. 1 — illustration, no solver
│   ├── scaling_run.py        # solve native + lifted        → data/scaling.csv
│   ├── scaling_plot.py       # data/scaling.csv             → figures/scaling.{pdf,png}, scaling_times.tex
│   ├── energy_hub_run.py     # solve native + lifted        → data/energy_hub.csv, energy_hub.json
│   └── energy_hub_plot.py    # data/energy_hub.*            → figures/energy_hub.{pdf,png}
├── notebooks/
│   └── mpc_example.ipynb     # self-contained soft-constrained MPC tutorial
├── data/                     # committed, compact (target: < 2 MB total)
└── figures/                  # committed: PDFs/.tex for the paper, PNGs for the README
```

Each experiment is split into a **run** script (slow, machine-dependent timings)
and a **plot** script (fast, deterministic). Since the data is committed, anyone
can regenerate the paper figures without re-running benchmarks.

## 3. Code content

### `osqpslack/penalties.py`
One small table that covers the paper's appendix. For each penalty
(quadratic, ℓ1, elastic net, Huber, ℓ∞):
- **native**: the arguments for `solver.setup_penalty(...)`;
- **lifted**: `lift(P, q, A, l, u, soft_rows, penalty) -> (P', q', A', l', u')`,
  i.e. the standard QP reformulation from the appendix's lifting table, plus
  the index slice to recover `x` from the lifted solution.

This replaces today's `penalties.py`, `penalty_families.py`, `formulations.py`,
`osqp_adapter.py`, `utilities.py` (≈1.3k lines). Plain functions and
`scipy.sparse` only: no dataclass hierarchy, no `SolutionMap` abstraction.

### `osqpslack/random_qp.py`
One function, `random_qp(n, seed) -> P, q, A, l, u`, implementing exactly the
generator described in the paper (Q = LᵀL + 0.1I with 8 % density, A with 5 % density
and ≥1 entry per row, anchor/target points, band bounds).

### `osqpslack/energy_hub.py`
`build_energy_hub() -> (P, q, A, l, u, penalty spec per row group, index info)`
for the 2-zone building + battery + grid problem (1 day, 15 min). It also has
helpers to extract trajectories from `x` for plotting. This is a cleaned-up
version of today's 764-line `energy_hub.py`, with the paper parameters
hardcoded as module constants.

### Experiments
- **scaling**: n ∈ {25, 50, 100, 200, 400, 800} × 5 penalties × 100 seeds,
  all rows soft. For each instance: solve native and lifted with identical
  settings (eps_abs = eps_rel = 1e-6, adaptive ρ, ρ₀ = 0.1). Record setup/solve
  times, iterations, and the time breakdown (linear system / projection /
  rest). A sanity check asserts that native and lifted objectives agree; this
  replaces the separate "equivalence" experiment.
  Single-threaded env vars (`OMP_NUM_THREADS=1`, …) are set inside the script.
- **energy_hub**: one native solve (trajectories for the plot) and a timing
  comparison of native and lifted (the 21 ms vs 41 ms numbers in the text).
- **slack_projection**: moved from the paper's `figures/plot_slack_projection.py`.

### `notebooks/mpc_example.ipynb` (new)
A self-contained tutorial that depends only on `osqp` (the fork), numpy, scipy
and matplotlib, and does **not** import `osqpslack`. Sketch:
1. A small linear system, e.g. a double integrator or oscillating masses, with state
   constraints that become infeasible under a disturbance.
2. Hard-constrained MPC → infeasible / fails.
3. Same MPC with `setup_penalty` (quadratic vs ℓ1 vs Huber) → closed-loop
   simulation, plots of violation patterns.
4. Show the online workflow: `update(q=…, l=…, u=…)`, warm starting,
   `update_penalty_params`, recovering slacks with `solver.slack(x)`.

Committed **with outputs** so it renders on GitHub.

### Dropped from the old codebase
Equivalence, MPC, building_control and lane_change experiments; `mpc.py`,
`benchmarking.py`, `metrics.py`, `results.py` (dataset writer/loader), the
`design/` docs, and the `scaling_30seeds_*` variants.

## 4. Data and figures (committed)

| File | Content | Approx. size |
|---|---|---|
| `data/scaling.csv` | one row per (n, penalty, seed, formulation): times, iterations, breakdown | ~6k rows, < 1 MB |
| `data/energy_hub.csv` | time series needed for the plot (96 steps) | < 50 kB |
| `data/energy_hub.json` | native/lifted solve times and iterations | < 1 kB |
| `figures/slack_projection.pdf`, `scaling.pdf`, `energy_hub.pdf`, `scaling_times.tex` | paper figures/table | < 1 MB |
| `figures/slack_projection.png`, `scaling.png`, `energy_hub.png` | same figures for the README (150–200 dpi) | < 1 MB |

Every plot script saves both formats from the same figure (one shared
`save(fig, name)` helper in `plotting.py`). The README embeds the PNGs, since
GitHub does not render PDFs inline.

The new code reruns the benchmarks (on the paper's machine) to produce the
committed data. The paper's numbers (speedups, 21 vs 41 ms, the timing table)
are then updated to match. (Today's raw scaling data is
15 MB in a different format and is not carried over.)

## 5. Build and reproduce

```bash
git clone --recursive https://github.com/A2R-Lab/OSQPSlack.git
cd OSQPSlack
python -m venv .venv && source .venv/bin/activate
make install   # pip install; builds the forked osqp from external/ (needs CMake + C compiler)
make figures   # data/ → figures/          (seconds)
make data      # re-run benchmarks → data/  (slow; machine-dependent timings)
```

- `make install` wraps
  `SKBUILD_CMAKE_DEFINE="OSQP_SOURCE_DIR=$PWD/external/osqp" pip install -r requirements.txt`,
  so the bindings compile against the submodule's C code. The README shows
  this one-liner too, for users without `make`.
- Plain pip, no uv required (`uv pip install -r requirements.txt` works as well).
  `requirements.txt` pins exact versions of the direct dependencies; that is
  enough reproducibility for this small set.
- `requirements.txt` lists the fork as `./external/osqp-python` (relative paths
  are allowed there, unlike in `pyproject.toml` dependencies). It installs under
  the normal name `osqp`, inside the venv only.
- Builtin algebra backend only (no MKL/CUDA).

## 6. Implementation order

1. Scaffold OSQPSlack: submodules, `requirements.txt`, `pyproject.toml`, `Makefile`, and verify that
   `make install` builds.
2. `penalties.py` + `random_qp.py` → scaling run/plot; compare with the paper.
3. `energy_hub.py` → energy hub run/plot; compare with the paper.
4. Move the slack projection figure.
5. MPC notebook.
6. README.
