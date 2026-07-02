# AI Implementation Context

This file is for AI agents or developers planning implementation work in this
directory. It summarizes the project shape, core APIs, benchmark semantics,
known results, and pitfalls that matter when extending Lambert-based
optimization or transfer-table code.

## Scope And Source Of Truth

This file assumes the project root is the directory containing
`benchmark_numba_lambert.py`. All paths in this document are relative to that
project root.

It is a standalone Numba-based companion project for pykep Lambert workloads.
An AI working on this project should need only:

- the files in this project
- the public pykep 2.6 implementation of `lambert_problem`
- the public package APIs listed in the dependency section

Do not assume access to files outside this project root.

The code implements a Numba port following the numerical structure of pykep
2.6's Izzo Lambert solver, then uses it in Keplerian TSP-style transfer
benchmarks bundled with this project. The table benchmark is self-contained,
runs in-memory, and excludes `.npz` or other disk serialization from timed
measurements.

## Dependencies

The benchmark expects:

- Python 3.12
- `numpy`
- `numba`
- `pykep` 2.6
- `scipy`
- `fcmaes`
- local bundled KTTSP files in `problems/`

The code has been checked on an AMD 9950X 16-core CPU. Thread counts in the
benchmarks are hardware-specific measurements, not universal tuning advice.

## File Map

- `pk_lambert.py`
  Numba implementation of pykep 2.6's Izzo Lambert structure plus Keplerian
  propagation and fused objective kernels.

- `numba_transfer.py`
  Python wrapper around `pk_lambert.py`. Provides a `NumbaTransfer` class with
  scalar, batch, and fixed-leg objective APIs.

- `utils_users.py`
  Local KTTSP problem loader and pykep reference implementation. Parses bundled
  `.kttsp` files, builds pykep planets, classifies inner/outer/switch transfer
  types, and provides `TomatoProblem.compute_transfer()`.

- `benchmark_numba_lambert.py`
  Main benchmark driver. Covers Lambert-only calls, full `compute_transfer`,
  Numba table creation, optional original-pykep table creation, CMA ask/tell,
  and BiteOpt.

- `README.md`
  Short human-facing overview and headline result.

- `NUMBA_LAMBERT_BENCHMARKS.md`
  Detailed derivation and benchmark report.

- `results/benchmark_latest.json`
  Default no-argument benchmark output.

- `results/benchmark_32768.json`
  Copy of the latest default 32768-sample run.

- `results/benchmark_table_pykep.json`
  Optional slow pykep table-creation comparison.

## Problem Semantics

Bundled KTTSP headers in `problems/*.kttsp` use this format:

```text
p kttsp t0[d] min_tof[d] max_total_time[d] dv_threshold[m/s] dv_exception[m/s] n_exceptions
```

Important details:

- `min_tof` is in days, not seconds.
- Public Python APIs take times in days.
- Low-level Numba kernels take times in seconds.
- The conversion constant is `DAY = 86400.0`.
- `TomatoProblem.max_revs` can be scalar or a 3-vector.
- Transfer types are:
  - `0`: inner -> inner
  - `1`: outer -> outer
  - `2`: cluster switch
- With scalar `max_revs`, switch transfers add one revolution. With vector
  `max_revs`, the value is indexed by transfer type.
- Default benchmark `max_revs` is `[5, 2, 0]`.

For pykep 2.6 `planet.keplerian`, the six orbital elements are
`a,e,i,W,w,M`; the sixth value is mean anomaly. This is distinct from lower
level element-to-Cartesian routines such as `par2ic`, which operate on
eccentric anomaly after pykep converts `M` with `m2e()`. The Numba path mirrors
that split: it stores `M0`, advances it as `M0 + n * dt`, solves Kepler's
equation to obtain eccentric anomaly `E`, and then forms the Cartesian state.

The local code classifies inner/outer clusters by the largest gap in
semi-major axis. Do not remove `transfer_type()`, `is_inner()`, `is_outer()`,
or `is_cluster_switch()`; `NumbaTransfer` depends on them.

## Core APIs

Use `TomatoProblem` as the reference oracle:

```python
problem = TomatoProblem("problems/easy.kttsp", max_revs=[5, 2, 0])
dv = problem.compute_transfer(i_from, i_to, t_start_days, tof_days)
```

Use `NumbaTransfer` for accelerated scalar and batch transfer evaluation:

```python
transfer = NumbaTransfer(problem)

dv = transfer.compute_transfer(i_from, i_to, t_start_days, tof_days)

dvs = transfer.compute_transfer_batch([
    (i_from, i_to, t_start_days, tof_days),
    ...
])
```

Low-level kernels in `pk_lambert.py`:

- `lambert_dv(r1, v_dep, r2, v_arr, tof_sec, mu, multi_revs)`
  Best two-impulse delta-v over both Lambert branches for already propagated
  states.

- `lambert_dv_batch(r1, v_dep, r2, v_arr, tof_sec, mu, multi_revs, out)`
  Parallel batch version for already propagated states.

- `dv_full(...)`
  Scalar propagation plus Lambert solve for one transfer.

- `dv_full_batch(A, E, I, O, Oc, M0, NN, mu, i_idx, j_idx, td_sec, tof_sec,
  max_revs_pair, out)`
  Parallel batch full transfer evaluation.

- `fixed_leg_objective_batch(...)`
  Fused CMA population objective for one fixed leg.

- `fixed_leg_objective_one(...)`
  Fused scalar objective for scalar optimizers like BiteOpt.

Use `numba.set_num_threads()` through the helper in the benchmark when timing.
Do not assume all machines benefit from the same thread count.

## Optimization Patterns

The important performance lesson is not "replace only Lambert". The largest
gain appears when the whole hot path is compiled or batched:

- ephemeris propagation
- Lambert solve
- delta-v reduction
- constraint penalty calculation
- objective-array fill
- population or table-cell evaluation

For fcmaes CMA-ES ask/tell:

- Use population size 32 for the included two-variable example.
- Use `NumbaTransfer.fixed_leg_objective_batch()` for the Numba backend.
- On the AMD 9950X, 8 Numba threads were a good default for the small CMA
  population. Larger populations or other CPUs may differ.

For BiteOpt:

- Use `_optimize_leg()` in `benchmark_numba_lambert.py`.
- It uses `fcmaes.bitecpp.minimize()`.
- The Numba backend calls `fixed_leg_objective_one()`.
- BiteOpt avoids ask/tell overhead but gives up population batching.

The objective form in the benchmark is:

```text
arrival + penalty_linear * max(0, dv - dv_limit)
        + penalty_quad * max(0, dv - dv_limit)^2
```

Default fixed-leg optimization settings:

- Leg: `0 -> 1`
- `max_wait = 12 d`
- `max_tof = 12 d`
- `dv_limit = 600 m/s`
- `optimizer_evals = 4096`
- `opt_popsize = 32`

## Transfer Table Creation

The table benchmark is a self-contained Keplerian transfer-table workload:

1. Build all ordered pairs `i != j`.
2. Optionally filter by cluster: `all`, `inner`, `outer`, or `switch`.
3. Apply an analytic plane-change lower-bound filter using transfer-type DV
   limits.
4. Build per-transfer-type departure and TOF grids.
5. Evaluate only cells where `departure + tof <= target_max_time`.
6. Run Lambert transfer evaluation.
7. Keep cells with `dv <= dv_limit[transfer_type]`.
8. Store either all kept cells or the best TOF per departure.

Default table settings:

```text
cluster: all
target_max_time: 200 d
min_tof: [0.05, 0.2, 0.35]
max_tof: [1.5, 5.0, 0.65]
tof_step: [0.05, 0.2, 0.1]
t_start_step: [0.05, 0.2, 0.1]
dv_limit: [105, 105, 605] m/s
store_mode: best-per-departure
table_workers: 16
table_batch_chunk: 1000000
```

For `problems/easy.kttsp`, these settings produce:

```text
filtered ordered pairs: 474
pair types: [36 inner, 174 outer, 264 switch]
departure grid sizes: [3999, 999, 1997]
TOF grid sizes: [30, 25, 4]
valid table cells evaluated: 10709178
nonempty pairs: 301
retained cells: 175150
```

The Numba table benchmark uses one process and 16 Numba threads. The original
pykep comparison uses a warmed 16-process multiprocessing pool. Avoid mixing a
multiprocessing pool with Numba parallel kernels unless you explicitly limit
threads per process; otherwise oversubscription can dominate timings.

The table benchmark intentionally uses flattened valid cells with
`dv_full_batch()` instead of the rectangular `grid_dv()` helper, because this
benchmark skips cells that exceed the mission horizon.

## Benchmark Commands

Default benchmark:

```bash
python benchmark_numba_lambert.py
```

Default behavior:

- Uses bundled `problems/easy.kttsp`.
- Runs 32768 random Lambert/transfer samples.
- Uses 8 workers/Numba threads for Lambert and `compute_transfer` batch tests.
- Uses CMA population size 32.
- Runs the Numba table-creation benchmark with 16 Numba threads.
- Writes `results/benchmark_latest.json`.
- Does not run the slow pykep table-creation comparison.

Optional pykep table comparison:

```bash
python benchmark_numba_lambert.py \
  --include-pykep-table \
  --json-out results/benchmark_table_pykep.json
```

Fast smoke for table-code changes:

```bash
python benchmark_numba_lambert.py \
  --samples 128 \
  --optimizer-evals 128 \
  --table-target-max-time 2 \
  --json-out results/smoke_table.json
```

Fast smoke including pykep table comparison:

```bash
python benchmark_numba_lambert.py \
  --samples 128 \
  --optimizer-evals 128 \
  --table-target-max-time 2 \
  --include-pykep-table \
  --json-out results/smoke_pykep_table.json
```

Always run at least:

```bash
python -m py_compile *.py
python -m json.tool results/benchmark_latest.json >/dev/null
```

## Current Headline Results

On AMD 9950X, 16-core CPU:

Same table workload, same number of CPU threads/processes, disk I/O excluded:

| implementation | CPU threads/processes | table cells | seconds | cells/s | retained cells |
|---|---:|---:|---:|---:|---:|
| original pykep | 16 processes | 10709178 | 24.998271 | 428396.8 | 175150 |
| Numba port | 16 threads | 10709178 | 1.941224 | 5516713.8 | 175150 |

Speedup: `12.88x`. Both paths retained exactly the same cells over the same
301 nonempty ordered pairs.

Default Numba-only table run:

```text
numba table batch 16: 10709178 cells, 1.873333 s, 5716645.0 cells/s
```

The difference between `1.94 s` and `1.87 s` is normal run-to-run variance; use
the pykep comparison result when quoting the direct pykep-vs-Numba table
speedup.

Other default benchmark rows from `benchmark_latest.json`:

```text
Lambert API:
  pykep serial: 39027.9 calls/s
  pykep batch 8: 179433.9 calls/s
  numba serial: 523617.4 calls/s
  numba batch 8: 4417722.3 calls/s

compute_transfer:
  pykep serial: 38734.8 calls/s
  pykep batch 8: 271346.4 calls/s
  numba serial: 308533.9 calls/s
  numba batch 8: 2034060.6 calls/s

optimizer:
  CMA pykep: 67151.7 eval/s
  CMA numba: 387845.7 eval/s
  BiteOpt pykep: 67574.4 eval/s
  BiteOpt numba: 284578.8 eval/s
```

## Extending The Code

When adding a new optimizer benchmark:

- Decide whether the optimizer is batch-oriented or scalar.
- For batch ask/tell optimizers, prefer a fused Numba batch objective.
- For scalar optimizers, add a scalar Numba objective like
  `fixed_leg_objective_one()`.
- Record the same best feasible information for pykep and Numba.
- Keep pykep as the reference oracle.
- Avoid adding slow optional benchmarks to the no-argument default.

When adding a new table benchmark:

- Reuse the table pair/filter/grid semantics above unless intentionally testing
  a different table family.
- Record pair count, pair type counts, evaluated cells, retained cells,
  retained-by-type counts, nonempty pairs, and thread/process count.
- Exclude disk I/O unless the benchmark is explicitly about serialization.
- For pykep comparisons, keep them opt-in and document that they are slow.
- If comparing pykep and Numba, assert or inspect that evaluated and retained
  cell counts match.

When adding a new Numba kernel:

- Keep arrays contiguous where practical.
- Convert days to seconds at the wrapper boundary.
- Avoid Python objects inside `@njit` kernels.
- Warm the kernel before timed measurements.
- Use `None` in JSON for unavailable values; do not serialize `NaN` in JSON.
- Keep a pykep numerical validation path.

## Common Pitfalls

- Do not assume current working directory. Defaults are script-relative.
- Do not treat `min_tof` as seconds. It is days in the bundled KTTSP files.
- Do not make `--include-pykep-table` part of the default run.
- Do not compare table timings if one path writes `.npz` and the other does
  not.
- Do not use a pykep multiprocessing pool and Numba parallel threads together
  without controlling oversubscription.
- Do not change transfer-type max-revolution logic without checking both
  `TomatoProblem.compute_transfer()` and `NumbaTransfer._max_revs()`.
- Do not trust a single very small batch timing; Numba thread scheduling
  overhead dominates tiny batches.
- Do not assume 8 or 16 threads are optimal on another CPU.

## Recommended Verification After Changes

Minimum:

```bash
python -m py_compile *.py
python benchmark_numba_lambert.py \
  --samples 128 \
  --optimizer-evals 128 \
  --table-target-max-time 2 \
  --json-out results/smoke_table.json
python -m json.tool results/smoke_table.json >/dev/null
```

For table comparison changes:

```bash
python benchmark_numba_lambert.py \
  --samples 128 \
  --optimizer-evals 128 \
  --table-target-max-time 2 \
  --include-pykep-table \
  --json-out results/smoke_pykep_table.json
python -m json.tool results/smoke_pykep_table.json >/dev/null
```

For final documented numbers:

```bash
python benchmark_numba_lambert.py
cp results/benchmark_latest.json results/benchmark_32768.json

python benchmark_numba_lambert.py \
  --include-pykep-table \
  --json-out results/benchmark_table_pykep.json
```

Update `README.md` and `NUMBA_LAMBERT_BENCHMARKS.md` whenever published
benchmark numbers change.
