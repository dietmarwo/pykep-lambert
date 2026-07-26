# pykep Lambert Python, Numba, and Rust Benchmarks

Python results generated on 2026-07-26 from:

```bash
python benchmark_numba_lambert.py
```

Machine-readable results are in `results/benchmark_latest.json`; the same run
is also copied to `results/benchmark_32768.json`.

The native Rust results are medians of five warmed release-mode runs from:

```bash
cd rust
cargo run --release --locked
```

The original pykep table-creation comparison is intentionally not part of the
default command because it is much slower. It was run separately with:

```bash
python benchmark_numba_lambert.py \
  --include-pykep-table \
  --json-out results/benchmark_table_pykep.json
```

## What Was Ported

`pk_lambert.py` is a Numba implementation of the same numerical structure used
by pykep 2.6's `lambert_problem` implementation:

- Izzo's Lambert parameterization is preserved: the non-dimensional time of
  flight transform, the Battin/Lagrange branches, Householder iteration, and
  the multi-revolution solution enumeration.
- Both clockwise senses are evaluated, matching the original KTTSP
  `compute_transfer` loop over `cw=False` and `cw=True`.
- The best total two-impulse delta-v is selected from all returned
  multi-revolution candidates, matching the pykep-facing code path.
- The Kepler propagation side was implemented in Numba using the same orbital
  element convention used by the KTTSP problem files and pykep
  `planet.keplerian`: `a,e,i,W,w,M`, where the sixth element is mean anomaly.
  Lower-level state conversion uses eccentric anomaly only after solving
  Kepler's equation, matching pykep 2.6's `M -> E -> Cartesian` path.
- `dv_full_batch` and `lambert_dv_batch` wrap the scalar kernels in `prange` so
  a CMA population, transfer table row, or benchmark batch can be evaluated in
  one threaded call.

The practical result is that the original pykep implementation remains the
reference oracle, while the Numba path is usable for high-volume search and can
be checked against pykep at the boundary.

The benchmark driver and table-creation workload are derived from real SpOC 4
competition code used by team fcmaes, not from a toy standalone example. The
measured paths therefore include the same kinds of objective, transfer-grid,
and sparse-table loops that appear in competition search workflows.
That SpOC 4 transfer-table experience motivated the deliberate batch extension
maintained in
[dietmarwo/pykep-lambert](https://github.com/dietmarwo/pykep-lambert).

Two published Rust deployment paths are included:

- `pykep_rust_transfer.py` exercises the `pykep-rust 0.1.1` PyPI wheel through
  its Python `Planet` and `LambertProblem` APIs.
- `rust/` is an independent Cargo project pinned to crates.io
  `pykep-core 0.1.2` and `fcmaes-core 0.1.3`. It has no path or Git
  dependencies on either development repository.

Some KTTSP input rows use non-canonical inclination representations accepted
by original pykep. The wheel backend therefore seeds its native Keplerian
providers once from original-pykep Cartesian states. This initialization is
outside every timed region; all timed propagation and Lambert solves use the
published wheel.

## Benchmark Setup

- Problem: `problems/easy.kttsp`
- pykep: `2.6`
- pykep-rust wheel: `0.1.1`
- pykep-core crate: `0.1.2`
- fcmaes-core crate: `0.1.3`
- Python: `3.12.13`
- Rust: release profile, Rust `1.88` minimum
- Hardware checked: AMD 9950X, 16-core CPU
- Lambert revolution setting: `[5, 2, 0]`
- Random transfer samples: `32768`
- Lambert and `compute_transfer` batch width: `8` workers/threads
- Transfer-table creation batch width: `16` Numba threads
- Seed: `12345`
- Optimizer evaluation budget: `4096`
- CMA-ES population size: `32`
- Numba compile time: excluded by warm-up
- pykep batch mode: warmed 8-process `multiprocessing` pool
- Numba batch mode: one `prange` kernel using 8 Numba threads

The selected thread counts were checked on that AMD 9950X machine. They should
be treated as benchmark settings, not universal tuning advice: the optimal
number of Numba threads can differ with CPU architecture, memory bandwidth,
core count, scheduler behavior, and the surrounding objective workload.

## Lambert API Results

These timings use already-propagated position/velocity states. They isolate the
pykep `lambert_problem` API plus Python-side min-DV reduction from KTTSP
ephemeris propagation.

Scope note: this is not a pure `kep3::lambert_problem` C++ arithmetic
benchmark. The pykep column still pays for Python object construction,
`get_v1()` / `get_v2()` materialization, and Python reduction over returned
branch velocities. The speedups should be read as Python-user workload
speedups for this API shape.

| mode | samples | seconds | calls/s | speedup vs pykep serial |
|---|---:|---:|---:|---:|
| pykep serial | 32768 | 0.834809 | 39,252.1 | 1.00x |
| pykep batch 8 | 32768 | 0.190363 | 172,134.4 | 4.39x |
| pykep-rust wheel serial | 32768 | 0.351271 | 93,284.2 | 2.38x |
| numba serial | 32768 | 0.062034 | 528,230.2 | 13.46x |
| numba batch 8 | 32768 | 0.007784 | 4,209,411.6 | 107.24x |

Accuracy versus pykep serial:

- finite cases compared: `32768`
- max absolute delta-v difference: `4.18367562815547e-11 m/s`
- mean absolute delta-v difference: `5.126010293315941e-14 m/s`
- max relative difference: `1.7903088754118263e-14`

Published `pykep-rust` wheel accuracy versus pykep serial:

- finite cases compared: `32768`
- max absolute delta-v difference: `6.230038707144558e-11 m/s`
- mean absolute delta-v difference: `2.3796586988249213e-13 m/s`
- max relative difference: `4.732783496926049e-14`

## compute_transfer Results

These timings include KTTSP ephemeris propagation plus Lambert. This is the
drop-in replacement path for `TomatoProblem.compute_transfer`.

| mode | samples | seconds | calls/s | speedup vs pykep serial |
|---|---:|---:|---:|---:|
| pykep serial | 32768 | 0.839684 | 39,024.2 | 1.00x |
| pykep batch 8 | 32768 | 0.112355 | 291,646.6 | 7.47x |
| pykep-rust wheel serial | 32768 | 0.306110 | 107,046.6 | 2.74x |
| numba serial | 32768 | 0.105633 | 310,205.4 | 7.95x |
| numba batch 8 | 32768 | 0.016178 | 2,025,427.4 | 51.90x |

Accuracy versus pykep serial:

- finite cases compared: `32768`
- max absolute delta-v difference: `3.2755951906437986e-07 m/s`
- mean absolute delta-v difference: `1.7450147126981189e-10 m/s`
- max relative difference: `1.6662656465202954e-10`

Published `pykep-rust` wheel accuracy versus pykep serial:

- finite cases compared: `32768`
- max absolute delta-v difference: `3.1397348720929585e-7 m/s`
- mean absolute delta-v difference: `4.4653163287712745e-10 m/s`
- max relative difference: `1.5170505910298076e-10`

The full `compute_transfer` path is less extreme than the Lambert API benchmark because
state propagation and Python wrapper work are part of the measurement. The
published wheel gives `2.74x` the scalar throughput of original pykep. The
batched Numba path gives roughly `6.94x` the throughput of warmed pykep batch
mode and `51.90x` the throughput of pykep serial mode.

## Lambert Table Creation Results

This benchmark mirrors the broad `trans_table.py` table-building pattern from
the KTTSP workspace: build ordered pairs with the same geometric plane-change
prefilter, form per-transfer-type departure and TOF grids, evaluate only cells
whose arrival stays inside the mission horizon, apply the final DV limits, and
store the best TOF per departure. It is an in-memory timing benchmark and does
not write an `.npz` table, so disk I/O is excluded.

Settings:

- Cluster: `all`
- Target horizon: `200 d`
- DV limits: `[105, 105, 605] m/s`
- Departure grid sizes: `[3999, 999, 1997]`
- TOF grid sizes: `[30, 25, 4]`
- Filtered ordered pairs: `474` (`36` inner, `174` outer, `264` switch)
- Valid table cells evaluated: `10709178`
- Storage mode: `best-per-departure`
- Numba threads: `16`

| mode | pairs | evaluated cells | seconds | cells/s | nonempty pairs | retained cells |
|---|---:|---:|---:|---:|---:|---:|
| numba table batch 16 | 474 | 10709178 | 1.873507 | 5,716,112.1 | 301 | 175150 |

Cells evaluated by transfer type: `4304304` inner, `4297626` outer, and
`2107248` switch. Retained cells by transfer type: `65861` inner, `71768`
outer, and `37521` switch.

The optional original-pykep comparison uses the same pair list, grids, mission
horizon filter, DV limits, and `best-per-departure` retention. It runs pykep in
a warmed 16-process pool and still excludes `.npz` disk I/O.

| mode | pairs | evaluated cells | seconds | cells/s | nonempty pairs | retained cells | speedup vs pykep |
|---|---:|---:|---:|---:|---:|---:|---:|
| pykep table batch 16 | 474 | 10709178 | 24.998271 | 428,396.8 | 301 | 175150 | 1.00x |
| numba table batch 16 | 474 | 10709178 | 1.941224 | 5,516,713.8 | 301 | 175150 | 12.88x |

Both paths retained exactly the same `175150` table cells over the same `301`
nonempty ordered pairs in this benchmark.

## Optimizer Examples

`benchmark_numba_lambert.py` also includes a small fixed-leg optimization with
two fcmaes optimizers and three Python-callable transfer backends:

- `cma_ask_tell`: `fcmaes.cmaescpp.ACMA_C` using ask/tell population batches.
- `bite_minimize`: `fcmaes.bitecpp.minimize`, a scalar `fun(x)` API with no
  ask/tell loop.
- Backends: original `pykep`, the published `pykep-rust` wheel, and the local
  Numba implementation.

The decision vector is:

```text
x = [wait_days, tof_days]
```

The objective is arrival time plus a delta-v penalty:

```text
arrival + 1000 * max(0, dv - dv_limit) + max(0, dv - dv_limit)^2
```

The demo optimizes leg `0 -> 1`, with `max_wait=12`, `max_tof=12`, and
`dv_limit=600 m/s`. The pykep backend evaluates scalar `compute_transfer`
calls. The Numba CMA backend evaluates each population with one compiled
fixed-leg objective kernel, including transfer computation, penalty
calculation, and objective-array fill. The Numba BiteOpt backend evaluates one
compiled scalar fixed-leg objective per BiteOpt call.

| optimizer | backend | evals | seconds | evals/s | feasible | arrival | dv | wait | tof |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| cma_ask_tell | pykep | 3972 | 0.055301 | 71,824.5 | True | 0.969180817 | 600.000000 | 0.000000000 | 0.969180817 |
| cma_ask_tell | pykep-rust wheel | 3972 | 0.023186 | 171,308.4 | True | 0.969180817 | 600.000000 | 0.000000000 | 0.969180817 |
| cma_ask_tell | numba | 3972 | 0.005821 | 682,390.4 | True | 0.969180817 | 600.000000 | 0.000000000 | 0.969180817 |
| bite_minimize | pykep | 4096 | 0.060279 | 67,950.9 | True | 0.969180817 | 600.000000 | 0.000000000 | 0.969180817 |
| bite_minimize | pykep-rust wheel | 4096 | 0.026642 | 153,740.8 | True | 0.969180817 | 600.000000 | 0.000000000 | 0.969180817 |
| bite_minimize | numba | 4096 | 0.012895 | 317,641.9 | True | 0.969180817 | 600.000000 | 0.000000000 | 0.969180817 |

The wheel backend is `2.38x` faster than original pykep in CMA ask/tell and
`2.26x` faster in BiteOpt. Numba remains faster inside Python because it fuses
population evaluation and avoids per-solution Python object materialization.

## Native Rust Batch And Optimizer Results

The `rust/` executable removes the Python boundary and uses `fcmaes-core`
directly. It uses the ordered batch APIs published in `pykep-core 0.1.2`:
ephemeris requests are grouped by provider and evaluated through
`Ephemeris::states_parallel`, while clockwise and counter-clockwise
`LambertRequest`s are evaluated through `solve_lambert_batch`. The local layer
then composes these into a staged full-transfer objective.

The following timings are medians of five warmed release-mode runs:

| path | evals | seconds | evals/s | feasible | arrival | dv |
|---|---:|---:|---:|---|---:|---:|
| propagation serial | 65536 | 0.009868 | 6,641,114 | n/a | n/a | n/a |
| propagation pykep batch, 8 workers | 65536 | 0.002310 | 28,374,014 | n/a | n/a | n/a |
| Lambert serial | 32768 | 0.073927 | 443,248 | n/a | n/a | n/a |
| Lambert pykep batch, 8 workers | 32768 | 0.016869 | 1,942,456 | n/a | n/a | n/a |
| full transfer serial | 32768 | 0.068199 | 480,473 | n/a | n/a | n/a |
| full transfer, Rayon over scalar pykep | 32768 | 0.009211 | 3,557,499 | n/a | n/a | n/a |
| full transfer, staged pykep batch | 32768 | 0.015569 | 2,104,628 | n/a | n/a | n/a |
| CMA-ES, Rayon over scalar pykep | 3808 | 0.002336 | 1,629,796 | True | 0.969180817 | 600.000000 |
| CMA-ES, ask/tell pykep batch | 3744 | 0.005018 | 746,127 | True | 0.969180817 | 600.000000 |
| native BiteOpt | 4096 | 0.005690 | 719,882 | True | 0.969180817 | 600.000000 |

The native pykep batch speedups are `4.27x` for propagation, `4.38x` for
Lambert, and `4.38x` for the complete staged transfer. The comparison path
uses fcmaes' Rayon executor around complete scalar objectives, keeping every
pykep call single-threaded; it reaches `7.40x` over serial and `1.69x` the
staged pykep-batch throughput. The executable verifies exact equality among
all three objective arrays before printing each measurement.

The native optimizer implementations are not byte-for-byte identical to the
Python fcmaes wrappers, so evaluation counts can stop at slightly different
points. They use the same two decision variables, bounds, 4,096 evaluation
budget, seed, CMA population, Lambert branches, and penalty. The new Rust
ask/tell mode calls `Cmaes::ask()`, sends the complete population through
`FixedLegProblem::objective_batch`, and calls `Cmaes::tell()`. All three native
optimizers reached the same feasible boundary solution.

For this population of only 32 candidates, ask/tell batching reaches
`746,127 eval/s`, below the outer-Rayon CMA path's `1,629,796 eval/s`.
A generation requires two ephemeris group dispatches and one Lambert dispatch,
so this small population cannot amortize scheduling and intermediate-vector
costs. Large transfer grids still gain 4–5x; larger populations and more
expensive objective cases are the intended optimizer-batch use cases.

Other existing pykep computations with published 0.1.2 batch entry points
include vector operations, anomaly conversions,
Lagrangian/universal/Keplerian propagation (including the Lagrangian STM and
time-grid variants), ephemeris accelerations/periods/orbital elements, and ZOH
low-thrust mismatch evaluation. This project does not add eclipse, access, or
conjunction physics not already exposed by pykep. Scalar APIs remain
preferable for tiny batches where scheduling overhead dominates.

## Notes

- The benchmark intentionally separates the Lambert API layer from full
  `compute_transfer` because they answer different questions: solver throughput
  versus application-level throughput.
- The PyPI and crates.io consumers pin exact published versions so a successful
  clean build demonstrates that the deployed artifacts, rather than local
  development paths, are usable.
- The pykep batch numbers are not a straw man; they use a pre-warmed process
  pool. Numba still wins because it avoids Python object construction inside
  the hot loop and evaluates the population inside compiled code.
- For very small batches, either Numba or Rust batch mode can lose to its
  scalar/parallel-scalar alternative due to scheduling overhead. The advantage
  appears when a batch is large enough to amortize that overhead, as in
  transfer-table construction and beam search. The measured Rust CMA
  population of 32 is a concrete counterexample, documented above.
- Final contest submissions should still be validated against the pykep
  reference path. The observed numerical differences here are far below the
  KTTSP delta-v thresholds, but pykep remains the external reference.
