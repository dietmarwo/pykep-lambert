# Numba Lambert Port and Benchmarks

Generated on 2026-07-01 from:

```bash
python benchmark_numba_lambert.py
```

Machine-readable results are in `results/benchmark_latest.json`; the same run
is also copied to `results/benchmark_32768.json`.

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

## Benchmark Setup

- Problem: `problems/easy.kttsp`
- pykep: `2.6`
- Python: `3.12.13`
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
| pykep serial | 32768 | 0.839604 | 39,027.9 | 1.00x |
| pykep batch 8 | 32768 | 0.182619 | 179,433.9 | 4.60x |
| numba serial | 32768 | 0.062580 | 523,617.4 | 13.42x |
| numba batch 8 | 32768 | 0.007417 | 4,417,722.3 | 113.19x |

Accuracy versus pykep serial:

- finite cases compared: `32768`
- max absolute delta-v difference: `4.18367562815547e-11 m/s`
- mean absolute delta-v difference: `5.126010293315941e-14 m/s`
- max relative difference: `1.7903088754118263e-14`

## compute_transfer Results

These timings include KTTSP ephemeris propagation plus Lambert. This is the
drop-in replacement path for `TomatoProblem.compute_transfer`.

| mode | samples | seconds | calls/s | speedup vs pykep serial |
|---|---:|---:|---:|---:|
| pykep serial | 32768 | 0.845958 | 38,734.8 | 1.00x |
| pykep batch 8 | 32768 | 0.120761 | 271,346.4 | 7.01x |
| numba serial | 32768 | 0.106205 | 308,533.9 | 7.97x |
| numba batch 8 | 32768 | 0.016110 | 2,034,060.6 | 52.51x |

Accuracy versus pykep serial:

- finite cases compared: `32768`
- max absolute delta-v difference: `3.2755951906437986e-07 m/s`
- mean absolute delta-v difference: `1.7450147126981189e-10 m/s`
- max relative difference: `1.6662656465202954e-10`

The full `compute_transfer` path is less extreme than the Lambert API benchmark because
state propagation and Python wrapper work are part of the measurement. The
batched Numba path still gives roughly `7.49x` the throughput of warmed pykep
batch mode and roughly `52.51x` the throughput of pykep serial mode.

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
| numba table batch 16 | 474 | 10709178 | 1.873333 | 5,716,645.0 | 301 | 175150 |

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
two fcmaes optimizers:

- `cma_ask_tell`: `fcmaes.cmaescpp.ACMA_C` using ask/tell population batches.
- `bite_minimize`: `fcmaes.bitecpp.minimize`, a scalar `fun(x)` API with no
  ask/tell loop.

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
| cma_ask_tell | pykep | 3812 | 0.056767 | 67,151.7 | True | 0.969180817 | 600.000000 | 0.000000000 | 0.969180817 |
| cma_ask_tell | numba | 3876 | 0.009994 | 387,845.7 | True | 0.969180817 | 600.000000 | 0.000000000 | 0.969180817 |
| bite_minimize | pykep | 4096 | 0.060615 | 67,574.4 | True | 1.003516881 | 600.000000 | 0.459965771 | 0.543551110 |
| bite_minimize | numba | 4096 | 0.014393 | 284,578.8 | True | 1.003516881 | 600.000000 | 0.459961883 | 0.543554998 |

The direct BiteOpt path shows the cost trade-off between optimizer overhead and
batching. With this small two-variable example, pykep BiteOpt and pykep CMA
ask/tell have similar throughput, while Numba BiteOpt is `4.21x` faster than
pykep BiteOpt. For the Numba backend, CMA ask/tell remains faster overall
because it preserves population batching; with population size `32` and 8
Numba threads it is `5.78x` faster than pykep CMA ask/tell and `1.36x` faster
than Numba BiteOpt.

## Notes

- The benchmark intentionally separates the Lambert API layer from full
  `compute_transfer` because they answer different questions: solver throughput
  versus application-level throughput.
- The pykep batch numbers are not a straw man; they use a pre-warmed process
  pool. Numba still wins because it avoids Python object construction inside
  the hot loop and evaluates the population inside compiled code.
- For very small batches, Numba batch mode can lose to scalar Numba due to
  scheduling overhead. The advantage appears when the batch is large enough to
  amortize that overhead, as in transfer-table construction, beam search, and
  CMA population evaluation.
- Final contest submissions should still be validated against the pykep
  reference path. The observed numerical differences here are far below the
  KTTSP delta-v thresholds, but pykep remains the external reference.
