# pykep Lambert Numba Benchmark

This directory contains a Numba port of pykep 2.6's Izzo Lambert solver plus
small tooling to compare it against the original pykep implementation.
See the related post [Lunar Tomato War](https://www.linkedin.com/pulse/algorithms-ai-lunar-tomato-war-final-debrief-from-esas-wolz-qbxle) 

## Main Result

Using the same number of CPU threads to create a Lambert transfer table for the
Keplerian TSP task of SpOC 4, the Numba implementation is substantially faster
than the original pykep path.

| implementation | CPU threads | table cells | seconds | cells/s | retained cells |
|---|---:|---:|---:|---:|---:|
| original pykep | 16 | 10,709,178 | 24.998271 | 428,396.8 | 175,150 |
| Numba port | 16 | 10,709,178 | 1.941224 | 5,516,713.8 | 175,150 |

The Numba table build is `12.88x` faster in this benchmark, with identical
retained-cell counts.

## Files

- `pk_lambert.py`: scalar and batched Numba kernels for Lambert-only and full
  KTTSP transfer evaluation.
- `numba_transfer.py`: local drop-in `compute_transfer` wrapper with scalar and
  batched APIs.
- `benchmark_numba_lambert.py`: solver, transfer, table-creation, and fcmaes
  optimizer benchmarks.
- `NUMBA_LAMBERT_BENCHMARKS.md`: derivation notes, benchmark numbers, and
  interpretation.
- `results/benchmark_latest.json`: latest machine-readable benchmark output.

## Reproduce

```bash
python benchmark_numba_lambert.py
```

By default, the script uses the bundled `problems/easy.kttsp`, runs 32768
samples, uses 8 workers/Numba threads, keeps the CMA-ES population size at 32,
adds a 16-thread Numba transfer-table creation benchmark, and writes
`results/benchmark_latest.json`. The script warms Numba kernels before timing.

The original pykep table-creation comparison is intentionally opt-in because it
is much slower:

```bash
python benchmark_numba_lambert.py \
  --include-pykep-table \
  --json-out results/benchmark_table_pykep.json
```
