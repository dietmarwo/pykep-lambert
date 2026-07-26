# pykep Lambert Python, Numba, and Rust Benchmark

This directory contains a Numba port of pykep 2.6's Izzo Lambert solver plus
tooling to compare it against the original pykep implementation, the published
`pykep-rust` Python wheel, and a standalone native Rust consumer.

See the related post [Lunar Tomato War](https://www.linkedin.com/pulse/algorithms-ai-lunar-tomato-war-final-debrief-from-esas-wolz-qbxle).
Experience building the transfer table during the SpOC 4 competition motivated
the parallel batch extension developed in
[dietmarwo/pykep-lambert](https://github.com/dietmarwo/pykep-lambert).

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

For 32,768 serial full-transfer calls on the same machine, the published
`pykep-rust 0.1.1` wheel is `2.74x` faster than original pykep while preserving
the Python API boundary:

| implementation | samples | seconds | calls/s | speedup |
|---|---:|---:|---:|---:|
| original pykep serial | 32,768 | 0.839684 | 39,024.2 | 1.00x |
| `pykep-rust` wheel serial | 32,768 | 0.306110 | 107,046.6 | 2.74x |
| Numba serial | 32,768 | 0.105633 | 310,205.4 | 7.95x |
| Numba batch 8 | 32,768 | 0.016178 | 2,025,427.4 | 51.90x |

The native `rust/` consumer uses the crates.io releases directly. Across five
release-mode runs, median serial and eight-worker batch full-transfer
throughput reached `485,426` and `2,875,181 eval/s`; native CMA-ES and BiteOpt
reached `1,771,331` and `724,012 eval/s`, respectively.

The Rust consumer now deliberately extends the original scalar pykep model
with ordered native batches. With 32,768 transfers and eight worker threads,
five release-mode runs produced these medians on the same benchmark host:

| Rust stage | serial items/s | batch items/s | batch speedup |
|---|---:|---:|---:|
| Keplerian propagation | 7,278,981 | 54,381,577 | 7.47x |
| Lambert evaluation | 547,264 | 4,061,278 | 7.42x |
| full propagation + Lambert + objective | 485,426 | 2,875,181 | 5.92x |

These are ordered, numerically identical batches. The full path first
propagates every departure and arrival state in parallel and then evaluates
every valid Lambert transfer in parallel. This is a project-level extension
built on the published scalar `pykep-core 0.1.1` APIs, not a claim that the
original Python pykep API provides native batching.

## Where Native Batching Also Makes Sense

The same pattern is useful whenever one call supplies enough independent work
to amortize scheduling and allocation:

- KTTSP transfer-table generation over all body pairs, departure epochs, and
  times of flight;
- pork-chop plots and launch-window scans;
- propagation of one body at many epochs or many bodies at shared epochs;
- Lambert sweeps over time of flight, direction, and revolution count;
- complete CMA-ES, DE, PGPE, and MODE populations;
- Monte Carlo trajectory ensembles and robust/worst-case objective scenarios;
- transfer encodings, flyby constraints/Jacobians, and MIMA estimates; and
- low-thrust leg mismatch/constraint evaluation across independent legs or
  candidates.

This experiment motivated the generic ordered batch layer prepared for
`pykep-core 0.1.2`. The benchmark remains pinned to published
`pykep-core 0.1.1` so its recorded measurements are reproducible. Scalar APIs
remain useful: tiny batches can be slower because thread scheduling and
intermediate arrays dominate.

## Files

- `pk_lambert.py`: scalar and batched Numba kernels for Lambert-only and full
  KTTSP transfer evaluation.
- `numba_transfer.py`: local drop-in `compute_transfer` wrapper with scalar and
  batched APIs.
- `pykep_rust_transfer.py`: scalar transfer backend using the published
  `pykep-rust` wheel.
- `benchmark_numba_lambert.py`: solver, transfer, table-creation, and fcmaes
  optimizer benchmarks for pykep, `pykep-rust`, and Numba.
- `test_pykep_rust_transfer.py`: installed-wheel integration and parity test.
- `rust/`: standalone native benchmark and ordered parallel propagation,
  Lambert, and full-transfer batch APIs using crates.io `pykep-core 0.1.1` and
  `fcmaes-core 0.1.3`, with no path dependencies.
- `NUMBA_LAMBERT_BENCHMARKS.md`: derivation notes, benchmark numbers, and
  interpretation.
- `results/benchmark_latest.json`: latest machine-readable benchmark output.

## Reproduce

```bash
python -m pip install "pykep-rust==0.1.1"
python -m pytest -q test_pykep_rust_transfer.py
python benchmark_numba_lambert.py
```

By default, the script uses the bundled `problems/easy.kttsp`, runs 32768
samples, uses 8 workers/Numba threads, keeps the CMA-ES population size at 32,
adds a 16-thread Numba transfer-table creation benchmark, and writes
`results/benchmark_latest.json`. The script warms Numba kernels before timing.

Build and run the independent crates.io consumer:

```bash
cd rust
cargo test --locked
cargo run --release --locked
```

The original pykep table-creation comparison is intentionally opt-in because it
is much slower:

```bash
python benchmark_numba_lambert.py \
  --include-pykep-table \
  --json-out results/benchmark_table_pykep.json
```
