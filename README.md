# pykep Lambert Python, Numba, and Rust Benchmark

This directory contains a Numba port of pykep 2.6's Izzo Lambert solver plus
tooling to compare it against the original pykep implementation, the published
`pykep-rust` Python wheel, and a standalone native Rust consumer.

See the related post [Lunar Tomato War](https://www.linkedin.com/pulse/algorithms-ai-lunar-tomato-war-final-debrief-from-esas-wolz-qbxle).

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
release-mode runs, median fixed-leg objective throughput was `484,279 eval/s`;
native CMA-ES and BiteOpt reached `1,241,658` and `729,254 eval/s`,
respectively.

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
- `rust/`: standalone native benchmark using crates.io `pykep-core 0.1.1` and
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
