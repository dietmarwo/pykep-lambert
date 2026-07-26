# Native Rust benchmark

This standalone crate ports the fixed-leg `easy.kttsp` optimization workload
to native Rust. It deliberately uses only the published registry packages:

- `pykep-core = 0.1.1` for Keplerian propagation and Lambert solutions;
- `fcmaes-core = 0.1.3` for CMA-ES and BiteOpt.

No path or Git dependencies point at either development repository. A normal
build therefore verifies that the crates.io packages are usable by a separate
project.

```bash
cd rust
cargo test --locked
cargo run --release --locked
```

The benchmark evaluates the fixed `0 -> 1` leg with the same wait/time-of-flight
bounds, delta-v penalty, optimizer budget, CMA population, and seed used by the
Python optimizer example. Use `cargo run --release --locked -- --help` for
shorter diagnostic runs and optimizer selection.

The raw objective result is a native scalar throughput measurement. The
optimizer rows are useful comparisons with the Python benchmark when both are
run on the same machine, but they are not transfer-table measurements.

## Measured result

On the AMD 9950X benchmark host, five warmed release runs gave these median
internal timings:

| path | evaluations | seconds | evaluations/s |
|---|---:|---:|---:|
| fixed-leg objective | 32,768 | 0.067663 | 484,279 |
| native CMA-ES | 3,808 | 0.003067 | 1,241,658 |
| native BiteOpt | 4,096 | 0.005617 | 729,254 |

Both optimizers returned the same feasible boundary solution as the Python
comparison: arrival `0.969180817124 d` at `600.000000000 m/s`.
