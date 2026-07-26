# Native Rust benchmark

This standalone crate ports the fixed-leg `easy.kttsp` optimization workload
to native Rust. It deliberately uses only the published registry packages:

- `pykep-core = 0.1.1` for Keplerian propagation and Lambert solutions;
- `fcmaes-core = 0.1.3` for CMA-ES and BiteOpt.

No path or Git dependencies point at either development repository. A normal
build therefore verifies that the crates.io packages are usable by a separate
project.

The original pykep-facing APIs are scalar. Experience creating the Lambert
transfer table during the SpOC 4 competition motivated a deliberate native
batch extension in
[dietmarwo/pykep-lambert](https://github.com/dietmarwo/pykep-lambert):

- `FixedLegProblem::propagate_batch` evaluates ordered body/epoch requests;
- `lambert_delta_v_batch` evaluates ordered pre-propagated transfers;
- `FixedLegProblem::evaluate_batch` performs all propagations as one batch,
  then all valid Lambert computations as a second batch; and
- `FixedLegProblem::objective_batch` returns optimizer-ready scalar values.

These project-level helpers use the cached Rayon executor published by
`fcmaes-core`. They consume the scalar `pykep-core 0.1.1` API and therefore do
not imply that original Python pykep or that published crate version already
has equivalent batch entry points. Output order always matches input order;
invalid rows remain in place as `None` or the optimizer's finite replacement.

```bash
cd rust
cargo test --locked
cargo run --release --locked
```

The benchmark evaluates the fixed `0 -> 1` leg with the same wait/time-of-flight
bounds, delta-v penalty, optimizer budget, CMA population, and seed used by the
Python optimizer example. Use `cargo run --release --locked -- --help` for
shorter diagnostic runs and optimizer selection.

The benchmark reports serial and parallel propagation, Lambert-only, and
complete-transfer throughput. The optimizer rows are useful comparisons with
the Python benchmark when both are run on the same machine, but they are not
transfer-table measurements.

## Measured result

On the AMD 9950X benchmark host, five warmed release runs gave these median
internal timings:

| path | evaluations | seconds | evaluations/s |
|---|---:|---:|---:|
| propagation serial | 65,536 | 0.009003 | 7,278,981 |
| propagation batch, 8 workers | 65,536 | 0.001205 | 54,381,577 |
| Lambert serial | 32,768 | 0.059876 | 547,264 |
| Lambert batch, 8 workers | 32,768 | 0.008068 | 4,061,278 |
| full transfer serial | 32,768 | 0.067504 | 485,426 |
| full transfer batch, 8 workers | 32,768 | 0.011397 | 2,875,181 |
| native CMA-ES | 3,808 | 0.002150 | 1,771,331 |
| native BiteOpt | 4,096 | 0.005657 | 724,012 |

The batch speedups are 7.47x for propagation, 7.42x for Lambert, and 5.92x
for the complete staged transfer. The benchmark asserts exact serial/batch
equality before reporting each result.

Both optimizers returned the same feasible boundary solution as the Python
comparison: arrival `0.969180817124 d` at `600.000000000 m/s`.

## Further batch candidates

High-value candidates for a future public `pykep-core` batch layer are:

- general ephemeris requests spanning arbitrary providers, bodies, and epochs;
- Lambert grids spanning geometry, time of flight, direction, and revolutions;
- fused body-pair/departure/TOF transfer tables and pork-chop plots;
- optimizer populations and robust Monte Carlo scenario ensembles;
- transfer encodings, flyby constraints/Jacobians, and MIMA estimates; and
- independent low-thrust leg propagation and mismatch constraints.

The generic batch layer prepared for `pykep-core 0.1.2` exposes these patterns
to Python with NumPy arrays and one GIL release per native batch. Keep scalar
entry points for latency-sensitive calls and small batches where parallel
scheduling costs more than it saves.
