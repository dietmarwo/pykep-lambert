# Native Rust benchmark

This standalone crate ports the fixed-leg `easy.kttsp` optimization workload
to native Rust. It deliberately uses only published registry packages:

- `pykep-core = "=0.1.2"` for Keplerian propagation and Lambert solutions;
- `fcmaes-core = "=0.1.3"` for CMA-ES and BiteOpt.

No path or Git dependencies point at either development repository. A normal
build therefore verifies that the crates.io packages are usable by a separate
project.

The benchmark composes the batch APIs added in `pykep-core 0.1.2`:

- `FixedLegProblem::propagate_batch` groups requests by ephemeris and calls
  `Ephemeris::states_parallel`;
- `lambert_delta_v_batch` maps each transfer to clockwise and
  counter-clockwise `LambertRequest`s and calls `solve_lambert_batch`;
- `FixedLegProblem::evaluate_batch` performs the staged propagation and
  Lambert batches while retaining invalid rows in their original positions;
- `FixedLegProblem::objective_batch` returns optimizer-ready scalar values;
  and
- the `cma-batch` mode uses `Cmaes::ask()`, evaluates each complete population
  through `objective_batch`, and returns it with `Cmaes::tell()`.

Output order always matches input order. Invalid rows remain in place as
`None` or the optimizer's finite replacement. The project wrappers add
KTTSP-specific grouping, reduction over the two Lambert directions, and
objective construction; the numerical batch computations themselves now use
the published `pykep-core` APIs.

```bash
cd rust
cargo test --locked
cargo run --release --locked
```

The default `both` selection runs parallel-scalar CMA-ES, ask/tell batch
CMA-ES, and BiteOpt. An individual optimizer can be selected explicitly:

```bash
cargo run --release --locked -- --optimizer cma-batch
```

Use `cargo run --release --locked -- --help` for shorter diagnostic runs,
worker selection, and the other optimizer choices. The benchmark evaluates
the fixed `0 -> 1` leg with the same wait/time-of-flight bounds, delta-v
penalty, optimizer budget, CMA population, and seed used by the Python
optimizer example.

## Measured result

On the AMD 9950X benchmark host, five warmed release runs gave these median
internal timings:

| path | evaluations | seconds | evaluations/s |
|---|---:|---:|---:|
| propagation serial | 65,536 | 0.009868 | 6,641,114 |
| propagation pykep batch, 8 workers | 65,536 | 0.002310 | 28,374,014 |
| Lambert serial | 32,768 | 0.073927 | 443,248 |
| Lambert pykep batch, 8 workers | 32,768 | 0.016869 | 1,942,456 |
| full transfer serial | 32,768 | 0.068199 | 480,473 |
| full transfer, Rayon over scalar pykep | 32,768 | 0.009211 | 3,557,499 |
| full transfer, staged pykep batch | 32,768 | 0.015569 | 2,104,628 |
| CMA-ES, Rayon over scalar pykep | 3,808 | 0.002336 | 1,629,796 |
| CMA-ES, ask/tell pykep batch | 3,744 | 0.005018 | 746,127 |
| native BiteOpt | 4,096 | 0.005690 | 719,882 |

The pykep batch speedups are 4.27x for propagation, 4.38x for Lambert, and
4.38x for the complete staged transfer. Parallelizing complete scalar
objectives with Rayon reaches 7.40x over serial and is 1.69x faster than the
staged pykep batch on this workload. The executable asserts exact equality
among the serial, outer-Rayon, and staged-batch objective arrays before
reporting them.

All three optimizers returned the same feasible boundary solution: arrival
`0.969180817124 d` at `600.000000000 m/s`. The ask/tell implementation proves
that a complete Rust optimizer population can use the new pykep batches, but
it is not faster for this two-variable, population-32 example. The Rayon
parallel-scalar CMA path reaches `1,629,796 eval/s`, versus `746,127 eval/s`
for ask/tell batching. Each generation performs two ephemeris group dispatches
and one Lambert dispatch, so the small population does not amortize the
scheduling and intermediate-vector overhead. Larger populations or more
expensive cases are the intended use case; keep the outer-Rayon mode for this
small, fully independent objective.

## Other existing 0.1.2 batch surfaces

`pykep-core 0.1.2` also exposes ordered batches for vector operations, anomaly
conversions, Lagrangian/universal/Keplerian propagation (including the
Lagrangian STM and time-grid variants), ephemeris accelerations/periods/orbital
elements, and ZOH low-thrust mismatch evaluation. Those APIs can be useful in
future benchmarks when the corresponding existing pykep computation is part
of a real workload. This project intentionally does not invent new eclipse,
access, or conjunction physics.

Scalar entry points remain preferable for latency-sensitive calls and small
batches where scheduling costs more than they save.
