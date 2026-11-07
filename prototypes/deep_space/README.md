# Deep-space autonomous swarm runtime (Rust research prototype)

A `no_std`, allocation-free, dependency-free execution core for testing
whether swarm optimization can drive long-duration spacecraft autonomy
safely. The mission briefing and design rationale are in
[`docs/deep_space_mission.md`](../../docs/deep_space_mission.md).

> Research prototype. Not flight software, not flight-qualified, and the tests
> here are not qualification tests.

## Layout

| File | Role |
|---|---|
| `src/baseline.rs` | Deterministic value-per-energy observation picker (not PSO) |
| `src/pso.rs` | Bounded global-best PSO, const-generic swarm size and dimension |
| `src/planner.rs` | PSO over observation subsets; *proposes* plans |
| `src/safety.rs` | Independent gate; *approves or rejects* every plan |
| `src/envelope.rs` | Versioned 32-byte CRC-32 message envelope; sliding-window replay guard |
| `src/state.rs` | A/B persistent snapshots with CRC and generation counter |
| `src/rng.rs` | SplitMix64, single-`u64` persisted state |
| `src/sim.rs` | Three-spacecraft mission with deterministic fault injection |
| `src/main.rs` | Desktop driver (`std`): mission table and benchmark mode |
| `tests/mission.rs` | Mission-level fault and safety tests |
| `compare_python.py` | Rust PSO vs. Python SwarmOpt on seeded benchmarks |

## Run

```bash
cd prototypes/deep_space
cargo test --release             # unit + mission tests
cargo run --release              # mission comparison table
cargo run --release -- mission 7 # with another seed
```

Rust ↔ Python comparison (from the repo root, with `pip install -e .`):

```bash
cargo build --release --manifest-path prototypes/deep_space/Cargo.toml
python prototypes/deep_space/compare_python.py --seeds 30
```

The Python side dominates the runtime (about 12 minutes for 30 seeds on the
dev container, mostly 10-D Rosenbrock); use `--seeds 10` for a quick look.

## Results (Linux x86_64, rustc 1.97, run for this PR)

`cargo test --release`: 25 unit tests and 16 mission tests pass. The existing
Python suite (`python -m pytest tests/unit`) is unaffected: 93 passed.

### Mission (seed 2026, 400 ticks, 3 spacecraft)

```
faults    strategy  science   obs   dup  energy   sent accept decErr replay  safety  minMarg us/decide
nominal   Baseline   2500.2   466    28   685.1    932    932      0      0       0     0.00       0.5
nominal   Pso        2412.7   450    24   685.4    900    900      0      0      20     0.00      31.2
harsh     Baseline   2165.1   478    99   685.0    956    567     68    112       0     0.00       0.5
harsh     Pso        2158.4   462    93   685.8    924    548     64    112      42     0.00      32.4
```

- `science` counts first observations of a target in each epoch only.
  `dup` counts wasted re-observations.
- `safety` is how many proposed plans the gate rejected. Each one fell back
  to the baseline.
- `minMarg` is the lowest `energy − reserve` after any action, and is never
  negative. It reaches ~0 because both strategies spend right up to reserve,
  which is allowed.
- Fixed simulation state: 16,368 bytes. No heap.
- Timing is desktop wall-clock and only indicative.

**The PSO planner does not beat the ratio-greedy baseline here.** Across
seeds 1–5 (nominal and harsh), the baseline earns more science in 8 of 10
runs. PSO usually duplicates fewer observations (8 of 10 runs). See the briefing for the likely cause: a
per-tick knapsack is myopic.

### Rust PSO vs. Python SwarmOpt

30 seeds, 30 particles, 100 epochs, c1 = c2 = 2, linear inertia 0.9 → 0.4,
velocity bound 20% of the range. Median best cost [interquartile range]:

```
function    dims                              Rust                   Python SwarmOpt
sphere         2    3.599e-12 [2.26e-13, 9.76e-12]    9.801e-13 [1.13e-13, 1.28e-11]
sphere        10    9.008e-04 [4.89e-04, 1.74e-03]    1.121e-03 [6.17e-04, 2.62e-03]
rosenbrock     2    2.510e-05 [6.51e-06, 1.71e-04]    1.314e-04 [3.08e-05, 6.92e-04]
rosenbrock    10    1.087e+01 [8.75e+00, 4.54e+01]    9.656e+00 [8.57e+00, 1.09e+01]
```

The two implementations use different random generators, so only the
distributions are comparable, not individual runs. Medians agree to within
the same order of magnitude. Rust's wider upper quartile on 10-D Rosenbrock
is consistent with its absorbing position walls (Python does not clamp
positions), but that has not been isolated yet. Rust also uses `f32` and
Python uses `f64`.

## Not yet verified

- The bare-metal builds (`thumbv7em-none-eabihf`, `riscv32imac-unknown-none-elf`)
  are set up in CI (`.github/workflows/rust-prototype.yml`) but were not run
  locally, because those targets could not be installed in the dev
  environment.
- Cross-platform bit-identical determinism is only asserted within one
  platform. The CI matrix runs the tests on Linux, macOS and Windows, but
  does not yet compare outputs across them.
