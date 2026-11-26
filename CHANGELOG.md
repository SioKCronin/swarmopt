# Changelog

All notable changes to SwarmOpt will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Security and data-integrity audit (`docs/SECURITY_AUDIT.md`) and crash-safe
persistence for unattended hardware.

### Added
- `swarmopt.nio`: the nature-inspired extension pack, merged from
  `nature-inspired-optimisation`. 57 population optimizers behind one
  `run(iterations)` contract and a name registry (`OPTIMIZERS`), Boids and
  Vicsek collective-motion steps, ant-colony and river-formation pathfinders,
  and `swarmopt.nio.rl.TrainingEnv` with a trainable linear policy. Standard
  library only. References in `swarmopt/nio/README.md`.
- `Swarm(algo=...)` runs any `swarmopt.nio` optimizer by short name over the
  same bounds, seed and epochs, with `algo_params` for per-algorithm
  settings. `swarmopt.algorithms(family=None)` lists every name.
- One step contract for all 57 nio optimizers (`initialise`, argument-free
  `step`, `best_position`, `best_fitness`) and `swarmopt.nio.drive`, which
  adds checkpoint and resume (bit-for-bit, strict JSON, classes allow-listed
  to `swarmopt.nio`). Works through `Swarm.optimize(checkpoint=...)`.
- `swarmopt.nio.FAMILIES`: each optimizer's metaphor family.
- `swarmopt.nio.compare` and `benchmarks/nio_comparison.py`: every algorithm
  at an equal evaluation budget; results in `docs/nio_benchmarks.md`.

- Communication topologies (`ring`, `random-k`, `geometric`, `dynamic`) with a
  counted message channel, loss, delay, a lying peer, and a spot-check
  defence. `topology='knn-oracle'` keeps the previous all-pairs neighbourhood.
  See `benchmarks/dice_first_figure.py`.
- `rust/persist` (`swarmopt-persist`): a Rust crate with Python bindings providing a
  replicated, CRC-checked, append-only run log and power-cut-safe A/B
  checkpoints. It has storage budgets, needs no clock, and has no
  dependencies in the core.
- `swarmopt.persist`: strict-JSON payload encoding (never pickle),
  `RunLog`, `read_run_log` and `SwarmCheckpoint`.
- `Swarm.optimize(checkpoint=..., checkpoint_every=N)` resumes an
  interrupted run so it finishes bit-for-bit as if uninterrupted (algo
  `global`, `local`, `unified`, `sa`).
- `benchmarks/run_benchmarks.py --log-dir` writes every trial to a durable
  run log.
- CI: `persist.yml` (fmt, clippy, fault-injection tests, Python bindings);
  Dependabot for Cargo.

### Changed
- `Swarm(algo='pso')` is the core global-best PSO and `algo='ppso'` the core
  proactive PSO, so each has one implementation behind `Swarm`.
- Bat, Cultural, Dragonfly, Firefly, IWD-CO, Philippine Eagle, PPSO and Water
  Cycle now follow the nio step contract, so they also work in
  `nio.rl.rollout`, which previously failed for them.
- `Swarm` refuses communication topologies, `respect_boundary`, diversity
  monitoring, variation and delegates for nio algorithms instead of ignoring
  them.

### Fixed
- The core PSO loop rescanned every particle's neighbourhood after each
  particle moved, so one epoch cost O(n^3) even for `algo='global'`, which
  never reads the local best. `global` and `sa` now skip that scan and the
  global best is updated incrementally; `local` and `unified` keep the same
  update order with a vectorized neighbour search. Seeded results are
  unchanged by this fix.
- Core PSO particles (`global`, `local`, `unified`, `sa`, `cpso`) could leave
  the search box. Positions are now clipped to `velocity_clamp`, matching the
  nio optimizers, PPSO and HHOA. This changes seeded results for runs that
  reached the bounds.
- The respect-boundary penalty multiplied the cost, so it rewarded entering
  the boundary when the cost was negative and vanished at zero. It is now
  additive: `cost + 10 * ((r - d) / r)^2 * (1 + |cost|)`.
- `sa` never found its worst particle (`worst_cost` started at +inf), so it
  ran exactly as `global`. The worst particle is now found each epoch.
- An objective that returns NaN no longer crashes the core PSO; NaN is
  treated as +inf and `best_pos` is always set.
- `Swarm` rejects non-positive `n_particles`, `dims` or `epochs`, an empty or
  infinite `velocity_clamp`, and unknown `inertia_func` or
  `velocity_clamp_func` names, which used to fall back to a default silently.
- Collective Animal Behavior kept its "memory" as references to agents that
  then moved, so the memory changed under it and the rebuilt herd could hold
  the same agent twice. It now keeps copies, as in Cuevas et al. (2012).
- **`best_pos` could disagree with `best_cost`.** Particle position arrays
  were aliased and updated in place. Results that relied on `best_pos` may
  have been wrong. Seeded results change with this fix.
- Local-neighbourhood bests paired a neighbour's best cost with its current
  position; the `sa` branch updated `best_cost` without `best_pos`.
- The benchmark runners recorded a seed that had no effect on the run
  (they seeded NumPy's global state, which `Swarm` ignores). The seed is
  now passed to `Swarm`, and recorded seeds reproduce results.
- Benchmark and leaderboard outputs are written atomically and never
  silently overwritten. CSV cells are protected against formula injection.
  NaN/inf rows can no longer top the leaderboard. JSON output is
  standards-compliant.
- `benchmarks/generate_scores.py` no longer crashes.

## [0.3.0] - 2026-11-16

Foundation release: correctness, reproducibility, and supply-chain hygiene.

### Added
- `Swarm(seed=...)` makes runs reproducible. All randomness comes from a
  `numpy.random.Generator`; SwarmOpt no longer reads or changes NumPy's or
  Python's global random state.
- `Swarm(respect_boundary=...)` sets the standoff distance explicitly
  (default remains 10% of the search-space diagonal when `target_position`
  is given). Invalid distances raise `ValueError`.
- CI: unit tests on Python 3.10–3.13 (plus macOS and Windows), a build and
  wheel smoke test, and a weekly integration run.
- Supply chain: OpenSSF Scorecard, Dependabot, pinned action SHAs,
  `SECURITY.md` with private reporting, and build-provenance attestations
  on releases.

### Fixed
- `get_optimal_position` returned wrong optima for Dixon-Price (0-based
  index) and Perm (this is the Perm d,β form, optimum `(1, 2, ..., n)`).
- `dimension_wise_clamping` crashed on scalar bounds.
- `diversity_monitoring=True` crashed when an intervention fired (the
  intervention methods were defined on `Particle` instead of `Swarm`).

### Changed
- Requires Python 3.10+. Packaging moved to `pyproject.toml`; the version
  lives in `swarmopt.__version__`.
- `inertia.random_inertia_weight(s)` uses a private generator, so seeded
  values differ from 0.2.0.
- `algo='multiswarm'` (DMS-PSO) raises `NotImplementedError`. It crashed on
  construction in 0.2.0 and is being reworked.

### Removed
- The optional ETDA hook that modified `sys.path` on import, and the
  unused `etda` submodule entry.

## [0.2.0] - 2024-12-14

### 🎯 Major Features Added

#### Safety & Control
- **Respect Boundary**: Safety-critical standoff distance optimization
  - Automatic enforcement when target position specified
  - Default: 10% of search space diagonal
  - Applications: Satellites, robots, collision avoidance
  - Cannot be disabled for safety-critical applications

#### Multiobjective Optimization
- **Multiobjective PSO**: Handle multiple conflicting objectives
  - NSGA-II inspired algorithm
  - SPEA2 inspired algorithm
  - Pareto dominance and ranking
  - Hypervolume indicator
  - External archive management
  - ZDT and DTLZ benchmark functions

#### Advanced PSO Algorithms
- **Cooperative PSO (CPSO)**: Multiple collaborating swarms
  - Dimension-based swarm assignment
  - Communication strategies: best, random, tournament
  - Scalable for high-dimensional problems

- **Proactive PSO (PPSO)**: Knowledge gain guided exploration
  - Mixed swarms (proactive + reactive particles)
  - Gaussian Process-inspired exploration
  - Sample density awareness
  - Adaptive exploration weights

#### Diversity & Variation
- **Diversity Monitoring**: Prevent premature convergence
  - 9 diversity metrics (Euclidean, Manhattan, Variance, etc.)
  - Automatic intervention triggers
  - Diversity-based particle restart

- **Variation Operators**: Escape local optima
  - 14 variation strategies
  - Gaussian, Uniform, Polynomial variations
  - Escape and diversity-preserving variations
  - Adaptive variation strength
  - Stagnation detection

#### Inertia & Velocity Control
- **8 Inertia Weight Variations**:
  - Constant, Linear Decreasing, Chaotic
  - Random, Adaptive, Chaotic-Random
  - Exponential Decreasing, Sigmoid Decreasing

- **11 Velocity Clamping Strategies**:
  - No Clamping, Basic, Adaptive
  - Exponential, Sigmoid, Random
  - Chaotic, Dimension-wise, Soft
  - Hybrid, Convergence-based

#### Application-Specific
- **Airfoil Optimization Suite**:
  - CST (Class-Shape Transformation) parameterization
  - Hicks-Henne bump functions
  - Bézier curve representation
  - NACA-k modifications
  - Multiple objective functions
  - Constraint handling

### 📊 Visualizations
- Interactive HTML animation viewer
- Respect boundary visualizations
- Satellite positioning examples
- Airfoil optimization viewer
- GIF generation for swarm paths
- Comparison visualizations

### 📚 Documentation
- Comprehensive README with examples
- Respect boundary guide (`docs/RESPECT_BOUNDARY_README.md`)
- Cancer TDA project plan (`docs/CANCER_TDA_PROJECT_PLAN.md`)
- Publishing guide (`PUBLISHING.md`)
- Test suite documentation
- Interactive visualization gallery

### 🧪 Testing
- 10+ comprehensive test scripts
- Test index for easy navigation (`tests_scripts/index.py`)
- Quick test runner (`run_tests.py`)
- Examples for all major features
- Installation verification scripts

### 🗂️ Project Organization
- Created `docs/` directory for documentation
- Organized `tests_scripts/` for all test files
- Centralized `swarm_visualizations/` for all visualizations
- Clean repository structure
- Improved navigation and discoverability

### 🔧 Improvements
- Better error handling
- Performance optimizations
- Enhanced code documentation
- Modular utility structure
- Backward compatibility maintained

### 🐛 Bug Fixes
- Fixed particle update logic in various algorithms
- Corrected local best position tracking
- Fixed SA algorithm implementation
- Improved multiswarm regrouping
- Various performance and stability fixes

### 📦 Dependencies
- Core: `numpy>=1.19.0`
- Optional: `matplotlib>=3.3.0` (for visualizations)
- Optional: `scipy>=1.5.0` (for multiobjective)
- Optional: `giotto-tda>=0.5.0` (for TDA applications)

## [0.1.0] - 2018-XX-XX

### Added
- Initial release
- Basic PSO algorithms:
  - Global Best PSO
  - Local Best PSO
  - Unified PSO
  - Dynamic Multi-Swarm PSO
  - Simulated Annealing PSO
- Benchmark functions:
  - Sphere, Rosenbrock, Ackley
  - Griewank, Rastrigin, Weierstrass
- Basic particle and swarm classes
- Initial documentation

---

## Version Numbering

- **Major version** (X.0.0): Breaking changes
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes only

## Links

- **PyPI**: https://pypi.org/project/swarmopt/
- **GitHub**: https://github.com/siokcronin/swarmopt
- **Documentation**: https://github.com/siokcronin/swarmopt/tree/main/docs
- **Issues**: https://github.com/siokcronin/swarmopt/issues

