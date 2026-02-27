# SwarmOpt benchmark suite (for researchers)

This directory provides a **reproducible benchmark suite** to compare different PSO models (algorithms × functions × runs) and export results for papers or analysis.

## Requirements

- SwarmOpt and its dependencies (e.g. `numpy`). Some SwarmOpt features (e.g. diversity monitoring) require `scipy`; install with `pip install scipy` if you see import errors.

## Quick start

From the **project root**:

```bash
# Quick run (few algorithms × 3 functions × 2 runs), prints summary table
python tests/benchmarks/run_suite.py

# Save CSV/JSON to a directory
python tests/benchmarks/run_suite.py --output results/

# Use a built-in config: medium, full, unimodal, multimodal
python tests/benchmarks/run_suite.py --config medium --output results/
python tests/benchmarks/run_suite.py --config unimodal
python tests/benchmarks/run_suite.py --config full --runs 5
```

## What it does

- Runs **algorithm × function × N runs** (with fixed seeds for reproducibility).
- Writes **benchmark_results.csv** and **benchmark_results.json** (when `--output` is set).
- Prints a **summary table** (mean ± std of best cost per algorithm/function).

## Available algorithms (presets)

List them:

```bash
python tests/benchmarks/run_suite.py --list-algorithms
```

Current presets include: `global_constant`, `global_linear`, `global_adaptive`, `global_exponential`, `local_linear`, `unified_linear`, `cpso_best`, `cpso_tournament`, `ppso`.

## Available functions

List them:

```bash
python tests/benchmarks/run_suite.py --list-functions
```

Functions are drawn from `swarmopt.functions` (sphere, rosenbrock, ackley, rastrigin, griewank, weierstrass, etc.). You can extend `BENCHMARK_FUNCTIONS` in `run_suite.py` to add more.

## Configs

- **Built-in (no file):** `quick`, `medium`, `full`, `unimodal`, `multimodal`.
- **JSON files:** `configs/quick.json`, `configs/unimodal.json`, `configs/multimodal.json`.

Example custom config (JSON):

```json
{
  "algorithms": ["global_linear", "cpso_best"],
  "functions": ["sphere", "ackley"],
  "dims": 10,
  "n_particles": 30,
  "epochs": 50,
  "runs_per_cell": 5,
  "velocity_clamp": [-5, 5],
  "seed": 42
}
```

Save as `configs/my_experiment.json` and run:

```bash
python tests/benchmarks/run_suite.py --config configs/my_experiment.json --output results/
```

## Output format

- **CSV:** one row per run; columns: `algorithm`, `function`, `best_cost`, `runtime_sec`, `seed`, `dims`, `n_particles`, `epochs`.
- **JSON:** same rows as a list of objects.

Use the CSV in R, Python (pandas), or Excel for further analysis and figures.

## Extending the suite

1. **New algorithm preset:** add a key to `ALGORITHM_PRESETS` in `run_suite.py` with the same kwargs you would pass to `Swarm(...)`.
2. **New function:** add the callable to `BENCHMARK_FUNCTIONS` in `run_suite.py` (and ensure bounds are in `FUNCTION_METADATA` if needed).
3. **New config:** add a JSON file under `configs/` or a new key in the `builtin` dict in `load_config()`.

## Reproducibility

- Each run uses a deterministic **seed** (configurable via `seed` in config).
- Record your SwarmOpt version and config when publishing results.
