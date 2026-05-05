# Benchmarks

Empirical comparison of SwarmOpt algorithms on standard test functions. The design follows the spirit of **Dewancker et al., [A Stratified Analysis of Bayesian Optimization Methods](https://arxiv.org/abs/1603.09441)** (arXiv:1603.09441): repeated independent runs, reporting both **central tendency and spread**, **stratifying** results by classes of test functions, and summarizing performance with **ranks within each stratum** (per function, then averaged over functions in that stratum).

Additional function ideas: [SigOpt evalset test functions](https://github.com/sigopt/evalset/blob/master/evalset/test_funcs.py).

## Requirements

Running the full benchmark driver (`run_benchmarks.py`) imports `Swarm`, which currently pulls in optional stack pieces (e.g. **SciPy** via `utils/diversity`). If import fails, install:

```bash
pip install scipy
```

(or your project’s usual extra, e.g. `pip install -e ".[multiobjective]"` where SciPy is listed).

## Suites (`suite.py`)

| Suite        | Contents |
|-------------|----------|
| `default`   | Nine n-dimensional problems: sphere, rosenbrock, ackley, griewank, rastrigin, schwefel, levy, weierstrass, michalewicz (with metadata-driven bounds). |
| `2d`        | Extra 2D-only problems (e.g. beale, booth) with `fixed_dims=2`. |
| `default+2d`| `default` + `2d`. |
| `full`      | `default` plus sum_squares, rotated_hyper_ellipsoid, zakharov, and `2d`. |

**Strata** (for grouping and mean-rank tables):

- **Coarse `stratum`:** `unimodal` vs `multimodal` (from `FUNCTION_METADATA` in `swarmopt/functions.py`).
- **`stratum_fine`:** Finer “genre” tags (e.g. oscillatory, ridged, deceptive) to mirror the paper’s idea of function genres without requiring BO-specific metrics.

**Note:** The Branin function is not included in the automated suite, because it needs *per-dimension* bounds and `Swarm` currently uses a scalar search box for all dimensions.

## Running benchmarks (`run_benchmarks.py`)

From the **repository root**:

```bash
python benchmarks/run_benchmarks.py
```

Useful options:

| Option | Meaning |
|--------|---------|
| `--suite` | `default`, `2d`, `default+2d`, or `full`. |
| `--algos` | Space-separated: `global`, `local`, `hhoa`, `cpso`, `unified`. |
| `--n-trials` | Number of random restarts (default 20). |
| `--seed` | Base integer seed; each trial uses a **deterministic** offset (not Python’s `hash()`). |
| `--dims` | Search space dimension for problems that are not `fixed_dims`. |
| `--epochs`, `--n-particles`, `--c1`, `--c2`, `--w` | Swarm hyperparameters. |
| `--only` | Comma-separated function names to restrict the suite. |
| `--quick` | Smoke test: 2 trials, 3 epochs, `sphere` only, `global`+`hhoa`. |

**Outputs (under `csvfiles/`):**

1. **`benchmark_stratified_<timestamp>.csv`** — One row per (algorithm, function) with `avg_cost`, `std_cost`, `median_cost`, time stats, `mean_regret` when the global minimum is a known scalar in metadata.
2. **`legacy_leaderboard_<timestamp>.csv`** — Subset of columns for the HTML leaderboard: `algo`, `function`, `stratum`, `avg_cost`, `avg_time`, `std_cost`.

At the end of a run, **mean rank per coarse stratum** is printed (rank 1 = best mean cost on a function, then averaged over functions in that stratum).

The legacy entry point `generate_scores.py` forwards to `run_benchmarks.py`.

## Leaderboard (local HTML)

Aggregate CSVs and open a static page:

```bash
python benchmarks/leaderboard.py
python benchmarks/leaderboard.py --open
```

`leaderboard.html` shows optional **Stratum** and **Std cost** columns when present in any CSV. Older four-column files still work.

## Metrics (`metrics.py`)

- **Simple regret** — When `optimal_value` in metadata is numeric: `max(0, f_best − f*)`.
- **Summary statistics** — Mean, sample standard deviation, and median over trials.
- **Mean ranks by stratum** — For each stratum, rank algorithms on each function, then average ranks across functions in that stratum (see paper’s stratified comparison idea).

## Citation (methodology)

If you use this benchmark protocol, consider citing the stratified-empirical methodology from:

[I. Dewancker et al., *A Stratified Analysis of Bayesian Optimization Methods*, arXiv:1603.09441 (2016)](https://arxiv.org/abs/1603.09441).

Here we reuse the *repeated runs, stratum labels, and rank summaries* style for swarm optimizers (final values after fixed budgets), not the paper’s full Bayesian-optimization run-length models.
