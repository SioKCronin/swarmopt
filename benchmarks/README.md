# Benchmarks

This is where I will be benchmarking performance between PSO variations. 

Here is some evaluation [inspiration](https://github.com/sigopt/evalset/blob/master/evalset/test_funcs.py) to springboard from, and a [paper](https://arxiv.org/abs/1603.09441) outlining methods for comparing empirical performance on optimizers.

## Leaderboard (local HTML)

Benchmark runs are stored as CSV under `csvfiles/` (see `generate_scores.py`). To build a static leaderboard page and print a `file://` URL you can paste into the browser:

```bash
python benchmarks/leaderboard.py
```

To regenerate the page and open it in your default browser in one step:

```bash
python benchmarks/leaderboard.py --open
```

Output is written to `benchmarks/leaderboard.html`.
