import numpy as np
import math
import sys
from swarmopt import Swarm, functions
import csv
from time import gmtime, strftime
import os
file_path = 'csvfiles/{}.csv'.format(strftime("%Y-%m-%d-%H-%M-%S"))

def run_20(n, dims, c1, c2, w, iters, obj_func, v_clamp):
    costs = []
    runtimes = []
    for _ in range(20):
        s = Swarm(n, dims, c1, c2, w, iters, obj_func, v_clamp)
        s.optimize()
        costs.append(s.best_cost)
        runtimes.append(s.runtime)
    return np.mean(costs), np.mean(runtimes)


def run_all_tests(n, dims, c1, c2, w, iters):
    funcs = [
        [functions.sphere, [-5.12, 5.12]],
        [functions.ackley, [-32.768, 32.768]],
        [functions.griewank, [-600, 600]],
        [functions.rastrigin, [-5.12, 5.12]],
        [functions.weierstrass, [-0.5, 0.5]],
    ]

    algos = [["global_best", Swarm]]

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["algo", "function", "avg_cost", "avg_time"])

        for algo_name, algo in algos:
            for func in funcs:
                print("--------------------------")
                print("Testing %s" % func[0])
                s = algo(n, dims, c1, c2, w, iters, func[0], func[1])
                s.optimize()
                print("Single run Best Cost:", s.best_cost)
                print("Single run Runtime:", s.runtime)

                avg_cost, avg_runtime = run_20(n, dims, c1, c2, w, iters, func[0], func[1])
                print("Run 20 Average Cost:", avg_cost)
                print("Run 20 Average Runtime:", avg_runtime)

                writer.writerow([algo_name, func[0].__name__, avg_cost, avg_runtime])

def leaderboard(csv_dir="csvfiles"):
    """Compare algorithms across benchmark functions using normalized scores.

    Reads all CSV files in csv_dir, applies min-max normalization per function
    so algorithms are ranked fairly across functions with different cost scales,
    then prints a leaderboard sorted by mean normalized score.
    """
    import glob

    # {func_name: {algo_name: [costs]}}
    data = {}
    pattern = os.path.join(csv_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        print("No CSV files found in %s. Run generate_scores first." % csv_dir)
        return

    for fpath in files:
        with open(fpath, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                algo = row.get("algo")
                func = row.get("function")
                try:
                    cost = float(row["avg_cost"])
                except (ValueError, KeyError, TypeError):
                    continue
                if algo and func:
                    data.setdefault(func, {}).setdefault(algo, []).append(cost)

    if not data:
        print("No valid data found in CSV files.")
        return

    # Average repeated runs per (algo, func) pair
    avg_data = {
        func: {algo: np.mean(costs) for algo, costs in algos.items()}
        for func, algos in data.items()
    }

    all_algos = sorted({algo for funcs in avg_data.values() for algo in funcs})
    all_funcs = sorted(avg_data.keys())

    # Min-max normalize per function: score 1.0 = best (lowest cost), 0.0 = worst
    normalized = {algo: [] for algo in all_algos}
    for func in all_funcs:
        func_costs = avg_data[func]
        values = list(func_costs.values())
        lo, hi = min(values), max(values)
        for algo in all_algos:
            if algo in func_costs:
                cost = func_costs[algo]
                score = 1.0 - (cost - lo) / (hi - lo) if hi > lo else 1.0
                normalized[algo].append(score)

    rankings = [
        (algo, np.mean(scores))
        for algo, scores in normalized.items()
        if scores
    ]
    rankings.sort(key=lambda x: x[1], reverse=True)

    print("\nAlgorithm Comparison Leaderboard")
    print("Score: 1.0 = best, 0.0 = worst per function (min-max normalized)")
    print("Functions: %s" % ", ".join(all_funcs))
    print()
    print("%-5s %-30s %s" % ("Rank", "Algorithm", "Mean Score"))
    print("-" * 50)
    for i, (algo, score) in enumerate(rankings, 1):
        print("%-5d %-30s %.4f" % (i, algo, score))
    print()

    return rankings


if __name__ == '__main__':
    if "--leaderboard" in sys.argv:
        leaderboard()
    else:
        run_all_tests(30, 2, 0.5, 0.3, 0.9, 2000)
