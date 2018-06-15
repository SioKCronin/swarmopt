import numpy as np
import math
import sys
from swarmopt import Swarm, functions
import csv
from time import gmtime, strftime
import os
#script_dir = os.path.dirname(__file__)
#file_path = os.path.join(script_dir, 'data/{}.csv'.format(strftime("%Y-%m-%d-%H-%M-%S")))
file_path = '{}.csv'.format(strftime("%Y-%m-%d-%H-%M-%S"))

def run_all_tests(n, dims, c1, c2, w, iters):

    def run_20(n, dims, c1, c2, w, iters, obj_func, v_clamp):
        costs = []
        runtimes = []
        for i in range(20):
            s = Swarm(n, dims, c1, c2, w, iters, obj_func, v_clamp)
            s.optimize()
            costs.append(s.best_cost)
            runtimes.append(s.runtime)
        return [np.mean(costs), np.mean(runtimes)]

    funcs = [
        [functions.sphere, [-5.12, 5.12]],
        [functions.ackley, [-32.768, 32.768]]
        #[functions.griewank, [-600, 600]],
        #[functions.rastrigin, [-5.12, 5.12]],
        #[functions.weierstrass, [-0.5, 0.5]],
    ]

    algos = [["global_best", Swarm]]


    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["algo", "function", "avg_cost", "avg_time"])

    for algo_name, algo in algos:

        algo_name = []

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

            #algo_name.append({func.__name__: [avg_cost, avg_runtime]})

        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([algo.__name__, func[0].__name__, avg_cost, avg_runtime])

if __name__ == '__main__':
    run_all_tests(30, 2, 0.5, 0.3, 0.9, 2000)
