import numpy as np
import math
from random import shuffle
from dmspso.dmspso import dynamic_multiswarm_pso
from common.functions.single_objective import sphere_func, rosenbrock_func,\
                                              griewank_func, ackley_func,\
                                              rastrigin_func, weierstrass_func

algo = dynamic_multiswarm_pso

def run_20(n, m, c1, c2, w, R, iters, dims, obj_func, val_min, val_max):
    '''
    n: Number of groups in swarm (10, 20, 50 tested)
    m: Each swarm's particles (2, 3, 5 tested)
    c1: Cognitive weight (how much each particle references their memory)
    c2: Social weight (how much each particle references swarm/group memory)
    w: Velocity weight
    R: Swarm reshuffling interval
    iters: 2000
    dims: 2
    '''
    print("m=%i"%m)
    vals = []
    for i in range(20):
        vals.append(algo(n, m, c1, c2, w, R, iters, dims,
                         obj_func, val_min, val_max))
    print("Min value:",min(vals))
    print("Mean value:",np.mean(vals))

def run_all_tests(obj_func, val_min, val_max):
    n = 30
    c1 = 0.5
    c2 = 0.3
    w = 0.9
    R = 10
    iters = 2000
    dims = 2

    print("--------------------------")
    print("Testing %s" %obj_func)
    print("Single run test:", algo(n, 2, c1, c2, w, R, iters, dims,
                                   obj_func, val_min, val_max))
    for x in [2, 3, 5]:
        print("M=%i:"%x, run_20(n, x, c1, c2, w, R, iters, dims,
                                obj_func, val_min, val_max))

run_all_tests(sphere_func, -5.12, 5.12)
run_all_tests(rosenbrock_func, -2.048, 2.048)
run_all_tests(ackley_func, -32.768, 32.768)
run_all_tests(griewank_func, -600, 600)
run_all_tests(rastrigin_func, -5.12, 5.12)
run_all_tests(weierstrass_func, -0.5, 0.5)
