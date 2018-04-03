import numpy as np
import math
from dmspso import dynamic_multiswarm_pso
from common.functions.single_objective import sphere_func, griewank_func, ackley_func, rastrigin_func, weierstrass_func
from random import shuffle

# Run DMSPSO 20 times on each function and take an average

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
        vals.append(dynamic_multiswarm_pso(n, m, c1, c2, w, R, iters, dims, obj_func, val_min, val_max))
    print(min(vals))

c1 = 0.5
c2 = 0.3
w = 0.9
R = 10
iters = 2000
dims = 2

print("FUNC 1: Sphere Function")
obj_func = sphere_func
val_min = -5.12
val_max = 5.12
n = 30

print("Testing Sphere Function")
print(dynamic_multiswarm_pso(n, 2, c1, c2, w, R, iters, dims, obj_func, val_min, val_max))
run_20(n, 2, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
run_20(n, 3, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
run_20(n, 5, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)

print("--------------------------------")

# FUNC 2: Rosenbrock Function

#print("--------------------------------")
print("FUNC 3: Ackley's Function")
obj_func = ackley_func
val_min = -32.768
val_max = 32.768
n = 30

print("Testing Ackley's Function")
print(dynamic_multiswarm_pso(n, 2, c1, c2, w, R, iters, dims, obj_func, val_min, val_max))
run_20(n, 2, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
run_20(n, 3, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
run_20(n, 5, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)

print("--------------------------------")
print("FUNC 4: Griewank's Function")
obj_func = griewank_func
val_min = -600
val_max = 600
n = 30

print("Testing Griewank Function")
print(dynamic_multiswarm_pso(n, 2, c1, c2, w, R, iters, dims, obj_func, val_min, val_max))
run_20(n, 2, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
run_20(n, 3, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
run_20(n, 5, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)

print("--------------------------------")
print("FUNC 5: Rastrigin Function")
obj_func = griewank_func
val_min = -5.12
val_max = 5.12
n = 30

print("Testing Rastrigin Function")
print(dynamic_multiswarm_pso(n, 2, c1, c2, w, R, iters, dims, obj_func, val_min, val_max))
run_20(n, 2, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
run_20(n, 3, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
run_20(n, 5, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)

print("--------------------------------")
print("FUNC 6: Weierstrass Function")
obj_func = weierstrass_func
val_min = -0.5
val_max = 0.5
n = 30

print("Testing Weierstrass Function")
print(dynamic_multiswarm_pso(n, 2, c1, c2, w, R, iters, dims, obj_func, val_min, val_max))
run_20(n, 2, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
run_20(n, 3, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
run_20(n, 5, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
