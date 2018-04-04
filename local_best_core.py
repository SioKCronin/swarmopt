import numpy as np
import math
from pso.local_best_pso import local_best_pso
from common.functions.single_objective import sphere_func, rosenbrock_func, griewank_func, ackley_func, rastrigin_func, weierstrass_func

# Run Local Best PSO 20 times on each function and take an average

def run_20(n, dims, c1, c2, w, k, iters, obj_func):
    '''
    n: Number of particles in swarm
    dims: Number of dimensions
    c1: Cognitive weight (how much each particle references their memory)
    c2: Social weight (how much each particle references swarm/group memory)
    w: Velocity weight
    k: number of nearest neighbors to evaluate
    iters: Number of iterations
    '''
    vals = []
    for i in range(20):
        vals.append(local_best_pso(n, dims, c1, c2, w, k, iters, obj_func))
    print("Mean value:",np.mean(vals))

n = 30
dims = 2
c1 = 0.5
c2 = 0.3
w = 0.9
k = 2
iters = 2000

print("FUNC 1: Sphere Function")
obj_func = sphere_func
val_min = -5.12
val_max = 5.12

print("Testing Sphere Function")
print(local_best_pso(n, dims, c1, c2, w, k, iters, obj_func))
run_20(n, dims, c1, c2, w, k, iters, obj_func)

print("--------------------------------")
print("FUNC 2: Rosenbrock's Function")
obj_func = rosenbrock_func
val_min = -2.048
val_max = 2.048

print("Testing Rosenbrock's Function")
print(local_best_pso(n, dims, c1, c2, w, k, iters, obj_func))
run_20(n, dims, c1, c2, w, k, iters, obj_func)

#print("--------------------------------")
print("FUNC 3: Ackley's Function")
obj_func = ackley_func
val_min = -32.768
val_max = 32.768

print("Testing Ackley's Function")
print(local_best_pso(n, dims, c1, c2, w, k, iters, obj_func))
run_20(n, dims, c1, c2, w, k, iters, obj_func)

print("--------------------------------")
print("FUNC 4: Griewank's Function")
obj_func = griewank_func
val_min = -600
val_max = 600

print("Testing Griewank Function")
print(local_best_pso(n, dims, c1, c2, w, k, iters, obj_func))
run_20(n, dims, c1, c2, w, k, iters, obj_func)

print("--------------------------------")
print("FUNC 5: Rastrigin Function")
obj_func = rastrigin_func
val_min = -5.12
val_max = 5.12

print("Testing Rastrigin Function")
print(local_best_pso(n, dims, c1, c2, w, k, iters, obj_func))
run_20(n, dims, c1, c2, w, k, iters, obj_func)

print("--------------------------------")
print("FUNC 6: Weierstrass Function")
obj_func = weierstrass_func
val_min = -0.5
val_max = 0.5

print("Testing Weierstrass Function")
print(local_best_pso(n, dims, c1, c2, w, k, iters, obj_func))
run_20(n, dims, c1, c2, w, k, iters, obj_func)
