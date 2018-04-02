import numpy as np
import math
from dmspso import dynamic_multiswarm_pso
from common.functions.single_objective import sphere_func
from random import shuffle

# n - Number of groups in swarm (10, 20, 50 tested)
# m - Each swarm's particles (2, 3, 5 tested)
c1 = 0.5 # Cognitive weight (how much each particle references their memory)
c2 = 0.3 # Social weight (how much each particle references swarm/group memory)
w = 0.9 # Velocity weight
R = 10 # Swarm reshuffling interval
iters = 2000
dims = 2

# FUNC 1: Sphere Function
print("FUNC 1: Sphere Function")
obj_func = sphere_func
val_min = -5.12
val_max = 5.12
search_range = 10.24
n = 30
print("m=2")
dynamic_multiswarm_pso(n, 2, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
print("m=3")
dynamic_multiswarm_pso(n, 3, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
print("m=5")
dynamic_multiswarm_pso(n, 5, c1, c2, w, R, iters, dims, obj_func, val_min, val_max)
