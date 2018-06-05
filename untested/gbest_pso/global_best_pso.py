"""
Global Best Particle Swarm Optimization (Global PSO)

Particle positions are updated based on their relative performance to global
best.

    n : int
      number of particles in the swarm
    dims : int
      number of dimensions in the space
    c1 : float
      cognitive parameter
    c2: float
      social parameter
    w : float
      inertia parameter
    k : int
      number of neighbors to be considered
    iters: int
      number of iterations
    obj_func: function
      objective function
    val_min : float
      minimum evaluatable value for obj_func
    val_max : float
      maximum evaluatable value for obj_func
"""

import numpy as np
from random import shuffle
from .utils.distance import euclideanDistance, getNeighbors
from .utils.swarm import initialize_swarm, update_pbest, generate_weights, \
                        update_position, calculate_swarm_best

def global_best_pso(n, dims, c1, c2, w, iters, obj_func, val_min, val_max):
    v_clamp = 0.2 * (val_max - val_min)
    swarm = initialize_swarm(n, val_min, val_max, dims, v_clamp, obj_func)
    swarm_best_pos, swarm_best_cost = calculate_swarm_best(dims, obj_func)
    epoch = 1

    while epoch <= iters:
        for idx, particle in enumerate(swarm):
            swarm = update_pbest(swarm, idx, obj_func)
            swarm, swarm_best_pos, swarm_best_cos = \
                    update_position(swarm, idx, w, k, c1, c2, 
                            swarm_best_pos, swarm_best_cost)

        epoch += 1

    return swarm_best_cost
