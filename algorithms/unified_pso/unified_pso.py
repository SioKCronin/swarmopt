"""Unified PSO"""

import numpy as np
from random import shuffle
from utils import euclideanDistance, getNeighbors, \
                        initialize_swarm, generate_weights, optimize_swarm,
                        update_position, calculate_swarm_best, global_best,
                        local_best, calculate_unified_velocity

POS = 0
VEL = 1
B_POS = 2
B_COST = 3

def unified_pso(n, dims, c1, c2, w, u, k, mu, std, weight,
                           iters, obj_func, val_min, val_max):
    '''
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
    '''

    v_clamp = 0.2 * (val_max - val_min)
    swarm = initialize_swarm(n, val_min, val_max, dims, v_clamp, obj_func)
    global_best_pos, global_best_cost = calculate_swarm_best(dims, obj_func)
    print(optimize_swarm(iters, swarm, obj_func, w, k, c1, c2, swarm_best_pos, swarm_best_cost))

    x = 0

    while x < iters:
        for idx, particle in enumerate(swarm):

            def calculate_personal_best(idx, obj_func, swarm):
                if obj_func.__name__ == 'rosenbrock_func':
                    if idx == len(swarm) - 1:
                        pass
                    else:
                        current_cost = obj_func(particle[P_POS_IDX], swarm[idx + 1][P_POS_IDX])
                else:
                    current_cost = obj_func(particle[P_POS_IDX])
                personal_best_cost = swarm[idx][B_COST]

                if current_cost < personal_best_cost:
                    swarm[idx][B_POS] = swarm[idx][P_POS_IDX]
                    swarm[idx][B_COST] = current_cost

                return personal_best_pos, personal_best_cost ## FIX these

        global_best_pos, global_best_cost = global_best(idx, personal_best_cost, global_best_cost, swarm)
        local_best_pos, local_best_cost = local_best(particle, swarm, k)
        velocity = compute_unified_velocity(c1, c2, idx, swarm, global_best_pos, local_best_pos, g)
        swarm[idx][VEL] = clamp_velocity(velocity)
        swarm[idx][POS] += swarm[idx][VEL]

        x += 1

    print(swarm_best_cost)
