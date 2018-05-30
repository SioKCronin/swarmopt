"""Unified PSO"""

import numpy as np
from random import shuffle
from utils.distance import euclideanDistance, getNeighbors
from utils.swarm import initialize_swarm, update_pbest, generate_weights,
                        update_position, calculate_swarm_best

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

    for particle in range(n):
        pos = np.random.uniform(val_min, val_max, dims)
        velocity = np.random.uniform(val_min, val_max, dims)
        p_best_pos = np.random.uniform(val_min, val_max, dims)
        if obj_func.__name__ == 'rosenbrock_func':
            p_best_cost = obj_func(p_best_pos, p_best_pos)
        else:
            p_best_cost = obj_func(p_best_pos)

        swarm.append([pos, velocity, p_best_pos, p_best_cost])

    x = 0

    while x < iters:
        for idx, particle in enumerate(swarm):
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

            def global_best(idx, personal_best_cost, global_best_cost, swarm):
                if personal_best_cost < global_best_cost:
                    global_best_pos = swarm[idx][B_POS]
                    global_best_cost = personal_best_cost
                return global_best_pos, global_best_cost

            def local_best(particle, swarm, k):
                best_neighbors = getNeighbors(particle[B_POS], swarm, k)
                best_cost = particle[B_COST]
                best_pos = particle[B_POS]
                for y in range(len(best_neighbors)):
                    if swarm[y][B_COST] < best_cost:
                        best_pos = swarm[y][B_POS]
                        best_cost = swarm[y][B_COST]
                return best_pos, best_cost

            def compute_unified_velocity(c1, c2, idx, swarm):
                cognitive = (c1 * np.random.uniform(0, 1, 2)) * (swarm[idx][B_POS] - swarm[idx][P_POS_IDX])
                social_global = (c2 * np.random.uniform(0, 1, 2)) * (swarm_best_pos - swarm[idx][P_POS_IDX])
                social_local = (c2 * np.random.uniform(0, 1, 2)) * (best_pos - swarm[idx][P_POS_IDX])
                if weight == g:
                    unified = np.random.normal(mu, std, dims) * u * social_global \
                              + (1-u) * social_local
                else:
                    unified = u * social_global + \
                            np.random.normal(mu, std, dims) * (1-u) * social_local
                velocity = (w * swarm[idx][VEL]) + unified

            velocity = compute_unified_velocity(c1, c2, idx, swarm)

            if velocity[0] < v_clamp_min:
                swarm[idx][VEL][0] = v_clamp_min
            if velocity[1] < v_clamp_min:
                swarm[idx][VEL][1] = v_clamp_min
            if velocity[0] > v_clamp_max:
                swarm[idx][VEL][0] = v_clamp_max
            if velocity[1] > v_clamp_max:
                swarm[idx][VEL][1] = v_clamp_max
            else:
                swarm[idx][VEL] = velocity

            # Update poss
            swarm[idx][POS] += swarm[idx][VEL]

        x += 1

    return swarm_best_cost
