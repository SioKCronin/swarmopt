# Dynamic Neighborhood PSO

import numpy as np
import math
from common.utils.distance import euclideanDistance, getNeighbors

def dynamic_neighborhood_pso(n, dims, c1, c2, w, k, iters,
                   obj_func1, obj_func2, val_min, val_max):

    search_range = val_max - val_min
    v_clamp = 0.2 * search_range

    swarm = []
    swarm_best_pos = np.array([0]*dims)
    swarm_best_cost = 5

    P_POS_IDX = 0
    P_VELOCITY_IDX = 1
    P_BEST_POS_IDX = 2
    P_BEST_COST_IDX = 3

    for particle in range(n):
        pos = np.random.uniform(val_min, val_max, dims)
        velocity = np.random.uniform(-v_clamp, v_clamp, dims)
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
                    current_cost = obj_func1(particle[P_POS_IDX], swarm[idx + 1][P_POS_IDX])
            else:
                current_cost = obj_func1(particle[P_POS_IDX])
            personal_best_cost = particle[P_BEST_COST_IDX]
            if current_cost < personal_best_cost:
                swarm[idx][P_BEST_POS_IDX] = particle[P_POS_IDX]
                swarm[idx][P_BEST_COST_IDX] = current_cost

            # Update swarm local best
            best_neighbors = getNeighbors(particle[P_BEST_POS_IDX], swarm, k)
            best_cost = particle[P_BEST_COST_IDX]
            best_pos = particle[P_BEST_POS_IDX]

            # Here's the magic - we select best from the 2nd obj_func
            for y in range(len(best_neighbors)):
                if obj_func2(swarm[y][P_POS_IDX]) < best_cost:
                    best_pos = swarm[y][P_POS_IDX]

            if personal_best_cost < swarm_best_cost:
                swarm_best_cost = personal_best_cost
                swarm_best_pos = particle[P_BEST_POS_IDX]

            # Compute velocity
            cognitive = (c1 * np.random.uniform(0, 1, 2)) * (swarm[idx][P_BEST_POS_IDX] - swarm[idx][P_POS_IDX])
            social = (c2 * np.random.uniform(0, 1, 2)) * (best_pos - swarm[idx][P_POS_IDX])
            swarm[idx][P_VELOCITY_IDX] = (w * swarm[idx][P_VELOCITY_IDX]) + cognitive + social

            # Update poss
            swarm[idx][P_POS_IDX] += swarm[idx][P_VELOCITY_IDX]

        x += 1

    return swarm_best_cost
