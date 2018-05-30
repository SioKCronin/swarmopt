# Simulated Annealing PSO

import numpy as np
from random import shuffle
from common.utils.distance import euclideanDistance, getNeighbors

def global_best_pso(n, dims, c1, c2, w, iters, obj_func, val_min, val_max):

    search_range = val_max - val_min
    v_clamp_min = - (0.2 * search_range)
    v_clamp_max = 0.2 * search_range

    swarm = []
    swarm_best_pos = np.array([0]*dims)
    swarm_best_idx = 0
    swarm_worst_new_pos = np.array([0]*dims)
    swarm_worst_old_pos = np.array([0]*dims)
    swarm_worst_idx = 0
    if obj_func.__name__ == 'rosenbrock_func':
        swarm_best_cost = obj_func(swarm_best_pos, swarm_best_pos)
        swarm_worst_cost = obj_func(swarm_worst_old_pos, swarm_worst_old_pos)
    else:
        swarm_best_cost = obj_func(swarm_best_pos)
        swarm_worst_cost = obj_func(swarm_worst_old_pos)

    P_POS_IDX = 0
    P_VELOCITY_IDX = 1
    P_BEST_POS_IDX = 2
    P_BEST_COST_IDX = 3

    for particle in range(n):
        pos = np.random.uniform(val_min, val_max, dims)
        velocity = np.random.uniform(val_min, val_max, dims)
        p_best_pos = np.random.uniform(val_min, val_max, dims)
        if obj_func.__name__ == 'rosenbrock_func':
            p_best_cost = obj_func(p_best_pos, p_best_pos)
        else:
            p_best_cost = obj_func(p_best_pos)

        swarm.append([pos, velocity, p_best_pos, p_best_cost])

    def calculate_velocity(idx):
        cognitive = (c1 * np.random.uniform(0, 1, 2)) * (swarm[idx][P_BEST_POS_IDX] - swarm[idx][P_POS_IDX])
        social = (c2 * np.random.uniform(0, 1, 2)) * (swarm_best_pos - swarm[idx][P_POS_IDX])
        velocity = (w * swarm[idx][P_VELOCITY_IDX]) + cognitive + social
        if velocity[0] < v_clamp_min:
            swarm[idx][P_VELOCITY_IDX][0] = v_clamp_min
        if velocity[1] < v_clamp_min:
            swarm[idx][P_VELOCITY_IDX][1] = v_clamp_min
        if velocity[0] > v_clamp_max:
            swarm[idx][P_VELOCITY_IDX][0] = v_clamp_max
        if velocity[1] > v_clamp_max:
            swarm[idx][P_VELOCITY_IDX][1] = v_clamp_max
        else:
            swarm[idx][P_VELOCITY_IDX] = velocity

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

            # Update swarm global best
            if personal_best_cost < swarm_best_cost:
                swarm_best_cost = personal_best_cost
                swarm_best_pos = swarm[idx][P_BEST_POS_IDX]
                swarm_best_idx = idx

            # Update swarm global worst
            if personal_best_cost > swarm_worst_cost:
                swarm_worst_cost = personal_best_cost
                swarm_worst_pos = swarm[idx][P_BEST_POS_IDX]
                swarm_worst_idx = idx

        for idx, particle in enumerate(swarm):
            if idx == swarm_best_idx:
                swarm[idx][P_POS_IDX] = np.random.uniform(val_min, val_max, dims)

            elif idx == swarm_worst_idx:
                old_pos = swarm[idx][P_POS_IDX]
                new_pos = swarm[idx][P_POS_IDX] + calculate_velocity(idx)
                old_swarm_worst_cost = obj_func(old_pos)
                new_swarm_worst_cost = obj_funct(new_pos)
                if new_warm_worst_cost - old_swarm_worst_cost < alpha:
                    swarm[idx][P_POS_IDX] = new_pos
                else:
                    best_neighbors = getNeighbors(particle[P_BEST_POS_IDX], swarm, k)
                    best_cost = particle[P_BEST_COST_IDX]
                    best_pos = particle[P_BEST_POS_IDX]
                    for y in range(len(best_neighbors)):
                        if swarm[y][P_BEST_COST_IDX] < best_cost:
                            best_pos = swarm[y][P_BEST_POS_IDX]
            else:
                if obj_func.__name__ == 'rosenbrock_func':
                    if idx == len(swarm) - 1:
                        pass
                    else:
                        current_cost = obj_func(particle[P_POS_IDX], swarm[idx + 1][P_POS_IDX])
                else:
                    current_cost = obj_func(particle[P_POS_IDX])
                personal_best_cost = swarm[idx][P_BEST_COST_IDX]

                # Update personal best
                if current_cost < personal_best_cost:
                    swarm[idx][P_BEST_POS_IDX] = swarm[idx][P_POS_IDX]
                    swarm[idx][P_BEST_COST_IDX] = current_cost

                if personal_best_cost < swarm_best_cost:
                    swarm_best_cost = personal_best_cost
                    swarm_best_pos = particle[P_BEST_POS_IDX]

                # Update pos
                swarm[idx][P_POS_IDX] += swarm[idx][P_VELOCITY_IDX]

        x += 1

    return swarm_best_cost
