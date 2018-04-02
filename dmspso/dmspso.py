# dmspso

import numpy as np
from random import shuffle
from pyswarms.utils.functions.single_obj import sphere_func

def dynamic_multiswarm_pso(n, m, c1, c2, w, R, iters, dims, obj_func, val_min, val_max):

    search_range = 0.2 * (val_max - val_min)

    swarm = []
    swarm_best_pos = np.random.uniform(val_min, val_max, dims)
    swarm_best_cost = obj_func(swarm_best_pos)

    group_data = []

    P_POS_IDX = 0
    P_VELOCITY_IDX = 1
    P_BEST_POS_IDX = 2
    P_BEST_COST_IDX = 3

    G_BEST_POS_IDX = 0
    G_BEST_COST_IDX = 1

    decrement = 0.7 / n

    for group in range(n):
        w = w - decrement
        group = []
        group_best_pos = np.random.uniform(val_min, val_max, dims)
        group_best_cost = obj_func(group_best_pos)
        group_data.append([group_best_pos, group_best_cost])

        for particle in range(m):
            pos = np.random.uniform(val_min, val_max, dims)
            velocity = np.random.uniform(-search_range, search_range, dims)
            p_best_pos = np.random.uniform(val_min, val_max, dims)
            p_best_cost = obj_func(p_best_pos)

            group.append([pos, velocity, p_best_pos, p_best_cost])

        swarm.append(group)

    # Species bests for the first 0.9 iterations (following the UPSO scheme)
    for x in range(int(0.9 * iters)):

        if (x % R)== 0 and x != 0:
            particles = [particle for group in swarm for particle in group]
            shuffle(particles)
            swarm = [particles[i:i + m] for i in range(0, len(particles), m)]

        for group_idx, group in enumerate(swarm):
            group_bests = group_data[group_idx]

            for particle in group:
                current_cost = obj_func(particle[P_POS_IDX])
                personal_best_cost = particle[P_BEST_COST_IDX]
                if current_cost < personal_best_cost:
                    particle[P_BEST_POS_IDX] = particle[P_POS_IDX]
                    particle[P_BEST_COST_IDX] = current_cost

                # Update swarm global best
                if personal_best_cost < group_bests[G_BEST_COST_IDX]:
                    group_bests[G_BEST_COST_IDX] = personal_best_cost
                    group_bests[G_BEST_POS_IDX] = particle[P_BEST_POS_IDX]

                # Calculate cognitive and social
                cognitive = (c1 * np.random.uniform(0, 1, 2)) * \
                     (particle[P_BEST_POS_IDX] - particle[P_POS_IDX])
                social = (c2 * np.random.uniform(0, 1, 2)) * \
                         (group_bests[G_BEST_POS_IDX] - particle[P_POS_IDX])

                # Calculate velocity
                if particle[P_POS_IDX][0] > search_range:
                    particle[P_POS_IDX][0] > search_range
                if particle[P_POS_IDX][1] > search_range:
                    particle[P_POS_IDX][1] > search_range
                particle[P_VELOCITY_IDX] = (w * particle[P_VELOCITY_IDX]) + \
                                                cognitive + social

                # Update pos
                particle[P_POS_IDX] += particle[P_VELOCITY_IDX]

    # Global bests for the last 0.1 iterations
    for x in range(int(0.1 * iters)):
        for group in swarm:

            for particle in group:
                current_cost = obj_func(particle[P_POS_IDX])
                personal_best_cost = particle[P_BEST_COST_IDX]
                if current_cost < personal_best_cost:
                    particle[P_BEST_POS_IDX] = particle[P_POS_IDX]
                    particle[P_BEST_COST_IDX] = current_cost

                # Update swarm global best
                if swarm_best_cost == None:
                    swarm_best_cost = personal_best_cost
                if personal_best_cost < swarm_best_cost:
                    swarm_best_cost = personal_best_cost
                    swarm_best_pos = particle[P_BEST_POS_IDX]

                # Compute velocity
                cognitive = (c1 * np.random.uniform(0, 1, 2)) * \
                            (particle[P_BEST_POS_IDX] - particle[P_POS_IDX])
                social = (c2 * np.random.uniform(0, 1, 2)) * \
                         (swarm_best_pos - particle[P_POS_IDX])
                particle[P_VELOCITY_IDX] = (w * particle[P_VELOCITY_IDX]) + \
                                            cognitive + social

                # Update poss
                particle[P_POS_IDX] += particle[P_VELOCITY_IDX]

    print("Best cost:", swarm_best_cost)
    print("Best position:", swarm_best_pos)
