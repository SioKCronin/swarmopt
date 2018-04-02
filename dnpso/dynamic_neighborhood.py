# Dynamic Neighbor

import numpy as np
from random import shuffle
from functions import multiobjective

n1 = 100 # Particle population size for Lis and Eiben tests
n2 = 200 # Particle population size for Zitzler tests
m = 2 # Neighbor size
c1 = 1.49445 # Cognitive weight (how much each particle references their memory)
c2 = 1.49445 # Social weight (how much each particle references swarm/group memory)
w = (0.5 + i/2) # Velocity weight (base 0.5 plus additional as rounds increase)
k = 4 # neighborhood size
dimensions = 2

# Velocity clamping
search_range = np.array([0]*dimensions) # Set to the dynamic range of on each dim (what is that?!!)

swarm = []
swarm_best_pos = np.array([0]*dimensions)
swarm_best_cost = 5 # How high should the init be set? (is this in the lit?)

group_data = []

P_POS_IDX = 0
P_VELOCITY_IDX = 1
P_BEST_POS_IDX = 2
P_BEST_COST_IDX = 3

G_BEST_POS_IDX = 0
G_BEST_COST_IDX = 1

for group in range(n):
    group = []
    group_best_pos = np.array([0, 0])
    group_best_cost = 5
    group_data.append([group_best_pos, group_best_cost])

    for particle in range(m):
        pos = np.random.uniform(-1, 1, dimensions)
        velocity = np.random.uniform(-1, 1, dimensions)
        p_best_pos = np.random.uniform(-1, 1, dimensions)
        p_best_cost = 1

        group.append([pos, velocity, p_best_pos, p_best_cost])

    swarm.append(group)

# LSPO for the first 0.9 iterations (following the UPSO scheme)
for x in range(int(0.9 * iters)):

    if (x % R)== 0 and x != 0:
        particles = [particle for group in swarm for particle in group]
        shuffle(particles)
        swarm = [particles[i:i + m] for i in range(0, len(particles), m)]
        print("Sample swarm", swarm[0])

    for group_idx, group in enumerate(swarm):
        group_bests = group_data[group_idx]

        for particle in group:
            current_cost = sum([i**2 for i in (particle[P_POS_IDX])])
            personal_best_cost = particle[P_BEST_COST_IDX]
            if current_cost < personal_best_cost:
                particle[P_BEST_POS_IDX] = particle[P_POS_IDX]
                particle[P_BEST_COST_IDX] = current_cost

            # Update swarm global best
            if personal_best_cost < group_bests[G_BEST_COST_IDX]:
                group_bests[G_BEST_COST_IDX] = personal_best_cost
                group_bests[G_BEST_POS_IDX] = particle[P_BEST_POS_IDX]

	    # Compute velocity
            cognitive = (c1 * np.random.uniform(0, 1, 2)) * \
                 (particle[P_BEST_POS_IDX] - particle[P_POS_IDX])
            social = (c2 * np.random.uniform(0, 1, 2)) * \
                     (group_bests[G_BEST_POS_IDX] - particle[P_POS_IDX])

            # Check to see if velocity is within search range
            if particle[P_POS_IDX][0] > search_range[0] or \
               particle[P_POS_IDX][1] > search_range[1]:
                # ??? search range
                particle[P_VELOCITY_IDX] = (w * search_range) + \
                                            cognitive + social
            else:
                particle[P_VELOCITY_IDX] = (w * particle[P_VELOCITY_IDX]) + \
                                            cognitive + social

            # Update pos
            particle[P_POS_IDX] += particle[P_VELOCITY_IDX]

            # Update search range
            if particle[P_POS_IDX][0] > search_range[0]:
                search_range[0] = particle[P_POS_IDX][0]
            if particle[P_POS_IDX][1] > search_range[1]:
                search_range[1] = particle[P_POS_IDX][1]

# GPSO for the last 0.1 iterations
for x in range(int(0.1 * iters)):
    for group in swarm:

        for particle in group:
            current_cost = sum([i**2 for i in (particle[P_POS_IDX])])
            personal_best_cost = particle[P_BEST_COST_IDX]
            if current_cost < personal_best_cost:
                particle[P_BEST_POS_IDX] = particle[P_POS_IDX]
                particle[P_BEST_COST_IDX] = current_cost

            # Update swarm global best
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
