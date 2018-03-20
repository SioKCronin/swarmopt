# DMPSO

import numpy as np
from random import shuffle

n = 10 # Number of groups in swarm (10, 20, 50 tested)
m = 3 # Each swarm's population size (2, 3, 5 tested)
dimensions = 2
c1 = 0.5 # Cognitive weight (how much each particle references their memory)
c2 = 0.3 # Social weight (how much each particle references swarm/metaswarm memory)
w = 0.9 # Velocity weight
R = 1000 # Swarm reshuffling interval
iters = 2000 

swarm = []
swarm_best_pos = [0, 0]
swarm_best_cost = [5]

group_data = []

P_POS_IDX = 0
P_VELOCITY_IDX = 1
P_BEST_POS_IDX = 2
P_BEST_COST_IDX = 3

GROUP_BEST_POS_IDX = 0
GROUP_BEST_COST_IDX = 1

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

x = 0

# LSPO for the first 0.9 iterations
while x < (0.9 * iters):

    if (x % R)== 0:
        flattened_list = [particle for group in swarm for particle in group]
        shuffle(flattened_list)
        swarm = [flattened_list[i:i + n] for i in range(0, len(flattened_list), 3)]

    for group in range(n):
        if group == 1:

        for particle in range(m):
            current_cost = sum([i**2 for i in (swarm[group][particle][P_POS_IDX])])
            personal_best_cost = swarm[group][particle][P_BEST_COST_IDX]
            if current_cost < personal_best_cost:
                swarm[group][particle][P_BEST_POS_IDX] = swarm[group][particle][P_POS_IDX]
                swarm[group][particle][P_BEST_COST_IDX] = current_cost

            # Update swarm global best
            if personal_best_cost < group_data[group][GROUP_BEST_COST_IDX]:
                group_data[group][GROUP_BEST_COST_IDX] = personal_best_cost
                group_data[group][GROUP_BEST_POS_IDX] = swarm[group][particle][P_BEST_POS_IDX]

	    # Compute velocity
            cognitive = (c1 * np.random.uniform(0, 1, 2)) * (swarm[group][particle][P_BEST_POS_IDX] - swarm[group][particle][P_POS_IDX])
            social = (c2 * np.random.uniform(0, 1, 2)) * (group_data[group][GROUP_BEST_POS_IDX] - swarm[group][particle][P_POS_IDX])
            swarm[group][particle][P_VELOCITY_IDX] = (w * swarm[group][particle][P_VELOCITY_IDX]) + cognitive + social

            # Update poss
            swarm[group][particle][P_POS_IDX] += swarm[group][particle][P_VELOCITY_IDX]

    x += 1

x = 0

# GPSO for the last 0.1 iterations
while x < (0.1 * iters):
    for group in range(n):

        for particle in range(m):
            current_cost = sum([i**2 for i in (swarm[group][particle][P_POS_IDX])])
            personal_best_cost = swarm[group][particle][P_BEST_COST_IDX]
            if current_cost < personal_best_cost:
                swarm[group][particle][P_BEST_POS_IDX] = swarm[group][particle][P_POS_IDX]
                swarm[group][particle][P_BEST_COST_IDX] = current_cost

            # Update swarm global best
            if personal_best_cost < swarm_best_cost:
                swarm_best_cost = personal_best_cost
                swarm_best_pos = swarm[group][particle][P_BEST_POS_IDX]

	    # Compute velocity
            cognitive = (c1 * np.random.uniform(0, 1, 2)) * (swarm[group][particle][P_BEST_POS_IDX] - swarm[group][particle][P_POS_IDX])
            social = (c2 * np.random.uniform(0, 1, 2)) * (group_data[group][GROUP_BEST_POS_IDX] - swarm[group][particle][P_POS_IDX])
            swarm[group][particle][P_VELOCITY_IDX] = (w * swarm[group][particle][P_VELOCITY_IDX]) + cognitive + social

            # Update poss
            swarm[group][particle][P_POS_IDX] += swarm[group][particle][P_VELOCITY_IDX]

    x += 1

print(swarm_best_cost)
