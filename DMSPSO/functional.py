# Functional PSO

import numpy as np
from scipy.spatial import KDTree

m = 3 # Each swarm's population size (2, 3, 5 tested)
n = 10 # Number of swarms (10, 20, 50 tested)
c1 = 0.5
c2 = 0.3
w = 0.9
p = 2
k = 1
R = 10
max_gen = 100
dimensions = 2

pos = []
personal_best_pos = []
pbest_cost = []
best_cost = []
best_pos = [[0, 0]]
velocity = []

for i in range(n):
    subswarm_pos = []
    subswarm_personal_best_pos = []
    subswarm_pbest_cost = []
    subswarm_best_cost = []
    subswarm_velocity = []

    for i in range(m):
        subswarm_pos.append(np.random.uniform(-1, 1, dimensions))
        subswarm_personal_best_pos.append(np.random.uniform(-1, 1, dimensions))
        subswarm_pbest_cost.append(1)
        subswarm_velocity.append(np.random.uniform(-1, 1, dimensions))

    pos.append(subswarm_pos)
    personal_best_pos.append(subswarm_personal_best_pos)
    pbest_cost.append(subswarm_pbest_cost)
    velocity.append(subswarm_velocity)

i = 1

while i < (0.9 * max_gen):

    for i in range(n):
        for j in range(m):
            current_cost = sum([i**2 for i in (pos[i][j])])
            personal_best_cost = sum([i**2 for i in (personal_best_pos[i][j])])
            if current_cost < personal_best_cost:
                personal_best_pos[i][j] = pos[i][j]
                pbest_cost[i][j] = current_cost
            if personal_best_cost < best_cost:
                best_cost = pbest_cost[i][j]
                best_pos = personal_best_pos[i][j]

	    # Compute for cognitive and social terms
            cognitive = (c1 * np.random.uniform(0, 1, 2)) * (personal_best_pos[i][j] - pos[i][j])
            social = (c2 * np.random.uniform(0, 1, 2)) * (best_pos - pos[i][j])
            velocity = w + cognitive + social

            # Update positions

            pos += velocity

    i += 1

while i < (0.1 * max_gen):
    for i in range(n):
        for j in range(m):
            current_cost = sum([i**2 for i in (pos[i][j])])
            pbest_cost = sum([i**2 for i in (personal_best_pos[i][j])])
            if current_cost < pbest_cost:
                best_pos[i][j] = pos[i][j]
                best_cost[i][j] = current_cost

            if np.min(pbest_cost) < best_cost:
                best_cost = np.min(pbest_cost)
                best_pos = personal_best_pos[np.argmin(pbest_cost)]

	    # Compute for cognitive and social terms
            cognitive = (c1 * np.random.uniform(0, 1, n) * (personal_best_pos - pos))
            social = (c2 * np.random.uniform(0, 1, n) * (best_pos - pos))
            velocity = (w * velocity) + cognitive + social

            # Update positions

            pos += velocity

    i += 1

print(best_pos)
