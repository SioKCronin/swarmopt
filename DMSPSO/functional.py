# Functional PSO

import numpy as np
from scipy.spatial import KDTree

m = 3 # Each swarm's population size (2, 3, 5 tested)
n = 10 # Number of swarms (10, 20, 50 tested)
c1 = 0.5
c2 = 0.3
w = 0.9
p = 2
k = 2
R = 10
max_gen = 10
dimensions = 2

pos = []
personal_best_pos = []
pbest_cost = []
best_cost = []
velocity = []

def _get_neighbors(pbest_cost, i):
    # Use cKDTree to get the indices of the nearest neighbors
    tree = KDTree(pos[i])
    _, idx = tree.query(pos, p, k)

    # Map the computed costs to the neighbour indices and take the
    # argmin. If k-neighbors is equal to 1, then the swarm acts
    # independently of each other.
    if k == 1:
            # The minimum index is itself, no mapping needed.
            best_neighbor = pbest_cost[idx][:, np.newaxis].argmin(axis=1)
    else:
        idx_min = pbest_cost[idx].argmin(axis=1)
        best_neighbor = idx[np.arange(len(idx)), idx_min]

    return best_neighbor

for i in range(n):
    subswarm_pos = []
    subswarm_personal_best_pos = []
    subswarm_pbest_cost = []
    subswarm_best_cost = []
    subswarm_velocity = []

    for i in range(m):
        subswarm_pos.append(np.random.uniform(-1, 1, dimensions))
        subswarm_personal_best_pos.append([0 for i in range(dimensions)])
        subswarm_pbest_cost.append(0)
        subswarm_best_cost.append(0)
        subswarm_velocity.append([0.0 for i in range(dimensions)])

    pos.append(subswarm_pos)
    personal_best_pos.append(subswarm_personal_best_pos)
    pbest_cost.append(subswarm_pbest_cost)
    best_cost.append(subswarm_best_cost)
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

            # Get neighbors
            nmin_idx = _get_neighbors(pbest_cost, i)
            best_cost[i][j] = pbest_cost[nmin_idx]
            best_pos[i][j] = best_pos[nmin_idx]

            # Update velocity

	    # Compute for cognitive and social terms
            cognitive = (c1 * np.random.uniform(0, 1, n) * (personal_best_pos - pos))
            social = (c2 * np.random.uniform(0, 1, n) * (best_pos - pos))
            velocity = (w * velocity) + cognitive + social

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
