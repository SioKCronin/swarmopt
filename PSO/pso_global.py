# PSO Global

import numpy as np

m = 3 # Each swarm's population size (2, 3, 5 tested)
n = 10 # Number of swarms (10, 20, 50 tested)
c1 = 0.5
c2 = 0.3
w = 0.9
iters = 100
dimensions = 2

pos = []
velocity = []
personal_best_pos = []
personal_best_cost = []
global_best_pos = [0, 0]
global_best_cost = [5]

x = 0
# GPSO for the last 0.1 iterations
while x < iters:
    for i in range(n):
        for j in range(m):
            current_cost = sum([i**2 for i in (pos[i][j])])
            pbest_cost = sum([i**2 for i in (personal_best_pos[i][j])])
            if current_cost < pbest_cost:
                best_pos[i][j] = pos[i][j]
                best_cost = current_cost

            # Update global best (across all swarms) personal best is better
            if np.min(pbest_cost) < best_cost:
                print("pbest", np.min(pbest_cost))
                best_cost = np.min(pbest_cost)
                best_pos = personal_best_pos[np.argmin(pbest_cost)]

	    # Compute for cognitive and social terms
            cognitive = (c1 * np.random.uniform(0, 1, 2) * (personal_best_pos[i][j] - pos[i][j]))
            social = (c2 * np.random.uniform(0, 1, 2) * (best_pos - pos[i][j]))
            velocity[i][j] = (w * velocity[i][j]) + cognitive + social

            # Update positions
            pos += velocity

    x += 1

print(best_cost)
