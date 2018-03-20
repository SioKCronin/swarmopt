# DMPSO

import numpy as np

m = 3 # Each swarm's population size (2, 3, 5 tested)
n = 10 # Number of swarms (10, 20, 50 tested)
c1 = 0.5
c2 = 0.3
w = 0.9
p = 2
k = 1
R = 5
iters = 2000
dimensions = 2

metaswarm = []
global_best_pos = [0, 0]
global_best_cost = [5]

swarm_data = []

for i in range(n):
    swarm = []
    swarm_best_pos = np.array([0, 0])
    swarm_best_cost = 5
    swarm_data.append([swarm_best_pos, swarm_best_cost])

    for i in range(m):
        position = np.random.uniform(-1, 1, dimensions)
        velocity = np.random.uniform(-1, 1, dimensions)
        personal_best_pos = np.random.uniform(-1, 1, dimensions)
        personal_best_cost = 1
        swarm.append([position, velocity, personal_best_pos, personal_best_cost])

    metaswarm.append(swarm)

x = 0

# LSPO for the first 0.9 iterations
while x < (0.9 * iters):

    if (x % R)== 0:
        pass

    for i in range(n):

        for j in range(m):
            current_cost = sum([i**2 for i in (metaswarm[i][j][0])])
            personal_best_cost = metaswarm[i][j][3]
            if current_cost < personal_best_cost:
                metaswarm[i][j][2] = metaswarm[i][j][0]
                metaswarm[i][j][3] = current_cost

            # print("Swarm data", swarm_data[i])

            # Update swarm global best
            if personal_best_cost < swarm_data[i][1]:
                swarm_data[i][1] = personal_best_cost
                swarm_data[i][0] = metaswarm[i][j][2]

	    # Compute velocity
            cognitive = (c1 * np.random.uniform(0, 1, 2)) * (metaswarm[i][j][2] - metaswarm[i][j][0])
            social = (c2 * np.random.uniform(0, 1, 2)) * (global_best_pos - metaswarm[i][j][0])
            metaswarm[i][j][1] = (w * metaswarm[i][j][1]) + cognitive + social

            # Update positions
            metaswarm[i][j][0] += metaswarm[i][j][1]

    x += 1

x = 0

# GPSO for the last 0.1 iterations
while x < (0.1 * iters):
    for i in range(n):
        for j in range(m):
            current_cost = sum([i**2 for i in (metaswarm[i][j][0])])
            personal_best_cost = metaswarm[i][j][3]
            if current_cost < personal_best_cost:
                metaswarm[i][j][2] = metaswarm[i][j][0]
                metaswarm[i][j][3] = current_cost

            # Update swarm global best
            if personal_best_cost < global_best_cost:
                global_best_cost = personal_best_cost
                global_best_pos = metaswarm[i][j][2]

	    # Compute velocity
            cognitive = (c1 * np.random.uniform(0, 1, 2)) * (metaswarm[i][j][2] - metaswarm[i][j][0])
            social = (c2 * np.random.uniform(0, 1, 2)) * (global_best_pos - metaswarm[i][j][0])
            metaswarm[i][j][1] = (w * metaswarm[i][j][1]) + cognitive + social

            # Update positions
            metaswarm[i][j][0] += metaswarm[i][j][1]

    x += 1

print(global_best_cost)
