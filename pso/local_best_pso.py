# Local Best PSO

import numpy as np
import math
from operator import itemgetter
from random import shuffle

m = 10 # Swarm population size
dimensions = 2
c1 = 0.5 # Cognitive weight (how much each particle references their memory)
c2 = 0.3 # Social weight (how much each particle references swarm/metaswarm memory)
w = 0.9 # Velocity weight
p = 4
k = 2
iters = 2000 

swarm = []
swarm_best_pos = np.array([0, 0])
swarm_best_cost = 5

P_POS_IDX = 0
P_VELOCITY_IDX = 1
P_BEST_POS_IDX = 2
P_BEST_COST_IDX = 3

def euclideanDistance(point1, point2):
    distance = 0
    for x in range(dimensions):
        distance += pow((point1[x] - point2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(target, k):
    distances = []
    for x in range(m):
        distances.append((swarm[x][P_POS_IDX], euclideanDistance(swarm[x][P_POS_IDX], target)))
    sorted(distances,key=itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

for particle in range(m):
    pos = np.random.uniform(-1, 1, dimensions)
    velocity = np.random.uniform(-1, 1, dimensions)
    p_best_pos = np.random.uniform(-1, 1, dimensions)
    p_best_cost = 1

    swarm.append([pos, velocity, p_best_pos, p_best_cost])

x = 0

while x < iters:

    for particle in range(m):
        current_cost = sum([i**2 for i in (swarm[particle][P_POS_IDX])])
        personal_best_cost = swarm[particle][P_BEST_COST_IDX]
        if current_cost < personal_best_cost:
            swarm[particle][P_BEST_POS_IDX] = swarm[particle][P_POS_IDX]
            swarm[particle][P_BEST_COST_IDX] = current_cost

        # Update swarm local best
        best_neighbors = getNeighbors(swarm[particle][P_BEST_POS_IDX], k+1)
        best_cost = swarm[particle][P_BEST_COST_IDX]
        best_pos = swarm[particle][P_BEST_POS_IDX]
        for y in range(len(best_neighbors)):
            if swarm[y][P_BEST_COST_IDX] < best_cost:
                best_pos = swarm[y][P_BEST_POS_IDX]

        if personal_best_cost < swarm_best_cost:
            swarm_best_cost = personal_best_cost
            swarm_best_pos = swarm[particle][P_BEST_POS_IDX]

        # Compute velocity
        cognitive = (c1 * np.random.uniform(0, 1, 2)) * (swarm[particle][P_BEST_POS_IDX] - swarm[particle][P_POS_IDX])
        social = (c2 * np.random.uniform(0, 1, 2)) * (best_pos - swarm[particle][P_POS_IDX])
        swarm[particle][P_VELOCITY_IDX] = (w * swarm[particle][P_VELOCITY_IDX]) + cognitive + social

        # Update poss
        swarm[particle][P_POS_IDX] += swarm[particle][P_VELOCITY_IDX]

    x += 1

print(swarm_best_cost)
