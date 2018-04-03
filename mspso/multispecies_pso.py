# MSPSO

import numpy as np
from random import shuffle
from pyswarms.utils.functions.single_obj import sphere_func, ackley_func, \ 
                                                rosenbrock_func, beale_func

n = 10 # Number of species in swarm
m = 3 # Each swarm's particles
c1 = 0.5 # Cognitive weight (how much each particle references their memory)
c2 = 0.3 # Social weight (how much each particle references swarm/group memory)
w = 0.9 # Velocity weight
n_influence = 0.5 # Neighbor influence constant
n_objective_funcs = 4 # Number of objective functions
obs_influence_constant = 2 # Observation influence constant

iters = 2000
dimensions = 2

search_range = np.array([0]*dimensions)
vmax = 0.2 * search_range # Use if lower than returned velocity

swarm = []
swarm_best_pos = np.array([0]*dimensions)
swarm_best_cost = 5

species_data = []

P_POS_IDX = 0
P_VELOCITY_IDX = 1
P_BEST_POS_IDX = 2
P_BEST_COST_IDX = 3

G_BEST_POS_IDX = 0
G_BEST_COST_IDX = 1

# Objective functions

obj_func1 =
obj_func2 =
obj_func3 =
obj_func4 =

# Define neighbors




for specie in range(n):
    specie = []
    specie_best_pos = np.array([0, 0])
    specie_best_cost = 5
    specie_data.append([specie_best_pos, specie_best_cost])

    for particle in range(m):
        pos = np.random.uniform(-1, 1, dimensions)
        velocity = np.random.uniform(-1, 1, dimensions)
        p_best_pos = np.random.uniform(-1, 1, dimensions)
        p_best_cost = 1

        specie.append([pos, velocity, p_best_pos, p_best_cost])

    swarm.append(specie)

for x in range(iters):

    for group_idx, group in enumerate(swarm):
        group_bests = group_data[group_idx]

        for particle in group:
            current_cost = sum([i**2 for i in (particle[P_POS_IDX])])
            personal_best_cost = particle[P_BEST_COST_IDX]
            if current_cost < personal_best_cost:
                particle[P_BEST_POS_IDX] = particle[P_POS_IDX]
                particle[P_BEST_COST_IDX] = current_cost

            # Update specie best
            if personal_best_cost < specie_bests[G_BEST_COST_IDX]:
                specie_bests[G_BEST_COST_IDX] = personal_best_cost
                specie_bests[G_BEST_POS_IDX] = particle[P_BEST_POS_IDX]

            # Update swarm global best
            if personal_best_cost < swarm_best_cost:
                swarm_best_cost = personal_best_cost
                swarm_best_pos = particle[P_BEST_POS_IDX]

	    # Compute velocity
            cognitive = (c1 * np.random.uniform(0, 1, 2)) * \
                 (particle[P_BEST_POS_IDX] - particle[P_POS_IDX])
            social = (c2 * np.random.uniform(0, 1, 2)) * \
                     (group_bests[G_BEST_POS_IDX] - particle[P_POS_IDX])
            # TODO
            neighbor = (n_influence * np.random.uniform(0, 1, 2))) * \
                       (neighbor_best[particle][N_BEST_POS_IDX] - particle[P_POS_IDX])
            particle[P_VELOCITY_IDX] = (w * particle[P_VELOCITY_IDX]) + \
                                            cognitive + social

            # Update pos
            particle[P_POS_IDX] += particle[P_VELOCITY_IDX]

print("Best cost:", swarm_best_cost)
print("Best position:", swarm_best_pos)
