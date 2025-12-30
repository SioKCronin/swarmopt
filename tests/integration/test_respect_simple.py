#!/usr/bin/env python3
"""Simple test for respect boundary"""

import numpy as np
from swarmopt import Swarm

# Define target and boundary
target = np.array([5.0, 5.0])
respect_distance = 2.0

# Define objective
def distance_to_target(x):
    return np.linalg.norm(x - target)

print("Testing respect boundary...")
print(f"Target: {target}")
print(f"Respect distance: {respect_distance}")

# Test WITH respect boundary
swarm = Swarm(
    n_particles=10,
    dims=2,
    c1=2.0, c2=2.0, w=0.9,
    epochs=20,
    obj_func=distance_to_target,
    algo='global',
    respect_boundary=respect_distance,
    target_position=target
)

print("Running optimization...")
swarm.optimize()

final_distance = np.linalg.norm(swarm.best_pos - target)

print(f"\nResults:")
print(f"Converged to: {swarm.best_pos}")
print(f"Distance from target: {final_distance:.4f}")
print(f"Respect boundary: {respect_distance}")
print(f"Boundary respected: {final_distance >= respect_distance}")

