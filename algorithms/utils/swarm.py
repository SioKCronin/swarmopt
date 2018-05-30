"""
Core components for building a swarm
"""

def initialize_swarm(n, val_min, val_max, dims, v_clamp, obj_func):
    swarm = []
    for particle in range(n):
        pos = np.random.uniform(val_min, val_max, dims)
        velocity = np.random.uniform(-v_clamp, v_clamp, dims)
        p_best_pos = np.random.uniform(val_min, val_max, dims)
        if obj_func.__name__ == 'rosenbrock_func':
            p_best_cost = obj_func(p_best_pos, p_best_pos)
        else:
            p_best_cost = obj_func(p_best_pos)

        swarm.append([pos, velocity, p_best_pos, p_best_cost])
    return swarm

def update_pbest(swarm, idx, obj_func):
    """Update personal best based on current positions of particles"""
    if obj_func.__name__ == 'rosenbrock_func':
        if idx == len(swarm) - 1:
            current_cost = obj_func(swarm[idx][0], swarm[idx][0])
        else:
            current_cost = obj_func(swarm[idx][0], swarm[idx + 1][0])
    else:
        current_cost = obj_func(swarm[idx][0])

    if current_cost < swarm[idx][3]:
        swarm[idx][2] = swarm[idx][0]
        swarm[idx][3] = current_cost
    return swarm

def generate_weights(swarm, idx, best_pos, c1, c2):
    """Generate weights with cognitive (c1) and social (c2) weights"""
    cognitive = (c1 * np.random.uniform(0, 1, 2)) \
                * (swarm[idx][2] - swarm[idx][0])
    social = (c2 * np.random.uniform(0, 1, 2)) \
             * ( best_pos - swarm[idx][0])
    return cognitive, social

def calculate_lbest_pos(swarm, idx, k):
    """Calculate local best score from neighbors"""
    best_neighbors = getNeighbors(swarm[idx][2], swarm, k)
    best_cost = swarm[idx][3]
    best_pos = swarm[idx][2]

    for y in range(len(best_neighbors)):
        if swarm[y][3] < best_cost:
            return swarm[y][2]

    return best_pos

def update_position(swarm, idx, w, k, c1, c2, swarm_best_pos, swarm_best_cost):
    best_pos = calculate_best_pos(swarm,idx, k)

    if swarm[idx][3] < swarm_best_cost:
        swarm_best_cost = swarm[idx][3]
        swarm_best_pos = swarm[idx][2]

    cognitive, social = generate_weights(swarm, idx, best_pos, c1, c2)

    swarm[idx][0] += (w * swarm[idx][1]) + cognitive + social

    # Add clamp!

    return swarm, swarm_best_pos, swarm_best_cost

def calculate_swarm_best(dims, obj_func):
    swarm_best_pos = np.array([0]*dims)

    if obj_func.__name__ == 'rosenbrock_func':
        swarm_best_cost = obj_func(swarm_best_pos, swarm_best_pos)
    else:
        swarm_best_cost = obj_func(swarm_best_pos)

    return swarm_best_pos, swarm_best_cost
