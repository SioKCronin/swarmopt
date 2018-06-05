"""
Core components for building a swarm
"""

import math

def initialize_swarm(n, val_min, val_max, dims, v_clamp, obj_func):
    swarm = []
    for particle in range(n):
        pos = np.random.uniform(val_min, val_max, dims)
        velocity = np.random.uniform(-v_clamp, v_clamp, dims)
        p_best_pos = pos
        if obj_func.__name__ == 'rosenbrock_func':
            p_best_cost = obj_func(p_best_pos, p_best_pos)
        else:
            p_best_cost = obj_func(p_best_pos)

        swarm.append([pos, velocity, p_best_pos, p_best_cost])

    #global_best_pos, global_best_cost = best_of_the_personal

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
    # Intertia weighting
    # Maybe mutation here?
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
    # Velocity clamping will come in here
    swarm[idx][1] = (w * swarm[idx][1]) + cognitive + social
    swarm[idx][0] += swarm[idx][1] 

    return swarm, swarm_best_pos, swarm_best_cost


def calculate_swarm_best(dims, obj_func):
    global_best = np.random.uniform(val_min, val_max, dims)

    if obj_func.__name__ == 'rosenbrock_func':
        swarm_best_cost = obj_func(swarm_best_pos, swarm_best_pos)
    else:
        swarm_best_cost = obj_func(swarm_best_pos)

    return swarm_best_pos, swarm_best_cost


def global_best(idx, personal_best_cost, global_best_cost, swarm):
    if personal_best_cost < global_best_cost:
        global_best_pos = swarm[idx][B_POS]
        global_best_cost = personal_best_cost
    return global_best_pos, global_best_cost


def local_best(particle, swarm, k):
    best_neighbors = getNeighbors(particle[B_POS], swarm, k)
    best_cost = particle[B_COST]
    best_pos = particle[B_POS]
    for y in range(len(best_neighbors)):
        if swarm[y][B_COST] < best_cost:
            best_pos = swarm[y][B_POS]
            best_cost = swarm[y][B_COST]
    return best_pos, best_cost


def compute_unified_velocity(c1, c2, idx, swarm, global_best_pos, local_best_pos, g):
    cognitive = (c1 * np.random.uniform(0, 1, 2)) * (swarm[idx][B_POS] - swarm[idx][P_POS_IDX])
    social_global = (c2 * np.random.uniform(0, 1, 2)) * (global_best_pos - swarm[idx][P_POS_IDX])
    social_local = (c2 * np.random.uniform(0, 1, 2)) * (local_best_pos - swarm[idx][P_POS_IDX])
    if weight == g:
        unified = np.random.normal(mu, std, dims) * u * social_global \
                  + (1-u) * social_local
    else:
        unified = u * social_global + \
                np.random.normal(mu, std, dims) * (1-u) * social_local
    velocity = (w * swarm[idx][VEL]) + unified


def clamp_velocity(velocity):
    if velocity[0] < v_clamp_min:
        velocity[0] = v_clamp_min
    if velocity[1] < v_clamp_min:
        velocity[1] = v_clamp_min
    if velocity[0] > v_clamp_max:
        velocity[0] = v_clamp_max
    if velocity[1] > v_clamp_max:
        velocity[0] = v_clamp_max
    return velocity


def optimize_swarm(iters, swarm, obj_func, w, k, c1, c2, swarm_best_pos, swarm_best_cost):
    epoch = 1
    while epoch <= iters:
        for idx, particle in enumerate(swarm):
            swarm = update_pbest(swarm, idx, obj_func)
            swarm, swarm_best_pos, swarm_best_cos = \
                    update_position(swarm, idx, w, k, c1, c2, 
                            swarm_best_pos, swarm_best_cost)
        epoch += 1
    return swarm_best_cost
