# PSO Policy

def model_based_computation(start_state, action sequence):
    R = 0
    k = 0
    for k < T:
        a = x[action]
        (s, r) = m(s, a)
        R = R + (y**k * r)
        k += 1
    return R

def psop(p):
    p = 0
    # init random positions and velocities

    for p < P:
        fitness = model_based_computation(s, particle)
        # update position
        # update best neighbor
        # update velocities
        # update positions
        # position clamps (check for this)
        p += 1
    return g_best


