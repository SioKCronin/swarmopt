"""Velocity Clamping Functions"""
import numpy as np

def no_clamping(velocity, velocity_bounds):
    """No velocity clamping - particles can move freely"""
    return velocity

def basic_clamping(velocity, velocity_bounds):
    """Basic velocity clamping to [-velocity_bounds, velocity_bounds]"""
    return np.clip(velocity, -velocity_bounds, velocity_bounds)

def adaptive_clamping(velocity, velocity_bounds, current_iter, max_iter):
    """Adaptive velocity clamping that decreases over time"""
    # Start with full clamping, gradually reduce
    adaptive_bounds = velocity_bounds * (1 - current_iter / max_iter)
    return np.clip(velocity, -adaptive_bounds, adaptive_bounds)

def exponential_clamping(velocity, velocity_bounds, current_iter, max_iter):
    """Exponential velocity clamping that decreases exponentially"""
    exp_bounds = velocity_bounds * np.exp(-2 * current_iter / max_iter)
    return np.clip(velocity, -exp_bounds, exp_bounds)

def sigmoid_clamping(velocity, velocity_bounds, current_iter, max_iter):
    """Sigmoid velocity clamping that decreases sigmoidally"""
    x = 2 * current_iter / max_iter - 1
    sigmoid_bounds = velocity_bounds / (1 + np.exp(x))
    return np.clip(velocity, -sigmoid_bounds, sigmoid_bounds)

def random_clamping(velocity, velocity_bounds):
    """Random velocity clamping with random bounds"""
    random_bounds = velocity_bounds * np.random.uniform(0.5, 1.5, len(velocity))
    return np.clip(velocity, -random_bounds, random_bounds)

def chaotic_clamping(velocity, velocity_bounds, z):
    """Chaotic velocity clamping using logistic map"""
    z = 4 * z * (1 - z)  # Logistic map
    chaotic_bounds = velocity_bounds * (0.5 + 0.5 * z)
    return np.clip(velocity, -chaotic_bounds, chaotic_bounds)

def dimension_wise_clamping(velocity, velocity_bounds):
    """Different clamping for each dimension"""
    if len(velocity_bounds) == 1:
        # If single value, apply to all dimensions
        return np.clip(velocity, -velocity_bounds, velocity_bounds)
    else:
        # If array, apply per dimension
        return np.clip(velocity, -velocity_bounds, velocity_bounds)

def soft_clamping(velocity, velocity_bounds, alpha=0.1):
    """Soft velocity clamping using tanh function"""
    return velocity_bounds * np.tanh(velocity / velocity_bounds)

def hybrid_clamping(velocity, velocity_bounds, current_iter, max_iter):
    """Hybrid clamping: adaptive for first half, exponential for second half"""
    if current_iter < max_iter / 2:
        # Adaptive clamping for first half
        adaptive_bounds = velocity_bounds * (1 - current_iter / max_iter)
        return np.clip(velocity, -adaptive_bounds, adaptive_bounds)
    else:
        # Exponential clamping for second half
        exp_bounds = velocity_bounds * np.exp(-2 * current_iter / max_iter)
        return np.clip(velocity, -exp_bounds, exp_bounds)

def convergence_based_clamping(velocity, velocity_bounds, best_cost, initial_cost):
    """Velocity clamping based on convergence progress"""
    if initial_cost > 0:
        convergence_ratio = (initial_cost - best_cost) / initial_cost
        # More convergence = less clamping
        adaptive_bounds = velocity_bounds * (1 - convergence_ratio)
        return np.clip(velocity, -adaptive_bounds, adaptive_bounds)
    return np.clip(velocity, -velocity_bounds, velocity_bounds)
