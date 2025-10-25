"""Inertia Weight Functions"""
import random
import numpy as np

def constant_inertia_weight(w):
    """Constant inertia weight"""
    return w

def linear_inertia_weight(w_start, w_end, max_iter, current_iter):
    """Linear decreasing inertia weight"""
    return w_start - (w_start - w_end) * (current_iter / max_iter)

def chaotic_inertia_weight(z, max_iter, current_iter):
    """Chaotic inertia weight - Feng et al. 2008"""
    z = 4 * z * (1 - z)  # Logistic map
    return 0.9 * z + 0.1

def random_inertia_weight(s=None):
    """Random inertia weight between 0.5 and 1.0"""
    if s is not None:
        random.seed(s)
    return 0.5 + random.random() / 2

def adaptive_inertia_weight(w_start, w_end, max_iter, current_iter, 
                           best_cost, initial_cost, current_cost):
    """Adaptive inertia weight based on convergence"""
    # Linear component
    linear_w = w_start - (w_start - w_end) * (current_iter / max_iter)
    
    # Adaptive component based on convergence
    if initial_cost > 0:
        convergence_ratio = (initial_cost - current_cost) / initial_cost
        adaptive_factor = 1.0 - convergence_ratio
        return linear_w * adaptive_factor
    return linear_w

def chaotic_random_inertia_weight(z, s=None):
    """Combination of chaotic and random inertia weight"""
    if s is not None:
        random.seed(s)
    z = 4 * z * (1 - z)  # Logistic map
    return 0.5 * random.random() + 0.5 * z

def exponential_inertia_weight(w_start, w_end, max_iter, current_iter):
    """Exponential decreasing inertia weight"""
    return w_end + (w_start - w_end) * np.exp(-3 * current_iter / max_iter)

def sigmoid_inertia_weight(w_start, w_end, max_iter, current_iter):
    """Sigmoid decreasing inertia weight"""
    x = 2 * current_iter / max_iter - 1
    return w_end + (w_start - w_end) / (1 + np.exp(x))






