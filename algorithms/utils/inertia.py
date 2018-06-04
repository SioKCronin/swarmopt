"""Intertia Weight Functions"""
import random

def constant_inertia_weight(c):
    try:
        c != None
        return c
    except TypeError:
        print("Function requires a constant")

def random_inertia_weight(s=None):
    if s != None:
        random.seed(s)
    return (0.5 + random.random()/2) 

def chaotic_inertia_weight(w1, w2, z, max_iter, current_iter):
    """Introduced by Feng et al. 2008"""
    z = 4 * z * (1-z)
    return (w1-w2)*((max_iter-current_iter)/max_iter)+(w2*z)



