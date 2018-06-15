"""Intertia Weight Functions"""
import random

def chaotic_inertia_weight(c1, c2, z, max_iter, current_iter):
    """Introduced by Feng et al. 2008"""
    z = 4 * z * (1-z)
    return (c1-c2)*((max_iter-current_iter)/max_iter)+(c2*z)


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




