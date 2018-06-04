"""Intertia Weight Functions"""
import random

def constant_intertia_weight(c):
    return c

def random_inertia_weight(s=None):
    random.seed(s)
    return (0.5 + random.random()/2)
