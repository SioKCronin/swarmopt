"""Single objective test functions"""

import numpy as np
import math

def sphere(x):
    return sum([y**2 for y in x])

def rosenbrock(x1, x2):
    return sum([100*((x**2 - y)**2) + (x - 1)**2 for (x, y) in zip(x1,x2)])

def ackley(x):
    return -20 * np.exp(-0.2 * (sum([y**2 for y in x])/len(x) ** 0.5)) - \
          np.exp(sum([np.cos(2 * math.pi * y)/len(x) for y in x])) + 20 + math.e

def griewank(x):
    return sum([y**2 / 4000 for y in x]) - np.prod([np.cos(y/((i+1)**0.5)) for i, y in enumerate(x)]) + 1

def rastrigin(x):
    return sum([y**2 - 10*np.cos(2*math.pi*y) + 10 for y in x])

def weierstrass(x):
    a = 0.5
    b = 3
    k_max = 20

    def sub_sum(x):
        return sum([a**k * np.cos(2*math.pi*(b**k)*(x + 0.5)) for k in range(k_max)])

    val = sum([sub_sum(x0) for x0 in x]) - (len(x) * sum([a**k * np.cos(2*math.pi*(b**k)*0.5) for k in range(k_max)]))

    return val



