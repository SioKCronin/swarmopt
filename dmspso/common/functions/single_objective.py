# Single objective

def sphere_func(x):
    return sum([y**2 for y in x])

# How will give it two particles just for this one?
def rosenbrock_func(x1, x2):
    return sum([100*((x**2 - y)**2) + (x - 1)**2 for (x, y) in zip(x1,x2)])

def ackley_func(x):
    return -20 * np.exp(-0.2 * (sum([y**2 for y in x])/len(x) ** 0.5)) - \
          np.exp(sum([np.cos(2 * math.pi * y)/len(x) for y in x])) + 20 + math.e

#def griewank_func(x):
 #   val = sum([y**2 / 4000 for y in x]) - np.prod([np.cos

def rastrigin_func(x):
    val = sum([y**2 - 10*np.cos(2*math.pi*y) + 10 for y in x])

def weierstrass_func(x):
    a = 0.5
    b = 3
    k_max = 20

