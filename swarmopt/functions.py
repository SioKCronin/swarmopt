"""Single objective test functions
Collection of 30 multidimensional functions for global optimization benchmarking
Based on: https://www.mdpi.com/2306-5729/7/4/46
"""

import numpy as np
import math

# ============================================================================
# Function Metadata
# ============================================================================

FUNCTION_METADATA = {
    'sphere': {
        'optimal_value': 0.0,
        'optimal_position': 'zeros',
        'bounds': [-5.12, 5.12],
        'dimensions': 'any',
        'type': 'unimodal'
    },
    'sum_squares': {
        'optimal_value': 0.0,
        'optimal_position': 'zeros',
        'bounds': [-10, 10],
        'dimensions': 'any',
        'type': 'unimodal'
    },
    'rotated_hyper_ellipsoid': {
        'optimal_value': 0.0,
        'optimal_position': 'zeros',
        'bounds': [-65.536, 65.536],
        'dimensions': 'any',
        'type': 'unimodal'
    },
    'zakharov': {
        'optimal_value': 0.0,
        'optimal_position': 'zeros',
        'bounds': [-5, 10],
        'dimensions': 'any',
        'type': 'unimodal'
    },
    'dixon_price': {
        'optimal_value': 0.0,
        'optimal_position': 'special',  # x_i* = 2^(-(2^i-2)/(2^i))
        'bounds': [-10, 10],
        'dimensions': 'any',
        'type': 'unimodal'
    },
    'powell': {
        'optimal_value': 0.0,
        'optimal_position': 'zeros',
        'bounds': [-4, 5],
        'dimensions': 'divisible_by_4',
        'type': 'unimodal'
    },
    'rosenbrock': {
        'optimal_value': 0.0,
        'optimal_position': [1.0],  # All ones
        'bounds': [-5, 10],
        'dimensions': 'any',
        'type': 'multimodal'
    },
    'ackley': {
        'optimal_value': 0.0,
        'optimal_position': 'zeros',
        'bounds': [-32.768, 32.768],
        'dimensions': 'any',
        'type': 'multimodal'
    },
    'griewank': {
        'optimal_value': 0.0,
        'optimal_position': 'zeros',
        'bounds': [-600, 600],
        'dimensions': 'any',
        'type': 'multimodal'
    },
    'rastrigin': {
        'optimal_value': 0.0,
        'optimal_position': 'zeros',
        'bounds': [-5.12, 5.12],
        'dimensions': 'any',
        'type': 'multimodal'
    },
    'schwefel': {
        'optimal_value': 0.0,
        'optimal_position': [420.9687],  # Approximately
        'bounds': [-500, 500],
        'dimensions': 'any',
        'type': 'multimodal'
    },
    'levy': {
        'optimal_value': 0.0,
        'optimal_position': [1.0],  # All ones
        'bounds': [-10, 10],
        'dimensions': 'any',
        'type': 'multimodal'
    },
    'michalewicz': {
        'optimal_value': 'dimension_dependent',
        'optimal_position': 'dimension_dependent',
        'bounds': [0, math.pi],
        'dimensions': 'any',
        'type': 'multimodal'
    },
    'perm': {
        'optimal_value': 0.0,
        'optimal_position': 'special',  # [1, 1/2, 1/3, ..., 1/n]
        'bounds': 'dimension_dependent',  # [-n, n]
        'dimensions': 'any',
        'type': 'multimodal'
    },
    'trid': {
        'optimal_value': 'dimension_dependent',  # -n(n+4)(n-1)/6
        'optimal_position': 'special',  # x_i* = i(n+1-i)
        'bounds': 'dimension_dependent',  # [-n^2, n^2]
        'dimensions': 'any',
        'type': 'multimodal'
    },
    'weierstrass': {
        'optimal_value': 0.0,
        'optimal_position': 'zeros',
        'bounds': [-0.5, 0.5],
        'dimensions': 'any',
        'type': 'multimodal'
    },
    'de_jong_f5': {
        'optimal_value': 0.998,
        'optimal_position': [-32, -32],
        'bounds': [-65.536, 65.536],
        'dimensions': 2,
        'type': 'multimodal'
    },
    'beale': {
        'optimal_value': 0.0,
        'optimal_position': [3, 0.5],
        'bounds': [-4.5, 4.5],
        'dimensions': 2,
        'type': 'multimodal'
    },
    'booth': {
        'optimal_value': 0.0,
        'optimal_position': [1, 3],
        'bounds': [-10, 10],
        'dimensions': 2,
        'type': 'multimodal'
    },
    'matyas': {
        'optimal_value': 0.0,
        'optimal_position': [0, 0],
        'bounds': [-10, 10],
        'dimensions': 2,
        'type': 'multimodal'
    },
    'three_hump_camel': {
        'optimal_value': 0.0,
        'optimal_position': [0, 0],
        'bounds': [-5, 5],
        'dimensions': 2,
        'type': 'multimodal'
    },
    'six_hump_camel': {
        'optimal_value': -1.0316,
        'optimal_position': [[0.0898, -0.7126], [-0.0898, 0.7126]],  # Multiple optima
        'bounds': [-3, 3],
        'dimensions': 2,
        'type': 'multimodal'
    },
    'easom': {
        'optimal_value': -1.0,
        'optimal_position': [math.pi, math.pi],
        'bounds': [-100, 100],
        'dimensions': 2,
        'type': 'multimodal'
    },
    'goldstein_price': {
        'optimal_value': 3.0,
        'optimal_position': [0, -1],
        'bounds': [-2, 2],
        'dimensions': 2,
        'type': 'multimodal'
    },
    'branin': {
        'optimal_value': 0.397887,
        'optimal_position': [[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]],
        'bounds': [[-5, 10], [0, 15]],  # Different bounds for each dimension
        'dimensions': 2,
        'type': 'multimodal'
    },
    'shubert': {
        'optimal_value': -186.7309,
        'optimal_position': 'multiple',  # Many global minima
        'bounds': [-10, 10],
        'dimensions': 2,
        'type': 'multimodal'
    },
    'hartmann_3d': {
        'optimal_value': -3.86278,
        'optimal_position': [0.114614, 0.555649, 0.852547],
        'bounds': [0, 1],
        'dimensions': 3,
        'type': 'multimodal'
    },
    'hartmann_6d': {
        'optimal_value': -3.32237,
        'optimal_position': [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
        'bounds': [0, 1],
        'dimensions': 6,
        'type': 'multimodal'
    },
    'shekel': {
        'optimal_value': 'm_dependent',  # Depends on m parameter
        'optimal_position': [4, 4, 4, 4],
        'bounds': [0, 10],
        'dimensions': 4,
        'type': 'multimodal'
    }
}


def get_function_metadata(func_name):
    """Get metadata for a test function"""
    return FUNCTION_METADATA.get(func_name, None)


def get_optimal_position(func_name, n_dims=None):
    """Get the optimal position for a function
    
    Args:
        func_name: Name of the function
        n_dims: Number of dimensions (required for some functions)
    
    Returns:
        List representing optimal position, or None if not available
    """
    meta = FUNCTION_METADATA.get(func_name, None)
    if not meta:
        return None
    
    opt_pos = meta['optimal_position']
    
    if opt_pos == 'zeros':
        dims = n_dims if n_dims else 2
        return [0.0] * dims
    elif opt_pos == 'special':
        if not n_dims:
            return None
        if func_name == 'dixon_price':
            # x_i* = 2^(-(2^i-2)/(2^i))
            return [2**(-(2**i - 2) / (2**i)) for i in range(n_dims)]
        elif func_name == 'perm':
            # x_i* = 1/(i+1)
            return [1.0 / (i + 1) for i in range(n_dims)]
        elif func_name == 'trid':
            # x_i* = i(n+1-i)
            return [(i + 1) * (n_dims - i) for i in range(n_dims)]
    elif isinstance(opt_pos, list):
        if len(opt_pos) == 1 and meta['dimensions'] == 'any':
            # All same value, replicate for n_dims
            dims = n_dims if n_dims else 2
            return opt_pos * dims
        elif isinstance(opt_pos[0], list):
            # Multiple optimal positions, return first
            return opt_pos[0]
        else:
            return opt_pos
    
    return opt_pos

# ============================================================================
# Unimodal Functions
# ============================================================================

def sphere(x):
    """Sphere function - Simple unimodal function
    Global minimum: f(0,...,0) = 0
    Search domain: [-5.12, 5.12]^n
    """
    return sum([y**2 for y in x])

def sum_squares(x):
    """Sum Squares function
    Global minimum: f(0,...,0) = 0
    Search domain: [-10, 10]^n
    """
    return sum([(i+1) * y**2 for i, y in enumerate(x)])

def rotated_hyper_ellipsoid(x):
    """Rotated Hyper-Ellipsoid function
    Global minimum: f(0,...,0) = 0
    Search domain: [-65.536, 65.536]^n
    """
    return sum([sum([x[j]**2 for j in range(i+1)]) for i in range(len(x))])

def zakharov(x):
    """Zakharov function
    Global minimum: f(0,...,0) = 0
    Search domain: [-5, 10]^n
    """
    sum1 = sum([y**2 for y in x])
    sum2 = sum([0.5 * (i+1) * y for i, y in enumerate(x)])
    return sum1 + sum2**2 + sum2**4

def dixon_price(x):
    """Dixon-Price function
    Global minimum: f(x*) = 0, where x_i* = 2^(-(2^i-2)/(2^i))
    Search domain: [-10, 10]^n
    """
    term1 = (x[0] - 1)**2
    term2 = sum([(i+1) * (2*x[i]**2 - x[i-1])**2 for i in range(1, len(x))])
    return term1 + term2

def powell(x):
    """Powell function (requires n divisible by 4)
    Global minimum: f(0,...,0) = 0
    Search domain: [-4, 5]^n
    """
    if len(x) % 4 != 0:
        raise ValueError("Powell function requires dimension divisible by 4")
    result = 0
    for i in range(0, len(x)-3, 4):
        result += (x[i] + 10*x[i+1])**2
        result += 5 * (x[i+2] - x[i+3])**2
        result += (x[i+1] - 2*x[i+2])**4
        result += 10 * (x[i] - x[i+3])**4
    return result

# ============================================================================
# Multimodal Functions
# ============================================================================

def rosenbrock(x):
    """Rosenbrock function (Banana function)
    Global minimum: f(1,...,1) = 0
    Search domain: [-5, 10]^n
    """
    return sum([100*((x[i]**2 - x[i+1])**2) + (x[i] - 1)**2 for i in range(len(x)-1)])

def ackley(x):
    """Ackley function
    Global minimum: f(0,...,0) = 0
    Search domain: [-32.768, 32.768]^n
    """
    n = len(x)
    sum1 = sum([y**2 for y in x])
    sum2 = sum([np.cos(2 * math.pi * y) for y in x])
    return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + math.e

def griewank(x):
    """Griewank function
    Global minimum: f(0,...,0) = 0
    Search domain: [-600, 600]^n
    """
    sum_term = sum([y**2 / 4000 for y in x])
    prod_term = np.prod([np.cos(y / np.sqrt(i+1)) for i, y in enumerate(x)])
    return sum_term - prod_term + 1

def rastrigin(x):
    """Rastrigin function
    Global minimum: f(0,...,0) = 0
    Search domain: [-5.12, 5.12]^n
    """
    n = len(x)
    return 10*n + sum([y**2 - 10*np.cos(2*math.pi*y) for y in x])

def schwefel(x):
    """Schwefel function
    Global minimum: f(420.9687,...,420.9687) ≈ 0
    Search domain: [-500, 500]^n
    """
    n = len(x)
    return 418.9829 * n - sum([y * np.sin(np.sqrt(abs(y))) for y in x])

def levy(x):
    """Levy function
    Global minimum: f(1,...,1) = 0
    Search domain: [-10, 10]^n
    """
    w = [1 + (y - 1) / 4 for y in x]
    term1 = np.sin(math.pi * w[0])**2
    term2 = sum([(w[i] - 1)**2 * (1 + 10 * np.sin(math.pi * w[i] + 1)**2) 
                 for i in range(len(w)-1)])
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * math.pi * w[-1])**2)
    return term1 + term2 + term3

def michalewicz(x, m=10):
    """Michalewicz function
    Global minimum: depends on dimension
    Search domain: [0, π]^n
    """
    return -sum([np.sin(y) * (np.sin((i+1) * y**2 / math.pi))**(2*m) 
                 for i, y in enumerate(x)])

def perm(x, beta=10):
    """Perm function
    Global minimum: f(1, 1/2, 1/3, ..., 1/n) = 0
    Search domain: [-n, n]^n
    """
    n = len(x)
    result = 0
    for k in range(1, n+1):
        inner_sum = sum([((i+1)**k + beta) * ((x[i] / (i+1))**k - 1) 
                         for i in range(n)])
        result += inner_sum**2
    return result

def trid(x):
    """Trid function
    Global minimum: f(x*) = -n(n+4)(n-1)/6, where x_i* = i(n+1-i)
    Search domain: [-n^2, n^2]^n
    """
    term1 = sum([(y - 1)**2 for y in x])
    term2 = sum([x[i] * x[i-1] for i in range(1, len(x))])
    return term1 - term2

def weierstrass(x):
    """Weierstrass function
    Global minimum: f(0,...,0) = 0
    Search domain: [-0.5, 0.5]^n
    """
    a = 0.5
    b = 3
    k_max = 20

    def sub_sum(x_val):
        return sum([a**k * np.cos(2*math.pi*(b**k)*(x_val + 0.5)) 
                   for k in range(k_max)])

    val = sum([sub_sum(x0) for x0 in x]) - \
          (len(x) * sum([a**k * np.cos(2*math.pi*(b**k)*0.5) 
                         for k in range(k_max)]))
    return val

def de_jong_f5(x):
    """De Jong function F5 (Shekel's Foxholes)
    Global minimum: f(-32, -32) ≈ 0.998
    Search domain: [-65.536, 65.536]^2
    Note: This is a 2D function
    """
    if len(x) != 2:
        raise ValueError("De Jong F5 requires exactly 2 dimensions")
    a = np.array([[-32, -16, 0, 16, 32] * 5,
                  [-32]*5 + [-16]*5 + [0]*5 + [16]*5 + [32]*5]).T
    result = 0.002
    for j in range(25):
        sum_term = sum([(x[i] - a[j, i])**6 for i in range(2)])
        result += 1.0 / (j + 1 + sum_term)
    return 1.0 / result

# ============================================================================
# Low-Dimensional Functions (2D-6D)
# ============================================================================

def beale(x):
    """Beale function (2D)
    Global minimum: f(3, 0.5) = 0
    Search domain: [-4.5, 4.5]^2
    """
    if len(x) != 2:
        raise ValueError("Beale function requires exactly 2 dimensions")
    return (1.5 - x[0] + x[0]*x[1])**2 + \
           (2.25 - x[0] + x[0]*x[1]**2)**2 + \
           (2.625 - x[0] + x[0]*x[1]**3)**2

def booth(x):
    """Booth function (2D)
    Global minimum: f(1, 3) = 0
    Search domain: [-10, 10]^2
    """
    if len(x) != 2:
        raise ValueError("Booth function requires exactly 2 dimensions")
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def matyas(x):
    """Matyas function (2D)
    Global minimum: f(0, 0) = 0
    Search domain: [-10, 10]^2
    """
    if len(x) != 2:
        raise ValueError("Matyas function requires exactly 2 dimensions")
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def three_hump_camel(x):
    """Three-Hump Camel function (2D)
    Global minimum: f(0, 0) = 0
    Search domain: [-5, 5]^2
    """
    if len(x) != 2:
        raise ValueError("Three-Hump Camel function requires exactly 2 dimensions")
    return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2

def six_hump_camel(x):
    """Six-Hump Camel function (2D)
    Global minimum: f(±0.0898, ∓0.7126) ≈ -1.0316
    Search domain: [-3, 3]^2
    """
    if len(x) != 2:
        raise ValueError("Six-Hump Camel function requires exactly 2 dimensions")
    return (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + \
           (-4 + 4*x[1]**2)*x[1]**2

def easom(x):
    """Easom function (2D)
    Global minimum: f(π, π) = -1
    Search domain: [-100, 100]^2
    """
    if len(x) != 2:
        raise ValueError("Easom function requires exactly 2 dimensions")
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - math.pi)**2 + 
                                                     (x[1] - math.pi)**2))

def goldstein_price(x):
    """Goldstein-Price function (2D)
    Global minimum: f(0, -1) = 3
    Search domain: [-2, 2]^2
    """
    if len(x) != 2:
        raise ValueError("Goldstein-Price function requires exactly 2 dimensions")
    term1 = (1 + (x[0] + x[1] + 1)**2 * 
             (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2))
    term2 = (30 + (2*x[0] - 3*x[1])**2 * 
             (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2))
    return term1 * term2

def branin(x):
    """Branin function (2D)
    Global minimum: f(-π, 12.275) ≈ 0.397887, f(π, 2.275) ≈ 0.397887, f(9.42478, 2.475) ≈ 0.397887
    Search domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    """
    if len(x) != 2:
        raise ValueError("Branin function requires exactly 2 dimensions")
    a, b, c, r, s, t = 1, 5.1/(4*math.pi**2), 5/math.pi, 6, 10, 1/(8*math.pi)
    return a*(x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s

def shubert(x):
    """Shubert function (2D)
    Global minimum: f(x*) ≈ -186.7309 (multiple global minima)
    Search domain: [-10, 10]^2
    """
    if len(x) != 2:
        raise ValueError("Shubert function requires exactly 2 dimensions")
    sum1 = sum([i * np.cos((i+1)*x[0] + i) for i in range(1, 6)])
    sum2 = sum([i * np.cos((i+1)*x[1] + i) for i in range(1, 6)])
    return sum1 * sum2

def hartmann_3d(x):
    """Hartmann 3D function
    Global minimum: f(0.114614, 0.555649, 0.852547) ≈ -3.86278
    Search domain: [0, 1]^3
    """
    if len(x) != 3:
        raise ValueError("Hartmann 3D function requires exactly 3 dimensions")
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673],
                        [4699, 4387, 7470],
                        [1091, 8732, 5547],
                        [381, 5743, 8828]])
    result = 0
    for i in range(4):
        inner_sum = sum([A[i, j] * (x[j] - P[i, j])**2 for j in range(3)])
        result -= alpha[i] * np.exp(-inner_sum)
    return result

def hartmann_6d(x):
    """Hartmann 6D function
    Global minimum: f(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573) ≈ -3.32237
    Search domain: [0, 1]^6
    """
    if len(x) != 6:
        raise ValueError("Hartmann 6D function requires exactly 6 dimensions")
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                        [2329, 4135, 8307, 3736, 1004, 9991],
                        [2348, 1451, 3522, 2883, 3047, 6650],
                        [4047, 8828, 8732, 5743, 1091, 381]])
    result = 0
    for i in range(4):
        inner_sum = sum([A[i, j] * (x[j] - P[i, j])**2 for j in range(6)])
        result -= alpha[i] * np.exp(-inner_sum)
    return result

def shekel(x, m=10):
    """Shekel function (4D, but can be extended)
    Global minimum: depends on m
    Search domain: [0, 10]^4
    """
    if len(x) != 4:
        raise ValueError("Shekel function requires exactly 4 dimensions")
    beta = np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]) / 10.0
    C = np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                  [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                  [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                  [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])
    result = 0
    for i in range(min(m, 10)):
        inner_sum = sum([(x[j] - C[j, i])**2 for j in range(4)])
        result -= 1.0 / (inner_sum + beta[i])
    return result



