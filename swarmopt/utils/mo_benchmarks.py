"""
Challenging Multiobjective Benchmark Functions

This module implements the most challenging multiobjective test problems
from literature to properly evaluate MOPSO algorithms.

References:
- ZDT Suite: Zitzler, E., Deb, K., & Thiele, L. (2000)
- DTLZ Suite: Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2005)
- WFG Suite: Huband, S., Hingston, P., Barone, L., & While, L. (2006)
"""

import numpy as np
from typing import Tuple

# ============================================================================
# ZDT SUITE - Classic multiobjective benchmarks
# ============================================================================

def zdt4(x: np.ndarray) -> np.ndarray:
    """
    ZDT4 - Multiple local Pareto fronts (multimodal)
    
    This is one of the most challenging ZDT functions due to
    21^9 ≈ 10^12 local Pareto fronts!
    
    Dimensions: 10
    Objectives: 2
    Bounds: x[0] ∈ [0, 1], x[i] ∈ [-5, 5] for i > 0
    """
    n = len(x)
    f1 = x[0]
    
    g = 1 + 10 * (n - 1) + np.sum(x[1:]**2 - 10 * np.cos(4 * np.pi * x[1:]))
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    
    return np.array([f1, f2])

def zdt6(x: np.ndarray) -> np.ndarray:
    """
    ZDT6 - Non-uniform search space, biased Pareto front
    
    Challenges:
    - Non-uniform density of solutions
    - Low density near Pareto front
    - Biased search space
    
    Dimensions: 10
    Objectives: 2
    """
    n = len(x)
    f1 = 1 - np.exp(-4 * x[0]) * (np.sin(6 * np.pi * x[0]))**6
    
    g = 1 + 9 * ((np.sum(x[1:]) / (n - 1))**0.25)
    h = 1 - (f1 / g)**2
    f2 = g * h
    
    return np.array([f1, f2])

# ============================================================================
# DTLZ SUITE - Scalable many-objective benchmarks
# ============================================================================

def dtlz3(x: np.ndarray, n_obj: int = 3) -> np.ndarray:
    """
    DTLZ3 - Multimodal, 3^k local Pareto fronts
    
    Extremely challenging due to:
    - Highly multimodal landscape
    - 3^k local Pareto fronts (k = dims - n_obj + 1)
    - Easy to get stuck in local optima
    
    Dimensions: n_obj + k - 1 (typically 12 for 3 objectives)
    Objectives: n_obj
    """
    k = len(x) - n_obj + 1
    
    # Highly multimodal g function
    g = 100 * (k + np.sum([
        (x[i] - 0.5)**2 - np.cos(20 * np.pi * (x[i] - 0.5))
        for i in range(n_obj - 1, len(x))
    ]))
    
    objectives = []
    for m in range(n_obj):
        if m == 0:
            f = (1 + g) * np.prod([np.cos(x[j] * np.pi / 2) for j in range(n_obj - 1)])
        elif m < n_obj - 1:
            f = (1 + g) * np.prod([np.cos(x[j] * np.pi / 2) for j in range(n_obj - m - 1)])
            f *= np.sin(x[n_obj - m - 1] * np.pi / 2)
        else:
            f = (1 + g) * np.sin(x[0] * np.pi / 2)
        
        objectives.append(f)
    
    return np.array(objectives)

def dtlz4(x: np.ndarray, n_obj: int = 3, alpha: float = 100.0) -> np.ndarray:
    """
    DTLZ4 - Biased density of solutions
    
    Challenges:
    - Non-uniform distribution
    - Difficult to maintain diversity
    - Alpha parameter controls bias
    
    Dimensions: n_obj + k - 1
    Objectives: n_obj
    """
    k = len(x) - n_obj + 1
    g = np.sum([(x[i] - 0.5)**2 for i in range(n_obj - 1, len(x))])
    
    objectives = []
    for m in range(n_obj):
        if m == 0:
            f = (1 + g) * np.prod([np.cos((x[j]**alpha) * np.pi / 2) for j in range(n_obj - 1)])
        elif m < n_obj - 1:
            f = (1 + g) * np.prod([np.cos((x[j]**alpha) * np.pi / 2) for j in range(n_obj - m - 1)])
            f *= np.sin((x[n_obj - m - 1]**alpha) * np.pi / 2)
        else:
            f = (1 + g) * np.sin((x[0]**alpha) * np.pi / 2)
        
        objectives.append(f)
    
    return np.array(objectives)

def dtlz7(x: np.ndarray, n_obj: int = 3) -> np.ndarray:
    """
    DTLZ7 - Disconnected Pareto front
    
    Most challenging DTLZ function:
    - 2^(M-1) disconnected Pareto regions
    - Mixed shape/convergence difficulty
    - Requires good diversity maintenance
    
    Dimensions: n_obj + k - 1
    Objectives: n_obj
    """
    k = len(x) - n_obj + 1
    g = 1 + (9.0 / k) * np.sum(x[n_obj - 1:])
    
    objectives = []
    for m in range(n_obj - 1):
        objectives.append(x[m])
    
    # Last objective with g function
    h = n_obj - np.sum([
        (objectives[j] / (1 + g)) * (1 + np.sin(3 * np.pi * objectives[j]))
        for j in range(n_obj - 1)
    ])
    
    objectives.append((1 + g) * h)
    
    return np.array(objectives)

# ============================================================================
# WFG SUITE - Walking Fish Group benchmarks
# ============================================================================

def wfg1(x: np.ndarray, n_obj: int = 2) -> np.ndarray:
    """
    WFG1 - Highly biased, mixed Pareto front geometry
    
    Challenges:
    - Highly biased search space
    - Mixed convex/concave regions
    - Difficult convergence
    
    Dimensions: k + l (k = n_obj - 1, l = 10)
    Objectives: n_obj
    """
    k = n_obj - 1
    l = len(x) - k
    
    # Shape functions (simplified)
    t = x.copy()
    
    # Simple transformation
    for i in range(len(t)):
        t[i] = t[i] / (i + 1)
    
    # Calculate objectives
    objectives = []
    for m in range(1, n_obj + 1):
        if m == n_obj:
            f = t[-1]
        else:
            f = t[m - 1]
        objectives.append(f)
    
    return np.array(objectives)

# ============================================================================
# DIFFICULT VARIANTS
# ============================================================================

def fonseca_fleming(x: np.ndarray) -> np.ndarray:
    """
    Fonseca-Fleming - Classic 2-objective function
    
    Challenges:
    - Concave Pareto front
    - Good diversity test
    
    Dimensions: n (typically 3)
    Objectives: 2
    """
    n = len(x)
    
    f1 = 1 - np.exp(-np.sum((x - 1/np.sqrt(n))**2))
    f2 = 1 - np.exp(-np.sum((x + 1/np.sqrt(n))**2))
    
    return np.array([f1, f2])

def kursawe(x: np.ndarray) -> np.ndarray:
    """
    Kursawe - Non-convex, disconnected Pareto front
    
    Challenges:
    - Highly non-convex
    - Disconnected Pareto regions
    - Requires good diversity
    
    Dimensions: 3
    Objectives: 2
    """
    n = len(x)
    
    f1 = np.sum([
        -10 * np.exp(-0.2 * np.sqrt(x[i]**2 + x[i+1]**2))
        for i in range(n - 1)
    ])
    
    f2 = np.sum([
        np.abs(x[i])**0.8 + 5 * np.sin(x[i]**3)
        for i in range(n)
    ])
    
    return np.array([f1, f2])

def viennet(x: np.ndarray) -> np.ndarray:
    """
    Viennet - Three objectives, complex geometry
    
    Challenges:
    - Three conflicting objectives
    - Complex Pareto surface
    - Non-convex regions
    
    Dimensions: 2
    Objectives: 3
    """
    x1, x2 = x[0], x[1]
    
    f1 = 0.5 * (x1**2 + x2**2) + np.sin(x1**2 + x2**2)
    f2 = ((3*x1 - 2*x2 + 4)**2 / 8) + ((x1 - x2 + 1)**2 / 27) + 15
    f3 = 1 / (x1**2 + x2**2 + 1) - 1.1 * np.exp(-(x1**2 + x2**2))
    
    return np.array([f1, f2, f3])

# ============================================================================
# EXTREME CHALLENGES - Many objectives
# ============================================================================

def dtlz5_degenerate(x: np.ndarray, n_obj: int = 3) -> np.ndarray:
    """
    DTLZ5 - Degenerate Pareto front
    
    Challenges:
    - Pareto front is a curve in M-dimensional space (not a surface)
    - Tests algorithm's ability to handle degenerate cases
    - Difficult to maintain diversity
    
    Dimensions: n_obj + k - 1
    Objectives: n_obj
    """
    k = len(x) - n_obj + 1
    g = np.sum([(x[i] - 0.5)**2 for i in range(n_obj - 1, len(x))])
    
    # Adjust positions
    theta = []
    theta.append(x[0] * np.pi / 2)
    
    for i in range(1, n_obj - 1):
        theta_i = (1 + 2 * g * x[i]) / (2 * (1 + g)) * np.pi / 2
        theta.append(theta_i)
    
    objectives = []
    for m in range(n_obj):
        if m == 0:
            f = (1 + g) * np.prod([np.cos(theta[j]) for j in range(n_obj - 1)])
        elif m < n_obj - 1:
            f = (1 + g) * np.prod([np.cos(theta[j]) for j in range(n_obj - m - 1)])
            f *= np.sin(theta[n_obj - m - 1])
        else:
            f = (1 + g) * np.sin(theta[0])
        
        objectives.append(f)
    
    return np.array(objectives)

def many_objective_dtlz2(x: np.ndarray, n_obj: int = 5) -> np.ndarray:
    """
    Many-Objective DTLZ2 - 5+ objectives
    
    Challenges:
    - High-dimensional objective space
    - Difficult visualization
    - Curse of dimensionality
    - Balance exploration/exploitation
    
    Dimensions: n_obj + 9
    Objectives: n_obj (5-10)
    """
    k = len(x) - n_obj + 1
    g = np.sum([(x[i] - 0.5)**2 for i in range(n_obj - 1, len(x))])
    
    objectives = []
    for m in range(n_obj):
        if m == 0:
            f = (1 + g) * np.prod([np.cos(x[j] * np.pi / 2) for j in range(n_obj - 1)])
        elif m < n_obj - 1:
            f = (1 + g) * np.prod([np.cos(x[j] * np.pi / 2) for j in range(n_obj - m - 1)])
            f *= np.sin(x[n_obj - m - 1] * np.pi / 2)
        else:
            f = (1 + g) * np.sin(x[0] * np.pi / 2)
        
        objectives.append(f)
    
    return np.array(objectives)

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def inverted_generational_distance(obtained_front: np.ndarray, 
                                   true_front: np.ndarray) -> float:
    """
    Inverted Generational Distance (IGD)
    
    Measures convergence and diversity of obtained solutions.
    Lower is better.
    
    Parameters:
    -----------
    obtained_front : np.ndarray
        Objectives of obtained Pareto front (N x M)
    true_front : np.ndarray
        True Pareto front objectives (N_true x M)
    
    Returns:
    --------
    float : IGD value
    """
    if len(obtained_front) == 0 or len(true_front) == 0:
        return float('inf')
    
    # For each point in true front, find minimum distance to obtained front
    distances = []
    for true_point in true_front:
        min_dist = float('inf')
        for obtained_point in obtained_front:
            dist = np.linalg.norm(true_point - obtained_point)
            min_dist = min(min_dist, dist)
        distances.append(min_dist)
    
    return np.mean(distances)

def spacing_metric(front: np.ndarray) -> float:
    """
    Spacing (SP) - Measures uniformity of solution distribution
    
    Lower is better (more uniform distribution).
    
    Parameters:
    -----------
    front : np.ndarray
        Objectives of Pareto front (N x M)
    
    Returns:
    --------
    float : Spacing value
    """
    if len(front) < 2:
        return 0.0
    
    # Calculate distances to nearest neighbor
    distances = []
    for i, point in enumerate(front):
        min_dist = float('inf')
        for j, other_point in enumerate(front):
            if i != j:
                dist = np.linalg.norm(point - other_point)
                min_dist = min(min_dist, dist)
        distances.append(min_dist)
    
    # Calculate spacing metric
    d_mean = np.mean(distances)
    sp = np.sqrt(np.mean([(d - d_mean)**2 for d in distances]))
    
    return sp

def hypervolume_wfg(front: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Hypervolume using WFG algorithm (simplified)
    
    Measures volume of objective space dominated by Pareto front.
    Higher is better.
    
    Parameters:
    -----------
    front : np.ndarray
        Objectives of Pareto front (N x M)
    reference_point : np.ndarray
        Reference point (worst acceptable values)
    
    Returns:
    --------
    float : Hypervolume value
    """
    if len(front) == 0:
        return 0.0
    
    # Simplified hypervolume (exact calculation is NP-hard for M>3)
    n_obj = front.shape[1]
    
    # For 2D, use exact calculation
    if n_obj == 2:
        # Sort by first objective
        sorted_front = front[front[:, 0].argsort()]
        
        hv = 0.0
        for i in range(len(sorted_front)):
            if i == 0:
                width = sorted_front[i, 0] - 0
            else:
                width = sorted_front[i, 0] - sorted_front[i-1, 0]
            
            height = reference_point[1] - sorted_front[i, 1]
            hv += width * height
        
        return hv
    
    # For higher dimensions, use bounding box approximation
    dominated_volume = 1.0
    for obj_idx in range(n_obj):
        dominated_volume *= (reference_point[obj_idx] - np.min(front[:, obj_idx]))
    
    return dominated_volume

def spread_metric(front: np.ndarray) -> float:
    """
    Spread (Δ) - Measures extent and distribution
    
    Combines boundary coverage and distribution uniformity.
    Lower is better.
    
    Parameters:
    -----------
    front : np.ndarray
        Objectives of Pareto front (N x M)
    
    Returns:
    --------
    float : Spread value
    """
    if len(front) < 3:
        return float('inf')
    
    n_obj = front.shape[1]
    
    # Calculate distances between consecutive solutions
    distances = []
    for i in range(len(front) - 1):
        dist = np.linalg.norm(front[i+1] - front[i])
        distances.append(dist)
    
    d_mean = np.mean(distances)
    
    # Extreme points (simplified)
    d_f = 0.0
    d_l = 0.0
    
    # Calculate spread
    numerator = d_f + d_l + np.sum([abs(d - d_mean) for d in distances])
    denominator = d_f + d_l + len(distances) * d_mean
    
    if denominator == 0:
        return float('inf')
    
    return numerator / denominator

# ============================================================================
# TRUE PARETO FRONTS (for IGD calculation)
# ============================================================================

def generate_true_pareto_front(function_name: str, n_points: int = 100) -> np.ndarray:
    """
    Generate true Pareto front for benchmark functions
    
    Parameters:
    -----------
    function_name : str
        Name of benchmark function
    n_points : int
        Number of points on true front
    
    Returns:
    --------
    np.ndarray : True Pareto front objectives
    """
    if function_name == 'zdt1':
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        return np.column_stack([f1, f2])
    
    elif function_name == 'zdt2':
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - f1**2
        return np.column_stack([f1, f2])
    
    elif function_name == 'zdt3':
        # Discontinuous front
        segments = []
        for segment in np.linspace(0, 1, 5):
            f1 = np.linspace(segment, segment + 0.15, n_points // 5)
            f1 = f1[f1 <= 1]
            f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
            segments.append(np.column_stack([f1, f2]))
        return np.vstack(segments)
    
    elif function_name == 'dtlz2':
        # Sphere in objective space
        theta = np.linspace(0, np.pi/2, n_points)
        f1 = np.cos(theta)
        f2 = np.sin(theta)
        f3 = np.zeros_like(theta)
        return np.column_stack([f1, f2, f3])
    
    else:
        raise ValueError(f"True Pareto front not implemented for {function_name}")

# ============================================================================
# CHALLENGE RATINGS
# ============================================================================

BENCHMARK_DIFFICULTY = {
    'zdt1': {'difficulty': 1, 'challenges': ['convex', 'continuous']},
    'zdt2': {'difficulty': 2, 'challenges': ['non-convex', 'continuous']},
    'zdt3': {'difficulty': 3, 'challenges': ['discontinuous', 'multi-modal']},
    'zdt4': {'difficulty': 5, 'challenges': ['21^9 local fronts', 'highly multimodal']},
    'zdt6': {'difficulty': 4, 'challenges': ['non-uniform', 'biased', 'low density']},
    'dtlz1': {'difficulty': 3, 'challenges': ['linear', 'multimodal']},
    'dtlz2': {'difficulty': 2, 'challenges': ['concave', 'spherical']},
    'dtlz3': {'difficulty': 5, 'challenges': ['3^k local fronts', 'highly multimodal']},
    'dtlz4': {'difficulty': 4, 'challenges': ['biased density', 'parameter sensitive']},
    'dtlz5': {'difficulty': 4, 'challenges': ['degenerate', 'curve not surface']},
    'dtlz7': {'difficulty': 5, 'challenges': ['2^(M-1) disconnected regions', 'mixed']},
    'fonseca_fleming': {'difficulty': 2, 'challenges': ['concave']},
    'kursawe': {'difficulty': 4, 'challenges': ['non-convex', 'disconnected']},
    'viennet': {'difficulty': 3, 'challenges': ['three objectives', 'complex']},
    'wfg1': {'difficulty': 4, 'challenges': ['biased', 'mixed geometry']},
    'many_objective': {'difficulty': 5, 'challenges': ['curse of dimensionality', '5+ objectives']},
}

def get_benchmark_info(function_name: str) -> dict:
    """Get difficulty rating and challenges for a benchmark function"""
    return BENCHMARK_DIFFICULTY.get(function_name, 
                                   {'difficulty': 0, 'challenges': ['unknown']})
