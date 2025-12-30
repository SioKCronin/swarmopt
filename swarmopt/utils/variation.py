"""
Variation Operators for Particle Swarm Optimization

This module implements various variation operators that can be applied
to PSO particles to improve exploration and escape local optima.
"""

import numpy as np
import inspect
from typing import Tuple, Optional, Callable

def gaussian_variation(particle_pos: np.ndarray, variation_rate: float = 0.1, 
                     variation_strength: float = 0.1) -> np.ndarray:
    """
    Apply Gaussian variation to particle position
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    variation_rate : float
        Probability of mutating each dimension
    variation_strength : float
        Standard deviation of Gaussian noise
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    
    # Apply variation to each dimension with given probability
    for i in range(len(particle_pos)):
        if np.random.random() < variation_rate:
            noise = np.random.normal(0, variation_strength)
            mutated_pos[i] += noise
    
    return mutated_pos

def uniform_variation(particle_pos: np.ndarray, variation_rate: float = 0.1,
                    variation_range: Tuple[float, float] = (-1, 1)) -> np.ndarray:
    """
    Apply uniform variation to particle position
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    variation_rate : float
        Probability of mutating each dimension
    variation_range : Tuple[float, float]
        Range for uniform variation
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    
    for i in range(len(particle_pos)):
        if np.random.random() < variation_rate:
            variation_value = np.random.uniform(variation_range[0], variation_range[1])
            mutated_pos[i] += variation_value
    
    return mutated_pos

def polynomial_variation(particle_pos: np.ndarray, variation_rate: float = 0.1,
                       eta: float = 20.0, bounds: Tuple[float, float] = (-5, 5)) -> np.ndarray:
    """
    Apply polynomial variation (common in NSGA-II)
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    variation_rate : float
        Probability of mutating each dimension
    eta : float
        Distribution index for polynomial variation
    bounds : Tuple[float, float]
        Bounds for the search space
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    x_min, x_max = bounds
    
    for i in range(len(particle_pos)):
        if np.random.random() < variation_rate:
            u = np.random.random()
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            
            mutated_pos[i] += delta * (x_max - x_min)
            mutated_pos[i] = np.clip(mutated_pos[i], x_min, x_max)
    
    return mutated_pos

def cauchy_variation(particle_pos: np.ndarray, variation_rate: float = 0.1,
                   scale: float = 0.1) -> np.ndarray:
    """
    Apply Cauchy variation (heavy-tailed distribution)
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    variation_rate : float
        Probability of mutating each dimension
    scale : float
        Scale parameter for Cauchy distribution
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    
    for i in range(len(particle_pos)):
        if np.random.random() < variation_rate:
            noise = np.random.standard_cauchy() * scale
            mutated_pos[i] += noise
    
    return mutated_pos

def levy_variation(particle_pos: np.ndarray, variation_rate: float = 0.1,
                 alpha: float = 1.5, beta: float = 1.0) -> np.ndarray:
    """
    Apply Levy variation (Levy flight)
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    variation_rate : float
        Probability of mutating each dimension
    alpha : float
        Stability parameter (0 < alpha <= 2)
    beta : float
        Scale parameter
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    
    for i in range(len(particle_pos)):
        if np.random.random() < variation_rate:
            # Generate Levy random number
            u = np.random.normal(0, 1)
            v = np.random.normal(0, 1)
            s = u / (abs(v) ** (1 / alpha))
            levy_noise = beta * s
            mutated_pos[i] += levy_noise
    
    return mutated_pos

def adaptive_variation(particle_pos: np.ndarray, current_iter: int, max_iter: int,
                     variation_rate: float = 0.1, variation_strength: float = 0.1) -> np.ndarray:
    """
    Apply adaptive variation that decreases over time
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    current_iter : int
        Current iteration number
    max_iter : int
        Maximum number of iterations
    variation_rate : float
        Initial variation rate
    variation_strength : float
        Initial variation strength
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    # Adaptive parameters that decrease over time
    progress = current_iter / max_iter
    adaptive_rate = variation_rate * (1 - progress)
    adaptive_strength = variation_strength * (1 - progress)
    
    return gaussian_variation(particle_pos, adaptive_rate, adaptive_strength)

def chaotic_variation(particle_pos: np.ndarray, variation_rate: float = 0.1,
                    chaos_param: float = 3.9) -> np.ndarray:
    """
    Apply chaotic variation using logistic map
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    variation_rate : float
        Probability of mutating each dimension
    chaos_param : float
        Chaos parameter for logistic map
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    
    for i in range(len(particle_pos)):
        if np.random.random() < variation_rate:
            # Generate chaotic sequence
            x = np.random.random()
            for _ in range(10):  # Iterate logistic map
                x = chaos_param * x * (1 - x)
            
            # Scale to variation range
            variation_value = (x - 0.5) * 2  # Scale to [-1, 1]
            mutated_pos[i] += variation_value
    
    return mutated_pos

def differential_variation(particle_pos: np.ndarray, population: list,
                          variation_rate: float = 0.1, f: float = 0.5) -> np.ndarray:
    """
    Apply differential evolution variation
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    population : list
        Population of particles for differential variation
    variation_rate : float
        Probability of applying variation
    f : float
        Differential evolution scaling factor
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    if len(population) < 3:
        return particle_pos.copy()
    
    mutated_pos = particle_pos.copy()
    
    if np.random.random() < variation_rate:
        # Select three random particles
        candidates = [p for p in population if not np.array_equal(p, particle_pos)]
        if len(candidates) >= 3:
            x1, x2, x3 = np.random.choice(len(candidates), 3, replace=False)
            x1, x2, x3 = candidates[x1], candidates[x2], candidates[x3]
            
            # Differential variation: x1 + f * (x2 - x3)
            mutated_pos = x1 + f * (x2 - x3)
    
    return mutated_pos

def boundary_variation(particle_pos: np.ndarray, bounds: Tuple[float, float],
                     variation_rate: float = 0.1) -> np.ndarray:
    """
    Apply boundary variation (random value within bounds)
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    bounds : Tuple[float, float]
        Search space bounds
    variation_rate : float
        Probability of mutating each dimension
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    x_min, x_max = bounds
    
    for i in range(len(particle_pos)):
        if np.random.random() < variation_rate:
            mutated_pos[i] = np.random.uniform(x_min, x_max)
    
    return mutated_pos

def non_uniform_variation(particle_pos: np.ndarray, current_iter: int, max_iter: int,
                        variation_rate: float = 0.1, bounds: Tuple[float, float] = (-5, 5),
                        b: float = 2.0) -> np.ndarray:
    """
    Apply non-uniform variation that decreases over time
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    current_iter : int
        Current iteration number
    max_iter : int
        Maximum number of iterations
    variation_rate : float
        Probability of mutating each dimension
    bounds : Tuple[float, float]
        Search space bounds
    b : float
        Shape parameter for non-uniform variation
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    x_min, x_max = bounds
    
    for i in range(len(particle_pos)):
        if np.random.random() < variation_rate:
            r1 = np.random.random()
            r2 = np.random.random()
            
            if r1 < 0.5:
                delta = (2 * r2) ** (1 / (b + 1)) - 1
            else:
                delta = 1 - (2 * (1 - r2)) ** (1 / (b + 1))
            
            # Scale by iteration progress
            progress = current_iter / max_iter
            delta *= (1 - progress)
            
            mutated_pos[i] += delta * (x_max - x_min)
            mutated_pos[i] = np.clip(mutated_pos[i], x_min, x_max)
    
    return mutated_pos

# Variation strategy dispatcher
VARIATION_STRATEGIES = {
    'gaussian': gaussian_variation,
    'uniform': uniform_variation,
    'polynomial': polynomial_variation,
    'cauchy': cauchy_variation,
    'levy': levy_variation,
    'adaptive': adaptive_variation,
    'chaotic': chaotic_variation,
    'differential': differential_variation,
    'boundary': boundary_variation,
    'non_uniform': non_uniform_variation
}

def apply_variation(particle_pos: np.ndarray, strategy: str, **kwargs) -> np.ndarray:
    """
    Apply variation using specified strategy
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    strategy : str
        Variation strategy to use
    **kwargs : dict
        Additional parameters for the variation strategy
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    if strategy not in VARIATION_STRATEGIES:
        raise ValueError(f"Unknown variation strategy: {strategy}")
    
    variation_func = VARIATION_STRATEGIES[strategy]

    # Filter kwargs to only those accepted by the variation function
    sig = inspect.signature(variation_func)
    accepted = {name for name in sig.parameters if name != 'particle_pos'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}

    return variation_func(particle_pos, **filtered_kwargs)

def detect_stalled_particles(particles: list, stagnation_threshold: int = 10) -> list:
    """
    Detect particles that have stalled (not improved for several iterations)
    
    Parameters:
    -----------
    particles : list
        List of particle objects
    stagnation_threshold : int
        Number of iterations without improvement to consider stalled
        
    Returns:
    --------
    list : List of stalled particle indices
    """
    stalled_indices = []
    
    for i, particle in enumerate(particles):
        if hasattr(particle, 'stagnation_count'):
            if particle.stagnation_count >= stagnation_threshold:
                stalled_indices.append(i)
        else:
            # Initialize stagnation counter if not exists
            particle.stagnation_count = 0
    
    return stalled_indices

def detect_converged_particles(particles: list, convergence_threshold: float = 1e-6) -> list:
    """
    Detect particles that have converged to local optima
    
    Parameters:
    -----------
    particles : list
        List of particle objects
    convergence_threshold : float
        Velocity threshold below which particle is considered converged
        
    Returns:
    --------
    list : List of converged particle indices
    """
    converged_indices = []
    
    for i, particle in enumerate(particles):
        if hasattr(particle, 'velocity'):
            velocity_magnitude = np.linalg.norm(particle.velocity)
            if velocity_magnitude < convergence_threshold:
                converged_indices.append(i)
    
    return converged_indices

def escape_local_optima_variation(particle_pos: np.ndarray, bounds: Tuple[float, float],
                                escape_strength: float = 2.0) -> np.ndarray:
    """
    Strong variation to escape local optima
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    bounds : Tuple[float, float]
        Search space bounds
    escape_strength : float
        Strength of escape variation
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    x_min, x_max = bounds
    mutated_pos = particle_pos.copy()
    
    # Strong random displacement
    for i in range(len(particle_pos)):
        displacement = np.random.uniform(-escape_strength, escape_strength)
        mutated_pos[i] += displacement
        mutated_pos[i] = np.clip(mutated_pos[i], x_min, x_max)
    
    return mutated_pos

def diversity_preserving_variation(particle_pos: np.ndarray, population: list,
                                variation_rate: float = 0.3) -> np.ndarray:
    """
    Variation that preserves population diversity
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    population : list
        Population of particles
    variation_rate : float
        Probability of variation
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    if len(population) < 2:
        return particle_pos.copy()
    
    mutated_pos = particle_pos.copy()
    
    if np.random.random() < variation_rate:
        # Find the most different particle
        max_distance = 0
        most_different = None
        
        for other in population:
            if not np.array_equal(other, particle_pos):
                distance = np.linalg.norm(other - particle_pos)
                if distance > max_distance:
                    max_distance = distance
                    most_different = other
        
        if most_different is not None:
            # Move towards the most different particle
            direction = most_different - particle_pos
            mutated_pos += 0.5 * direction
    
    return mutated_pos

def restart_variation(particle_pos: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    """
    Complete restart variation (random reinitialization)
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    bounds : Tuple[float, float]
        Search space bounds
        
    Returns:
    --------
    np.ndarray : Completely new random position
    """
    x_min, x_max = bounds
    return np.random.uniform(x_min, x_max, len(particle_pos))

def adaptive_variation_strength(particle_pos: np.ndarray, current_iter: int, max_iter: int,
                              base_strength: float = 0.1, bounds: Tuple[float, float] = (-5, 5)) -> np.ndarray:
    """
    Adaptive variation that increases strength when particles are stuck
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    current_iter : int
        Current iteration number
    max_iter : int
        Maximum iterations
    base_strength : float
        Base variation strength
    bounds : Tuple[float, float]
        Search space bounds
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    # Increase variation strength as optimization progresses
    progress = current_iter / max_iter
    adaptive_strength = base_strength * (1 + 2 * progress)
    
    return gaussian_variation(particle_pos, variation_rate=0.2, variation_strength=adaptive_strength)

def opposition_based_variation(particle_pos: np.ndarray, bounds: Tuple[float, float],
                            variation_rate: float = 0.1) -> np.ndarray:
    """
    Opposition-based variation (explore opposite region)
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    bounds : Tuple[float, float]
        Search space bounds
    variation_rate : float
        Probability of applying opposition variation
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    x_min, x_max = bounds
    mutated_pos = particle_pos.copy()
    
    if np.random.random() < variation_rate:
        # Calculate opposite position
        opposite_pos = x_min + x_max - particle_pos
        mutated_pos = 0.5 * (particle_pos + opposite_pos)
    
    return mutated_pos

def hybrid_variation(particle_pos: np.ndarray, current_iter: int, max_iter: int,
                   bounds: Tuple[float, float], population: list = None) -> np.ndarray:
    """
    Hybrid variation that combines multiple strategies
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    current_iter : int
        Current iteration number
    max_iter : int
        Maximum iterations
    bounds : Tuple[float, float]
        Search space bounds
    population : list
        Population for diversity-based variations
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    progress = current_iter / max_iter
    
    # Choose variation strategy based on optimization progress
    if progress < 0.3:
        # Early stage: gentle exploration
        return gaussian_variation(particle_pos, variation_rate=0.1, variation_strength=0.05)
    elif progress < 0.7:
        # Middle stage: balanced exploration/exploitation
        if population and len(population) > 1:
            return diversity_preserving_variation(particle_pos, population, variation_rate=0.2)
        else:
            return adaptive_variation_strength(particle_pos, current_iter, max_iter)
    else:
        # Late stage: strong escape from local optima
        return escape_local_optima_variation(particle_pos, bounds, escape_strength=1.0)

# Enhanced variation strategies for local optima escape
ENHANCED_VARIATION_STRATEGIES = {
    'escape_local_optima': escape_local_optima_variation,
    'diversity_preserving': diversity_preserving_variation,
    'restart': restart_variation,
    'adaptive_strength': adaptive_variation_strength,
    'opposition_based': opposition_based_variation,
    'hybrid': hybrid_variation
}

# Update the main strategies dictionary
VARIATION_STRATEGIES.update(ENHANCED_VARIATION_STRATEGIES)
