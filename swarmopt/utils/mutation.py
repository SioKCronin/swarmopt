"""
Mutation Operators for Particle Swarm Optimization

This module implements various mutation operators that can be applied
to PSO particles to improve exploration and escape local optima.
"""

import numpy as np
from typing import Tuple, Optional, Callable

def gaussian_mutation(particle_pos: np.ndarray, mutation_rate: float = 0.1, 
                     mutation_strength: float = 0.1) -> np.ndarray:
    """
    Apply Gaussian mutation to particle position
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    mutation_rate : float
        Probability of mutating each dimension
    mutation_strength : float
        Standard deviation of Gaussian noise
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    
    # Apply mutation to each dimension with given probability
    for i in range(len(particle_pos)):
        if np.random.random() < mutation_rate:
            noise = np.random.normal(0, mutation_strength)
            mutated_pos[i] += noise
    
    return mutated_pos

def uniform_mutation(particle_pos: np.ndarray, mutation_rate: float = 0.1,
                    mutation_range: Tuple[float, float] = (-1, 1)) -> np.ndarray:
    """
    Apply uniform mutation to particle position
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    mutation_rate : float
        Probability of mutating each dimension
    mutation_range : Tuple[float, float]
        Range for uniform mutation
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    
    for i in range(len(particle_pos)):
        if np.random.random() < mutation_rate:
            mutation_value = np.random.uniform(mutation_range[0], mutation_range[1])
            mutated_pos[i] += mutation_value
    
    return mutated_pos

def polynomial_mutation(particle_pos: np.ndarray, mutation_rate: float = 0.1,
                       eta: float = 20.0, bounds: Tuple[float, float] = (-5, 5)) -> np.ndarray:
    """
    Apply polynomial mutation (common in NSGA-II)
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    mutation_rate : float
        Probability of mutating each dimension
    eta : float
        Distribution index for polynomial mutation
    bounds : Tuple[float, float]
        Bounds for the search space
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    x_min, x_max = bounds
    
    for i in range(len(particle_pos)):
        if np.random.random() < mutation_rate:
            u = np.random.random()
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            
            mutated_pos[i] += delta * (x_max - x_min)
            mutated_pos[i] = np.clip(mutated_pos[i], x_min, x_max)
    
    return mutated_pos

def cauchy_mutation(particle_pos: np.ndarray, mutation_rate: float = 0.1,
                   scale: float = 0.1) -> np.ndarray:
    """
    Apply Cauchy mutation (heavy-tailed distribution)
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    mutation_rate : float
        Probability of mutating each dimension
    scale : float
        Scale parameter for Cauchy distribution
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    
    for i in range(len(particle_pos)):
        if np.random.random() < mutation_rate:
            noise = np.random.standard_cauchy() * scale
            mutated_pos[i] += noise
    
    return mutated_pos

def levy_mutation(particle_pos: np.ndarray, mutation_rate: float = 0.1,
                 alpha: float = 1.5, beta: float = 1.0) -> np.ndarray:
    """
    Apply Levy mutation (Levy flight)
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    mutation_rate : float
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
        if np.random.random() < mutation_rate:
            # Generate Levy random number
            u = np.random.normal(0, 1)
            v = np.random.normal(0, 1)
            s = u / (abs(v) ** (1 / alpha))
            levy_noise = beta * s
            mutated_pos[i] += levy_noise
    
    return mutated_pos

def adaptive_mutation(particle_pos: np.ndarray, current_iter: int, max_iter: int,
                     mutation_rate: float = 0.1, mutation_strength: float = 0.1) -> np.ndarray:
    """
    Apply adaptive mutation that decreases over time
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    current_iter : int
        Current iteration number
    max_iter : int
        Maximum number of iterations
    mutation_rate : float
        Initial mutation rate
    mutation_strength : float
        Initial mutation strength
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    # Adaptive parameters that decrease over time
    progress = current_iter / max_iter
    adaptive_rate = mutation_rate * (1 - progress)
    adaptive_strength = mutation_strength * (1 - progress)
    
    return gaussian_mutation(particle_pos, adaptive_rate, adaptive_strength)

def chaotic_mutation(particle_pos: np.ndarray, mutation_rate: float = 0.1,
                    chaos_param: float = 3.9) -> np.ndarray:
    """
    Apply chaotic mutation using logistic map
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    mutation_rate : float
        Probability of mutating each dimension
    chaos_param : float
        Chaos parameter for logistic map
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    
    for i in range(len(particle_pos)):
        if np.random.random() < mutation_rate:
            # Generate chaotic sequence
            x = np.random.random()
            for _ in range(10):  # Iterate logistic map
                x = chaos_param * x * (1 - x)
            
            # Scale to mutation range
            mutation_value = (x - 0.5) * 2  # Scale to [-1, 1]
            mutated_pos[i] += mutation_value
    
    return mutated_pos

def differential_mutation(particle_pos: np.ndarray, population: list,
                          mutation_rate: float = 0.1, f: float = 0.5) -> np.ndarray:
    """
    Apply differential evolution mutation
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    population : list
        Population of particles for differential mutation
    mutation_rate : float
        Probability of applying mutation
    f : float
        Differential evolution scaling factor
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    if len(population) < 3:
        return particle_pos.copy()
    
    mutated_pos = particle_pos.copy()
    
    if np.random.random() < mutation_rate:
        # Select three random particles
        candidates = [p for p in population if not np.array_equal(p, particle_pos)]
        if len(candidates) >= 3:
            x1, x2, x3 = np.random.choice(len(candidates), 3, replace=False)
            x1, x2, x3 = candidates[x1], candidates[x2], candidates[x3]
            
            # Differential mutation: x1 + f * (x2 - x3)
            mutated_pos = x1 + f * (x2 - x3)
    
    return mutated_pos

def boundary_mutation(particle_pos: np.ndarray, bounds: Tuple[float, float],
                     mutation_rate: float = 0.1) -> np.ndarray:
    """
    Apply boundary mutation (random value within bounds)
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    bounds : Tuple[float, float]
        Search space bounds
    mutation_rate : float
        Probability of mutating each dimension
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    x_min, x_max = bounds
    
    for i in range(len(particle_pos)):
        if np.random.random() < mutation_rate:
            mutated_pos[i] = np.random.uniform(x_min, x_max)
    
    return mutated_pos

def non_uniform_mutation(particle_pos: np.ndarray, current_iter: int, max_iter: int,
                        mutation_rate: float = 0.1, bounds: Tuple[float, float] = (-5, 5),
                        b: float = 2.0) -> np.ndarray:
    """
    Apply non-uniform mutation that decreases over time
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    current_iter : int
        Current iteration number
    max_iter : int
        Maximum number of iterations
    mutation_rate : float
        Probability of mutating each dimension
    bounds : Tuple[float, float]
        Search space bounds
    b : float
        Shape parameter for non-uniform mutation
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    mutated_pos = particle_pos.copy()
    x_min, x_max = bounds
    
    for i in range(len(particle_pos)):
        if np.random.random() < mutation_rate:
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

# Mutation strategy dispatcher
MUTATION_STRATEGIES = {
    'gaussian': gaussian_mutation,
    'uniform': uniform_mutation,
    'polynomial': polynomial_mutation,
    'cauchy': cauchy_mutation,
    'levy': levy_mutation,
    'adaptive': adaptive_mutation,
    'chaotic': chaotic_mutation,
    'differential': differential_mutation,
    'boundary': boundary_mutation,
    'non_uniform': non_uniform_mutation
}

def apply_mutation(particle_pos: np.ndarray, strategy: str, **kwargs) -> np.ndarray:
    """
    Apply mutation using specified strategy
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    strategy : str
        Mutation strategy to use
    **kwargs : dict
        Additional parameters for the mutation strategy
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    if strategy not in MUTATION_STRATEGIES:
        raise ValueError(f"Unknown mutation strategy: {strategy}")
    
    mutation_func = MUTATION_STRATEGIES[strategy]
    return mutation_func(particle_pos, **kwargs)

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

def escape_local_optima_mutation(particle_pos: np.ndarray, bounds: Tuple[float, float],
                                escape_strength: float = 2.0) -> np.ndarray:
    """
    Strong mutation to escape local optima
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    bounds : Tuple[float, float]
        Search space bounds
    escape_strength : float
        Strength of escape mutation
        
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

def diversity_preserving_mutation(particle_pos: np.ndarray, population: list,
                                mutation_rate: float = 0.3) -> np.ndarray:
    """
    Mutation that preserves population diversity
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    population : list
        Population of particles
    mutation_rate : float
        Probability of mutation
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    if len(population) < 2:
        return particle_pos.copy()
    
    mutated_pos = particle_pos.copy()
    
    if np.random.random() < mutation_rate:
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

def restart_mutation(particle_pos: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    """
    Complete restart mutation (random reinitialization)
    
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

def adaptive_mutation_strength(particle_pos: np.ndarray, current_iter: int, max_iter: int,
                              base_strength: float = 0.1, bounds: Tuple[float, float] = (-5, 5)) -> np.ndarray:
    """
    Adaptive mutation that increases strength when particles are stuck
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    current_iter : int
        Current iteration number
    max_iter : int
        Maximum iterations
    base_strength : float
        Base mutation strength
    bounds : Tuple[float, float]
        Search space bounds
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    # Increase mutation strength as optimization progresses
    progress = current_iter / max_iter
    adaptive_strength = base_strength * (1 + 2 * progress)
    
    return gaussian_mutation(particle_pos, mutation_rate=0.2, mutation_strength=adaptive_strength)

def opposition_based_mutation(particle_pos: np.ndarray, bounds: Tuple[float, float],
                            mutation_rate: float = 0.1) -> np.ndarray:
    """
    Opposition-based mutation (explore opposite region)
    
    Parameters:
    -----------
    particle_pos : np.ndarray
        Current particle position
    bounds : Tuple[float, float]
        Search space bounds
    mutation_rate : float
        Probability of applying opposition mutation
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    x_min, x_max = bounds
    mutated_pos = particle_pos.copy()
    
    if np.random.random() < mutation_rate:
        # Calculate opposite position
        opposite_pos = x_min + x_max - particle_pos
        mutated_pos = 0.5 * (particle_pos + opposite_pos)
    
    return mutated_pos

def hybrid_mutation(particle_pos: np.ndarray, current_iter: int, max_iter: int,
                   bounds: Tuple[float, float], population: list = None) -> np.ndarray:
    """
    Hybrid mutation that combines multiple strategies
    
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
        Population for diversity-based mutations
        
    Returns:
    --------
    np.ndarray : Mutated position
    """
    progress = current_iter / max_iter
    
    # Choose mutation strategy based on optimization progress
    if progress < 0.3:
        # Early stage: gentle exploration
        return gaussian_mutation(particle_pos, mutation_rate=0.1, mutation_strength=0.05)
    elif progress < 0.7:
        # Middle stage: balanced exploration/exploitation
        if population and len(population) > 1:
            return diversity_preserving_mutation(particle_pos, population, mutation_rate=0.2)
        else:
            return adaptive_mutation_strength(particle_pos, current_iter, max_iter)
    else:
        # Late stage: strong escape from local optima
        return escape_local_optima_mutation(particle_pos, bounds, escape_strength=1.0)

# Enhanced mutation strategies for local optima escape
ENHANCED_MUTATION_STRATEGIES = {
    'escape_local_optima': escape_local_optima_mutation,
    'diversity_preserving': diversity_preserving_mutation,
    'restart': restart_mutation,
    'adaptive_strength': adaptive_mutation_strength,
    'opposition_based': opposition_based_mutation,
    'hybrid': hybrid_mutation
}

# Update the main strategies dictionary
MUTATION_STRATEGIES.update(ENHANCED_MUTATION_STRATEGIES)
