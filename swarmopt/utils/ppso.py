"""
Proactive Particle Swarm Optimization (PPSO)

This module implements PPSO with mixed swarms containing both reactive and proactive particles.
Proactive particles use knowledge gain metrics to explore regions with low sample density,
inspired by Gaussian Process acquisition functions.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
import warnings

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback implementations
    def pdist(X, metric='euclidean'):
        """Fallback pdist implementation"""
        n = len(X)
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(X[i] - X[j])
                distances.append(dist)
        return np.array(distances)
    
    def squareform(distances):
        """Fallback squareform implementation"""
        n = int((1 + np.sqrt(1 + 8 * len(distances))) // 2)
        matrix = np.zeros((n, n))
        k = 0
        for i in range(n):
            for j in range(i+1, n):
                matrix[i, j] = distances[k]
                matrix[j, i] = distances[k]
                k += 1
        return matrix

class KnowledgeGainCalculator:
    """
    Calculate knowledge gain metrics for proactive particles
    """
    
    def __init__(self, method: str = 'gaussian', kernel_width: float = 1.0):
        self.method = method
        self.kernel_width = kernel_width
        self.sample_history = []
        self.fitness_history = []
    
    def add_sample(self, position: np.ndarray, fitness: float):
        """Add a new sample to the history"""
        self.sample_history.append(position.copy())
        self.fitness_history.append(fitness)
    
    def calculate_knowledge_gain(self, position: np.ndarray) -> float:
        """
        Calculate knowledge gain at a given position
        
        Parameters:
        -----------
        position : np.ndarray
            Position to evaluate knowledge gain for
            
        Returns:
        --------
        float : Knowledge gain value
        """
        if len(self.sample_history) < 2:
            return 1.0  # Maximum knowledge gain for unexplored regions
        
        if self.method == 'gaussian':
            return self._gaussian_knowledge_gain(position)
        elif self.method == 'inverse_distance':
            return self._inverse_distance_knowledge_gain(position)
        elif self.method == 'entropy':
            return self._entropy_knowledge_gain(position)
        elif self.method == 'acquisition':
            return self._acquisition_knowledge_gain(position)
        else:
            raise ValueError(f"Unknown knowledge gain method: {self.method}")
    
    def _gaussian_knowledge_gain(self, position: np.ndarray) -> float:
        """Gaussian Process-inspired knowledge gain"""
        if len(self.sample_history) == 0:
            return 1.0
        
        # Calculate distances to all previous samples
        distances = [np.linalg.norm(position - sample) for sample in self.sample_history]
        min_distance = min(distances)
        
        # Gaussian kernel: higher gain for larger distances
        knowledge_gain = np.exp(-(min_distance ** 2) / (2 * self.kernel_width ** 2))
        
        # Normalize by the number of samples (more samples = less gain)
        sample_density = len(self.sample_history) / (4 * np.pi * self.kernel_width ** 2)
        knowledge_gain *= (1.0 / (1.0 + sample_density))
        
        return knowledge_gain
    
    def _inverse_distance_knowledge_gain(self, position: np.ndarray) -> float:
        """Inverse distance-based knowledge gain"""
        if len(self.sample_history) == 0:
            return 1.0
        
        distances = [np.linalg.norm(position - sample) for sample in self.sample_history]
        min_distance = min(distances)
        
        # Avoid division by zero
        if min_distance < 1e-10:
            return 0.0
        
        # Inverse distance with normalization
        knowledge_gain = 1.0 / (1.0 + min_distance)
        return knowledge_gain
    
    def _entropy_knowledge_gain(self, position: np.ndarray) -> float:
        """Entropy-based knowledge gain"""
        if len(self.sample_history) < 2:
            return 1.0
        
        # Calculate entropy of the region around the position
        distances = [np.linalg.norm(position - sample) for sample in self.sample_history]
        
        # Create a histogram of distances
        hist, _ = np.histogram(distances, bins=10)
        hist = hist[hist > 0]  # Remove zero counts
        
        if len(hist) == 0:
            return 1.0
        
        # Calculate entropy
        probs = hist / np.sum(hist)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Higher entropy = more uncertainty = higher knowledge gain
        max_entropy = np.log2(len(hist))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def _acquisition_knowledge_gain(self, position: np.ndarray) -> float:
        """Gaussian Process acquisition function-inspired knowledge gain"""
        if len(self.sample_history) < 2:
            return 1.0
        
        # Simple GP mean and variance approximation
        distances = [np.linalg.norm(position - sample) for sample in self.sample_history]
        fitnesses = self.fitness_history
        
        # Weighted mean based on distance
        weights = np.exp(-np.array(distances) / self.kernel_width)
        weights = weights / (np.sum(weights) + 1e-10)
        
        mean_prediction = np.sum(weights * fitnesses)
        
        # Variance approximation (higher for unexplored regions)
        min_distance = min(distances)
        variance = np.exp(-min_distance / self.kernel_width)
        
        # Expected Improvement-like acquisition
        best_fitness = min(fitnesses)
        improvement = best_fitness - mean_prediction
        
        # Knowledge gain is higher for regions with high uncertainty and potential improvement
        knowledge_gain = variance * max(0, improvement)
        
        return knowledge_gain

class ProactiveParticle:
    """
    Proactive particle that uses knowledge gain for exploration
    """
    
    def __init__(self, position: np.ndarray, velocity: np.ndarray, 
                 obj_func: Callable, bounds: Tuple[float, float],
                 knowledge_calculator: KnowledgeGainCalculator,
                 exploration_weight: float = 0.5):
        self.pos = position.copy()
        self.velocity = velocity.copy()
        self.obj_func = obj_func
        self.bounds = bounds
        self.knowledge_calculator = knowledge_calculator
        self.exploration_weight = exploration_weight
        
        # Traditional PSO attributes
        self.best_pos = position.copy()
        self.best_cost = obj_func(position)
        self.cost = self.best_cost
        
        # Proactive-specific attributes
        self.knowledge_gain = 0.0
        self.exploration_direction = np.zeros_like(position)
        self.adaptation_rate = 0.1
        
    def update(self, global_best_pos: np.ndarray, global_best_cost: float,
               c1: float, c2: float, w: float, current_iter: int, max_iter: int):
        """
        Update proactive particle using knowledge gain
        """
        # Calculate knowledge gain at current position
        self.knowledge_gain = self.knowledge_calculator.calculate_knowledge_gain(self.pos)
        
        # Update exploration direction based on knowledge gain
        self._update_exploration_direction()
        
        # Adaptive exploration weight based on progress
        adaptive_exploration = self._calculate_adaptive_exploration_weight(current_iter, max_iter)
        
        # Combine traditional PSO with proactive exploration
        cognitive_component = c1 * np.random.random() * (self.best_pos - self.pos)
        social_component = c2 * np.random.random() * (global_best_pos - self.pos)
        exploration_component = adaptive_exploration * self.exploration_direction
        
        # Update velocity
        self.velocity = (w * self.velocity + 
                        cognitive_component + 
                        social_component + 
                        exploration_component)
        
        # Update position
        self.pos += self.velocity
        
        # Apply bounds
        self.pos = np.clip(self.pos, self.bounds[0], self.bounds[1])
        
        # Evaluate fitness
        self.cost = self.obj_func(self.pos)
        
        # Update personal best
        if self.cost < self.best_cost:
            self.best_cost = self.cost
            self.best_pos = self.pos.copy()
        
        # Add sample to knowledge calculator
        self.knowledge_calculator.add_sample(self.pos, self.cost)
    
    def _update_exploration_direction(self):
        """Update exploration direction based on knowledge gain"""
        # Sample random directions
        n_directions = 10
        directions = np.random.randn(n_directions, len(self.pos))
        directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10)
        
        # Calculate knowledge gain for each direction
        step_size = 0.1
        knowledge_gains = []
        
        for direction in directions:
            test_position = self.pos + step_size * direction
            test_position = np.clip(test_position, self.bounds[0], self.bounds[1])
            kg = self.knowledge_calculator.calculate_knowledge_gain(test_position)
            knowledge_gains.append(kg)
        
        # Choose direction with highest knowledge gain
        best_idx = np.argmax(knowledge_gains)
        self.exploration_direction = directions[best_idx] * self.knowledge_gain
    
    def _calculate_adaptive_exploration_weight(self, current_iter: int, max_iter: int) -> float:
        """Calculate adaptive exploration weight"""
        # Start with high exploration, decrease over time
        progress = current_iter / max_iter
        base_exploration = self.exploration_weight * (1.0 - progress)
        
        # Increase exploration if knowledge gain is high
        knowledge_bonus = self.knowledge_gain * 0.5
        
        # Increase exploration if particle is stuck
        if hasattr(self, 'stagnation_count') and self.stagnation_count > 5:
            stagnation_bonus = 0.3
        else:
            stagnation_bonus = 0.0
        
        return base_exploration + knowledge_bonus + stagnation_bonus

class ReactiveParticle:
    """
    Traditional reactive particle for comparison
    """
    
    def __init__(self, position: np.ndarray, velocity: np.ndarray, 
                 obj_func: Callable, bounds: Tuple[float, float]):
        self.pos = position.copy()
        self.velocity = velocity.copy()
        self.obj_func = obj_func
        self.bounds = bounds
        
        self.best_pos = position.copy()
        self.best_cost = obj_func(position)
        self.cost = self.best_cost
    
    def update(self, global_best_pos: np.ndarray, global_best_cost: float,
               c1: float, c2: float, w: float, current_iter: int, max_iter: int):
        """Traditional PSO update"""
        cognitive_component = c1 * np.random.random() * (self.best_pos - self.pos)
        social_component = c2 * np.random.random() * (global_best_pos - self.pos)
        
        self.velocity = w * self.velocity + cognitive_component + social_component
        self.pos += self.velocity
        self.pos = np.clip(self.pos, self.bounds[0], self.bounds[1])
        
        self.cost = self.obj_func(self.pos)
        
        if self.cost < self.best_cost:
            self.best_cost = self.cost
            self.best_pos = self.pos.copy()

class PPSO:
    """
    Proactive Particle Swarm Optimization with mixed swarms
    """
    
    def __init__(self, n_particles: int, dims: int, obj_func: Callable,
                 bounds: Tuple[float, float] = (-5, 5),
                 proactive_ratio: float = 0.25,
                 knowledge_method: str = 'gaussian',
                 exploration_weight: float = 0.5,
                 c1: float = 2.0, c2: float = 2.0, w: float = 0.9,
                 epochs: int = 100):
        """
        Initialize PPSO
        
        Parameters:
        -----------
        n_particles : int
            Total number of particles
        dims : int
            Problem dimensions
        obj_func : Callable
            Objective function
        bounds : Tuple[float, float]
            Search space bounds
        proactive_ratio : float
            Ratio of proactive particles (0.2-0.3 recommended)
        knowledge_method : str
            Knowledge gain calculation method
        exploration_weight : float
            Weight for exploration component
        c1, c2, w : float
            PSO parameters
        epochs : int
            Number of iterations
        """
        self.n_particles = n_particles
        self.dims = dims
        self.obj_func = obj_func
        self.bounds = bounds
        self.proactive_ratio = proactive_ratio
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.epochs = epochs
        
        # Calculate number of proactive particles
        self.n_proactive = int(n_particles * proactive_ratio)
        self.n_reactive = n_particles - self.n_proactive
        
        # Initialize knowledge calculator
        self.knowledge_calculator = KnowledgeGainCalculator(method=knowledge_method)
        
        # Initialize particles
        self.particles = []
        self._initialize_particles()
        
        # Best solution tracking
        self.global_best_pos = None
        self.global_best_cost = float('inf')
        self.best_particle_type = None
        
        # Statistics
        self.knowledge_gain_history = []
        self.exploration_history = []
        
    def _initialize_particles(self):
        """Initialize mixed swarm of proactive and reactive particles"""
        # Initialize positions randomly
        positions = np.random.uniform(
            self.bounds[0], self.bounds[1], 
            (self.n_particles, self.dims)
        )
        
        # Initialize velocities
        velocities = np.random.uniform(
            -abs(self.bounds[1] - self.bounds[0]), 
            abs(self.bounds[1] - self.bounds[0]), 
            (self.n_particles, self.dims)
        )
        
        # Create proactive particles
        for i in range(self.n_proactive):
            particle = ProactiveParticle(
                positions[i], velocities[i], self.obj_func, self.bounds,
                self.knowledge_calculator, exploration_weight=0.5
            )
            self.particles.append(particle)
        
        # Create reactive particles
        for i in range(self.n_proactive, self.n_particles):
            particle = ReactiveParticle(
                positions[i], velocities[i], self.obj_func, self.bounds
            )
            self.particles.append(particle)
    
    def optimize(self) -> Dict:
        """
        Run PPSO optimization
        
        Returns:
        --------
        Dict : Optimization results
        """
        import time
        start_time = time.time()
        
        # Initialize global best
        self._update_global_best()
        
        for iteration in range(self.epochs):
            # Update all particles
            for particle in self.particles:
                particle.update(
                    self.global_best_pos, self.global_best_cost,
                    self.c1, self.c2, self.w, iteration, self.epochs
                )
            
            # Update global best
            self._update_global_best()
            
            # Track statistics
            self._track_statistics(iteration)
        
        end_time = time.time()
        
        return {
            'best_pos': self.global_best_pos,
            'best_cost': self.global_best_cost,
            'best_particle_type': self.best_particle_type,
            'runtime': end_time - start_time,
            'knowledge_gain_history': self.knowledge_gain_history,
            'exploration_history': self.exploration_history,
            'n_proactive': self.n_proactive,
            'n_reactive': self.n_reactive
        }
    
    def _update_global_best(self):
        """Update global best solution"""
        for i, particle in enumerate(self.particles):
            if particle.best_cost < self.global_best_cost:
                self.global_best_cost = particle.best_cost
                self.global_best_pos = particle.best_pos.copy()
                self.best_particle_type = 'proactive' if i < self.n_proactive else 'reactive'
    
    def _track_statistics(self, iteration: int):
        """Track optimization statistics"""
        # Calculate average knowledge gain for proactive particles
        proactive_particles = self.particles[:self.n_proactive]
        if proactive_particles:
            avg_knowledge_gain = np.mean([p.knowledge_gain for p in proactive_particles])
            self.knowledge_gain_history.append(avg_knowledge_gain)
        
        # Calculate exploration activity
        exploration_activity = np.mean([
            np.linalg.norm(p.velocity) for p in self.particles
        ])
        self.exploration_history.append(exploration_activity)
    
    def get_statistics(self) -> Dict:
        """Get optimization statistics"""
        return {
            'final_knowledge_gain': self.knowledge_gain_history[-1] if self.knowledge_gain_history else 0,
            'avg_knowledge_gain': np.mean(self.knowledge_gain_history) if self.knowledge_gain_history else 0,
            'final_exploration': self.exploration_history[-1] if self.exploration_history else 0,
            'avg_exploration': np.mean(self.exploration_history) if self.exploration_history else 0,
            'proactive_contribution': self._calculate_proactive_contribution()
        }
    
    def _calculate_proactive_contribution(self) -> float:
        """Calculate how much proactive particles contributed to the best solution"""
        if self.best_particle_type == 'proactive':
            return 1.0
        elif self.best_particle_type == 'reactive':
            return 0.0
        else:
            # Calculate based on proximity to best solution
            best_pos = self.global_best_pos
            proactive_distances = []
            reactive_distances = []
            
            for i, particle in enumerate(self.particles):
                distance = np.linalg.norm(particle.best_pos - best_pos)
                if i < self.n_proactive:
                    proactive_distances.append(distance)
                else:
                    reactive_distances.append(distance)
            
            if proactive_distances and reactive_distances:
                avg_proactive_dist = np.mean(proactive_distances)
                avg_reactive_dist = np.mean(reactive_distances)
                return avg_reactive_dist / (avg_proactive_dist + avg_reactive_dist)
            else:
                return 0.5
