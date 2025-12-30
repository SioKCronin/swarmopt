"""
Simplified Multiobjective Particle Swarm Optimization

This module implements a simplified multiobjective PSO that avoids
complex indexing issues while providing core functionality.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
import warnings

class SimpleMultiObjectivePSO:
    """
    Simplified Multiobjective PSO implementation
    """
    
    def __init__(self, n_particles: int, dims: int, obj_func: Callable,
                 bounds: Tuple[float, float] = (-5, 5),
                 c1: float = 2.0, c2: float = 2.0, w: float = 0.9,
                 epochs: int = 100, archive_size: int = 100):
        """
        Initialize Simple Multiobjective PSO
        
        Parameters:
        -----------
        n_particles : int
            Number of particles
        dims : int
            Problem dimensions
        obj_func : Callable
            Objective function returning multiple objectives
        bounds : Tuple[float, float]
            Search space bounds
        c1, c2, w : float
            PSO parameters
        epochs : int
            Number of iterations
        archive_size : int
            Size of external archive
        """
        self.n_particles = n_particles
        self.dims = dims
        self.obj_func = obj_func
        self.bounds = bounds
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.epochs = epochs
        self.archive_size = archive_size
        
        # Initialize particles
        self.particles = []
        self._initialize_particles()
        
        # External archive
        self.archive = []
        
        # Statistics
        self.hypervolume_history = []
    
    def _initialize_particles(self):
        """Initialize particle swarm"""
        for _ in range(self.n_particles):
            position = np.random.uniform(self.bounds[0], self.bounds[1], self.dims)
            velocity = np.random.uniform(-abs(self.bounds[1] - self.bounds[0]), 
                                       abs(self.bounds[1] - self.bounds[0]), self.dims)
            
            particle = {
                'pos': position,
                'velocity': velocity,
                'best_pos': position.copy(),
                'best_objectives': self.obj_func(position),
                'objectives': self.obj_func(position)
            }
            self.particles.append(particle)
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (minimization assumed)"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def _update_particles(self, iteration: int):
        """Update all particles"""
        for particle in self.particles:
            # Select global best from archive
            global_best = self._select_global_best()
            
            # Standard PSO update
            cognitive_component = self.c1 * np.random.random() * (particle['best_pos'] - particle['pos'])
            social_component = self.c2 * np.random.random() * (global_best['best_pos'] - particle['pos'])
            
            particle['velocity'] = (self.w * particle['velocity'] + 
                                  cognitive_component + social_component)
            particle['pos'] += particle['velocity']
            
            # Apply bounds
            particle['pos'] = np.clip(particle['pos'], self.bounds[0], self.bounds[1])
            
            # Evaluate new objectives
            particle['objectives'] = self.obj_func(particle['pos'])
            
            # Update personal best (Pareto dominance)
            if self._dominates(particle['objectives'], particle['best_objectives']):
                particle['best_pos'] = particle['pos'].copy()
                particle['best_objectives'] = particle['objectives'].copy()
            elif not self._dominates(particle['best_objectives'], particle['objectives']):
                # Non-dominated, randomly choose
                if np.random.random() < 0.5:
                    particle['best_pos'] = particle['pos'].copy()
                    particle['best_objectives'] = particle['objectives'].copy()
    
    def _select_global_best(self) -> Dict:
        """Select global best using tournament selection"""
        if len(self.archive) == 0:
            # No solutions in archive, use random particle
            return np.random.choice(self.particles)
        
        # Tournament selection
        tournament_size = min(3, len(self.archive))
        tournament = np.random.choice(self.archive, tournament_size, replace=False)
        
        # Select best from tournament (random for simplicity)
        return np.random.choice(tournament)
    
    def _update_archive(self):
        """Update external archive"""
        # Add all current particle positions to archive
        for particle in self.particles:
            solution = {
                'pos': particle['pos'].copy(),
                'objectives': particle['objectives'].copy(),
                'best_pos': particle['best_pos'].copy(),
                'best_objectives': particle['best_objectives'].copy()
            }
            self.archive.append(solution)
        
        # Remove dominated solutions
        self._remove_dominated_solutions()
        
        # Maintain archive size
        if len(self.archive) > self.archive_size:
            self._maintain_archive_size()
    
    def _remove_dominated_solutions(self):
        """Remove dominated solutions from archive"""
        non_dominated = []
        
        for i, sol1 in enumerate(self.archive):
            is_dominated = False
            for j, sol2 in enumerate(self.archive):
                if i != j and self._dominates(sol2['objectives'], sol1['objectives']):
                    is_dominated = True
                    break
            
            if not is_dominated:
                non_dominated.append(sol1)
        
        self.archive = non_dominated
    
    def _maintain_archive_size(self):
        """Maintain archive size by random selection"""
        if len(self.archive) > self.archive_size:
            # Randomly select solutions to keep
            indices = np.random.choice(len(self.archive), self.archive_size, replace=False)
            self.archive = [self.archive[i] for i in indices]
    
    def _calculate_hypervolume(self) -> float:
        """Calculate simplified hypervolume indicator"""
        if len(self.archive) == 0:
            return 0.0
        
        # Simple hypervolume approximation
        objectives = np.array([sol['objectives'] for sol in self.archive])
        ref_point = np.max(objectives, axis=0) + 1.0
        
        # Use volume of hyperrectangle
        volume = 1.0
        for obj_idx in range(objectives.shape[1]):
            volume *= ref_point[obj_idx] - np.min(objectives[:, obj_idx])
        
        return volume
    
    def optimize(self) -> Dict:
        """
        Run Simple Multiobjective PSO optimization
        
        Returns:
        --------
        Dict : Optimization results
        """
        import time
        start_time = time.time()
        
        for iteration in range(self.epochs):
            # Update particles
            self._update_particles(iteration)
            
            # Update archive
            self._update_archive()
            
            # Track statistics
            hypervolume = self._calculate_hypervolume()
            self.hypervolume_history.append(hypervolume)
        
        end_time = time.time()
        
        return {
            'pareto_front': self.archive,
            'hypervolume_history': self.hypervolume_history,
            'runtime': end_time - start_time,
            'n_objectives': len(self.archive[0]['objectives']) if self.archive else 0
        }

# Multiobjective benchmark functions
def zdt1(x: np.ndarray) -> np.ndarray:
    """ZDT1 benchmark function"""
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.array([f1, f2])

def zdt2(x: np.ndarray) -> np.ndarray:
    """ZDT2 benchmark function"""
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1 - (f1 / g) ** 2)
    return np.array([f1, f2])

def zdt3(x: np.ndarray) -> np.ndarray:
    """ZDT3 benchmark function"""
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
    return np.array([f1, f2])

def dtlz1(x: np.ndarray, n_obj: int = 3) -> np.ndarray:
    """DTLZ1 benchmark function"""
    k = len(x) - n_obj + 1
    g = 100 * (k + np.sum([(x[i] - 0.5)**2 - np.cos(20 * np.pi * (x[i] - 0.5)) 
                          for i in range(n_obj-1, len(x))]))
    
    objectives = []
    for i in range(n_obj):
        if i == n_obj - 1:
            f = 0.5 * (1 + g) * np.prod([x[j] for j in range(n_obj-1)])
        else:
            f = 0.5 * (1 + g) * np.prod([x[j] for j in range(n_obj-1) if j != i]) * (1 - x[i])
        objectives.append(f)
    
    return np.array(objectives)

def dtlz2(x: np.ndarray, n_obj: int = 3) -> np.ndarray:
    """DTLZ2 benchmark function"""
    k = len(x) - n_obj + 1
    g = np.sum([(x[i] - 0.5)**2 for i in range(n_obj-1, len(x))])
    
    objectives = []
    for i in range(n_obj):
        if i == n_obj - 1:
            f = (1 + g) * np.prod([np.cos(x[j] * np.pi / 2) for j in range(n_obj-1)])
        else:
            f = (1 + g) * np.prod([np.cos(x[j] * np.pi / 2) for j in range(n_obj-1) if j != i]) * np.sin(x[i] * np.pi / 2)
        objectives.append(f)
    
    return np.array(objectives)
