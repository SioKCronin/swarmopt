"""
Horse Herd Optimization Algorithm (HHOA)

This module implements the Horse Herd Optimization Algorithm, a bio-inspired technique
that mimics the motion cycles of an entire herd of horses. Based on the paper:

"A high-speed MPPT based horse herd optimization algorithm with dynamic linear active 
disturbance rejection control for PV battery charging system"
https://www.nature.com/articles/s41598-025-85481-6

The algorithm models three main behaviors:
1. Grazing behavior (exploration)
2. Leadership behavior (exploitation around leaders)
3. Following behavior (social learning)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
import time


class Horse:
    """
    Represents a single horse in the herd
    """
    
    def __init__(self, position: np.ndarray, obj_func: Callable, bounds: Tuple[float, float]):
        """
        Initialize a horse
        
        Parameters:
        -----------
        position : np.ndarray
            Initial position of the horse
        obj_func : Callable
            Objective function to optimize
        bounds : Tuple[float, float]
            Search space bounds
        """
        self.pos = position.copy()
        self.obj_func = obj_func
        self.bounds = bounds
        
        # Evaluate initial fitness
        self.fitness = self.obj_func(self.pos)
        self.best_pos = self.pos.copy()
        self.best_fitness = self.fitness
        
        # Horse-specific attributes
        self.role = 'follower'  # 'leader', 'follower', 'grazing'
        self.energy = 1.0  # Energy level (affects behavior)
        self.age = 0  # Iterations since last improvement
        self.velocity = np.zeros_like(position)
        
    def update_fitness(self):
        """Update fitness and personal best"""
        self.fitness = self.obj_func(self.pos)
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_pos = self.pos.copy()
            self.age = 0
        else:
            self.age += 1


class HHOA:
    """
    Horse Herd Optimization Algorithm
    
    Implements the three-phase behavior model:
    1. Grazing phase: Exploration of search space
    2. Leadership phase: Exploitation around best solutions
    3. Following phase: Social learning from leaders
    """
    
    def __init__(self, n_horses: int, dims: int, obj_func: Callable,
                 bounds: Tuple[float, float] = (-5, 5),
                 n_leaders: int = None,
                 grazing_rate: float = 0.3,
                 leadership_rate: float = 0.4,
                 following_rate: float = 0.3,
                 epochs: int = 100):
        """
        Initialize HHOA
        
        Parameters:
        -----------
        n_horses : int
            Number of horses in the herd
        dims : int
            Problem dimensions
        obj_func : Callable
            Objective function to optimize
        bounds : Tuple[float, float]
            Search space bounds
        n_leaders : int, optional
            Number of leader horses (default: 10% of herd)
        grazing_rate : float
            Probability of grazing behavior (exploration)
        leadership_rate : float
            Probability of leadership behavior (exploitation)
        following_rate : float
            Probability of following behavior (social learning)
        epochs : int
            Number of iterations
        """
        self.n_horses = n_horses
        self.dims = dims
        self.obj_func = obj_func
        self.bounds = bounds
        self.epochs = epochs
        
        # Behavior rates (should sum to 1.0)
        total_rate = grazing_rate + leadership_rate + following_rate
        self.grazing_rate = grazing_rate / total_rate
        self.leadership_rate = leadership_rate / total_rate
        self.following_rate = following_rate / total_rate
        
        # Number of leaders (default: 10% of herd, minimum 2)
        if n_leaders is None:
            self.n_leaders = max(2, int(0.1 * n_horses))
        else:
            self.n_leaders = min(n_leaders, n_horses - 1)
        
        # Initialize herd
        self.horses = []
        self._initialize_herd()
        
        # Global best tracking
        self.global_best_pos = None
        self.global_best_fitness = float('inf')
        self._update_global_best()
        
        # Statistics
        self.fitness_history = []
        self.convergence_history = []
        
    def _initialize_herd(self):
        """Initialize the horse herd"""
        for _ in range(self.n_horses):
            position = np.random.uniform(self.bounds[0], self.bounds[1], self.dims)
            horse = Horse(position, self.obj_func, self.bounds)
            self.horses.append(horse)
    
    def _update_global_best(self):
        """Update global best solution"""
        for horse in self.horses:
            if horse.best_fitness < self.global_best_fitness:
                self.global_best_fitness = horse.best_fitness
                self.global_best_pos = horse.best_pos.copy()
    
    def _select_leaders(self) -> List[Horse]:
        """Select leader horses based on fitness"""
        # Sort horses by fitness
        sorted_horses = sorted(self.horses, key=lambda h: h.best_fitness)
        return sorted_horses[:self.n_leaders]
    
    def _grazing_behavior(self, horse: Horse, current_iter: int, max_iter: int):
        """
        Grazing behavior: Exploration phase
        
        Horses graze (explore) in random directions with decreasing intensity
        """
        # Exploration decreases over time
        exploration_factor = 1.0 - (current_iter / max_iter)
        exploration_factor = max(0.1, exploration_factor)  # Minimum 10% exploration
        
        # Random grazing direction
        grazing_direction = np.random.randn(self.dims)
        grazing_direction /= np.linalg.norm(grazing_direction) + 1e-10
        
        # Grazing step size (decreases over time)
        step_size = exploration_factor * (self.bounds[1] - self.bounds[0]) * 0.1
        
        # Update position
        new_pos = horse.pos + step_size * grazing_direction
        
        # Apply bounds
        new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
        horse.pos = new_pos
        horse.update_fitness()
    
    def _leadership_behavior(self, horse: Horse, leaders: List[Horse], current_iter: int, max_iter: int):
        """
        Leadership behavior: Exploitation phase
        
        Leader horses exploit around their best positions
        """
        # Exploitation intensity increases over time
        exploitation_factor = current_iter / max_iter
        exploitation_factor = min(0.9, exploitation_factor)  # Maximum 90% exploitation
        
        # Local search around best position
        search_radius = (1.0 - exploitation_factor) * (self.bounds[1] - self.bounds[0]) * 0.05
        
        # Random perturbation around best position
        perturbation = np.random.randn(self.dims) * search_radius
        new_pos = horse.best_pos + perturbation
        
        # Apply bounds
        new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
        horse.pos = new_pos
        horse.update_fitness()
    
    def _following_behavior(self, horse: Horse, leaders: List[Horse], current_iter: int, max_iter: int):
        """
        Following behavior: Social learning phase
        
        Follower horses learn from and follow leader horses
        """
        # Select a random leader to follow
        if len(leaders) == 0:
            # No leaders, use global best
            target = self.global_best_pos
        else:
            # Weighted selection: better leaders more likely to be followed
            leader_fitnesses = [l.best_fitness for l in leaders]
            # Convert to probabilities (lower fitness = higher probability)
            max_fitness = max(leader_fitnesses)
            probabilities = [max_fitness - f + 1e-10 for f in leader_fitnesses]
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()
            
            selected_leader = np.random.choice(len(leaders), p=probabilities)
            target = leaders[selected_leader].best_pos
        
        # Follow the target with adaptive step size
        following_factor = 0.5 * (1.0 - current_iter / max_iter) + 0.1  # Decreases over time
        direction = target - horse.pos
        distance = np.linalg.norm(direction)
        
        if distance > 1e-10:
            direction /= distance
            step_size = following_factor * min(distance, (self.bounds[1] - self.bounds[0]) * 0.2)
            new_pos = horse.pos + step_size * direction
        else:
            # Already at target, add small random movement
            new_pos = horse.pos + np.random.randn(self.dims) * (self.bounds[1] - self.bounds[0]) * 0.01
        
        # Apply bounds
        new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
        horse.pos = new_pos
        horse.update_fitness()
    
    def _assign_roles(self, leaders: List[Horse], current_iter: int, max_iter: int):
        """
        Assign roles to horses based on fitness and iteration
        
        Better horses are more likely to be leaders, especially later in optimization
        """
        # Sort horses by fitness
        sorted_horses = sorted(self.horses, key=lambda h: h.best_fitness)
        
        # Assign leaders
        for i, horse in enumerate(sorted_horses[:self.n_leaders]):
            horse.role = 'leader'
        
        # Assign roles to remaining horses based on behavior rates
        remaining_horses = sorted_horses[self.n_leaders:]
        
        # Later in optimization, more horses follow (exploitation)
        # Earlier in optimization, more horses graze (exploration)
        time_factor = current_iter / max_iter
        
        for horse in remaining_horses:
            rand = np.random.random()
            
            # Adjust rates based on iteration
            adjusted_grazing = self.grazing_rate * (1.0 - time_factor)
            adjusted_following = self.following_rate * (0.5 + 0.5 * time_factor)
            adjusted_leadership = self.leadership_rate
            
            if rand < adjusted_grazing:
                horse.role = 'grazing'
            elif rand < adjusted_grazing + adjusted_following:
                horse.role = 'follower'
            else:
                horse.role = 'leader'  # Can promote to leader
    
    def optimize(self) -> Dict:
        """
        Run HHOA optimization
        
        Returns:
        --------
        Dict : Optimization results
        """
        start_time = time.time()
        
        for iteration in range(self.epochs):
            # Select leaders
            leaders = self._select_leaders()
            
            # Assign roles to all horses
            self._assign_roles(leaders, iteration, self.epochs)
            
            # Update each horse based on its role
            for horse in self.horses:
                if horse.role == 'grazing':
                    self._grazing_behavior(horse, iteration, self.epochs)
                elif horse.role == 'leader':
                    self._leadership_behavior(horse, leaders, iteration, self.epochs)
                else:  # follower
                    self._following_behavior(horse, leaders, iteration, self.epochs)
            
            # Update global best
            self._update_global_best()
            
            # Track statistics
            self.fitness_history.append(self.global_best_fitness)
            convergence = np.mean([h.best_fitness for h in self.horses])
            self.convergence_history.append(convergence)
        
        end_time = time.time()
        
        return {
            'best_pos': self.global_best_pos,
            'best_cost': self.global_best_fitness,
            'runtime': end_time - start_time,
            'fitness_history': self.fitness_history,
            'convergence_history': self.convergence_history,
            'n_horses': self.n_horses,
            'n_leaders': self.n_leaders
        }

