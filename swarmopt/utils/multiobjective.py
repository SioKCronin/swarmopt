"""
Multiobjective Particle Swarm Optimization (MOPSO)

This module implements various multiobjective optimization algorithms
including NSGA-II inspired approaches, Pareto dominance, and crowding distance.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
import warnings
from collections import defaultdict

class ParetoFront:
    """
    Represents a Pareto front with dominance relationships
    """
    
    def __init__(self):
        self.solutions = []  # List of (position, objectives, rank, crowding_distance)
        self.ranks = defaultdict(list)  # Solutions grouped by rank
    
    def add_solution(self, position: np.ndarray, objectives: np.ndarray):
        """Add a solution to the Pareto front"""
        rank = self._calculate_rank(objectives)
        crowding_distance = 0.0  # Will be calculated later
        
        solution = {
            'position': position.copy(),
            'objectives': objectives.copy(),
            'rank': rank,
            'crowding_distance': crowding_distance
        }
        
        self.solutions.append(solution)
        self.ranks[rank].append(len(self.solutions) - 1)
    
    def _calculate_rank(self, objectives: np.ndarray) -> int:
        """Calculate Pareto rank for a solution"""
        rank = 0
        for solution in self.solutions:
            if self._dominates(solution['objectives'], objectives):
                rank = max(rank, solution['rank'] + 1)
        return rank
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """
        Check if obj1 dominates obj2 (minimization assumed)
        obj1 dominates obj2 if obj1 is better in at least one objective
        and not worse in any objective
        """
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def update_crowding_distances(self):
        """Update crowding distances for all solutions"""
        if len(self.solutions) < 3:
            return
        
        # Reset crowding distances
        for solution in self.solutions:
            solution['crowding_distance'] = 0.0
        
        # Calculate crowding distance for each rank
        for rank, indices in self.ranks.items():
            if len(indices) < 3:
                continue
            
            # Get objectives for this rank
            objectives = np.array([self.solutions[i]['objectives'] for i in indices])
            n_obj = objectives.shape[1]
            
            # Calculate crowding distance for each objective
            for obj_idx in range(n_obj):
                # Sort by this objective
                sorted_indices = np.argsort(objectives[:, obj_idx])
                
                # Set boundary points to infinity
                self.solutions[indices[sorted_indices[0]]]['crowding_distance'] = float('inf')
                self.solutions[indices[sorted_indices[-1]]]['crowding_distance'] = float('inf')
                
                # Calculate distances for intermediate points
                obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]
                if obj_range > 0:
                    for i in range(1, len(sorted_indices) - 1):
                        distance = (objectives[sorted_indices[i+1], obj_idx] - 
                                  objectives[sorted_indices[i-1], obj_idx]) / obj_range
                        self.solutions[indices[sorted_indices[i]]]['crowding_distance'] += distance
    
    def get_pareto_front(self) -> List[Dict]:
        """Get solutions on the Pareto front (rank 0)"""
        return [sol for sol in self.solutions if sol['rank'] == 0]
    
    def get_best_solutions(self, n: int) -> List[Dict]:
        """Get n best solutions based on rank and crowding distance"""
        # Sort by rank first, then by crowding distance (descending)
        sorted_solutions = sorted(self.solutions, 
                                key=lambda x: (x['rank'], -x['crowding_distance']))
        return sorted_solutions[:n]

class MultiObjectiveParticle:
    """
    Particle for multiobjective optimization
    """
    
    def __init__(self, position: np.ndarray, velocity: np.ndarray, 
                 obj_funcs: List[Callable], bounds: Tuple[float, float]):
        self.pos = position.copy()
        self.velocity = velocity.copy()
        self.obj_funcs = obj_funcs
        self.bounds = bounds
        
        # Evaluate initial objectives
        self.objectives = np.array([func(self.pos) for func in obj_funcs])
        
        # Personal best
        self.best_pos = self.pos.copy()
        self.best_objectives = self.objectives.copy()
        
        # Multiobjective specific attributes
        self.rank = 0
        self.crowding_distance = 0.0
        self.dominated_count = 0
        self.dominates = []
    
    def update(self, global_best_pos: np.ndarray, global_best_objectives: np.ndarray,
               c1: float, c2: float, w: float, current_iter: int, max_iter: int):
        """Update particle position and velocity"""
        # Standard PSO update
        cognitive_component = c1 * np.random.random() * (self.best_pos - self.pos)
        social_component = c2 * np.random.random() * (global_best_pos - self.pos)
        
        self.velocity = w * self.velocity + cognitive_component + social_component
        self.pos += self.velocity
        
        # Apply bounds
        self.pos = np.clip(self.pos, self.bounds[0], self.bounds[1])
        
        # Evaluate new objectives
        self.objectives = np.array([func(self.pos) for func in self.obj_funcs])
        
        # Update personal best (Pareto dominance)
        if self._dominates(self.objectives, self.best_objectives):
            self.best_pos = self.pos.copy()
            self.best_objectives = self.objectives.copy()
        elif not self._dominates(self.best_objectives, self.objectives):
            # Non-dominated, keep both or use crowding distance
            if np.random.random() < 0.5:
                self.best_pos = self.pos.copy()
                self.best_objectives = self.objectives.copy()
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

class NSGA2PSO:
    """
    NSGA-II inspired Multiobjective PSO
    """
    
    def __init__(self, n_particles: int, dims: int, obj_funcs: List[Callable],
                 bounds: Tuple[float, float] = (-5, 5),
                 c1: float = 2.0, c2: float = 2.0, w: float = 0.9,
                 epochs: int = 100, archive_size: int = 100):
        """
        Initialize NSGA-II PSO
        
        Parameters:
        -----------
        n_particles : int
            Number of particles
        dims : int
            Problem dimensions
        obj_funcs : List[Callable]
            List of objective functions
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
        self.obj_funcs = obj_funcs
        self.n_objectives = len(obj_funcs)
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
        self.archive = ParetoFront()
        
        # Statistics
        self.hypervolume_history = []
        self.spread_history = []
        self.convergence_history = []
    
    def _initialize_particles(self):
        """Initialize particle swarm"""
        for _ in range(self.n_particles):
            position = np.random.uniform(self.bounds[0], self.bounds[1], self.dims)
            velocity = np.random.uniform(-abs(self.bounds[1] - self.bounds[0]), 
                                       abs(self.bounds[1] - self.bounds[0]), self.dims)
            
            particle = MultiObjectiveParticle(position, velocity, self.obj_funcs, self.bounds)
            self.particles.append(particle)
    
    def optimize(self) -> Dict:
        """
        Run NSGA-II PSO optimization
        
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
            
            # Select leaders
            self._select_leaders()
            
            # Track statistics
            self._track_statistics()
        
        end_time = time.time()
        
        return {
            'pareto_front': self.archive.get_pareto_front(),
            'all_solutions': self.archive.solutions,
            'hypervolume_history': self.hypervolume_history,
            'spread_history': self.spread_history,
            'convergence_history': self.convergence_history,
            'runtime': end_time - start_time,
            'n_objectives': self.n_objectives
        }
    
    def _update_particles(self, iteration: int):
        """Update all particles"""
        for particle in self.particles:
            # Select global best (leader selection)
            global_best = self._select_global_best(particle)
            
            # Update particle
            particle.update(
                global_best['position'], global_best['objectives'],
                self.c1, self.c2, self.w, iteration, self.epochs
            )
    
    def _select_global_best(self, particle: MultiObjectiveParticle) -> Dict:
        """Select global best for a particle using tournament selection"""
        # Tournament selection from archive
        tournament_size = min(5, len(self.archive.solutions))
        
        if tournament_size == 0:
            # No solutions in archive, use random particle
            random_particle = np.random.choice(self.particles)
            return {
                'position': random_particle.best_pos,
                'objectives': random_particle.best_objectives
            }
        
        # Select tournament participants
        tournament_indices = np.random.choice(len(self.archive.solutions), 
                                            tournament_size, replace=False)
        tournament_solutions = [self.archive.solutions[i] for i in tournament_indices]
        
        # Select best from tournament
        best_solution = min(tournament_solutions, 
                          key=lambda x: (x['rank'], -x['crowding_distance']))
        
        return best_solution
    
    def _update_archive(self):
        """Update external archive with new solutions"""
        # Add all current particle positions to archive
        for particle in self.particles:
            self.archive.add_solution(particle.pos, particle.objectives)
        
        # Update crowding distances
        self.archive.update_crowding_distances()
        
        # Maintain archive size
        if len(self.archive.solutions) > self.archive_size:
            self._maintain_archive_size()
    
    def _maintain_archive_size(self):
        """Maintain archive size using NSGA-II selection"""
        # Sort by rank and crowding distance
        sorted_solutions = sorted(self.archive.solutions,
                                key=lambda x: (x['rank'], -x['crowding_distance']))
        
        # Keep best solutions
        self.archive.solutions = sorted_solutions[:self.archive_size]
        
        # Rebuild ranks
        self.archive.ranks = defaultdict(list)
        for i, solution in enumerate(self.archive.solutions):
            self.archive.ranks[solution['rank']].append(i)
    
    def _select_leaders(self):
        """Select leaders for next iteration"""
        # This is handled in _select_global_best
        pass
    
    def _track_statistics(self):
        """Track optimization statistics"""
        if len(self.archive.solutions) > 0:
            # Hypervolume (simplified)
            hypervolume = self._calculate_hypervolume()
            self.hypervolume_history.append(hypervolume)
            
            # Spread (simplified)
            spread = self._calculate_spread()
            self.spread_history.append(spread)
            
            # Convergence (average distance to reference point)
            convergence = self._calculate_convergence()
            self.convergence_history.append(convergence)
    
    def _calculate_hypervolume(self) -> float:
        """Calculate hypervolume indicator (simplified)"""
        if len(self.archive.solutions) == 0:
            return 0.0
        
        # Simple hypervolume approximation
        pareto_front = self.archive.get_pareto_front()
        if len(pareto_front) == 0:
            return 0.0
        
        # Use volume of hyperrectangle
        objectives = np.array([sol['objectives'] for sol in pareto_front])
        ref_point = np.max(objectives, axis=0) + 1.0
        
        volume = 1.0
        for obj_idx in range(self.n_objectives):
            volume *= ref_point[obj_idx] - np.min(objectives[:, obj_idx])
        
        return volume
    
    def _calculate_spread(self) -> float:
        """Calculate spread indicator"""
        if len(self.archive.solutions) < 2:
            return 0.0
        
        pareto_front = self.archive.get_pareto_front()
        if len(pareto_front) < 2:
            return 0.0
        
        objectives = np.array([sol['objectives'] for sol in pareto_front])
        
        # Calculate average distance between consecutive solutions
        distances = []
        for i in range(len(objectives) - 1):
            dist = np.linalg.norm(objectives[i+1] - objectives[i])
            distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_convergence(self) -> float:
        """Calculate convergence indicator"""
        if len(self.archive.solutions) == 0:
            return float('inf')
        
        pareto_front = self.archive.get_pareto_front()
        if len(pareto_front) == 0:
            return float('inf')
        
        # Average distance to origin (reference point)
        objectives = np.array([sol['objectives'] for sol in pareto_front])
        distances = np.linalg.norm(objectives, axis=1)
        
        return np.mean(distances)

class SPEA2PSO:
    """
    SPEA2 inspired Multiobjective PSO
    """
    
    def __init__(self, n_particles: int, dims: int, obj_funcs: List[Callable],
                 bounds: Tuple[float, float] = (-5, 5),
                 c1: float = 2.0, c2: float = 2.0, w: float = 0.9,
                 epochs: int = 100, archive_size: int = 100):
        """
        Initialize SPEA2 PSO
        """
        self.n_particles = n_particles
        self.dims = dims
        self.obj_funcs = obj_funcs
        self.n_objectives = len(obj_funcs)
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
        self.fitness_history = []
    
    def _initialize_particles(self):
        """Initialize particle swarm"""
        for _ in range(self.n_particles):
            position = np.random.uniform(self.bounds[0], self.bounds[1], self.dims)
            velocity = np.random.uniform(-abs(self.bounds[1] - self.bounds[0]), 
                                       abs(self.bounds[1] - self.bounds[0]), self.dims)
            
            particle = MultiObjectiveParticle(position, velocity, self.obj_funcs, self.bounds)
            self.particles.append(particle)
    
    def optimize(self) -> Dict:
        """Run SPEA2 PSO optimization"""
        import time
        start_time = time.time()
        
        for iteration in range(self.epochs):
            # Update particles
            self._update_particles(iteration)
            
            # Update archive
            self._update_archive()
            
            # Track statistics
            self._track_statistics()
        
        end_time = time.time()
        
        return {
            'pareto_front': [sol for sol in self.archive if sol['fitness'] < 1.0],
            'all_solutions': self.archive,
            'fitness_history': self.fitness_history,
            'runtime': end_time - start_time,
            'n_objectives': self.n_objectives
        }
    
    def _update_particles(self, iteration: int):
        """Update all particles"""
        for particle in self.particles:
            # Select global best from archive
            global_best = self._select_global_best()
            
            # Update particle
            particle.update(
                global_best['position'], global_best['objectives'],
                self.c1, self.c2, self.w, iteration, self.epochs
            )
    
    def _select_global_best(self) -> Dict:
        """Select global best using fitness-based selection"""
        if len(self.archive) == 0:
            # No solutions in archive, use random particle
            random_particle = np.random.choice(self.particles)
            return {
                'position': random_particle.best_pos,
                'objectives': random_particle.best_objectives
            }
        
        # Select based on fitness (lower is better)
        valid_solutions = [sol for sol in self.archive if sol['fitness'] < 1.0]
        if not valid_solutions:
            valid_solutions = self.archive
        
        # Tournament selection
        tournament_size = min(3, len(valid_solutions))
        tournament = np.random.choice(valid_solutions, tournament_size, replace=False)
        best_solution = min(tournament, key=lambda x: x['fitness'])
        
        return best_solution
    
    def _update_archive(self):
        """Update external archive using SPEA2 selection"""
        # Combine current particles and archive
        all_solutions = []
        
        # Add current particle positions
        for particle in self.particles:
            all_solutions.append({
                'position': particle.pos.copy(),
                'objectives': particle.objectives.copy()
            })
        
        # Add archive solutions
        all_solutions.extend(self.archive)
        
        # Calculate fitness for all solutions
        self._calculate_fitness(all_solutions)
        
        # Select solutions for next archive
        self.archive = self._environmental_selection(all_solutions)
    
    def _calculate_fitness(self, solutions: List[Dict]):
        """Calculate SPEA2 fitness for all solutions"""
        n = len(solutions)
        
        # Calculate strength (number of solutions dominated)
        for i, sol1 in enumerate(solutions):
            sol1['strength'] = 0
            for j, sol2 in enumerate(solutions):
                if i != j and self._dominates(sol1['objectives'], sol2['objectives']):
                    sol1['strength'] += 1
        
        # Calculate raw fitness
        for sol in solutions:
            sol['raw_fitness'] = 0
            for other_sol in solutions:
                if self._dominates(other_sol['objectives'], sol['objectives']):
                    sol['raw_fitness'] += other_sol['strength']
        
        # Calculate density
        for sol in solutions:
            distances = []
            for other_sol in solutions:
                if other_sol != sol:
                    dist = np.linalg.norm(sol['objectives'] - other_sol['objectives'])
                    distances.append(dist)
            
            distances.sort()
            if len(distances) >= 2:
                sol['density'] = 1.0 / (distances[1] + 2.0)  # k-th nearest neighbor
            else:
                sol['density'] = 0.0
        
        # Calculate final fitness
        for sol in solutions:
            sol['fitness'] = sol['raw_fitness'] + sol['density']
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def _environmental_selection(self, solutions: List[Dict]) -> List[Dict]:
        """Environmental selection to maintain archive size"""
        # Sort by fitness
        sorted_solutions = sorted(solutions, key=lambda x: x['fitness'])
        
        # Select non-dominated solutions first
        non_dominated = [sol for sol in sorted_solutions if sol['fitness'] < 1.0]
        
        if len(non_dominated) <= self.archive_size:
            # Fill remaining slots with dominated solutions
            dominated = [sol for sol in sorted_solutions if sol['fitness'] >= 1.0]
            remaining = self.archive_size - len(non_dominated)
            return non_dominated + dominated[:remaining]
        else:
            # Too many non-dominated solutions, use crowding distance
            return self._truncate_by_crowding_distance(non_dominated)
    
    def _truncate_by_crowding_distance(self, solutions: List[Dict]) -> List[Dict]:
        """Truncate solutions using crowding distance"""
        if len(solutions) <= self.archive_size:
            return solutions
        
        # Calculate crowding distances
        objectives = np.array([sol['objectives'] for sol in solutions])
        n_obj = objectives.shape[1]
        
        crowding_distances = np.zeros(len(solutions))
        
        for obj_idx in range(n_obj):
            # Sort by this objective
            sorted_indices = np.argsort(objectives[:, obj_idx])
            
            # Set boundary points to infinity
            crowding_distances[sorted_indices[0]] = float('inf')
            crowding_distances[sorted_indices[-1]] = float('inf')
            
            # Calculate distances for intermediate points
            obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    distance = (objectives[sorted_indices[i+1], obj_idx] - 
                              objectives[sorted_indices[i-1], obj_idx]) / obj_range
                    crowding_distances[sorted_indices[i]] += distance
        
        # Sort by crowding distance (descending) and select best
        sorted_indices = np.argsort(-crowding_distances)
        return [solutions[i] for i in sorted_indices[:self.archive_size]]
    
    def _track_statistics(self):
        """Track optimization statistics"""
        if len(self.archive) > 0:
            avg_fitness = np.mean([sol['fitness'] for sol in self.archive])
            self.fitness_history.append(avg_fitness)

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
