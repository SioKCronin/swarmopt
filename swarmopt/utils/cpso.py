"""
Cooperative PSO (CPSO) Implementation

Cooperative Particle Swarm Optimization uses multiple collaborating swarms
that work together to solve optimization problems. Each swarm is responsible
for optimizing a subset of dimensions, and they share information to improve
overall performance.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional

class CooperativeSwarm:
    """
    A single swarm in the cooperative PSO system
    """
    
    def __init__(self, swarm_id: int, dimensions: List[int], n_particles: int, 
                 obj_func: Callable, c1: float = 2.0, c2: float = 2.0, 
                 w: float = 0.9, velocity_clamp: Tuple[float, float] = (-5, 5)):
        """
        Initialize a cooperative swarm
        
        Parameters:
        -----------
        swarm_id : int
            Unique identifier for this swarm
        dimensions : List[int]
            Dimensions this swarm is responsible for
        n_particles : int
            Number of particles in this swarm
        obj_func : Callable
            Objective function to optimize
        c1, c2 : float
            Cognitive and social acceleration coefficients
        w : float
            Inertia weight
        velocity_clamp : Tuple[float, float]
            Velocity bounds
        """
        self.swarm_id = swarm_id
        self.dimensions = dimensions
        self.n_particles = n_particles
        self.obj_func = obj_func
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.velocity_clamp = velocity_clamp
        
        # Initialize particles
        self.particles = []
        self.best_pos = None
        self.best_cost = float('inf')
        self.best_particle = None
        
        # Communication with other swarms
        self.global_context = None  # Will be set by CPSO coordinator
        
    def initialize_particles(self, full_dim: int):
        """Initialize particles for this swarm's dimensions"""
        self.particles = []
        for i in range(self.n_particles):
            # Initialize position for this swarm's dimensions
            pos = np.random.uniform(-5, 5, len(self.dimensions))
            particle = CooperativeParticle(pos, self.dimensions, self.velocity_clamp)
            particle.swarm_id = self.swarm_id
            
            # Initialize particle cost
            full_pos = np.random.uniform(-5, 5, full_dim)
            full_pos[self.dimensions] = pos
            particle.best_cost = self.obj_func(full_pos)
            
            self.particles.append(particle)
            
            # Update swarm best
            if particle.best_cost < self.best_cost:
                self.best_cost = particle.best_cost
                self.best_pos = particle.best_pos.copy()
                self.best_particle = particle
    
    def update_particles(self, global_context: np.ndarray, current_iter: int = 0):
        """Update all particles in this swarm"""
        for particle in self.particles:
            # Use particle's own best if swarm best is not available
            swarm_best = self.best_pos if self.best_pos is not None else particle.best_pos
            particle.update(global_context, swarm_best, self.c1, self.c2, 
                           self.w, current_iter, self.obj_func)
            
            # Update swarm best
            if particle.best_cost < self.best_cost:
                self.best_cost = particle.best_cost
                self.best_pos = particle.best_pos.copy()
                self.best_particle = particle

class CooperativeParticle:
    """
    A particle in the cooperative PSO system
    """
    
    def __init__(self, pos: np.ndarray, dimensions: List[int], 
                 velocity_clamp: Tuple[float, float]):
        """
        Initialize a cooperative particle
        
        Parameters:
        -----------
        pos : np.ndarray
            Initial position
        dimensions : List[int]
            Dimensions this particle is responsible for
        velocity_clamp : Tuple[float, float]
            Velocity bounds
        """
        self.pos = pos.copy()
        self.velocity = np.random.uniform(-1, 1, len(dimensions))
        self.dimensions = dimensions
        self.velocity_clamp = velocity_clamp
        
        # Particle's best
        self.best_pos = self.pos.copy()
        self.best_cost = float('inf')
        
        # Swarm information
        self.swarm_id = None
        
    def update(self, global_context: np.ndarray, swarm_best: np.ndarray,
               c1: float, c2: float, w: float, current_iter: int = 0, obj_func=None):
        """
        Update particle position and velocity
        
        Parameters:
        -----------
        global_context : np.ndarray
            Full-dimensional context from other swarms
        swarm_best : np.ndarray
            Best position in this swarm
        c1, c2 : float
            Cognitive and social acceleration coefficients
        w : float
            Inertia weight
        current_iter : int
            Current iteration number
        obj_func : Callable
            Objective function
        """
        # Create full-dimensional position for evaluation
        full_pos = global_context.copy()
        full_pos[self.dimensions] = self.pos
        
        # Evaluate current position
        current_cost = obj_func(full_pos)
        
        # Update particle best
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_pos = self.pos.copy()
        
        # Update velocity
        r1, r2 = np.random.random(2)
        
        # Cognitive component (particle's best)
        cognitive = c1 * r1 * (self.best_pos - self.pos)
        
        # Social component (swarm's best)
        social = c2 * r2 * (swarm_best - self.pos)
        
        # Update velocity
        self.velocity = w * self.velocity + cognitive + social
        
        # Apply velocity clamping
        if isinstance(self.velocity_clamp, (list, tuple)) and len(self.velocity_clamp) == 2:
            v_min, v_max = self.velocity_clamp
        else:
            v_min, v_max = -5, 5  # Default bounds
        self.velocity = np.clip(self.velocity, v_min, v_max)
        
        # Update position
        self.pos += self.velocity

class CPSO:
    """
    Cooperative Particle Swarm Optimization
    
    Uses multiple swarms that collaborate to solve optimization problems.
    Each swarm is responsible for a subset of dimensions.
    """
    
    def __init__(self, n_swarms: int, n_particles_per_swarm: int, 
                 total_dimensions: int, obj_func: Callable,
                 c1: float = 2.0, c2: float = 2.0, w: float = 0.9,
                 velocity_clamp: Tuple[float, float] = (-5, 5),
                 communication_strategy: str = 'best'):
        """
        Initialize Cooperative PSO
        
        Parameters:
        -----------
        n_swarms : int
            Number of cooperating swarms
        n_particles_per_swarm : int
            Number of particles in each swarm
        total_dimensions : int
            Total number of dimensions in the problem
        obj_func : Callable
            Objective function to optimize
        c1, c2 : float
            Cognitive and social acceleration coefficients
        w : float
            Inertia weight
        velocity_clamp : Tuple[float, float]
            Velocity bounds
        communication_strategy : str
            How swarms communicate ('best', 'random', 'tournament')
        """
        self.n_swarms = n_swarms
        self.n_particles_per_swarm = n_particles_per_swarm
        self.total_dimensions = total_dimensions
        self.obj_func = obj_func
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.velocity_clamp = velocity_clamp
        self.communication_strategy = communication_strategy
        
        # Initialize swarms
        self.swarms = []
        self.dimension_assignments = self._assign_dimensions()
        
        for i in range(n_swarms):
            swarm = CooperativeSwarm(
                swarm_id=i,
                dimensions=self.dimension_assignments[i],
                n_particles=n_particles_per_swarm,
                obj_func=obj_func,
                c1=c1, c2=c2, w=w,
                velocity_clamp=velocity_clamp
            )
            self.swarms.append(swarm)
        
        # Global best tracking
        self.global_best_pos = np.random.uniform(-5, 5, total_dimensions)
        self.global_best_cost = float('inf')
        self.global_context = np.random.uniform(-5, 5, total_dimensions)
        
        # Communication history
        self.communication_history = []
        
    def _assign_dimensions(self) -> List[List[int]]:
        """Assign dimensions to swarms"""
        dimensions_per_swarm = self.total_dimensions // self.n_swarms
        remainder = self.total_dimensions % self.n_swarms
        
        assignments = []
        start_dim = 0
        
        for i in range(self.n_swarms):
            # Some swarms get one extra dimension if there's a remainder
            n_dims = dimensions_per_swarm + (1 if i < remainder else 0)
            end_dim = start_dim + n_dims
            
            swarm_dims = list(range(start_dim, end_dim))
            assignments.append(swarm_dims)
            start_dim = end_dim
            
        return assignments
    
    def _update_global_context(self):
        """Update the global context that all swarms use"""
        if self.communication_strategy == 'best':
            # Use the best position from each swarm
            for swarm in self.swarms:
                if swarm.best_pos is not None:
                    self.global_context[swarm.dimensions] = swarm.best_pos
                    
        elif self.communication_strategy == 'random':
            # Use random particles from each swarm
            for swarm in self.swarms:
                if swarm.particles:
                    random_particle = np.random.choice(swarm.particles)
                    self.global_context[swarm.dimensions] = random_particle.pos
                    
        elif self.communication_strategy == 'tournament':
            # Use tournament selection
            for swarm in self.swarms:
                if len(swarm.particles) >= 2:
                    # Tournament between two random particles
                    p1, p2 = np.random.choice(swarm.particles, 2, replace=False)
                    winner = p1 if p1.best_cost < p2.best_cost else p2
                    self.global_context[swarm.dimensions] = winner.pos
                elif swarm.particles:
                    self.global_context[swarm.dimensions] = swarm.particles[0].pos
    
    def _evaluate_global_solution(self):
        """Evaluate the current global solution"""
        global_cost = self.obj_func(self.global_context)
        
        if global_cost < self.global_best_cost:
            self.global_best_cost = global_cost
            self.global_best_pos = self.global_context.copy()
            
        return global_cost
    
    def optimize(self, epochs: int, verbose: bool = True) -> dict:
        """
        Run cooperative PSO optimization
        
        Parameters:
        -----------
        epochs : int
            Number of optimization epochs
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        dict : Optimization results
        """
        import time
        start_time = time.time()
        
        # Initialize all swarms
        for swarm in self.swarms:
            swarm.initialize_particles(self.total_dimensions)
        
        # Initial global context
        self._update_global_context()
        self._evaluate_global_solution()
        
        if verbose:
            print(f"🎯 Starting Cooperative PSO with {self.n_swarms} swarms")
            print(f"   Total dimensions: {self.total_dimensions}")
            print(f"   Particles per swarm: {self.n_particles_per_swarm}")
            print(f"   Communication strategy: {self.communication_strategy}")
            print(f"   Initial global cost: {self.global_best_cost:.6f}")
        
        # Optimization loop
        for epoch in range(epochs):
            # Update global context
            self._update_global_context()
            
            # Update all swarms
            for swarm in self.swarms:
                swarm.update_particles(self.global_context, epoch)
            
            # Evaluate global solution
            current_cost = self._evaluate_global_solution()
            
            # Record communication
            self.communication_history.append({
                'epoch': epoch,
                'global_cost': current_cost,
                'swarm_costs': [swarm.best_cost for swarm in self.swarms]
            })
            
            if verbose and epoch % 10 == 0:
                print(f"   Epoch {epoch:3d}: Global cost = {current_cost:.6f}")
        
        runtime = time.time() - start_time
        
        if verbose:
            print(f"✅ Cooperative PSO completed!")
            print(f"   Final global cost: {self.global_best_cost:.6f}")
            print(f"   Runtime: {runtime:.3f}s")
        
        return {
            'best_cost': self.global_best_cost,
            'best_pos': self.global_best_pos,
            'runtime': runtime,
            'communication_history': self.communication_history,
            'swarm_results': [
                {
                    'swarm_id': swarm.swarm_id,
                    'dimensions': swarm.dimensions,
                    'best_cost': swarm.best_cost,
                    'best_pos': swarm.best_pos
                }
                for swarm in self.swarms
            ]
        }
    
    def get_swarm_statistics(self) -> dict:
        """Get statistics about swarm performance"""
        stats = {
            'n_swarms': self.n_swarms,
            'total_particles': sum(len(swarm.particles) for swarm in self.swarms),
            'dimension_assignments': self.dimension_assignments,
            'swarm_performance': []
        }
        
        for swarm in self.swarms:
            swarm_stats = {
                'swarm_id': swarm.swarm_id,
                'dimensions': swarm.dimensions,
                'n_particles': len(swarm.particles),
                'best_cost': swarm.best_cost,
                'particle_costs': [p.best_cost for p in swarm.particles]
            }
            stats['swarm_performance'].append(swarm_stats)
            
        return stats
