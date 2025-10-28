import numpy as np
import timeit
from random import shuffle

try:
    from .utils.distance import euclideanDistance
    from .utils.inertia import (
        constant_inertia_weight, linear_inertia_weight, chaotic_inertia_weight,
        random_inertia_weight, adaptive_inertia_weight, chaotic_random_inertia_weight,
        exponential_inertia_weight, sigmoid_inertia_weight
    )
    from .utils.velocity_clamping import (
        no_clamping, basic_clamping, adaptive_clamping, exponential_clamping,
        sigmoid_clamping, random_clamping, chaotic_clamping, dimension_wise_clamping,
        soft_clamping, hybrid_clamping, convergence_based_clamping
    )
    from .utils.cpso import CPSO
    from .utils.mutation import apply_mutation, MUTATION_STRATEGIES
    from .utils.diversity import DiversityMonitor, calculate_swarm_diversity
    from .utils.ppso import PPSO
    from .utils.simple_multiobjective import SimpleMultiObjectivePSO
except ImportError:
    from utils.distance import euclideanDistance
    from utils.inertia import (
        constant_inertia_weight, linear_inertia_weight, chaotic_inertia_weight,
        random_inertia_weight, adaptive_inertia_weight, chaotic_random_inertia_weight,
        exponential_inertia_weight, sigmoid_inertia_weight
    )
    from utils.velocity_clamping import (
        no_clamping, basic_clamping, adaptive_clamping, exponential_clamping,
        sigmoid_clamping, random_clamping, chaotic_clamping, dimension_wise_clamping,
        soft_clamping, hybrid_clamping, convergence_based_clamping
    )
    from utils.cpso import CPSO
    from utils.mutation import apply_mutation, MUTATION_STRATEGIES
    from utils.diversity import DiversityMonitor, calculate_swarm_diversity
    from utils.ppso import PPSO
    from utils.simple_multiobjective import SimpleMultiObjectivePSO


class Swarm:
    def __init__(self, n_particles, dims, c1, c2, w, epochs, obj_func,
                 algo='global', inertia_func='linear', velocity_clamp=(0,2),
                 k=5, u=0.5, m_swarms=3, hueristic_change=0.9, r=5,
                 w_start=0.9, w_end=0.4, z=0.5, velocity_clamp_func='basic',
                 n_swarms=None, communication_strategy='best',
                 mutation_strategy=None, mutation_rate=0.1, mutation_strength=0.1,
                 diversity_monitoring=False, diversity_threshold=0.1,
                 ppso_enabled=False, proactive_ratio=0.25, knowledge_method='gaussian',
                 exploration_weight=0.5,
                 multiobjective=False, mo_algorithm='nsga2', archive_size=100,
                 target_position=None, n_delegates=0, delegate_spread='uniform'):
        """Intialize the swarm

        Attributes
        ----------
        n_particles: int
            number of particles in the swarm
        dims: int
            dimensions of the space
        c1: float
            cognitive weight
        c2: float
            social weight
        w: float
            inertia weight
        epochs: int
            number of epochs
        obj_func: string
            name of objective function
        algo: string
            name of swarm algorithm
        intertia_func: string
            name of inertia function
        velocity_clamp: 2-tuple
            minimum velocity (first) and maximum velocity (second)
        k: int
            number of neighbors in a neighborhood
        u: float
            sets local/global ratio in unified (0 to 1)
        m_swarms: int
            number of swarms in 'multiswarm'
        hueristic_change: float
            sets ratio of first/second hueristic in 'multiswarm' (0 to 1)
        r: int
            reshuffle parameter in 'multiswarm'
        """

        self.algo = algo
        self.epochs = epochs
        self.n_particles = n_particles
        self.m_swarms = m_swarms
        self.hueristic_change = hueristic_change
        self.end = False
        self.r = r

        self.runtime = 0

        self.dims = dims
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.k = k
        self.u = u
        self.decrement = 0.7 / m_swarms
        
        # Inertia weight parameters
        self.inertia_func = inertia_func
        self.w_start = w_start
        self.w_end = w_end
        self.z = z
        self.initial_cost = None
        
        # Velocity clamping parameters
        self.velocity_clamp_func = velocity_clamp_func
        self.val_min, self.val_max = velocity_clamp
        self.velocity_bounds = 0.2 * (self.val_max - self.val_min)
        self.velocity_z = 0.5  # For chaotic clamping
        
        # CPSO parameters
        self.n_swarms = n_swarms
        self.communication_strategy = communication_strategy
        self.cpso = None
        
        # Mutation parameters
        self.mutation_strategy = mutation_strategy
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        
        # Diversity monitoring parameters
        self.diversity_monitoring = diversity_monitoring
        self.diversity_threshold = diversity_threshold
        self.diversity_monitor = None
        self.diversity_history = []
        
        # PPSO parameters
        self.ppso_enabled = ppso_enabled
        self.proactive_ratio = proactive_ratio
        self.knowledge_method = knowledge_method
        self.exploration_weight = exploration_weight
        self.ppso = None
        
        # Multiobjective parameters
        self.multiobjective = multiobjective
        self.mo_algorithm = mo_algorithm
        self.archive_size = archive_size
        self.mo_optimizer = None
        
        # Respect boundary parameters (mandatory for safety-critical applications)
        self.target_position = np.array(target_position) if target_position is not None else None
        self.n_delegates = n_delegates
        self.delegate_spread = delegate_spread
        self.delegate_positions = []
        
        # If target_position is provided, ALWAYS enforce respect boundary
        if target_position is not None:
            # Automatically calculate safe respect boundary (10% of search space diagonal)
            search_space_diagonal = np.sqrt(self.dims * (self.val_max - self.val_min)**2)
            self.respect_boundary = 0.1 * search_space_diagonal
            self.use_respect_boundary = True
            
            # Generate delegate positions if requested
            if n_delegates > 0:
                self.delegate_positions = self._generate_delegate_positions()
            
            # Log the automatic safety boundary
            import warnings
            delegate_info = f" with {n_delegates} delegate positions" if n_delegates > 0 else ""
            warnings.warn(
                f"Respect boundary automatically enabled for safety: {self.respect_boundary:.4f}. "
                f"Particles will maintain minimum distance from target at {target_position}"
                f"{delegate_info}. "
                f"This is mandatory for safety-critical applications and cannot be disabled.",
                UserWarning
            )
        else:
            self.respect_boundary = None
            self.use_respect_boundary = False

        self.obj_func = obj_func
        self.best_cost = float('inf')
        self.best_pos = None
        self.worst_cost = float('inf')
        self.worst_pos = None
        self.local_best_cost = float('inf')
        self.local_best_pos = None

        if self.algo == 'cpso':
            # Initialize Cooperative PSO
            if self.n_swarms is None:
                self.n_swarms = max(2, self.dims // 2)  # Default to 2 or dims/2
            self.cpso = CPSO(
                n_swarms=self.n_swarms,
                n_particles_per_swarm=self.n_particles,
                total_dimensions=self.dims,
                obj_func=self.obj_func,
                c1=self.c1, c2=self.c2, w=self.w,
                velocity_clamp=(self.val_min, self.val_max),
                communication_strategy=self.communication_strategy
            )
        elif self.algo == 'multiswarm':
            self.multiswarm = self.initialize_multiswarm()
        else:
            self.swarm = self.initialize_swarm()

        if self.algo != 'cpso' and not self.multiobjective:
            self.update_global_best_pos()

    def shape(self):
        return [self.n_particles, self.dims]
    
    def _generate_delegate_positions(self):
        """
        Generate delegate positions in polar/spherical coordinates around target
        
        Useful for positioning repair drones, service satellites, or redundant
        agents around a target while maintaining respect boundary distance.
        
        Returns:
        --------
        list : Delegate positions at respect boundary distance from target
        """
        delegate_positions = []
        
        if self.dims == 2:
            # 2D: Position delegates in circle around target
            for i in range(self.n_delegates):
                if self.delegate_spread == 'uniform':
                    # Evenly spaced around circle
                    angle = 2 * np.pi * i / self.n_delegates
                elif self.delegate_spread == 'random':
                    # Random angles
                    angle = 2 * np.pi * np.random.random()
                elif self.delegate_spread == 'opposite':
                    # Position delegates opposite each other (for redundancy)
                    angle = np.pi * i
                else:
                    angle = 2 * np.pi * i / self.n_delegates
                
                # Position at respect boundary distance
                offset = np.array([
                    self.respect_boundary * np.cos(angle),
                    self.respect_boundary * np.sin(angle)
                ])
                delegate_pos = self.target_position + offset
                delegate_positions.append(delegate_pos)
        
        elif self.dims == 3:
            # 3D: Position delegates on sphere around target
            for i in range(self.n_delegates):
                if self.delegate_spread == 'uniform':
                    # Fibonacci sphere for uniform distribution
                    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
                    y = 1 - (i / float(self.n_delegates - 1)) * 2  # y from 1 to -1
                    radius = np.sqrt(1 - y * y)
                    theta = phi * i
                    
                    x = np.cos(theta) * radius
                    z = np.sin(theta) * radius
                    
                elif self.delegate_spread == 'random':
                    # Random positions on sphere
                    theta = 2 * np.pi * np.random.random()
                    phi = np.arccos(2 * np.random.random() - 1)
                    
                    x = np.sin(phi) * np.cos(theta)
                    y = np.sin(phi) * np.sin(theta)
                    z = np.cos(phi)
                    
                elif self.delegate_spread == 'opposite':
                    # Position delegates opposite each other
                    if i % 2 == 0:
                        theta, phi = 0, np.pi * (i // 2) / max(1, self.n_delegates // 2)
                    else:
                        theta, phi = np.pi, np.pi * (i // 2) / max(1, self.n_delegates // 2)
                    
                    x = np.sin(phi) * np.cos(theta)
                    y = np.sin(phi) * np.sin(theta)
                    z = np.cos(phi)
                else:
                    # Default to uniform
                    phi = np.pi * (3. - np.sqrt(5.))
                    y = 1 - (i / float(self.n_delegates - 1)) * 2
                    radius = np.sqrt(1 - y * y)
                    theta = phi * i
                    
                    x = np.cos(theta) * radius
                    z = np.sin(theta) * radius
                
                # Scale to respect boundary distance
                offset = self.respect_boundary * np.array([x, y, z])
                delegate_pos = self.target_position + offset
                delegate_positions.append(delegate_pos)
        
        else:
            # Higher dimensions: Use hypersphere sampling
            for i in range(self.n_delegates):
                # Random point on unit hypersphere
                random_point = np.random.randn(self.dims)
                random_point /= np.linalg.norm(random_point)
                
                # Scale to respect boundary
                offset = self.respect_boundary * random_point
                delegate_pos = self.target_position + offset
                delegate_positions.append(delegate_pos)
        
        return delegate_positions
    
    def get_delegate_assignments(self):
        """
        Get the best particle assignments to delegate positions
        
        Returns:
        --------
        dict : Mapping of delegate_index -> closest_particle
        """
        if len(self.delegate_positions) == 0:
            return {}
        
        assignments = {}
        used_particles = set()
        
        # Assign each delegate position to nearest particle
        for i, delegate_pos in enumerate(self.delegate_positions):
            best_particle = None
            best_distance = float('inf')
            
            for j, particle in enumerate(self.swarm):
                if j in used_particles:
                    continue
                
                distance = np.linalg.norm(particle.pos - delegate_pos)
                if distance < best_distance:
                    best_distance = distance
                    best_particle = j
            
            if best_particle is not None:
                assignments[i] = {
                    'particle_index': best_particle,
                    'particle_pos': self.swarm[best_particle].pos,
                    'target_pos': delegate_pos,
                    'distance': best_distance
                }
                used_particles.add(best_particle)
        
        return assignments
    
    def objective_with_respect_boundary(self, position):
        """
        Evaluate objective function with respect boundary enforcement
        
        If respect_boundary is enabled, particles are penalized for getting
        closer than the respect distance to the target position.
        
        Parameters:
        -----------
        position : np.ndarray
            Particle position to evaluate
        
        Returns:
        --------
        float : Modified objective value
        """
        # Calculate base objective
        base_cost = self.obj_func(position)
        
        # If no respect boundary, return base cost
        if not self.use_respect_boundary:
            return base_cost
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(position - self.target_position)
        
        # If outside respect boundary, use base cost
        if distance_to_target >= self.respect_boundary:
            return base_cost
        
        # If inside respect boundary, add penalty
        # Penalty increases as particle gets closer to target
        violation = self.respect_boundary - distance_to_target
        penalty_factor = (violation / self.respect_boundary) ** 2
        
        # Scale penalty by base cost magnitude to be relative
        penalty = base_cost * (1.0 + 10.0 * penalty_factor)
        
        return penalty

    def initialize_swarm(self):
        swarm = []
        for particle in range(self.n_particles):
            swarm.append(Particle(self))
        return swarm

    def initialize_multiswarm(self):
        multiswarm = []
        for _ in range(self.m_swarms):
            multiswarm.append(self.initialize_swarm())
        return multiswarm

    def optimize(self):
        start = timeit.default_timer()
        
        # Handle Cooperative PSO
        if self.algo == 'cpso':
            results = self.cpso.optimize(self.epochs, verbose=False)
            self.best_cost = results['best_cost']
            self.best_pos = results['best_pos']
            self.runtime = results['runtime']
            return
        
        # Initialize diversity monitoring if enabled
        if self.diversity_monitoring:
            self.diversity_monitor = DiversityMonitor(
                diversity_threshold=self.diversity_threshold
            )
        
        # Initialize PPSO if enabled
        if self.ppso_enabled:
            self.ppso = PPSO(
                n_particles=self.n_particles,
                dims=self.dims,
                obj_func=self.obj_func,
                bounds=(self.val_min, self.val_max),
                proactive_ratio=self.proactive_ratio,
                knowledge_method=self.knowledge_method,
                exploration_weight=self.exploration_weight,
                c1=self.c1, c2=self.c2, w=self.w,
                epochs=self.epochs
            )
            
            # Run PPSO optimization
            results = self.ppso.optimize()
            self.best_cost = results['best_cost']
            self.best_pos = results['best_pos']
            self.runtime = results['runtime']
            return
        
        # Initialize multiobjective optimization if enabled
        if self.multiobjective:
            # Check if obj_func returns multiple objectives
            test_result = self.obj_func(np.random.uniform(self.val_min, self.val_max, self.dims))
            if not isinstance(test_result, np.ndarray) or len(test_result) < 2:
                raise ValueError("Multiobjective optimization requires obj_func to return multiple objectives")
            
            # Create multiobjective optimizer
            self.mo_optimizer = SimpleMultiObjectivePSO(
                n_particles=self.n_particles,
                dims=self.dims,
                obj_func=self.obj_func,
                bounds=(self.val_min, self.val_max),
                c1=self.c1, c2=self.c2, w=self.w,
                epochs=self.epochs,
                archive_size=self.archive_size
            )
            
            # Run multiobjective optimization
            results = self.mo_optimizer.optimize()
            self.best_cost = results['pareto_front'][0]['objectives'] if results['pareto_front'] else np.array([float('inf')])
            self.best_pos = results['pareto_front'][0]['pos'] if results['pareto_front'] else None
            self.runtime = results['runtime']
            return
        
        # Standard PSO algorithms
        for i in range(self.epochs):
            if i >= int(self.hueristic_change * self.epochs):
                self.end = True
            if i % self.r == 0:
                self.regroup = True
            for particle in self.swarm:
                particle.update(i)  # Pass current iteration for inertia weight calculation
            self.update_local_best_pos()
            self.update_global_best_pos()
            self.update_global_worst_pos()
            
            # Monitor diversity and apply interventions if needed
            if self.diversity_monitoring:
                diversity_result = self.diversity_monitor.update(self.swarm)
                self.diversity_history.append(diversity_result)
                
                # Apply diversity-based interventions
                if diversity_result['needs_intervention']:
                    self._apply_diversity_intervention(diversity_result, i)
        stop = timeit.default_timer()
        self.runtime = stop - start

    def get_best_neighbor(self, particle):
        distances = []
        for other in self.swarm:
            distances.append((other.pos, other.best_cost, euclideanDistance(other.pos, particle.pos)))
        sorted_distances = sorted(distances, key=lambda x: x[2])
        return sorted(sorted_distances[:self.k+1], key=lambda x: x[1])

    def update_global_best_pos(self):
        for particle in self.swarm:
            # Handle multiobjective case
            if self.multiobjective:
                # For multiobjective, we'll handle this in the multiobjective optimizer
                continue
            
            if particle.best_cost < self.best_cost:
                self.best_cost = particle.best_cost
                self.best_pos = particle.best_pos

    def update_local_best_pos(self):
        for particle in self.swarm:
            local_best = self.get_best_neighbor(particle)
            particle.local_best_cost = local_best[0][1]
            particle.local_best_pos = local_best[0][0]

    def update_global_worst_pos(self):
        for particle in self.swarm:
            if particle.best_cost > self.worst_cost:
                self.worst_cost = particle.best_cost
                self.worst_pos = particle.best_pos

    def get_inertia_weight(self, current_iter):
        """Calculate current inertia weight based on the selected function"""
        if self.initial_cost is None:
            self.initial_cost = self.best_cost
            
        if self.inertia_func == 'constant':
            return constant_inertia_weight(self.w)
        elif self.inertia_func == 'linear':
            return linear_inertia_weight(self.w_start, self.w_end, self.epochs, current_iter)
        elif self.inertia_func == 'chaotic':
            return chaotic_inertia_weight(self.z, self.epochs, current_iter)
        elif self.inertia_func == 'random':
            return random_inertia_weight()
        elif self.inertia_func == 'adaptive':
            return adaptive_inertia_weight(self.w_start, self.w_end, self.epochs, 
                                        current_iter, self.best_cost, self.initial_cost, self.best_cost)
        elif self.inertia_func == 'chaotic_random':
            return chaotic_random_inertia_weight(self.z)
        elif self.inertia_func == 'exponential':
            return exponential_inertia_weight(self.w_start, self.w_end, self.epochs, current_iter)
        elif self.inertia_func == 'sigmoid':
            return sigmoid_inertia_weight(self.w_start, self.w_end, self.epochs, current_iter)
        else:
            # Default to linear
            return linear_inertia_weight(self.w_start, self.w_end, self.epochs, current_iter)

    def apply_velocity_clamping(self, velocity, current_iter=0):
        """Apply velocity clamping based on the selected function"""
        if self.velocity_clamp_func == 'none':
            return no_clamping(velocity, self.velocity_bounds)
        elif self.velocity_clamp_func == 'basic':
            return basic_clamping(velocity, self.velocity_bounds)
        elif self.velocity_clamp_func == 'adaptive':
            return adaptive_clamping(velocity, self.velocity_bounds, current_iter, self.epochs)
        elif self.velocity_clamp_func == 'exponential':
            return exponential_clamping(velocity, self.velocity_bounds, current_iter, self.epochs)
        elif self.velocity_clamp_func == 'sigmoid':
            return sigmoid_clamping(velocity, self.velocity_bounds, current_iter, self.epochs)
        elif self.velocity_clamp_func == 'random':
            return random_clamping(velocity, self.velocity_bounds)
        elif self.velocity_clamp_func == 'chaotic':
            return chaotic_clamping(velocity, self.velocity_bounds, self.velocity_z)
        elif self.velocity_clamp_func == 'dimension_wise':
            return dimension_wise_clamping(velocity, self.velocity_bounds)
        elif self.velocity_clamp_func == 'soft':
            return soft_clamping(velocity, self.velocity_bounds)
        elif self.velocity_clamp_func == 'hybrid':
            return hybrid_clamping(velocity, self.velocity_bounds, current_iter, self.epochs)
        elif self.velocity_clamp_func == 'convergence_based':
            return convergence_based_clamping(velocity, self.velocity_bounds, self.best_cost, self.initial_cost)
        else:
            # Default to basic clamping
            return basic_clamping(velocity, self.velocity_bounds)


class Particle:
    def __init__(self, swarm):
        self.swarm = swarm
        self.dims = swarm.dims
        self.pos = self.best_pos = self.local_best_pos = np.random.uniform(
            swarm.val_min, swarm.val_max, swarm.dims
        )
        self.velocity = np.random.uniform(
            -swarm.velocity_bounds, swarm.velocity_bounds, swarm.dims
        )
        # Use respect boundary aware objective if enabled
        if swarm.use_respect_boundary:
            self.best_cost = swarm.objective_with_respect_boundary(self.best_pos)
        else:
            self.best_cost = swarm.obj_func(self.best_pos)

    def cognitive_weight(self):
        return (self.swarm.c1 * np.random.uniform(0, 1, self.dims)) * (
            self.best_pos - self.pos
        )

    def global_weight(self):
        return (self.swarm.c2 * np.random.uniform(0, 1, self.dims)) * (
            self.swarm.best_pos - self.pos
        )

    def local_weight(self):
        return (self.swarm.c2 * np.random.uniform(0, 1, self.dims)) * (
            self.local_best_pos - self.pos
        )

    def social_weight(self):
        return (self.swarm.c2 * np.random.uniform(0, 1, self.dims)) * (
            self.local_best_pos - self.pos
        )

    def update(self, current_iter=0):
        # Get current inertia weight
        current_w = self.swarm.get_inertia_weight(current_iter)
        
        if self.swarm.algo == 'global':
            self.velocity = (current_w * self.velocity) + (self.cognitive_weight() + self.global_weight())

        if self.swarm.algo == 'local':
            self.velocity = (current_w * self.velocity) + (self.cognitive_weight() + self.local_weight())

        if self.swarm.algo == 'unified':
            g_velocity = self.swarm.u * ((current_w * self.velocity) +
                         (self.cognitive_weight() + self.global_weight()))
            l_velocity = (1-self.swarm.u) * ((current_w * self.velocity) +
                         (self.cognitive_weight() + self.local_weight()))
            self.velocity = g_velocity + l_velocity

        if self.swarm.algo == 'sa':
            if np.array_equal(self.pos, self.swarm.worst_pos):
                new_pos = np.random.uniform(self.swarm.val_min, self.swarm.val_max, self.swarm.dims)
                # Use respect boundary aware objective if enabled
                if self.swarm.use_respect_boundary:
                    new_cost = self.swarm.objective_with_respect_boundary(new_pos)
                else:
                    new_cost = self.swarm.obj_func(new_pos)
                if new_cost < self.best_cost:
                    self.best_cost = new_cost
                    self.pos = new_pos
                if self.best_cost >= getattr(self.swarm, 'alpha', 0.5):
                    best_neighbor = self.swarm.get_best_neighbor(self)
                    self.pos = best_neighbor[0][0]
                    self.best_cost = best_neighbor[0][1]

            self.velocity = (current_w * self.velocity) + (self.cognitive_weight() + self.global_weight())

        if self.swarm.algo == 'multiswarm':
            """Reshuffling"""
            if hasattr(self.swarm, 'regroup') and self.swarm.regroup:
                particles = [particle for swarm in self.swarm.multiswarm for particle in swarm]
                shuffle(particles)
                m = len(particles) // self.swarm.m_swarms
                self.swarm.multiswarm = [particles[i:i+m] for i in range(0, len(particles), m)]

            """Starting heuristic"""
            if not self.swarm.end: 
                self.velocity = (current_w * self.velocity) + (self.cognitive_weight() + self.local_weight())

            """Ending heuristic"""
            if self.swarm.end:
                self.velocity = (current_w * self.velocity) + (self.cognitive_weight() + self.global_weight())


        # Apply velocity clamping
        self.velocity = self.swarm.apply_velocity_clamping(self.velocity, current_iter)
        
        self.pos += self.velocity
        
        # Apply mutation if enabled
        if self.swarm.mutation_strategy is not None:
            self.pos = self.apply_mutation(current_iter)
        
        # Update best position if current position is better
        # Use respect boundary aware objective if enabled
        if self.swarm.use_respect_boundary:
            current_cost = self.swarm.objective_with_respect_boundary(self.pos)
        else:
            current_cost = self.swarm.obj_func(self.pos)
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_pos = self.pos.copy()
            # Reset stagnation counter on improvement
            if hasattr(self, 'stagnation_count'):
                self.stagnation_count = 0
        else:
            # Increment stagnation counter
            if hasattr(self, 'stagnation_count'):
                self.stagnation_count += 1
            else:
                self.stagnation_count = 1
    
    def apply_mutation(self, current_iter: int):
        """Apply mutation to particle position"""
        from .utils.mutation import apply_mutation, detect_stalled_particles, detect_converged_particles
        
        # Get all particles for population-based mutations
        all_particles = [p.pos for p in self.swarm.swarm]
        
        # Check if this particle is stalled or converged
        is_stalled = hasattr(self, 'stagnation_count') and self.stagnation_count >= 10
        is_converged = np.linalg.norm(self.velocity) < 1e-6
        
        # Choose mutation strategy based on particle state
        if is_stalled or is_converged:
            # Strong mutation for stuck particles
            if self.swarm.mutation_strategy == 'hybrid':
                return apply_mutation(
                    self.pos, 'escape_local_optima',
                    bounds=(self.swarm.val_min, self.swarm.val_max),
                    escape_strength=2.0
                )
            elif self.swarm.mutation_strategy == 'adaptive_strength':
                return apply_mutation(
                    self.pos, 'adaptive_strength',
                    current_iter=current_iter, max_iter=self.swarm.epochs,
                    base_strength=self.swarm.mutation_strength,
                    bounds=(self.swarm.val_min, self.swarm.val_max)
                )
            else:
                # Default strong mutation
                return apply_mutation(
                    self.pos, self.swarm.mutation_strategy,
                    mutation_rate=0.3, mutation_strength=self.swarm.mutation_strength * 2,
                    bounds=(self.swarm.val_min, self.swarm.val_max),
                    current_iter=current_iter, max_iter=self.swarm.epochs,
                    population=all_particles
                )
        else:
            # Normal mutation
            return apply_mutation(
                self.pos, self.swarm.mutation_strategy,
                mutation_rate=self.swarm.mutation_rate,
                mutation_strength=self.swarm.mutation_strength,
                bounds=(self.swarm.val_min, self.swarm.val_max),
                current_iter=current_iter, max_iter=self.swarm.epochs,
                population=all_particles
            )
    
    def _apply_diversity_intervention(self, diversity_result: dict, current_iter: int):
        """Apply diversity-based intervention to improve swarm diversity"""
        intervention = diversity_result['recommended_intervention']
        stats = diversity_result['stats']
        
        if intervention == 'restart':
            # Complete restart of worst particles
            self._restart_worst_particles(0.3)  # Restart 30% of worst particles
            
        elif intervention == 'escape_local_optima':
            # Apply strong escape mutations to converged particles
            self._apply_escape_mutations(stats)
            
        elif intervention == 'diversity_preserving':
            # Apply diversity-preserving mutations
            self._apply_diversity_mutations()
            
        elif intervention == 'opposition_based':
            # Apply opposition-based mutations
            self._apply_opposition_mutations()
            
        elif intervention == 'adaptive_strength':
            # Increase mutation strength for all particles
            self._increase_mutation_strength()
    
    def _restart_worst_particles(self, restart_ratio: float):
        """Restart worst performing particles"""
        # Sort particles by fitness
        particles_with_costs = [(p, p.best_cost) for p in self.swarm]
        particles_with_costs.sort(key=lambda x: x[1], reverse=True)
        
        # Restart worst particles
        n_to_restart = int(len(self.swarm) * restart_ratio)
        for i in range(n_to_restart):
            particle = particles_with_costs[i][0]
            # Reinitialize position
            particle.pos = np.random.uniform(self.val_min, self.val_max, self.dims)
            particle.best_pos = particle.pos.copy()
            particle.best_cost = self.obj_func(particle.pos)
            particle.velocity = np.random.uniform(-self.velocity_bounds, self.velocity_bounds, self.dims)
            particle.stagnation_count = 0
    
    def _apply_escape_mutations(self, stats: dict):
        """Apply escape mutations to converged particles"""
        from .utils.mutation import apply_mutation
        
        for particle in self.swarm:
            if stats['convergence_metrics']['is_converged']:
                # Apply strong escape mutation
                particle.pos = apply_mutation(
                    particle.pos, 'escape_local_optima',
                    bounds=(self.val_min, self.val_max),
                    escape_strength=2.0
                )
    
    def _apply_diversity_mutations(self):
        """Apply diversity-preserving mutations"""
        from .utils.mutation import apply_mutation
        
        positions = [p.pos for p in self.swarm]
        for particle in self.swarm:
            particle.pos = apply_mutation(
                particle.pos, 'diversity_preserving',
                population=positions, mutation_rate=0.3
            )
    
    def _apply_opposition_mutations(self):
        """Apply opposition-based mutations"""
        from .utils.mutation import apply_mutation
        
        for particle in self.swarm:
            particle.pos = apply_mutation(
                particle.pos, 'opposition_based',
                bounds=(self.val_min, self.val_max),
                mutation_rate=0.2
            )
    
    def _increase_mutation_strength(self):
        """Increase mutation strength for all particles"""
        # This would be implemented by temporarily increasing mutation parameters
        # For now, we'll apply adaptive strength mutations
        from .utils.mutation import apply_mutation
        
        for particle in self.swarm:
            particle.pos = apply_mutation(
                particle.pos, 'adaptive_strength',
                current_iter=0, max_iter=self.epochs,
                base_strength=self.mutation_strength * 2,
                bounds=(self.val_min, self.val_max)
            )
