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


class Swarm:
    def __init__(self, n_particles, dims, c1, c2, w, epochs, obj_func,
                 algo='global', inertia_func='linear', velocity_clamp=(0,2),
                 k=5, u=0.5, m_swarms=3, hueristic_change=0.9, r=5,
                 w_start=0.9, w_end=0.4, z=0.5, velocity_clamp_func='basic'):
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

        self.obj_func = obj_func
        self.best_cost = float('inf')
        self.best_pos = None
        self.worst_cost = float('inf')
        self.worst_pos = None
        self.local_best_cost = float('inf')
        self.local_best_pos = None

        if self.algo == 'multiswarm':
            self.multiswarm = self.initialize_multiswarm()
        else:
            self.swarm = self.initialize_swarm()

        self.update_global_best_pos()

    def shape(self):
        return [self.n_particles, self.dims]

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
        
        # Update best position if current position is better
        current_cost = self.swarm.obj_func(self.pos)
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_pos = self.pos.copy()
