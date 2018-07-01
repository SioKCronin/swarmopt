import numpy as np
import timeit


class Swarm:
    def __init__(self, n_particles, dims, c1, c2, w, epochs, obj_func,
                 algo='global', inertia_func='linear', velocity_clamp=None,
                 k=5, u=0.5, m_swarms=3, hueristic_change=0.9, r=5):
        """Intialize the swarm.

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

        self.val_min, self.val_max = v_clamp
        self.velocity_bounds = 0.2 * (self.val_max - self.val_min)

        self.obj_func = obj_func
        self.best_cost = float('inf')
        self.best_pos = None
        self.worst_cost = float('inf')
        self.worst_pos = None
        self.local_best_cost = float('inf')
        self.local_best_pos = None

        if self.algo = 'multiswarm':
            self.multiswarm = self.initialize_multiswarm()
        else:
            self.swarm = self.initialize_swarm()

        self.update_best_pos()

    def shape(self):
        return [self.n_particles, self.dims]

    def initialize_swarm(self):
        swarm = []
        for particle in range(self.n_particles):
            swarm.append(Particle(self))
        return swarm

    def initialize_multiswarm(self):
        multiswarm = []
        for _ in self.m:
            m.append(intialize_swarm())
        return multiswarm

    def optimize(self):
        start = timeit.default_timer()
        for i in range(self.epochs):
            if i >= int(self.hueristic_change * self.epochs):
                self.end = True
            if i % self.r == 0:
                self.regroup = True
            for particle in self.swarm:
                particle.update()
                self.update_local_best_pos()
                self.update_global_best_pos()
                self.update_global_worst_pos()
        stop = timeit.default_timer()
        self.runtime = stop - start

    def get_best_neighbor(self, particle):
        distances = []
        for other in self.swarm:
            distances.append((other.pos, other.best_cost, calc_distance(other.pos, particle.pos)))
        sorted_distances = sorted(distances, key=lambda x: x[2])
        return sorted(sorted_distances[:self.k+1], key=lambda x: x[1])

    def update_global_best_pos(self):
        for particle in self.swarm:
            if particle.best_cost < self.best_cost:
                self.best_cost = particle.best_cost
                self.best_pos = particle.best_pos

    def update_local_best_pos(self):
        for particle in self.swarm:
            local_best = get_best_neighbor(particle)
            particle.local_best_cost = local_best[1]
            particle.local_best_pos = local_best[0]


class Particle:
    def __init__(self, swarm):
        self.swarm = swarm
        self.dims = swarm.dims
        self.pos = self.best_pos = np.random.uniform(
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

    def update(self):
        if self.swarm.algo == 'global':
            self.velocity = (self.swarm.w * self.velocity) +
                            (self.cognitive_weight() + self.global_weight())

        if self.swarm.algo == 'local':
            self.velocity = (self.swarm.w * self.velocity) +
                            (self.cognitive_weight() + self.local_weight())

        if self.swarm.algo == 'unified':
            g_velocity = self.u * ((self.swarm.w * self.velocity) +
                         (self.cognitive_weight() + self.global_weight()))
            l_velocity = (1-self.u) * ((self.swarm.w * self.velocity) +
                         (self.cognitive_weight() + self.local_weight()))
            self.velocity = g_velocity + l_velocity

        if self.swarm.algo == 'sa':
            if self.pos == self.swarm.worst_pos:
                new_pos = np.random.uniform(self.swarm.val_min, self.swarm.val_max, self.swarm.dims)
                new_cost = self.swarm.obj_func(new_pos)
                if new_cost < self.cost:
                    self.cost = new_cost
                    self.pos = new_pos
                if self.cost >= self.swarm.alpha:
                    best_neighbor = getNeighbor(self.pos)
                    self.pos = best_neighbor[0]
                    self.cost = best_neighbor[1]

            self.velocity = (self.swarm.w * self.velocity) +
                            (self.cognitive_weight() + self.global_weight())

        if self.swarm.algo == 'multiswarm':
            """Reshuffling"""
            if self.swarm.r:
                particles = [particle for swarm in self.swarm.multiswarm]
                shuffle(particles)
                self.swarm.multiswarm = [particles[i:i+m] for i in range(0, len(particles), m)]

            """Startng heuristic"""
            if not self.swarm.end: 
                self.velocity = (self.swarm.w * self.velocity) +
                                (self.cognitive_weight() + self.local_weight())

            """Ending hueristic"""
            if self.swarm.end:
                self.velocity = (self.swarm.w * self.velocity) +
                                (self.cognitive_weight() + self.global_weight())


        self.pos += self.velocity + self.pos
