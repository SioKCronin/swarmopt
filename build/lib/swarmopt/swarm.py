import numpy as np
import timeit


class Swarm:
    def __init__(self, n_particles, dims, c1, c2, w, epochs, obj_func, v_clamp):
        self.epochs = epochs
        self.n_particles = n_particles
        self.runtime = 0

        self.dims = dims
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.obj_func = obj_func
        self.val_min, self.val_max = v_clamp
        self.velocity_bounds = 0.2 * (self.val_max - self.val_min)

        self.best_cost = float('inf')
        self.best_pos = None
        self.swarm = self.initialize_swarm()
        self.update_best_pos()

    def shape(self):
        return [self.n_particles, self.dims]

    def initialize_swarm(self):
        swarm = []
        for particle in range(self.n_particles):
            swarm.append(Particle(self))
        return swarm

    def optimize(self):
        start = timeit.default_timer()
        for _ in range(self.epochs):
            for particle in self.swarm:
                particle.update()
            self.update_best_pos()
        stop = timeit.default_timer()
        self.runtime = stop - start

    def update_best_pos(self):
        for particle in self.swarm:
            if particle.best_cost < self.best_cost:
                self.best_cost = particle.best_cost
                self.best_pos = particle.best_pos


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

    def social_weight(self):
        return (self.swarm.c2 * np.random.uniform(0, 1, self.dims)) * (
            self.swarm.best_pos - self.pos
        )

    def weight(self):
        return self.cognitive_weight() + self.social_weight()


    def update(self):
        cost = self.swarm.obj_func(self.pos)

        if cost < self.best_cost:
            self.best_pos = self.pos
            self.best_cost = cost

        self.velocity = (self.swarm.w * self.velocity) + self.weight()
        self.pos += self.velocity
