import numpy as np

class Swarm:
    def __init__(self, n_particles, dims, c1, c2, w, epochs, obj_func, v_clamp):
        self.n_particles = n_particles
        self.dims = dims
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.epochs = epochs
        self.obj_func = obj_func
        self.val_min, self.val_max = v_clamp
        self.max_start_velocity = 0.2 * (self.val_max - self.val_min)
        self.swarm = self.initialize_swarm()

    def shape(self):
        return [self.n_particles, self.dims]

    def initialize_swarm(self):
        swarm = []
        for particle in range(self.n_particles):
            swarm.append(
                Particle(
                    self.val_min,
                    self.val_max,
                    self.max_start_velocity,
                    self.dims,
                    self.obj_func,
                )
            )
        return swarm

class Particle:
    def __init__(self, val_min, val_max, max_start_velocity, dims, obj_func, seed=None):
        random = np.random.RandomState(seed=seed)
        self.pos = self.best_pos = list(random.uniform(val_min, val_max, dims))
        self.velocity = list(random.uniform(-max_start_velocity, max_start_velocity, dims))
        self.best_cost = obj_func(self.best_pos)
