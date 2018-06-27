import numpy as np
import math
import sys
from swarmopt import Swarm, Particle, functions
import csv
from time import gmtime, strftime
import os

# class GobalSwarm(Swarm):

class LocalSwarm(Swarm):

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

class Unified(Swarm):

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

class DynamicMultiswarm(Swarm):

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


class SimulatedAnnealing(Swarm):

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
