# DMSPSO

import numpy as np
from pyswarms.utils.functions import single_obj as fx
from base import SwarmBase
from dynamic_lpso import DynamicPSO 
from random import shuffle

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'R': 10}

m = 2  # Each swarm's population size (2, 3, 5 tested)
n = 10 # Number of swarms (10, 20, 50 tested)
c1 = 0.5
c2 = 0.3
w = 0.9
R = 10
max_gen = 2000

params = []
best_params[]
cost = []
best_pos = 0
velocity = []


lpso_optimizer = ps.single.LocalBestPSO(n_particles=5, dimensions=2, options=options)



optimizer1 = ps.single.GlobalBestPSO(n_particles=5, dimensions=2, options=options)

params.append(np.random.uniform(-1,1,m*n))

params.append(all_subswarms)
shuffle(params)

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

swarm_params = [params[i:i + n] for i in range(0, len(params), n)]



cost, pos = optimizer1.optimize(fx.sphere_func, print_step100, iters=1000, verbose=3)

