#Benchmark PSO Global Best

import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k':2, 'p':2}

optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=2, options=options)

cost, pos = optimizer.optimize(fx.sphere_func, print_step=100, iters=1000, verbose=3)
