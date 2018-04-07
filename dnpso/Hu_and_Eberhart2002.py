import numpy as np
import math
import sys
sys.path.append('../')
from dnpso import dynamic_neighborhood_pso
from common.functions.multiobjective import *

def run_all_tests(algo, n, dims, c1, c2, w, k, iters, obj_func1, obj_func2):

    def run_20(algo, n, dims, c1, c2, w, iters, obj_func1, obj_func2,
                                                    val_min, val_max):
        vals = []
        for i in range(20):
            vals.append(algo(n, dims, c1, c2, w, iters, obj_func,\
                                                val_min, val_max))
        return np.mean(vals)

    lis_and_eiben = [[lis_and_eiben1, -5, 10],
                     [lis_and_eiben2, -5, 10],
                     [lis_and_eiben3, -100, 100],
                     [lis_and_eiben4, -100, 100]]

    #zitzler = [[]]

    for func in funcs:
        print("--------------------------")
        print("Testing %s" %func[0])
        print("Single run test:", algo(n, dims, c1, c2, w, iters, func[0],\
                                                       func[1], func[2]))
        print("Run 20:", run_20(algo, n, dims, c1, c2, w, iters, func[0],\
                                                       func[1], func[2]))

test_runner.run_all_tests(dynamic_neighborhood_pso,n=30,dim=2,
                          c1=0.5,c2=0.3,w=0.9,k=2,iters=2000,
                          obj_func1=lis_and_eiben1,obj_func2=lis_and_eiben2)
