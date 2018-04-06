import test.test_runner
from pso.global_best_pso import global_best_pso
from pso.local_best_pso import local_best_pso
from dmspso.dmspso import dynamic_multiswarm_pso

# Global Best
test_runner.run_all_tests(global_best_pso,
                          n=30,
                          dim=2,
                          c1=0.5,
                          c2=0.3,
                          w=0.9,
                          iters=2000)
# Local Best
test_runner.run_all_tests(local_best_pso,
                          n=30,
                          dim=2,
                          c1=0.5,
                          c2=0.3,
                          w=0.9,
                          iters=2000)

# Dynamic Multiswarm PSO - Lang & Suganthan 2015
test_runner.run_all_tests(dynamic_multiswarm_pso,
                          n=30,
                          dim=2,
                          c1=0.5,
                          c2=0.3,
                          w=0.9,
                          iters=2000)

# Unified PSO
## Set 1
test_runner.run_all_tests(unified_pso,
                          n=30,
                          dim=2,
                          c1=0.5,
                          c2=0.3,
                          w=0.9,
                          u=0.9,
                          mu=0,
                          std=0.01,
                          weight=g,
                          iters=2000)
## Set 2
test_runner.run_all_tests(unified_pso,
                          n=30,
                          dim=2,
                          c1=0.5,
                          c2=0.3,
                          w=0.9,
                          u=0.9,
                          mu=1,
                          std=0.01,
                          weight=l,
                          iters=2000)

