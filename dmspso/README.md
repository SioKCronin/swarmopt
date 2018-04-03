# Dynamic MultiSpecies Particle Swarm Optimization (DMSPSO)

This algorithm is defined by its shuffling of particles between parallel swarms for 90% of the iterations, followed by a combined global best search for the remaining 10% iterations. 

## Results 
DMSPSO is run 20 times on each function, and I believe the authors reported the minimum costs obtained for each run of 20. 

### Results achieved under different m (6 test funcs)
| L&S 2005  |  2 | 3 |  5 | 
|---                 |---|---|---|
|  1 | 0  |   0|  0 | 
|  2 |  3.0127e+000 | 1.3612e+000  |  9.9970e-001 | 
|  3 |  3.1974e-015 | 2.1316e-015  |  1.4211e-015 |
|  4 |  3.2935e-002 | 3.2496e-002  |  3.7349e-002 |
|  5 |  4.9325e+000 | 3.6068e+000  |  5.3728e+000 |
|  6 |  8.6547e-002 | 1.2269e-004  |  1.0408e-005 | 

|PSO Baselines   | 2  | 3  | 5 |
| ---              |---|---| ---|
| 1 |  |  | |
| 2 | 2.5009e-001 | 1.7143e-001 | 4.7632e-002 |
| 3 | 5.1050e-005 | 5.6564e-009 | 1.2719e-009 |
| 4 | 3.7622e-003 | 4.6382e-003 | |
| 5 | | | |
| 6 | | | |

### Results achieved under different R (6 test funcs)
| L&S 2005  |  2 | 3 |  5 | 10 | 20 | 50 
|---|---|---|---|---|---|---|
|  1 |  0 | 0 | 0 | 0 | 0 | 0 |
|  2 | 1.7154e+000 |1.5024e+000 | 1.0910e000 | 1.3612e+000 | 8.1698e-001 | 1.7155e+000 | 
|  3 |  0 |  0 | 0  | 2.1316e-015 | 3.1974e-015| 4.6185e-015| 
|  4 | 5.3793e-002 | 4.7304e-002 | 2.3389e-002 | 3.2496e-002 | 5.2159e-002 | 3.6415e-002 | 
|  5 | 5.9381e+000 | 3.2946e+000 | 3.2767e+000 | 3.6068e+000 | 4.7446e+000 | 5.5718e+000 | 
|  6 | 7.1016e-003 | 3.8061e-003 | 0           | 1.2269e-004 | 6.9781e-002 | 1.9784e-001 | 

|PSO Baselines   | 2  | 3  | 5 | 10 | 20 | 50|
| ---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
