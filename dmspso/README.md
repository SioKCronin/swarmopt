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
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |

### Results achieved under different R (6 test funcs)
| L&S 2005  |  2 | 3 |  5 | 10 | 20 | 50 
|---|---|---|---|---|---|---|
|  1 |   |   |   |  | | |
|  2 |   |   |   |  | | | 
|  3 |   |   |   |  | | | 
|  4 |   |   |   |  | | | 
|  5 |   |   |   |  | | | 
|  6 |   |   |   |  | | | 

|PSO Baselines   | 2  | 3  | 5 | 10 | 20 | 50|
| ---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
