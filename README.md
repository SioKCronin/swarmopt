# PSO Baselines

Particle swarm optimization (PSO) refers to one of several varations of the classic stochastic optimization heuristic developed by Kennedy and Eberhart. There are many particle swarm optimization (PSO) variations out there (prepare yourself for an onlsaught of acroynyms). 

These implementations require PySwarms, an open-source research toolkit. Some require gym. 

In addition to the implementations, I have included information in the docs about what differentiates them, and when each might be used (particularly in reinforcement learning). 

## Algorithms
### Single Objective 
| Name  | layers | params  | 
|---|---|---|
| GBEST |  2 | n, i, c1, c2, w  | 
| LBEST |  2 | n, i, c1, c2, w, k, p |
| DMS-PSO | 2 | n, R   |
| MLPSO | n |   |

### PSO + Q-learning combinations
| Name  | layers | params  |
|---|---|---|
| SRL-PSO |  | c1, c2, w |
| QSO|   |   |
| Intelligent PSO |   |   |

## Comparison Benchmark Functions

Single objective test functions:
* Sphere Function
* Rosenbrock's Function
* Ackley's Function
* Griewank's Function
* Rastrigin's Function
* Weierstrass Function

PSO + Q-learning:
* CartPole

## Questions to ask yourself

* What about the nature of this search space would lead us to select one method over another?
* Why might we be seeing the performance we're seeing?
* Are our metrics effectively capturing the learning rate/timecourse of learning?
* What other RL problems should we try these on?
* Any other notable PSO variants we should consider?
* How to best measure performance and compare across algorithms? 
