# Swarm Baselines

Swarm optimization refers to one of several variations of nature-inspired optimization algorithms/hueristics/meta-heuristics.
These range from particle swarm optimization (PSO) variations to the Krill herd algorithm (prepare yourself for an onlsaught of acroynyms), and while the implementations here represent a vast terrain of research, I have included this variety to encourage the overall observance of how stochastic process is leveraged in optimization. In the docs I have also information on what sets each algorithm apart, and when each might be used. 

## Algorithms
### Single Objective 
| Name  | layers | params  | 
|---|---|---|
| GBEST |  2 | n, i, c1, c2, w  | 
| LBEST |  2 | n, i, c1, c2, w, k, p |
| DMS-PSO | 2 | n, R   |
| MLPSO | n |   |

### PSO + Q-learning
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
