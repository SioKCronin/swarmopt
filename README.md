# Swarm Baselines

Swarm optimization refers to one of several variations of nature-inspired optimization algorithms/hueristics/meta-heuristics.
These range from particle swarm optimization (PSO) variations to the Krill herd algorithm (prepare yourself for an onlsaught of acroynyms), and while the implementations here represent a vast terrain of research, I have included this variety to encourage the overall observance of how stochastic process is leveraged in optimization. In the docs I have also information on what sets each algorithm apart, and when each might be used. 

## Algorithms
### Single Objective 

* Local Best (LBEST)
* Global Best (GBEST)
* Multispecies ([MSPSO](https://github.com/SioKCronin/swarm-baselines/tree/master/MSPSO))
* Dynamic MultiSpecies ([DMSPSO](https://github.com/SioKCronin/swarm-baselines/tree/master/DMSPSO))
* MultiLayer (MLPSO) 

### PSO + Q-learning

* Swarm RL (SRL-PSO)
* Q Swarm Optimizer (QSO) 
* Intelligent PSO

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
