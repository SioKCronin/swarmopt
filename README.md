# PSO Baselines

This repo is dedicated to providing high quality implementations of particle swarm optimization algorithms in Python, organized by their application. My hope is that by including a wide variety of algorithms I can help underscore the potential of stochastic process in optimization in general, and perhaps prime the canvas for algorithm innovation.

Particle swarm optimization (PSO) refers to one of several variations of nature-inspired optimization heuristics, originally presented by Eberhart and Kennedy in 1995. In the docs, I have included information on what sets each algorithm apart, and examples of when each might be best applied.  

## Algorithms
### Single Objective 

* Local Best (LBEST_PSO)
* Global Best (GBEST_PSO)
* Unified (UPSO)
* Multispecies ([MSPSO](https://github.com/SioKCronin/swarm-baselines/tree/master/MSPSO))
* Dynamic MultiSpecies ([DMSPSO](https://github.com/SioKCronin/swarm-baselines/tree/master/DMSPSO))
* MultiLayer (MLPSO) 

### Multi Objective

* Dynamic Neighborhood (DNPSO)

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

Multi objective test functions:
* Lis & Eiben
* Zitzler

PSO + Q-learning:
* CartPole

## Questions to ask yourself

* What should guide our decision process in selecting one method over another?
* If an algorithm is performing well in a particular context, why might that be? What is unique about that problem?
* Do our metrics effectively capturing the learning rate/timecourse of learning?
* What industry problems should we try these on?
* Are there any other notable swarm intelligence variants we should consider here?
* How can we best measure realtive performance across algorithms? 
