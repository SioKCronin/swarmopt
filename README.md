# Swarm Baselines

Swarm optimization refers to one of several variations of nature-inspired optimization heuristics, including particle swarm optimization (PSO), presented by Eberhart and Kennedy in 1995. In addition to PSO variants, this branch of optimization has sprouted many blossoms, ranging from ant colony optimization (ACO) to the Krill herd algorithm (prepare yourself for an onlsaught of acroynyms).  

I've started this repo to provide high quality implemnetations in Python, organized by their application. My hope is that by including a wide variety I can help underscore the potential of stochastic process in optimization, and perhaps prime the canvas for algorithm innovation.

In the docs, I ahve included information on what sets each algorithm apart, and examples of when each might be best applied.  

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

PSO + Q-learning:
* CartPole

## Questions to ask yourself

* What should guide our decision process in selecting one method over another?
* If an algorithm is performing well in a particular context, why might that be? What is unique about that problem?
* Do our metrics effectively capturing the learning rate/timecourse of learning?
* What industry problems should we try these on?
* Are there any other notable swarm intelligence variants we should consider here?
* How can we best measure realtive performance across algorithms? 
