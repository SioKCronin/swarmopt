![particles](https://github.com/SioKCronin/PSO-baselines/blob/master/common/media/particles.png)

# PSO Baselines

This repo is dedicated to providing high quality implementations of particle swarm optimization algorithms in Python, organized by their application. My hope is that by including a wide variety of algorithms I can help underscore the potential of stochastic process in optimization in general, and perhaps prime the canvas for algorithm innovation.

Particle swarm optimization (PSO) refers to one of several variations of nature-inspired optimization heuristics, originally presented by Eberhart and Kennedy in 1995. In the docs, I have included information on what sets each algorithm apart, and examples of when each might be best applied.  

## Algorithms
### Single Objective 

* Local Best PSO ([LBEST_PSO](https://github.com/SioKCronin/PSO-baselines/tree/master/PSO))
* Global Best PSO ([GBEST_PSO](https://github.com/SioKCronin/PSO-baselines/tree/master/PSO))
* Unified PSO ([UPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/UPSO))
* Dynamic Multi-Swarm PSO ([DMSPSO](https://github.com/SioKCronin/swarm-baselines/tree/master/DMSPSO))

### Multi Objective

* Dynamic Neighborhood ([DNPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/dnpso))
* Multispecies PSO ([MSPSO](https://github.com/SioKCronin/swarm-baselines/tree/master/MSPSO))

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
