![particles](https://github.com/SioKCronin/PSO-baselines/blob/master/common/media/particles.png)

# PSO Baselines

There have been many variations on the theme of particle swarm optimization since the original algorithm was presented by Eberhart and Kennedy in 1995, yet few open source attempts to compile code for these varitions in a single archive. This repo is dedicated to providing high quality implementations of PSO variations Python, organized by their application. My hope is that by including a wide variety of algorithms I can help underscore the potential of stochastic process in optimization in general, and perhaps prime the canvas for algorithm innovation.

## Algorithms
### Single Objective 

* Local Best PSO ([LBEST_PSO](https://github.com/SioKCronin/PSO-baselines/tree/master/pso))
* Global Best PSO ([GBEST_PSO](https://github.com/SioKCronin/PSO-baselines/tree/master/pso))
* Unified PSO ([UPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/upso))
* Dynamic Multi-Swarm PSO ([DMSPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/dmspso))

### Multi Objective

* Dynamic Neighborhood ([DNPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/dnpso))

## Benchmark Functions

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
