![particles](https://github.com/SioKCronin/PSO-baselines/blob/master/common/media/particles.png)

# Swarm Lab

Swarm Lab is dedicated to connecting academic swarm intelligence research with the developer community. The backbone of the lab is a collection of high quality implementations of swarm intelligence algorithms in Python, organized by application. We will also be adding visualization tools and a suite of real-world inspired testing functions to support ideation and development of new algorithms.

Swarm intelligence employs global population-based solutions that balance exploration and exploitation with respect to specified cost functions (or functions, in the multi-objective cases). The Particle Swarm Optimization (PSO) lineage was sparked by Eberhart and Kennedy in their original paper on PSOs in 1995, and the intervening years have seen many variations spring forth from their central idea. We think PSOs are a nice place to start when studying SI, as they are easy to understand and fiddle with, so we've begun there. 

## Algorithms
### Single Objective 

* PSO (global and local best)([PSO](https://github.com/SioKCronin/SwarmLab/tree/master/pso)) - Kennedy & Eberhart 1995
* Unified PSO ([UPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/upso)) - Parsopoulos &  Vrahatis 2004
* Dynamic Multi-Swarm PSO ([DMSPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/dmspso)) - Liang & Suganthan 2005
* Simulated Annealing PSO ([SAPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/sapso)) - Mu, Cao, & Wang 2009

### Multi Objective

* Dynamic Neighborhood PSO ([DNPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/dnpso)) - Hu & Eberhart 2002

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

RL benchmark environments:
* Mountain Car
* Cart Pole

## Visualization and Testing

Currently using [SwarmViz](https://github.com/epfl-disal/SwarmViz)

## On Deck

* Improved Particle Swarm Optimization (IPSO)
* Proactive Particles in Swarm Optimization (PPSO)
* Swarm Reinforcement Learning based on PSO (SRL-PSO)
* RL with PSO Policy (PSO-P)
* Dynamic Multiple Swarms in Multiobjective PSO (DMSMPSO)
* Cooperative Approach to PSO (CPSO) 
* Artificial Bee Colony (ABC)

## Application

* Scheduling for cloud computing
* Neural network number of layers and structure optimization
* Grid scheduling (load balancing)
* Routing in communication networks

## Citation

Siobhán K Cronin, Swarm Lab (2018), GitHub repository, https://github.com/SioKCronin/SwarmLab
