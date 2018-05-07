![particles](https://github.com/SioKCronin/PSO-baselines/blob/master/common/media/particles.png)

# Swarm Intelligence Baselines

This project is motivated by a desire to facilitate the design of SI algorithms, and to make it easier for developers to apply SI algorithms in novel applications. The repo includes high quality implementations of swarm intelligence algorithms in Python, and tools for testing new algorithms. Algorithms are organized by application. 

SI algorithms provide global population-based search solutions in contexts where we want to balance exploration with exploitation with respect to a cost functions (or functions, in the multi-objective cases). The PSO metahueristic family was launched by Eberhart and Kennedy in their original paper on PSOs in 1995, and the intervening years have seen many variations springing forth from the central idea of distributing search of optima across multiple agents. I think this is a nice place to start, as these approaches are easy to understand and fiddle with.

## Algorithms
### Single Objective 

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

## Application

* Scheduling for cloud computing
* Neural network number of layers and structure optimization
* Grid scheduling (load balancing)
* Routing in communication networks

## Citation

Siobhán K Cronin, SwarmLab (2018), GitHub repository, https://github.com/SioKCronin/SwarmLab
