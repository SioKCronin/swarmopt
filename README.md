![particles](https://github.com/SioKCronin/PSO-baselines/blob/master/media/swarmopt_lateral.png)

# SwarmOpt

SwarmOpt is a swarm intelligence optimizer for hyperparameter tuning. The project is currently in R&D, where I'm benchmarking variations, with a particular focus on hybrid topologies and velocity clamp variations. The current frontrunner is a simple global best, but I expect it won't stay out in front for long.  

Swarm intelligence leverages global population-based search solutions to balance exploration and exploitation with respect to specified cost functions. There are many exciting nooks and crannies to explore in the SI toplogy, yet I've chosen to kick things of with some Particle Swarm Optimization (PSO) algorithms, as they are easy to understand and fiddle with. The PSO lineage was sparked by Eberhart and Kennedy in their original paper on PSOs in 1995, and the intervening years have seen many variations spring from their central idea. 

## Installation

To install SwarmpOpt, run this command in your terminal:

```shell
$ pip install swarmopt
```

## Algorithms
* [Global Best PSO](https://github.com/SioKCronin/SwarmOpt/tree/master/algorithms/gbest_pso) - Kennedy & Eberhart 1995
* [Local Best PSO](https://github.com/SioKCronin/SwarmOpt/tree/master/algorithms/lbest_pso) - Kennedy & Eberhart 1995
* [Unified PSO](https://github.com/SioKCronin/SwarmOpt/tree/master/algorithms/unified_pso) - Parsopoulos &  Vrahatis 2004
* [Dynamic Multi-Swarm PSO](https://github.com/SioKCronin/SwarmOpt/tree/master/algorithms/dynamic_multiswarm_pso) - Liang & Suganthan 2005
* [Simulated Annealing PSO](https://github.com/SioKCronin/SwarmOpt/tree/master/algorithms/sim_annealing_pso) - Mu, Cao, & Wang 2009

## Benchmark Functions

Single objective test functions:
* Sphere Function
* Rosenbrock's Function
* Ackley's Function
* Griewank's Function
* Rastrigin's Function
* Weierstrass Function

## On Deck

* Cooperative Approach to PSO (CPSO)(multiple collaborating swarms)
* Proactive Particles in Swarm Optimization (PPSO) (self-tuning swarms)
* Inertia weight variations
* Mutation operator variations
* Velocity clamping variations
* Multiobjective variations
* Benchmark on something canonical like MNIST

## Applications

* Neural network number of layers and weight optimization
* Grid scheduling (load balancing)
* Routing in communication networks
* Anomaly detection

## Citation

Siobhán K Cronin, SwarmOpt (2018), GitHub repository, https://github.com/SioKCronin/SwarmOpt
