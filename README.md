![particles](https://github.com/SioKCronin/PSO-baselines/blob/master/media/swarmopt_lateral.png)

# SwarmOpt

SwarmOpt is a swarm intelligence optimizer for hyperparameter tuning. The project is currently in R&D, where I'm implementing swarm intelligence algorithms to find the best variation for hyperparemeter tuning. I'm particularly interested in comparing velocity clamp variations, and implementing test functions that most closely resemble the kinds of complex search environments encountered in hyperparameter tuning.  

Swarm intelligence leverages global population-based search solutions to balance exploration and exploitation with respect to specified cost functions. There are many exciting nooks and crannies to explore in the SI toplogy, yet I've chosen to kick things of with some Particle Swarm Optimization (PSO) algorithms, as they are easy to understand and fiddle with. The PSO lineage was sparked by Eberhart and Kennedy in their original paper on PSOs in 1995, and the intervening years have seen many variations spring from their central idea. 

## Algorithms
* PSO (global and local best)([PSO](https://github.com/SioKCronin/SwarmOpt/tree/master/algorithms/pso)) - Kennedy & Eberhart 1995
* Unified PSO ([UPSO](https://github.com/SioKCronin/SwarmOpt/tree/master/algorithms/upso)) - Parsopoulos &  Vrahatis 2004
* Dynamic Multi-Swarm PSO ([DMSPSO](https://github.com/SioKCronin/SwarmOpt/tree/master/algorithms/dmspso)) - Liang & Suganthan 2005
* Simulated Annealing PSO ([SAPSO](https://github.com/SioKCronin/SwarmOpt/tree/master/algorithms/sapso)) - Mu, Cao, & Wang 2009

## Benchmark Functions

Single objective test functions:
* Sphere Function
* Rosenbrock's Function
* Ackley's Function
* Griewank's Function
* Rastrigin's Function
* Weierstrass Function

## On Deck

* Improved Particle Swarm Optimization (IPSO)
* Proactive Particles in Swarm Optimization (PPSO)
* Cooperative Approach to PSO (CPSO) 
* Artificial Bee Colony (ABC)
* Ant Colony Optimization (ACO)

## Applications

* Neural network number of layers and structure optimization
* Grid scheduling (load balancing)
* Routing in communication networks

## Citation

Siobhán K Cronin, SwarmOpt (2018), GitHub repository, https://github.com/SioKCronin/SwarmOpt
