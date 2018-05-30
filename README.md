![particles](https://github.com/SioKCronin/PSO-baselines/blob/master/common/media/swarmopt_lateral.png)

# Swarm Opt

SwarmOpt is a swarm intelligence optimizer for hyperparameter tuning. The project is currently in R&D, where I'm implementing various swarm intelligence algorithms to find the best variation for tuning. 

Swarm intelligence leverages global population-based search solutions to balance exploration and exploitation with respect to specified cost functions. There are many exciting nooks and crannies to explore in the SI toplogy, yet I've chosen to kick things of with some Particle Swarm Optimization (PSO) algorithms, as they are easy to understand and fiddle with. The PSO lineage was sparked by Eberhart and Kennedy in their original paper on PSOs in 1995, and the intervening years have seen many variations spring from their central idea. 

The implementations reveal a lot of the shared architecture across these algorithms, and as I shift towards a tidy optimizer module, these implementations will shift to being housed at [SwarmLab](https://github.com/SioKCronin/SwarmLab), where I also aim to add some **visualization tools** and **a comparative analysis scoreboard**. Stay tuned and definitely reach out if you're up for collaborating!

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
