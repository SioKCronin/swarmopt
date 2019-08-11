![particles](https://github.com/SioKCronin/PSO-baselines/blob/master/media/swarmopt_lateral.png)

# SwarmOpt

SwarmOpt is undergoing an overhaul, whereby this library of swarm optimization algorithm implementations will be transformed
into a prodution-ready, Dockerized MOPSO algorithm (with multiple variations made available) designed to tune CNNs. 
Stay tuned! 

Swarm intelligence leverages global population-based search solutions to balance exploration and exploitation with respect 
to specified cost functions. The PSO lineage was sparked by Eberhart and Kennedy in their original paper on PSOs in 1995, 
and the intervening years have seen many variations spring from their central idea. 

## Installation

To install SwarmpOpt, run this command in your terminal:

```shell
$ pip install swarmopt
```

## Algorithms

### Single-Objective
* Global Best PSO - Kennedy & Eberhart 1995
* Local Best PSO - Kennedy & Eberhart 1995
* Unified PSO - Parsopoulos &  Vrahatis 2004
* Dynamic Multi-Swarm PSO - Liang & Suganthan 2005
* Simulated Annealing PSO - Mu, Cao, & Wang 2009

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
