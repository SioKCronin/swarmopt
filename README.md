![particles](https://github.com/SioKCronin/PSO-baselines/blob/master/common/media/particles.png)

# Swarm Intelligence Baselines

High quality implementations of Swarm Intelligence (SI) algorithms in Python, starting with some examples from the family of Particle Swarm Optimization (PSO) algorithms in Python. Algorithms are rganized by application, with results replicated from originating papers.  

SI algorithms allow us to search a solutions landscape in any context where we want to balance exploration with exploitation with respect to a cost functions (or functions, in the multi-objective cases). The PSO metahueristic family was launched by Eberhart and Kennedy in their original paper on PSOs in 1995, and the intervening years have seen many variations springing forth from the central idea of distributing search of optima across multiple agents. 

This project is motivated by a desire to facilitate the design of SI algorithm variations, and to make it easier for developers to apply SI algorithms in novel industry applications. 

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

## On Deck

* Improved Particle Swarm Optimization (IPSO)
* Proactive Particles in Swarm Optimization (PPSO)
* Swarm Reinforcement Learning based on PSO (SRL-PSO)
* RL with PSO Policy (PSO-P)
* Dynamic Multiple Swarms in Multiobjective PSO (DMSMPSO)
* Cooperative Approach to PSO (CPSO) 

## Citation

Siobhán K Cronin, Swarm Intelligence Baselines, (2018), GitHub repository, https://github.com/SioKCronin/swarm-intelligence-baselines
