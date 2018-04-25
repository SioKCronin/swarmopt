![particles](https://github.com/SioKCronin/PSO-baselines/blob/master/common/media/particles.png)

# Swarm Intelligence Baselines

High quality implementations of Swarm Intelligence (SI) algorithms in Python, starting with some examples from the family of Particle Swarm Optimization (PSO) algorithms in Python. Algorithms are rganized by application, and efforts are made to include code to replicate code from originating papers.  

SI algorithms allow us to search a solutions landscape in any context where we want to balance exploration with exploitation with respect to a cost functions (or functions, in the multi-objective cases). The PSO metahueristic family was launched by Eberhart and Kennedy in their original paper on PSOs in 1995, and the intervening years have seen many variations springing forth from the central idea of distributing search of optima across multiple agents. 

This project is motivated by a desire to support the advance of SI algorithm design and industry application. 

## Algorithms
### Single Objective 

* Unified PSO ([UPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/upso)) - Parsopoulos &  Vrahatis 2004
* Dynamic Multi-Swarm PSO ([DMSPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/dmspso)) - Liang & Suganthan 2005
* Simulated Annealing PSO ([SAPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/sapso)) - Mu, Cao, & Wang 2009

### Multi Objective

* Dynamic Neighborhood ([DNPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/dnpso)) - Hu & Eberhart 2002

### Rinforcement Learning as Optimization Task

* PSO Policy ([PSOP](https://github.com/SioKCronin/PSO-papers/tree/master/psop)) - Hein et al. 2016

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

## Citation

Siobhán K Cronin, PSO Baselines, (2018), GitHub repository, https://github.com/SioKCronin/PSO-baselines
