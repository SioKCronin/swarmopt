![particles](https://github.com/SioKCronin/PSO-baselines/blob/master/common/media/particles.png)

# PSO Baselines

High quality implementations of PSO variations in Python, organized by application, with tools to compare performance on benchmark functions and build intuition. This project is motivated by a desire to advance algorithm design by exploring the relative success of common varitions in PSOs, and to press for more robust measures of performance relative to specific objective functions.  

## Algorithms
### Single Objective 

* Unified PSO ([UPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/upso)) - Parsopoulos &  Vrahatis 2004
* Dynamic Multi-Swarm PSO ([DMSPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/dmspso)) - 

### Multi Objective

* Dynamic Neighborhood ([DNPSO](https://github.com/SioKCronin/PSO-baselines/tree/master/dnpso)) - Hu & Eberhart 2002

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

## Tools

* **Test Runner** - compare mean performance and convergence time for selected algorithms

## Collaborators welcome

Do you have a favorite PSO variation not implemented here that you'd like to see? Are there visualizations that would help you understand nuances of these algorithms? Feel free to add feature requests to issues, and drop me a line if you'd like to collaborate!
