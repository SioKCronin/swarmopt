# Swarm Reinforcement Learning

![cartpole](./media/cartpole.gif)
![cartpole](./media/cartpole.gif)
![cartpole](./media/cartpole.gif)
![cartpole](./media/cartpole.gif)

There are many particle swarm optimization (PSO) variations out there (prepare yourself for an onlsaught of acroynyms). 

When differentiates them? Are the reported observed differences between them truly significant? And when it comes to RL, how do they stack up against methods like Deep Q-learning?

## Particle Swarm Optimization (PSO) Variations

Particle swarm optimization (PSO) refers to one of several varations of the classic stochastic optimization heuristic developed by Kennedy and Eberhart. 

* PSO (local best, global best)
* MS-PSO (Chow & Tsui)
* MLPS (Wang, Yang & Chen) 
* DMS-PSO (Liang & Suganthan)
* SRL-PSOs (Iima & Kuroe)
* QSO (Hsieh & Su)
* Intelligent PSO (Khajenejad et al.) 

## Benchmark Functions

Optimization test functions used in our meta-analysis my:
* Sphere Function
* Rosenbrock's Function
* Ackley's Function
* Griewank's Function
* Rastrigin's Function
* Weierstrass Function

## RL Analysis

For this analysis I used Deep Q Network (with Adam) as the baseline. Analysis was done on best 100 consecutive episodes on gym's CartPole (final score and speed).  

The CartoPole is a benchmark problem in reinforcement learning. A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

## Tuning enhancements

* Gridsearch & RandomSearch to tune PSO hyperparameters
* PPSO (Nobiile, Pasi & Cazzaniga)

## Questions to ask throughout

* What about the nature of this search space would lead us to select one method over another?
* Why might we be seeing the performance we're seeing?
* Are our metrics effectively capturing the learning rate/timecourse of learning?
* What other RL problems should we try these on?
* Any other notable PSO variants we should consider?
* How to best measure performance (mean and sd of performance?)
