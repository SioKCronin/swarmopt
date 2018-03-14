# Swarm Reinforcement Learning

![cartpole](./media/cartpole.gif)
![cartpole](./media/cartpole.gif)
![cartpole](./media/cartpole.gif)
![cartpole](./media/cartpole.gif)

There are many particle swarm optimization (PSO) variations out there (prepare yourself for an onlsaught of acroynyms). 

When differentiates them, and are the reported observed differences between them truly significant? 

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
