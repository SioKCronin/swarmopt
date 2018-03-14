# PSO Baselines

![cartpole](./media/cartpole.gif)
![cartpole](./media/cartpole.gif)
![cartpole](./media/cartpole.gif)
![cartpole](./media/cartpole.gif)

Particle swarm optimization (PSO) refers to one of several varations of the classic stochastic optimization heuristic developed by Kennedy and Eberhart. There are many particle swarm optimization (PSO) variations out there (prepare yourself for an onlsaught of acroynyms). 

These implementations require PySwarms, an open-source research toolkit. Some require gym. 

In addition to the implementations, I have included information in the docs about what differentiates them, and when each might be used (particularly in reinforcement learning). 

## Variations
| Name  | layers | params  | X  | Y  |
|---|---|---|---|---|
| PSO_GBEST |   | c1, c2, w  |   |   |
| PSO_LBEST |   | c1, c2, w, k, p |   |   |
| MS-PSO | X | c1, c2, w, n |   |   |
| MLPSO | X |   |   |   |
| DMS-PSO |   |   |   |   |
| SRL-PSOs |   |   |   |   |
| QSO |   |   |   |   |

## Comparison Benchmark Functions

Optimization test functions used for comparison:
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
