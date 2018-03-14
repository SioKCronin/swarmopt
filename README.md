# Swarm Reinforcement Learning

![cartpole](./media/cartpole.gif)
![cartpole](./media/cartpole.gif)
![cartpole](./media/cartpole.gif)
![cartpole](./media/cartpole.gif)

Can swarm optimization improve upon Q-learning in a classic RL problem? 

If so, which variation? (prepare yourself for an onlsaught of acroynyms)

## Particle Swarm Optimization (PSO) Variations

Particle swarm optimization (PSO) refers to one of several varations of the classic stochastic optimization heuristic developed by Kennedy and Eberhart. 

* PSO (local best, global best)
* MS-PSO (Chow & Tsui)
* MLPS (Wang, Yang & Chen) 
* DMS-PSO (Liang & Suganthan)
* SRL-PSOs (Iima & Kuroe)
* QSO (Hsieh & Su)
* Intelligent PSO (Khajenejad et al.) 

## CartPole

The CartoPole is a benchmark problem in reinforcement learning. A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

## Tuning enhancements

* Gridsearch & RandomSearch for PSO tuning
* PPSO (Nobiile, Pasi & Cazzaniga)

## Questions

* What about the nature of this project would lead us to suspect one method over another?
* Why might we be seeing the performance we're seeing?
* Are our metrics effectively capturing the learning rate/timecourse of learning?
* What other RL problems should we try these on?
* Any other notable PSO variants we should consider?
* How to best measure performance (mean and sd of performance?)
