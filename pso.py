# PSO (inspired by an implementation by Rishav Chourasia)

import gym
import time
import pyswarms
from multiprocessing import Process, Pipe
from environments import EnvWorker, ParallelEnvironment
import numpy as np

# For multiproessing
num_envs = 10
NotDisplay = False
velocity_alpha = 0.01
env_name = "CartPole-v1"
p_env = ParallelEnvironment(env_name, num_envs, NotDisplay)

params = []
best_params = []
best_returns = []
best_pos = 0
velocity = []
iter_cnt = 0
intertia = 1
cognitive = 1
social = 1

for i in range(num_envs):
    # Random initialized for 4 cart params - TODO: use priors?
    params.append(np.random.uniform(-1,1,4))

    # Placeholder for best
    best_returns.append(0)

    # Placeholder for best returns
    best_params.append(np.array([0 for i in range(4)]))

    # Placeholder for velocity for each
    velocity.append(np.array([0.0 for i in range(4)]))

while True:
    returns = p_env.episode(params)

    iter_cnt +=1
    print("Number of batch episodes ran -> ", iter_cnt)
    # More prints here

    if returns==best_returns:
        print("Batch converged after {} iterations".format(iter_cnt))
        p_env.__del__()
        p_start, p_end = Pipe()
        env_worker = EnvWorker("CartPole-v1", p_end, name="Worker", NotDisplay=True, delay=0.02)
        env_worker.start()
        p_start.send(np.sum(best_params, axis=0)/num_envs)
        episode_return = p_start.recv()
        print("Reward for the final episode ->", episode_return)
        print("Best params:", best_params)
        p_start.send("EXIT")
        env_worker.terminate()

    for i in range(num_envs):
        if(returns[i]>=best_returns[i]):
            best_returns[i]=returns[i]
            best_params[i]=params[i]

    best_pos=returns.index(max(returns))

    # Could import PySwarms here
    for i in range(num_envs):
        velocity[i]=(intertia*velocity[i]
                +cognitive*np.random.uniform(0,velocity_alpha)*(best_params[i]-params[i])
                +social*np.random.uniform(0,velocity_alpha)*(best_params[best_pos]-params[i]))
        params[i] += velocity[i]

    pass
