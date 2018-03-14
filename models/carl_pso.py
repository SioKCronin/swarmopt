# PSO (inspired by an implementation by Rishav Chourasia)

import time
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import numpy as np
import gym

num_envs = 15
velocity_alpha = 0.01
env_name = "CartPole-v1"

params = []
best_params = []
best_returns = []
cost = []
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
    env = gym.make(env_name)

    def episode(env, params):
        returns = []
        for idx in range(num_envs):
            return_ = run(env, params[idx])
            env.reset()
            returns.append(return_)
        return returns

    def run(env, params):
        observation=env.reset()
        episode_return=0
        while True:
            decision = np.matmul(observation, param) # could use torch here
            action = 1 if decision>0 else 0
            observation, reward, done, _ = env.step(action)
            episode_return += reward
            env.render(close=True)

            if done:
                return episode_return

    returns = episode(env, params)

    iter_cnt +=1
    print("Number of batch episodes ran -> ", iter_cnt)
    print("Parameter for the batch for last episode ->")
    print(np.around(params, 3))
    print("Returns for the batch for last episode ->", returns)
    print("Returns for the batch for all episodes ->", best_returns)
    print("Rate of change of parameters for the batch ->", np.around(velocity,3))

    # When we've converged, run everything once more
    # with the final parameters and EXIT.
    if returns==best_returns:
        print("Batch converged after {} iterations".format(iter_cnt))
        env.render()
        break

    for i in range(num_envs):
        #optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
        #best_returns[i], best_params[i] = optimize(f, print_step=100, iters=1000, verbose=3)
        if(returns[i]>=best_returns[i]):
            best_returns[i]=returns[i]
            best_params[i]=params[i]

    best_pos=returns.index(max(returns))

    # Set up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    for i in range(num_envs):

        velocity[i]=(intertia*velocity[i]
                +cognitive*np.random.uniform(0,velocity_alpha)*(best_params[i]-params[i])
                +social*np.random.uniform(0,velocity_alpha)*(best_params[best_pos]-params[i]))
        params[i] += velocity[i]

    pass
