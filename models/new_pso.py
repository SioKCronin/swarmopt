# New PSO

import gym
import numpy as np
import pyswarms as ps

env_name = "CartPole-v1"

env = gym.make(env_name)

def forward(params):
    observation = env.reset()
    episode_return = 0

    while True:
        decision = np.matmul(observation, params)
        action = 1 if decision>0 else 0
        observation, reward, done, _ = env.step(action)
        episode_return += reward

        if done:
            return episode_return

def f(x):
    n_particles = x.shape[0]
    print("n_particles", n_particles)
    j = [forward(x[i]) for i in range(n_particles)]
    return np.array(j)

# Set hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=4, options=options)

cost, pos = optimizer.optimize(f, print_step=100, iters=68, verbose=3)

# env.render(close=False)

