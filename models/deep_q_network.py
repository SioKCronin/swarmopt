# Deep Q Network (DQN)

import gym
import numpy as np
from common.dqn_agent import DQNAgent
from collections import deque

if __name__ == "__main__":

    episodes = 1000

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    for e in range(episodes):

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for t in range(500):
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("episode: {}/{}, score: {}, e:".format(e, episodes, t))

                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

