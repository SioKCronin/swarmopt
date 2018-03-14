# Environments

import gym
import time
import numpy as np
from multiprocessing import Process, Pipe

class EnvWorker(Process):

    def __init__(self, env_name, pipe, name=None, NotDisplay=False, delay=0):
        Process.__init__(self, name=name)
        self.env = gym.make(env_name)
        self.pipe = pipe
        self.name = name
        self.Display = not NotDisplay
        self.delay = delay
        print("Environment initialized.", self.name)

    def run(self):
        observation=self.env.reset()
        param=self.pipe.recv()
        episode_return=0
        while True:
            time.sleep(self.delay)
            decision = np.matmul(observation, param) # could use torch here
            action = 1 if decision>0 else 0
            observation, reward, done, _ = self.env.step(action)
            episode_return += reward
            self.env.render(close=self.Display)

            if done:
                self.pipe.send(episode_return)
                episode_return=0
                param=self.pipe.recv()
                if param=="EXIT":
                    break
                self.env.reset()

class ParallelEnvironment(object):

    def __init__(self, env_name, num_env, NotDisplay):
        assert num_env>0, "Number of environments must be positive."
        self.num_env = num_env
        self.workers = []
        self.pipes = []
        for env_idx in range(num_env):
            p_start, p_end = Pipe()
            env_worker = EnvWorker(env_name, p_end, name="Worker"+str(env_idx), NotDisplay=NotDisplay)
            env_worker.start()
            self.workers.append(env_worker)
            self.pipes.append(p_start)

    def episode(self, params):
        returns = []
        for idx in range(self.num_env):
            self.pipes[idx].send(params[idx])
        for idx in range(self.num_env):
            return_ = self.pipes[idx].recv()
            returns.append(return_)
        return returns

    def __del__(self):
        for idx in range(self.num_env):
            self.pipes[idx].send("EXIT")

        for worker in self.workers:
            worker.terminate()
