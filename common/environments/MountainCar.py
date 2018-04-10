import numpy as np
import math
class MountainCar(object):
    def __init__(self, start, goal, Xrange, Vrange):
        self.start = start
        self.goal = goal
        self.Xrange = Xrange
        self.Vrange = Vrange
        self.num_actions = 3

    def _DoAction(self, action):
        position = self.state[0]
        speed = self.state[1]

        # position bounds
        bpleft = self.Xrange[0]

        # speed bounds
        bsleft = self.Vrange[0]
        bsright = self.Vrange[1]

        speedt1 = speed + (0.001 * action) + (-0.0025 * math.cos(3.0*position))
        speedt1 = speedt1 * 0.999 # thermodynamic law, for a more real system

        if speedt1 < bsleft:
            speedt1 = bsleft
        elif speedt1 > bsright:
            speedt1=bsright

        post1 = position + speedt1

        if post1<=bpleft:
            post1=bpleft
            speedt1=0.0

        xp = np.array([post1, speedt1])
        self.state = xp

    def _GetReward(self):
        position = self.state[0]
        bpright = self.goal

        r = -1
        f = False

        if position >= bpright:
            r = 100
            f = True

        return r, f

    def act(self, action):
        self._DoAction(action)
        reward, game_over = self._GetReward()
        return self.observe(), reward, game_over

    def observe(self):
        return self.state.reshape((1,-1))

    def reset(self):
        self.state = np.array([self.start[0], self.start[1]])
