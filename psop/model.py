import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from MountainCar import MountainCar

class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=0.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size = inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

if __name__ == "__main__":
    #parameters
    epsilon = 0.1 #exploration coeff
    num_actions = 3 # move left, stay put, move right
    epoch = 1000
    max_memory = 500
    hidden_size = 100
    batch_size = 50
    input_size = 2

    Xrange = [-1.5, 0.55]
    Vrange = [-0.7, 0.7]
    start = [-0.5, 0.0]
    goal = [0.45]

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(2, ), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=0.2),"mse")

    # Define environment
    env = MountainCar(start, goal, Xrange, Vrange)

    # Init experiene replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.0
        env.reset()
        game_over = False
        input_t = env.observe()

        step = 0
        while (not game_over):
            input_tm1 = input_t
            step += 1
            # get next action
            if np.random.rand() <= epsilon: # is this rand defined correctly?
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            if reward ==100:
                win_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)
        print("Step {} Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(step, e, loss, win_cnt))

    # Safe trained model weights and architecture (used for viz)
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

