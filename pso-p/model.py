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

            loss += model.train_on_batch(inputs, targets)[0]
        print("Step {} Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(step, e, loss, win_cnt))

    # Safe trained model weights and architecture (used for viz)
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

