import argparse
import os
import sys

import gym
import gym_minigrid

gym_minigrid
import matplotlib
import numpy as np
from keras import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Embedding, LSTM, concatenate, Activation
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce
import pickle


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr, vocab, n=20):
        """
        Initializes A2C.
        :param model: The actor model.
        :param lr:  Learning rate for the actor model.
        :param critic_model: The critic model.
        :param critic_lr: Learning rate for the critic model.
        :param n: The value of N in N-step A2C.
        """
        self.critic_model = critic_model
        self.n = n
        self.lr = lr
        super().__init__(model, lr, vocab)
        self.critic_optimizer = Adam(lr=critic_lr)
        self.critic_model.compile(loss='mean_squared_error', optimizer=self.critic_optimizer, metrics=['accuracy'])

    def train(self, env, episodes, env_name, gamma=1.0, render=False, reward_scale=1.0, without_mission=False):
        checkpointing = 500
        test_rewards = []
        train_rewards = []
        power_gamma = {k: gamma ** k for k in range(10000)}
        for episode in range(episodes + 1):
            if episode % checkpointing == 0:
                # Checkpoint
                self.save_weights("../pickles/a2c/%s/checkpoint/%s_n_%s_iter_%s.h5" % (env_name, "%s", self.n, episode))
                test_reward = []
                for _ in range(100):
                    _, _, rewards = self.generate_episode(env, reward_scale)
                    test_reward += [sum(rewards) * reward_scale]
                test_rewards.append((np.array(test_reward).mean(), np.array(test_reward).std()))
                print("Average test rewards = %s" % (str(test_rewards[-1])))
                np.save("../pickles/a2c/%s/test-rewards/n_%s_iter_%s.npy" % (env_name, self.n, episode),
                        np.array(test_rewards))
            states, actions, rewards = self.generate_episode(env, reward_scale, render=render)
            r = np.zeros(len(rewards))
            g = np.zeros(len(rewards))
            T = len(rewards)
            if without_mission:
                states_trasformed = np.array(states)
            else:
                im, descr = zip(*states)
                states_trasformed = [np.array(im), np.array(descr)]
            v = self.critic_model.predict(states_trasformed, batch_size=len(states)).flatten()
            for t in reversed(range(T)):
                v_end = 0 if (t + self.n >= T) else v[t + self.n]
                r[t] = power_gamma[self.n] * v_end + sum(
                    [(power_gamma[k] * rewards[t + k] if (t + k < T) else 0) for k in range(self.n)])
                g[t] = r[t] - v[t]
            history = self.model.fit(states_trasformed,
                                     np.array(np_utils.to_categorical(actions, num_classes=env.action_space.n)),
                                     epochs=1, batch_size=len(states), verbose=False, sample_weight=g)
            critic_history = self.critic_model.fit(states_trasformed, r, epochs=1, batch_size=len(states),
                                                   verbose=False)

            print("Episode %6d's, Steps = %3d, loss = %+.5f, critic_loss = %+.5f, cumulative reward:%+5.5f" % (
                episode, len(states), history.history['loss'][0], critic_history.history['loss'][0],
                sum(rewards) * reward_scale))
            train_rewards.append(sum(rewards) * reward_scale)
            np.save("../pickles/a2c/%s/n_%s_train-rewards.npy" % (env_name, self.n), np.array(train_rewards))

    def save_weights(self, name):
        self.model.save_weights(name % "actor")
        self.critic_model.save_weights(name % "critic")

    def load_weights(self, name):
        self.model.load_weights(name % "actor")
        self.model.compile(loss=categorical_crossentropy, optimizer=self.optimizer, metrics=['accuracy'])
        self.critic_model.load_weights(name % "critic")
        self.critic_model.compile(loss='mean_squared_error', optimizer=self.critic_optimizer, metrics=['accuracy'])


def createDirectories(l):
    for l in l:
        if not os.path.exists(l):
            os.makedirs(l)


def get_test_rewards(env, model, n=1):
    checkpointing = 500
    for episode in range(0, 60000, checkpointing):
        print("Episode: %s" % episode)
        model.load_weights("../pickles/a2c/checkpoint/%s_n_%s_iter_%s.h5" % ("%s", n, episode))
        test_rewards = []
        for _ in range(100):
            _, _, rewards = model.generate_episode(env)
            test_rewards.append(np.array(rewards))
        np.save("../pickles/a2c/test-rewards-lists/n_%s_iter_%s.npy" % (n, episode), np.array(test_rewards))


# def test_reward_with_error(n=1):
#     checkpointing = 500
#     test_rewards_mean = []
#     test_rewards_sd = []
#
#     for episode in range(0, 60000, checkpointing):
#         # print("Episode: %s" % episode)
#         test_rewards = np.array([_temp.sum()*100 for _temp in np.load("../pickles/a2c/test-rewards-lists/n_%s_iter_%s.npy" % (n, episode))])
#         test_rewards_mean.append(test_rewards.mean())
#         test_rewards_sd.append(test_rewards.std())
#
#     return test_rewards_mean, test_rewards_sd


def plot():
    n = 20
    # iteration = 86500 # n=100
    # iteration = 97000 # n = 1
    iteration = 100000
    r = np.load("../../n_%s_iter_%s.npy" % (n, iteration))
    y, err = list(zip(*[r[i] for i in range(len(r)) if i % 2 == 0]))
    x = list(range(0, len(y) * 1000, 1000))
    # x = list(range(0, len(y)*500, 500))
    plt.figure()
    plt.errorbar(x, y, yerr=err)
    plt.xlabel("Training episodes")
    plt.ylabel("Average reward over 100 episodes")
    plt.title("A2C cumulative reward for N=%s averaged over 100 episodes" % n)
    plt.savefig("a2c_n_%s.png" % n)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment-name', dest='environment_name',
                        type=str, default='MiniGrid-Fetch-6x6-N2-v0',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=100000, help="Number of episodes to train on.")
    parser.add_argument('--reward-scale', dest='reward_scale', type=float,
                        default=1, help="The scale factor for rewards")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-3, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-3, help="The critic's learning rate.")  # 5e-4 before
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=1, help="The value of gamma in A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--with-mission', dest='without_mission',
                              action='store_false',
                              help="Whether to use the mission string.")
    parser_group.add_argument('--without-mission', dest='without_mission',
                              action='store_true',
                              help="Whether to use the mission string.")
    parser.set_defaults(without_mission=False)

    return parser.parse_args()


def main(args, load_models=None):
    # Parse command-line arguments.
    args = parse_arguments()

    environment_name = args.environment_name
    print("Running env: %s, with reward scaling of: %s" % (environment_name, args.reward_scale))
    # Create the environment.
    env = gym.make(environment_name)
    dirs = [
        # "../pickles/a2c/weights",
        "../pickles/a2c/%s/checkpoint/" % environment_name,
        "../pickles/a2c/%s/test-rewards/" % environment_name,
        "../pickles/a2c/%s/test-rewards-lists/" % environment_name
    ]
    createDirectories(dirs)
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = 20  # args.n
    render = args.render

    print(
        "Training args: episodes=num_episodes, env_name=%s, render=%s, reward_scale=%s, without_mission=%s, gamma=%s" %
        (environment_name, render, args.reward_scale, args.without_mission, args.gamma))

    vocab = pickle.load(open('../data/vocab.p', 'rb'))

    # Load the actor model from file.
    def create_actor(with_language=True):
        # CNN input
        cnn_inputs = Input(env.observation_space.spaces['image'].shape,
                           name='vis_inp')  # env.observation_space.spaces['image'].spaces['image'].shape
        l1 = Conv2D(4, (2, 2), activation='relu')(cnn_inputs)
        l2 = MaxPooling2D(pool_size=(2, 2))(l1)
        l3 = Flatten()(l2)
        l5 = l3
        # lstm input
        if with_language:
            lstm_input = Input(shape=(12,), name='txt_inp')
            encoded_in = Embedding(len(vocab), 24)(lstm_input)
            l4 = LSTM(12)(encoded_in)
            l5 = concatenate([l3, l4])
        l6 = Dense(24, activation='relu')(l5)
        output = Activation('softmax')(Dense(env.action_space.n)(l6))
        if with_language:
            model = Model(inputs=[cnn_inputs, lstm_input], outputs=output)
        else:
            model = Model(inputs=[cnn_inputs], outputs=output)
        return model

    model = create_actor(True)

    # Critic model
    def create_critic(with_language=True):
        # CNN input
        cnn_inputs = Input(env.observation_space.spaces['image'].shape, name='vis_inp')
        l1 = Conv2D(4, (2, 2), activation='relu')(cnn_inputs)
        l2 = MaxPooling2D(pool_size=(2, 2))(l1)
        l3 = Flatten()(l2)
        l5 = l3
        # lstm input
        if with_language:
            lstm_input = Input(shape=(12,), name='txt_inp')
            encoded_in = Embedding(len(vocab), 24)(lstm_input)
            l4 = LSTM(12)(encoded_in)
            l5 = concatenate([l3, l4])
        l6 = Dense(24, activation='relu')(l5)
        l7 = Dense(30, activation='relu')(l6)
        output = Activation('linear')(Dense(1)(l7))
        if with_language:
            model = Model(inputs=[cnn_inputs, lstm_input], outputs=output)
        else:
            model = Model(inputs=[cnn_inputs], outputs=output)
        return model

    critic_model = create_critic(True)

    # critic_model.summary()
    # exit()

    # TODO: Train the model using A2C and plot the learning curves.
    a2c = A2C(model, lr, critic_model, critic_lr, vocab, n=n)
    if load_models is not None:
        a2c.load_weights(load_models)
        print("Loaded")

    a2c.train(env, episodes=num_episodes, env_name=environment_name, render=render, reward_scale=args.reward_scale,
              without_mission=args.without_mission, gamma=args.gamma)

    # for _n in [1, 20, 50, 100]:
    #     print("Starting for n=%s" % _n)
    #     get_test_rewards(env, a2c, n=_n)
    #
    # for _n in [1, 20, 50, 100]:
    #     print("Starting for n=%s" % _n)
    #     print(list(zip(*test_reward_with_error(n=_n))))

    # plot()


if __name__ == '__main__':
    main(sys.argv)
