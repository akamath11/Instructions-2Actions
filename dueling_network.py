"""
    Created by arvindsrikantan on 2018-04-12
"""
#!/usr/bin/env python
from __future__ import print_function
import os
import gym_minigrid
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing import sequence
from keras.layers import LSTM, Embedding, concatenate
from keras.layers.core import Flatten

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import gym
import keras, tensorflow as tf, sys, copy, argparse, pickle
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Merge, merge, Convolution2D, Flatten, Activation, BatchNormalization
from keras import backend as K
import numpy as np
import pdb
import collections
import time
import os
# from PIL import ImageOps
# import cv2
import pickle


class QNetwork:
    def __init__(self, environment_name, network_type):
        self.batch_size = 32
        self.optimizer = keras.optimizers.Adam(lr=0.0001)
        self.env_name = environment_name
        self.network_type = network_type
        self.env = gym.make(environment_name)
        self.vocab = pickle.load(open('vocab.p', 'rb'))

        if self.network_type == "dueling_dqn":
            # CNN input
            cnn_inputs = Input(self.env.observation_space.spaces['image'].shape, name='vis_inp')
            l1 = Conv2D(4, (2, 2), activation='relu')(cnn_inputs)
            l2 = MaxPooling2D(pool_size=(2,2))(l1)
            l3 = Flatten()(l2)
            # lstm input
            lstm_input = Input(shape=(12,),name='txt_inp')
            encoded_in = Embedding(len(self.vocab),24)(lstm_input)
            l4 = LSTM(12)(encoded_in)
            l5 = concatenate([l3,l4])
            l6 = Dense(24,activation='relu')(l5)
            advantage = Dense(self.env.action_space.n, activation='linear')(l6)
            value = Dense(1, activation='linear')(l6)
            outputs = Activation('linear')(Dense(self.env.action_space.n)(l6))
            def merge_final_layer(inp):
                advantage, value = inp
                avg = K.mean(advantage)
                centered_advantage = advantage - avg
                return centered_advantage + value

            q_values = merge([advantage, value], output_shape=(self.env.action_space.n,), mode=merge_final_layer)
            self.model = Model(inputs=[cnn_inputs,lstm_input], outputs=q_values)
            self.model.compile(optimizer=self.optimizer, loss='mse')

            print("In dqn")

        else:
            raise Exception("Unknown network architecture")

    def save_model(self, model_file):
        self.model.save(model_file)

    def load_model(self, model_file):
        self.model = keras.models.load_model(model_file)


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):
        self.D = collections.deque([], memory_size)
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        return (np.array(self.D)[np.random.choice(range(len(self.D)), batch_size)]).tolist()

    def append(self, transition):
        # Appends transition to the memory.
        self.D.append(transition)


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, gamma, epsilon, network_type, render=False):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.qnetwork = QNetwork(environment_name, network_type=network_type)
        self.epsilon = epsilon
        self.gamma = gamma
        self.render = render
        self.memory_size = 50000
        self.burn_in = 10
        self.states = []

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        q_values = q_values.flatten()
        if np.random.sample() > self.epsilon:
            return np.argmax(q_values)
        return self.qnetwork.env.action_space.sample()

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        q_values = q_values.flatten()
        return np.argmax(q_values)

    def get_state(self, state):
        # pdb.set_trace()
        mission = [self.qnetwork.vocab[word] for word in state['mission'].split()]
        mission = sequence.pad_sequences([mission], maxlen=12)
        return [np.array([state['image']]), np.array(mission)]
        # return np.expand_dims(np.array(state['image']), 0)

    def simulate(self, num_episodes=20, render=False):
        # TODO: add flag for greedy-epsilon greedy and include in workflow
        # TODO: make this independent of the current object, use a new instance of the same class to simulate
        episode = 0
        rewards = []
        env = gym.make(self.qnetwork.env_name)
        # pdb.set_trace()
        if render:
            env.render()
        while episode < num_episodes:
            state = env.reset()
            # pdb.set_trace()
            episode_r = 0
            done = False
            while not done:
                q_cur = self.qnetwork.model.predict(self.get_state(state))
                action = self.greedy_policy(q_cur)

                state, r, done, d = env.step(action)
                episode_r += r

                if render:
                    env.render()

            episode += 1
            rewards.append(episode_r)

        return sum(rewards) / num_episodes

    def video_capture(self, iteration, name):
        env = gym.make(self.qnetwork.env_name)
        root = 'video-captures-%s' % name
        if not os.path.exists(os.path.join(root, str(iteration))):
            os.makedirs(os.path.join(root, str(iteration)))

        directory = os.path.join(root, str(iteration))
        # Record the environment
        env = gym.wrappers.Monitor(env, directory, video_callable=lambda episode_id: True)  # ,force=True
        done = False
        state = env.reset()

        while not done:
            q_cur = self.qnetwork.model.predict(state.reshape(1, state.shape[0]))
            action = self.greedy_policy(q_cur)

            state, r, done, d = env.step(action)

        return

    def train(self, max_iterations=100, with_replay=False, use_target=False):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        env = self.qnetwork.env
        max_iterations += 1
        episodes = 0
        min_q = 0

        iterations = 0
        rewards = []
        iters = []
        losses = []

        sim_count = 0

        reward_q = collections.deque([], 100)

        episodes = 0
        checkpointing = 5000

        if not with_replay:
            setup = "noreplay_" + self.qnetwork.network_type
            while iterations < max_iterations:  # Convergence
                state = env.reset()
                print("Mission", state["mission"])
                r = 0.0
                r_sum = 0
                terminal = False
                cur_state = self.get_state(state)
                while not terminal and iterations < max_iterations:
                    if self.render == True:
                        env.render()
                    q_cur = self.qnetwork.model.predict(cur_state)
                    action = self.epsilon_greedy_policy(q_cur)
                    next_state, r, terminal, d = env.step(action)
                    next_state = self.get_state(next_state)
                    q_next = self.qnetwork.model.predict(next_state)
                    target = q_cur.copy()

                    target[0, action] = r + self.gamma * q_next.max()

                    history = self.qnetwork.model.fit(cur_state, target, verbose=False, epochs=1)

                    cur_state = next_state

                    iterations += 1
                    if self.epsilon > 0.05:
                        self.epsilon -= (0.5-0.05)/(1e6)  # * 2
                    r_sum += r

                episodes += 1
                print('Episodes : ', episodes, 'Epsilon : ', self.epsilon, 'Reward : ', r_sum)
                reward_q.append(r_sum)


        else:
            replay_memory = self.burn_in_memory()
            target_network = None
            update_stage = 10000
            setup = "replay_" + self.qnetwork.network_type
            iterations = 0
            while iterations < max_iterations:
                r_sum = 0
                print("Episode number: %s" % episodes)
                state = env.reset()
                print("Mission", state["mission"])
                terminal = False
                time = 0

                while not terminal:  # and time < env._max_episode_steps:

                    if iterations % checkpointing == 0:
                        # Checkpointing
                        # self.qnetwork.save_model("./model-checkpoints/%s_%s_replay_single_network" % (self.qnetwork.env_name, c))
                        self.qnetwork.save_model("./model-checkpoints_%s_%s/%s" % (self.qnetwork.env_name, setup, iterations))
                        np.save("./model-loss_%s_%s/%s" % (self.qnetwork.env_name, setup, iterations), np.array(losses))

                        # Simulating
                        print('Sim count : ', sim_count, 'iter count : ', iterations)
                        rewards.append(self.simulate(num_episodes=20))
                        # self.simulate(num_episodes=1, render=True)
                        iters.append(iterations)
                        np.save("./model-rewards_%s_%s/%s" % (self.qnetwork.env_name, setup, iterations), np.array(rewards))
                        print('Simulated rewards : %s'%rewards)

                        # Adding loss
                        if iterations > 0:
                            loss = history.history["loss"]
                            losses.append(loss)
                        sim_count += 1

                    if self.render == True:
                        env.render()
                    if iterations % update_stage == 0:
                        if use_target:
                            target_network = keras.models.clone_model(self.qnetwork.model)
                        else:
                            target_network = self.qnetwork.model
                    iterations += 1
                    time += 1

                    cur_state = self.get_state(state)
                    q_vals = self.qnetwork.model.predict(cur_state)
                    min_q = min(q_vals.min(), min_q)
                    action = self.epsilon_greedy_policy(q_vals)
                    next_state, r, terminal, _ = env.step(action)
                    replay_memory.append((cur_state, action, r, self.get_state(next_state), terminal))

                    samples = replay_memory.sample_batch(self.qnetwork.batch_size)
                    X_vis = []
                    X_txt = []
                    y = []
                    for cur_s, act, _r, next_s, term in samples:
                        # pdb.set_trace()
                        X_vis.append(cur_s[0][0])
                        X_txt.append(cur_s[1][0])

                        if term:
                            q_cur = self.qnetwork.model.predict(cur_s)
                            target = q_cur.flatten()
                            target[act] = _r
                            y.append(target)
                        else:
                            q_cur = self.qnetwork.model.predict(cur_s)
                            q_next_target = target_network.predict(next_s)
                            q_next_current = self.qnetwork.model.predict(next_s)
                            a_max = np.argmax(q_next_current.flatten())
                            target = q_cur.flatten()
                            target[act] = _r + self.gamma * q_next_target.flatten()[a_max]
                            y.append(target)
                    # pdb.set_trace()
                    history = self.qnetwork.model.fit([np.array(X_vis),np.array(X_txt)], np.array(y), verbose=False)

                    state = next_state

                    if self.epsilon > 0.05:
                        self.epsilon -= (0.5-0.05)/(1e6)  # * 2
                    r_sum += r

                episodes += 1
                reward_q.append(r_sum)
                print("episode=%s, Iteration = %s, loss = %s, epsilon = %s, avg_reward = %s, reward=%s" % (
                    episodes, iterations, history.history["loss"], self.epsilon, sum(reward_q) / float(len(reward_q)),
                    r_sum))


    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cumulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        env = gym.make(self.qnetwork.env_name)

        rewards = []
        for episode in range(100):
            state = env.reset()
            terminal = False
            rewards.append(0.0)

            if self.render:
                time.sleep(1)
                env.render()
                time.sleep(0.1)
            iterations = 0
            while not terminal:
                q_cur = self.qnetwork.model.predict(np.array([state]))
                # action = self.epsilon_greedy_policy(q_cur)
                action = self.greedy_policy(q_cur)
                state, r, terminal, d = env.step(action)
                rewards[-1] += r

                if self.render:
                    env.render()
                iterations += 1
            print("iterations=%s" % iterations)

        print("Average reward = %s" % (sum(rewards) / float(len(rewards))))

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        replay_memory = Replay_Memory(self.memory_size, self.burn_in)
        terminal = True
        env = self.qnetwork.env
        state = None
        for _ in range(self.burn_in):
            if terminal:
                state = env.reset()
                terminal = False
            cur_state = self.get_state(state)
            q_vals = self.qnetwork.model.predict(cur_state)
            action = self.epsilon_greedy_policy(q_vals)

            next_state, r, terminal, _ = env.step(action)

            replay_memory.append((cur_state, action, r, self.get_state(next_state), terminal))
            state = next_state

        return replay_memory


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    return parser.parse_args()


def main():
    environment_name = 'MiniGrid-Fetch-6x6-N2-v0'
    gamma = 1
    network_type = 'dueling_dqn'
    with_replay = True
    use_target = True
    rep = 'replay_' if with_replay else 'noreplay_'
    setup = rep + network_type

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    # gpu_ops = tf.GPUOptions(allow_growth=True)
    # config = tf.ConfigProto(gpu_options=gpu_ops)
    # sess = tf.Session(config=config)
    #
    # # Setting this as the default tensorflow session.
    # keras.backend.tensorflow_backend.set_session(sess)

    if not os.path.exists("./model-checkpoints_%s_%s"%(environment_name, setup)):
        os.makedirs("./model-checkpoints_%s_%s"%(environment_name, setup))

    if not os.path.exists("./model-loss_%s_%s"%(environment_name, setup)):
        os.makedirs("./model-loss_%s_%s"%(environment_name, setup))

    if not os.path.exists("./model-rewards_%s_%s"%(environment_name, setup)):
        os.makedirs("./model-rewards_%s_%s"%(environment_name, setup))

    agent = DQN_Agent(environment_name, gamma, 0.9, network_type, render=False)
    agent.qnetwork.load_model("./model-checkpoints_%s_%s/%s" % (environment_name, setup, 145000))
    agent.train(max_iterations=1000000, with_replay=with_replay, use_target=use_target)
    agent.qnetwork.save_model("models/%s_%s.model"%(environment_name, setup))#args.model_file)
    agent.test()


# You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
    main()