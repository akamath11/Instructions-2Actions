"""
    Created by arvindsrikantan on 2018-04-15
"""

import time

import gym
import gym_minigrid

gym_minigrid
import student_a2c_torch
import pickle


def run_random_policy():
    """Run a random policy for the given environment.

        Logs the total reward and the number of steps until the terminal
        state was reached.

        Parameters
        ----------
        env: gym.envs.Environment
          Instance of an OpenAI gym.

        Returns
        -------
        (float, int)
          First number is the total undiscounted reward received. The
          second number is the total number of actions taken before the
          episode finished.
        """
    env = gym.make('MiniGrid-Fetch-16x16-N4-v0')
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        env.render()
        print(nextstate)

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(0.1)

    print(total_reward)
    return total_reward, num_steps


def run_policy(env_name, episode):
    env = gym.make(env_name)
    vocab = pickle.load(open('../data/vocab.p', 'rb'))
    a2c = student_a2c_torch.A2C(student_a2c_torch.Actor(env_name, vocab, max_len=30), None,
                                student_a2c_torch.Critic(env_name, vocab, 30), None, vocab, max_len=30, n=20)
    # load_name = "../pickles/students/%s/checkpoint/%s_n_%s_iter_%s.h5" % (env_name, "%s", 20, episode)
    load_name = "../pickles/a2c/%s/checkpoint/%s_n_%s_iter_%s.h5" % ("MiniGrid-Fetch-6x6-N2-v0", "%s", 20, episode)
    a2c.load_weights(load_name)
    a2c.generate_episode(env, reward_scale=1, render=True)


if __name__ == "__main__":
    for _ in range(5):
        run_random_policy()
        # run_policy("MiniGrid-LockedRoom-v0", 100000)
