import sys
import time
from collections import defaultdict

import numpy as np

from simple_task_env import SimpleTaskEnv

from utils import *


def q_learning(env, num_episodes, gamma=1.0, alpha=0.1,
               start_eps=0.2, final_eps=0.1, annealing_steps=1000,
               max_episode_steps=200):
    '''
    Q-learning algorithm.

    Args:
        - env: The environment to train the agent on
        - num_episodes: The number of episodes to train the agent for
        - gamma: The discount factor
        - alpha: The stepsize
        - start_eps: The initial epsilon value for e-greedy action selection
        - final_eps: The final epsilon value for the e-greedy action selection
        - annealing_steps: The number of steps to anneal epsilon over
        - max_episode_steps: The maximum number of steps an episode can take

    Returns: (Q_func, episode_rewards, episode_lengths)
        - Q: Dictonary mapping state -> action values
        - episode_rewards: Numpy array containing the reward of each episode during training
        - episode_lengths: Numpy array containing the length of each episode during training
    '''
    Q = defaultdict(lambda: np.zeros(env.nA))
    episode_rewards = np.zeros(num_episodes)
    episode_lengths = np.zeros(num_episodes)

    exploration = LinearSchedule(annealing_steps, start_eps, final_eps)
    for episode in range(num_episodes):
        print '------Episode {} / {}------'.format(episode, num_episodes)
        s = round(env.reset(), 2)
        r_total = 0
        for step in range(max_episode_steps):
            e = exploration.value(step)
            a = eGreedyActionSelection(Q[s], e)
            s_, r, done, info = env.step(a)
            if not done:
                s_ = round(s_, 2)
                r_total += r
                Q[s][a] += alpha * (r + gamma * np.max(Q[s_]) - Q[s][a])
                s = s_
            else:
                r_total += r
                Q[s][a] += alpha * (r - Q[s][a])
            print 'step {}, {}, {}, {}, {}, {}'.format(step, s, a, s_, r, done)

            if done or step == max_episode_steps - 1:
                print '------Episode {} ended, total reward: {}, step: {}------'.format(episode, r_total, step)
                episode_rewards[episode] = r_total
                episode_lengths[episode] = step
                break

    return Q, episode_rewards, episode_lengths


def main():
    simple_task_env = SimpleTaskEnv()
    q_table, rewards, lengths = q_learning(simple_task_env, 1000, gamma=0.9)


if __name__ == '__main__':
    main()




