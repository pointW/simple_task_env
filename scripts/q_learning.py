import sys
import time
from collections import defaultdict

import numpy as np

from simple_task_env import SimpleTaskEnv


def greedyTieBreak(q_curr):
    max_q = np.max(q_curr)
    candidate = []
    for i, q_value in enumerate(q_curr):
        if q_value == max_q:
            candidate.append(i)
    return np.random.choice(candidate)


def eGreedyActionSelection(q_curr, eps):
    '''
    Preforms epsilon greedy action selectoin based on the Q-values.

    Args:
        q_curr: A numpy array that contains the Q-values for each action for a state.
        eps: The probability to select a random action. Float between 0 and 1.

    Returns:
        The selected action.
    '''

    if np.random.random() < eps:
        return np.random.randint(len(q_curr))
    else:
        return greedyTieBreak(q_curr)


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        '''
        Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Args:
            - schedule_timesteps: Number of timesteps for which to linearly anneal initial_p to final_p
            - initial_p: initial output value
            -final_p: final output value
        '''
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


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




