import time
import sys
import json

sys.path.append('../..')
from collections import defaultdict

from scoop_1d_env import ScoopEnv

from util.utils import *


def saveJson(Q):
    time_stamp = time.strftime('%Y%m%d%H%M%S', time.gmtime())
    json_path = '/home/ur5/thesis/simple_task/scoop_1d/data/q_learning/' + time_stamp + 'Q'
    json_q = {}
    for state in Q:
        json_q[str(state)] = list(Q[state])
    f = open(json_path, 'w')
    json.dump(json_q, f)


def q_learning(env, num_episodes, gamma=1.0, alpha=0.01,
               start_eps=1.0, final_eps=0.1, annealing_steps=10000,
               max_episode_steps=100):
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
    total_step = 0

    exploration = LinearSchedule(annealing_steps, final_eps, start_eps)
    for episode in range(num_episodes):
        print '------Episode {} / {}------'.format(episode, num_episodes)
        s = round(env.reset(), 2)
        r_total = 0
        for step in range(max_episode_steps):
            e = exploration.value(total_step)
            total_step += 1
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
                if episode % 100 == 0:
                    time_stamp = time.strftime('%Y%m%d%H%M%S', time.gmtime())
                    reward_path = '/home/ur5/thesis/simple_task/scoop_1d/data/q_learning/'+time_stamp+'reward'
                    length_path = '/home/ur5/thesis/simple_task/scoop_1d/data/q_learning/'+time_stamp+'length'
                    np.save(reward_path, episode_rewards)
                    np.save(length_path, episode_lengths)
                    saveJson(Q)
                break

    return Q, episode_rewards, episode_lengths


def main():
    simple_task_env = ScoopEnv(port=19999)
    q_table, rewards, lengths = q_learning(simple_task_env, 10000, gamma=0.9)


if __name__ == '__main__':
    main()




