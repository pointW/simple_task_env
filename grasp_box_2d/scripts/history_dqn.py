from collections import namedtuple
import random
import time
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.append('../..')

from util.utils import LinearSchedule
from simple_grasp_env import SimpleGraspEnv
from util.plot import plotLearningCurve
from agent.dqn import *


class HistoryDQN(torch.nn.Module):
    def __init__(self):
        super(HistoryDQN, self).__init__()

        self.fc1 = nn.Linear(49, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class HisDQNAgent(DQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=64, target_update_frequency=10):
        saving_dir = '/home/ur5/thesis/simple_task/grasp_box_2d/data/history_dqn'
        DQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                          target_update_frequency, saving_dir)

    def encodeState(self, observation, last_action, last_s):
        last_s_list = last_s.tolist()[0]
        s_list = last_s_list[5:] + [last_action] + list(observation)
        s_tensor = torch.tensor(s_list, device=self.device).unsqueeze(0)
        return s_tensor

    def initialState(self, observation):
        s_list = [0 for _ in range(45)] + list(observation)
        s_tensor = torch.tensor(s_list, device=self.device).unsqueeze(0)
        return s_tensor

    def train(self, num_episodes, max_episode_steps=100):
        while self.episodes_done < num_episodes:
            print '------Episode {} / {}------'.format(self.episodes_done, num_episodes)
            observation = self.env.reset()
            state = self.initialState(observation)
            r_total = 0
            for step in range(max_episode_steps):
                action, q = self.selectAction(state, require_q=True)
                observation_, r, done, info = self.env.step(action.item())
                print 'step {}, action: {}, q: {}, reward: {} done: {}'\
                    .format(step, action.item(), q, r, done)
                r_total += r
                observation = observation_
                if done or step == max_episode_steps - 1:
                    next_state = None
                else:
                    next_state = self.encodeState(observation_, action.item(), state)
                reward = torch.tensor([r], device=self.device, dtype=torch.float)
                self.memory.push(state, action, next_state, reward)
                self.optimizeModel()

                if done or step == max_episode_steps - 1:
                    print '------Episode {} ended, total reward: {}, step: {}------'.format(self.episodes_done, r_total, step)
                    self.episodes_done += 1
                    self.episode_rewards.append(r_total)
                    self.episode_lengths.append(step)
                    if self.episodes_done % 100 == 0:
                        self.save_checkpoint()
                    break
                state = next_state
            if self.episodes_done % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        self.save_checkpoint()


def main():
    # simple_task_env = SimpleGraspEnv(port=19998)
    # exploration = LinearSchedule(1000, initial_p=1.0, final_p=0.1)
    # agent = HisDQNAgent(HistoryDQN, env=simple_task_env, exploration=exploration)
    # agent.load_checkpoint('20181203140814')
    # agent.train(10000)

    agent = HisDQNAgent(HistoryDQN)
    agent.load_checkpoint('20181203140814')
    plotLearningCurve(agent.episode_rewards)
    plt.show()
    plotLearningCurve(agent.episode_lengths, label='length', color='r')
    plt.show()


if __name__ == '__main__':
    main()
