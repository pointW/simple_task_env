from collections import namedtuple
import random
import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from util.utils import LinearSchedule
from util.plot import plotLearningCurve
from agent.dqn import *
from scoop_1d_env import ScoopEnv


class HistoryDQN(torch.nn.Module):
    def __init__(self):
        super(HistoryDQN, self).__init__()

        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class HisDQNAgent(DQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=64, target_update_frequency=10):
        saving_dir = '/home/ur5/thesis/simple_task/scoop_1d/data/history_dqn'
        DQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                          target_update_frequency, saving_dir)

    def encodeState(self, theta, last_s):
        last_s_list = last_s.tolist()[0]
        s_list = last_s_list[1:] + [theta]
        s_tensor = torch.tensor(s_list, device=self.device).unsqueeze(0)
        return s_tensor

    def initialState(self, theta):
        s_list = [0, 0, 0, 0, theta]
        s_tensor = torch.tensor(s_list, device=self.device).unsqueeze(0)
        return s_tensor


def main():
    simple_task_env = ScoopEnv(port=19997)
    exploration = LinearSchedule(1000, initial_p=1.0, final_p=0.1)
    agent = HisDQNAgent(HistoryDQN, env=simple_task_env, exploration=exploration)
    agent.load_checkpoint('20181201155006')
    agent.save_checkpoint()
    agent.train(10000)

    # agent = HisDQNAgent(None, None)
    # agent.load_checkpoint('20181201155006')
    # plotLearningCurve(agent.episode_rewards)
    # plt.show()
    # plotLearningCurve(agent.episode_lengths, label='length', color='r')
    # plt.show()
    # pass


if __name__ == '__main__':
    main()
