from collections import namedtuple
import random
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from utils import *
from simple_task_env import SimpleTaskEnv

from plot import plotLearningCurve

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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


class HisDQNAgent:
    def __init__(self, env, exploration, gamma=0.99, model=None):
        self.env = env
        self.exploration = exploration
        self.episode_rewards = []
        self.episode_lengths = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = model
        self.target_net = None
        self.optimizer = None

        if model:
            self.target_net = HistoryDQN()
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net = self.policy_net.to(self.device)
            self.target_net = self.target_net.to(self.device)
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters())

        self.memory = ReplayMemory(10000)
        self.batch_size = 64
        self.gamma = gamma
        self.target_update = 10

        self.steps_done = 0
        self.episodes_done = 0

    def selectAction(self, state, require_q=False):
        e = self.exploration.value(self.steps_done)
        self.steps_done += 1
        with torch.no_grad():
            q_values = self.policy_net(state)
        if random.random() > e:
            action = q_values.max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
        q_value = q_values.gather(1, action).item()
        if require_q:
            return action, q_value
        return action

    def optimizeModel(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        mini_batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                mini_batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in mini_batch.next_state
                                           if s is not None])
        state_batch = torch.cat(mini_batch.state)
        action_batch = torch.cat(mini_batch.action)
        reward_batch = torch.cat(mini_batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    def encode_state(self, theta, last_s):
        last_s_list = last_s.tolist()[0]
        s_list = last_s_list[1:] + [theta]
        s_tensor = torch.tensor(s_list, device=self.device).unsqueeze(0)
        return s_tensor

    def get_initial_state(self, theta):
        s_list = [0, 0, 0, 0, theta]
        s_tensor = torch.tensor(s_list, device=self.device).unsqueeze(0)
        return s_tensor

    def train(self, num_episodes, max_episode_steps=100):
        while self.episodes_done < num_episodes:
            print '------Episode {} / {}------'.format(self.episodes_done, num_episodes)
            s = self.env.reset()
            state = self.get_initial_state(s)
            r_total = 0
            for step in range(max_episode_steps):
                action, q = self.selectAction(state, require_q=True)
                s_, r, done, info = self.env.step(action.item())
                print 'step {}, state: {}, action: {}, q: {}, next state: {}, reward: {} done: {}'\
                    .format(step, s, action.item(), q, s_, r, done)
                r_total += r
                s = s_
                if done or step == max_episode_steps - 1:
                    next_state = None
                else:
                    next_state = self.encode_state(s_, state)
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

    def save_checkpoint(self):
        time_stamp = time.strftime('%Y%m%d%H%M%S', time.gmtime())
        filename = '../data/history_dqn/checkpoint' + time_stamp + '.pth.tar'
        state = {
            'episode': self.episodes_done,
            'steps': self.steps_done,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'memory': self.memory,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
        torch.save(state, filename)

    def load_checkpoint(self, time_stamp):
        filename = '../data/history_dqn/checkpoint' + time_stamp + '.pth.tar'
        print 'loading checkpoint: ', filename
        checkpoint = torch.load(filename)
        self.episodes_done = checkpoint['episode']
        self.steps_done = checkpoint['steps']
        self.memory = checkpoint['memory']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']

        self.policy_net = HistoryDQN()
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_net = self.policy_net.to(self.device)
        self.policy_net.train()

        self.target_net = HistoryDQN()
        self.target_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net = self.target_net.to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def main():
    # simple_task_env = SimpleTaskEnv()
    # exploration = LinearSchedule(1000, initial_p=1.0, final_p=0.1)
    # agent = HisDQNAgent(simple_task_env, exploration)
    # agent.load_checkpoint('20181201155006')
    # agent.train(10000)

    agent = HisDQNAgent(None, None)
    agent.load_checkpoint('20181201155006')
    plotLearningCurve(agent.episode_rewards)
    plt.show()
    plotLearningCurve(agent.episode_lengths, label='length', color='r')
    plt.show()
    pass


if __name__ == '__main__':
    main()
