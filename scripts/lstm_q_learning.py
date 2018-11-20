from collections import namedtuple
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from utils import *
from simple_task_env import SimpleTaskEnv

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class LSTMQNet(torch.nn.Module):
    def __init__(self):
        super(LSTMQNet, self).__init__()

        self.lstm = nn.LSTMCell(1, 128)

        self.linear = nn.Linear(128, 2)

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return self.linear(x), (hx, cx)


class LSTMAgent:
    def __init__(self, model, env, exploration, gamma=0.9, alpha=0.01):
        self.model = model
        self.env = env
        self.exploration = exploration
        self.episode_rewards = []
        self.episode_lengths = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = self.model.to(self.device)
        self.target_net = self.model.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(1)
        self.batch_size = 1
        self.gamma = gamma
        self.alpha = alpha

        self.steps_done = 0

        self.hx = None
        self.cx = None

    def selectAction(self, state):
        e = self.exploration.value(self.steps_done)
        self.steps_done += 1
        data, (self.hx, self.cx) = self.policy_net((state, (self.hx, self.cx)))
        if random.random > e:
            with torch.no_grad():
                return data.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)

    def optimizeModel(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.double)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

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

    def train(self, num_episodes, max_episode_steps=200):
        for episode in range(num_episodes):
            self.hx = torch.zeros(1, 128, device=self.device)
            self.cx = torch.zeros(1, 128, device=self.device)
            # s = self.env.reset()
            s = 0.50
            state = torch.tensor(s, device=self.device).unsqueeze(0)
            r_total = 0
            for step in range(max_episode_steps):
                action = self.selectAction(state)
                s_, r, done, info = self.env.step(action.item())
                reward = torch.tensor([r], device=self.device)
                next_state = torch.tensor(s_, device=self.device).unsqueeze(0)
                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model()


def main():
    # simple_task_env = SimpleTaskEnv()
    agent = LSTMAgent(LSTMQNet(), None, LinearSchedule(1000, final_p=0.9, initial_p=0.1))
    agent.train(100)


if __name__ == '__main__':
    main()
