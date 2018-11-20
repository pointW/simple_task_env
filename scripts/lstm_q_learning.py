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
        self.memory = [[]]
        self.position = 0
        
    def push(self, *args):
        self.memory[-1].append(Transition(*args))
        if args[2] is None:
            self.memory.append([])
            if len(self.memory) > self.capacity:
                self.memory = self.memory[1:]
                
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# class ReplayMemory(object):
# 
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0
# 
#     def push(self, *args):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity
# 
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
# 
#     def __len__(self):
#         return len(self.memory)


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

        self.model = self.model.to(self.device)

        self.optimizer = optim.RMSprop(self.model.parameters())
        self.memory = ReplayMemory(100)
        self.batch_size = 10
        self.gamma = gamma
        self.alpha = alpha

        self.steps_done = 0

        self.hx = None
        self.cx = None
        
    def resetHidden(self):
        self.hx = torch.zeros(1, 128, device=self.device)
        self.cx = torch.zeros(1, 128, device=self.device)

    def selectAction(self, state):
        e = self.exploration.value(self.steps_done)
        self.steps_done += 1
        data, (self.hx, self.cx) = self.model((state, (self.hx, self.cx)))
        if random.random() > e:
            with torch.no_grad():
                return data.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)

    def optimizeModel(self):
        if len(self.memory) < self.batch_size:
            return
        memory = self.memory.sample(self.batch_size)

        loss = 0
        
        for transitions in memory:
            hx = torch.zeros(1, 128, device=self.device)
            cx = torch.zeros(1, 128, device=self.device)
            for transition in transitions:
                state = transition.state
                next_state = transition.next_state
                reward = transition.reward
                action = transition.action
                values, (hx, cx) = self.model((state, (hx, cx)))
                state_action_value = values.gather(1, action)
                if next_state is not None:
                    hx_copy = hx.clone()
                    cx_copy = cx.clone()
                    values, (_, _) = self.model((next_state, (hx_copy, cx_copy)))
                    next_state_value = values.max(1)[0].detach()
                else:
                    next_state_value = torch.zeros(1, device=self.device)
                expected_state_action_value = (next_state_value * self.gamma) + reward

                advantage = expected_state_action_value - state_action_value

                loss = loss + 0.5 * advantage.pow(2)

                # loss = F.smooth_l1_loss(state_action_value, expected_state_action_value.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, num_episodes, max_episode_steps=100):
        episode_rewards = []
        episode_lengths = []
        for episode in range(num_episodes):
            print '------Episode {} / {}------'.format(episode, num_episodes)
            self.resetHidden()
            s = self.env.reset()
            state = torch.tensor(s, device=self.device).unsqueeze(0)
            r_total = 0
            for step in range(max_episode_steps):
                action = self.selectAction(state)
                s_, r, done, info = self.env.step(action.item())
                print 'step {}, {}, {}, {}, {}, {}'.format(step, s, action.item(), s_, r, done)
                s = s_
                r_total += r
                if done or step == max_episode_steps - 1:
                    next_state = None
                else:
                    next_state = torch.tensor(s_, device=self.device).unsqueeze(0)
                reward = torch.tensor([r], device=self.device, dtype=torch.float)
                self.memory.push(state, action, next_state, reward)
                if done or step == max_episode_steps - 1:
                    print '------Episode {} ended, total reward: {}, step: {}------'.format(episode, r_total, step)
                    episode_rewards.append(r_total)
                    episode_lengths.append(step)
                    self.optimizeModel()
                    if episode % 100 == 0:
                        np.save('length', np.array(episode_lengths))
                        np.save('reward', np.array(episode_rewards))
                        torch.save(self.model.state_dict(), open('lstm_q.pt', 'w'))
                    break
                state = next_state
        np.save('length', np.array(episode_lengths))
        np.save('reward', np.array(episode_rewards))
        torch.save(self.model.state_dict(), open('lstm_q.pt', 'w'))


def main():
    simple_task_env = SimpleTaskEnv()
    agent = LSTMAgent(LSTMQNet(), simple_task_env, LinearSchedule(1000, final_p=0.1))
    agent.train(10000)


if __name__ == '__main__':
    main()
