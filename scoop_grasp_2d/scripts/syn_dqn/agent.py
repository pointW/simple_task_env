import sys
import os
import time
sys.path.append('..')

from collections import namedtuple
import random

from multiprocessing import Process, Pipe

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from worker import worker
from scoop_grasp_2d.scripts.scoop_2d_env import ScoopEnv
from util.utils import LinearSchedule

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        self.fc1 = nn.Linear(40, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SynDQNAgent:
    def __init__(self, port_start, num_worker, exploration, gamma=0.99, memory_size=10000, batch_size=64,
                 target_update_frequency=10, saving_dir=None):
        self.policy_net = HistoryDQN()
        self.policy_net.to(DEVICE)
        self.policy_net.train()
        self.target_net = HistoryDQN()
        self.target_net.to(DEVICE)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())

        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.num_worker = num_worker

        self.saving_dir = saving_dir

        self.parent_pipes = []
        self.workers = []
        self.worker_states = []

        self.exploration = exploration

        self.steps_done = 0
        self.episodes_done = 0

        for i in range(num_worker):
            parent_pipe, child_pipe = Pipe()
            self.parent_pipes.append(parent_pipe)
            p = Process(target=worker, args=(port_start+i, child_pipe))
            p.start()
            self.workers.append(p)
            self.worker_states.append(None)

        self.episode_rewards = []
        self.episode_lengths = []

    def resetEnvs(self):
        for send in self.parent_pipes:
            data = {'cmd': 'reset'}
            send.send(data)

        for i, recv in enumerate(self.parent_pipes):
            self.worker_states[i] = recv.recv()
            continue

    def getMakeUpState(self):
        s_list = [0. for _ in range(40)]
        s_tensor = torch.tensor(s_list).unsqueeze(0)
        return s_tensor

    def step(self):
        living_worker = []
        for i in range(self.num_worker):
            if self.worker_states[i] is not None:
                living_worker.append(i)

        inputs = [self.worker_states[i]
                  for i in living_worker]
        input_tensor = torch.cat(inputs).to(DEVICE)
        state_action_values = self.policy_net(input_tensor).to('cpu')
        e = self.exploration.value(self.steps_done)
        self.steps_done += 1
        for i in range(len(living_worker)):
            if random.random() > e:
                action = state_action_values[i].max(0)[1].view(1, 1)
            else:
                action = torch.tensor([[random.randrange(5)]], dtype=torch.long)

            q_value = state_action_values[i].unsqueeze(0).gather(1, action).item()
            print 'q: {}'.format(q_value)

            data = {'cmd': 'step',
                    'action': action}
            self.parent_pipes[living_worker[i]].send(data)
        for i in living_worker:
            ret_data = self.parent_pipes[i].recv()
            if not ret_data:
                continue
            t = ret_data['transition']
            self.memory.push(t[0], t[1], t[2], t[3])
            self.worker_states[i] = t[2]
            if t[2] is None:
                log = ret_data['log']
                self.episode_lengths.append(log['step'])
                self.episode_rewards.append(log['reward'])
                print 'worker done, step: {}, reward: {}'.format(log['step'], log['reward'])

    def optimizeModel(self):
        """
        one step update for the model
        :return: None
        """
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        mini_batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                mini_batch.next_state)), device=DEVICE, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in mini_batch.next_state
                                           if s is not None]).to(DEVICE)
        state_batch = torch.cat(mini_batch.state).to(DEVICE)
        action_batch = torch.cat(mini_batch.action).to(DEVICE)
        reward_batch = torch.cat(mini_batch.reward).to(DEVICE)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=DEVICE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    def train(self, num_episodes, max_episode_steps=200):
        while self.episodes_done < num_episodes:
            print '------Episode {} / {}------'.format(self.episodes_done, num_episodes)
            self.resetEnvs()
            for step in range(max_episode_steps):
                print 'step ', step
                self.step()
                self.optimizeModel()
                if all(state is None for state in self.worker_states):
                    self.episodes_done += 1
                    if self.episodes_done % self.target_update_frequency == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                    if self.episodes_done % 50 == 0:
                        self.save_checkpoint()
                    break

    def save_checkpoint(self):
        """
        save checkpoint in self.saving_dir
        :return: None
        """
        time_stamp = time.strftime('%Y%m%d%H%M%S', time.gmtime())
        filename = os.path.join(self.saving_dir, 'checkpoint' + time_stamp + '.pth.tar')
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
        """
        load checkpoint at input time stamp
        :param time_stamp: time stamp for the checkpoint
        :return: None
        """
        filename = os.path.join(self.saving_dir, 'checkpoint' + time_stamp + '.pth.tar')
        print 'loading checkpoint: ', filename
        checkpoint = torch.load(filename)
        self.episodes_done = checkpoint['episode']
        self.steps_done = checkpoint['steps']
        self.memory = checkpoint['memory']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']

        self.policy_net = HistoryDQN()
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_net = self.policy_net.to(DEVICE)
        self.policy_net.train()

        self.target_net = HistoryDQN()
        self.target_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net = self.target_net.to(DEVICE)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == '__main__':
    agent = SynDQNAgent(20005, 4, LinearSchedule(10000, 0.1), batch_size=256,
                        saving_dir='/home/ur5/thesis/simple_task/scoop_grasp_2d/data/sync_dqn')
    agent.load_checkpoint('20181205014300')
    agent.train(10000)







