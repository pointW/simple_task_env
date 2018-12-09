import sys
sys.path.append('../..')

from util.utils import LinearSchedule
from agent.dqn import *
from scoop_2d_env import ScoopEnv
import numpy as np


class HistoryActionDQN(torch.nn.Module):
    def __init__(self):
        super(HistoryActionDQN, self).__init__()

        self.fc1 = nn.Linear(90, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class HisActionDQNAgent(DQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=64, target_update_frequency=10):
        saving_dir = '/home/ur5/thesis/simple_task/scoop_grasp_2d/data/history_action_dqn'
        DQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                          target_update_frequency, saving_dir)

    def encodeState(self, observation, last_s, last_action):
        last_s_list = last_s.tolist()[0]
        last_action = np.identity(5)[last_action].tolist()
        s_list = last_s_list[9:] + last_action + list(observation)
        s_tensor = torch.tensor(s_list, device=self.device).unsqueeze(0)
        return s_tensor

    def initialState(self, observation):
        s_list = [0 for _ in range(86)] + list(observation)
        s_tensor = torch.tensor(s_list, device=self.device).unsqueeze(0)
        return s_tensor

    def train(self, num_episodes, max_episode_steps=100):
        while self.episodes_done < num_episodes:
            print '------Episode {} / {}------'.format(self.episodes_done, num_episodes)
            s = self.env.reset()
            state = self.initialState(s)
            r_total = 0
            for step in range(max_episode_steps):
                action, q = self.selectAction(state, require_q=True)
                s_, r, done, info = self.env.step(action.item())
                print 'step {}, action: {}, q: {}, reward: {} done: {}'\
                    .format(step, action.item(), q, r, done)
                r_total += r
                if done or step == max_episode_steps - 1:
                    next_state = None
                else:
                    next_state = self.encodeState(s_, state, action.item())
                reward = torch.tensor([r], device=self.device, dtype=torch.float)
                self.memory.push(state, action, next_state, reward)
                self.optimizeModel()

                if done or step == max_episode_steps - 1:
                    print '------Episode {} ended, total reward: {}, step: {}------'\
                        .format(self.episodes_done, r_total, step)
                    print '------Total steps done: {}, current e: {} ------'\
                        .format(self.steps_done, self.exploration.value(self.steps_done))
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


if __name__ == '__main__':
    # agent = HisActionDQNAgent(HistoryActionDQN, model=HistoryActionDQN(), env=ScoopEnv(port=20003),
    #                           exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.1), batch_size=128)
    # agent.train(10000)
    agent = HisActionDQNAgent(HistoryActionDQN)
    agent.load_checkpoint('20181209154025')
    plotLearningCurve(agent.episode_rewards)
    plt.show()
    plotLearningCurve(agent.episode_lengths, label='length', color='r')
    plt.show()
