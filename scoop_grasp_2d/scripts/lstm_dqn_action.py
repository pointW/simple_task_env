import sys
sys.path.append('../..')
from agent.drqn import *
from scoop_2d_env import ScoopEnv

import matplotlib.pyplot as plt
from util.plot import plotLearningCurve

class LSTMActionQNet(torch.nn.Module):
    def __init__(self):
        super(LSTMActionQNet, self).__init__()

        self.fc1 = nn.Linear(9, 32)
        self.lstm = nn.LSTMCell(32, 64)
        self.fc2 = nn.Linear(64, 5)

        self.hidden_size = 64

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.fc2(x), (hx, cx)


class LSTMActionDQNAgent(DRQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=1000, batch_size=10, target_update_frequency=10):
        saving_dir = '/home/ur5/thesis/simple_task/scoop_grasp_2d/data/lstm_action_dqn'
        DRQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                           target_update_frequency, saving_dir)

    def train(self, num_episodes, max_episode_steps=100):
        while self.episodes_done < num_episodes:
            print '------Episode {} / {}------'.format(self.episodes_done, num_episodes)
            self.hidden = self.resetHidden()
            s = [0 for _ in range(5)] + list(self.env.reset())
            state = torch.tensor(s, device=self.device).unsqueeze(0)
            r_total = 0
            for step in range(max_episode_steps):
                action, q = self.selectAction(state, require_q=True)
                s_, r, done, info = self.env.step(action.item())
                print 'step {}, action: {}, q: {}, reward: {} done: {}' \
                    .format(step, action.item(), q, r, done)
                r_total += r
                if done or step == max_episode_steps - 1:
                    next_state = None
                else:
                    next_state = torch.tensor(np.identity(5)[action.item()].tolist() + list(s_), device=self.device).unsqueeze(0)
                reward = torch.tensor([r], device=self.device, dtype=torch.float)
                self.memory.push(state, action, next_state, reward)
                self.optimizeModel()

                if done or step == max_episode_steps - 1:
                    print '------Episode {} ended, total reward: {}, step: {}------'\
                        .format(self.episodes_done, r_total, step)
                    print '------Total steps done: {}, current e: {} ------' \
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
    # agent = LSTMActionDQNAgent(LSTMActionQNet, model=LSTMActionQNet(), env=ScoopEnv(port=20111),
    #                            exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.1), batch_size=1)
    # agent.train(10000)

    agent = LSTMActionDQNAgent(LSTMActionQNet)
    agent.load_checkpoint('20181208183426')
    plotLearningCurve(agent.episode_rewards)
    plt.show()
    plotLearningCurve(agent.episode_lengths, label='length', color='r')
    plt.show()