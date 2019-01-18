import sys
sys.path.append('../..')

from util.utils import LinearSchedule

from agent.dqn import *
from scoop_grasp_1d_pid_env import ScoopEnv


class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(240, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent(DQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=64, target_update_frequency=10):
        saving_dir = '/home/ur5/thesis/simple_task/scoop_grasp_1d/data/dqn'
        DQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                          target_update_frequency, saving_dir)

    def train(self, num_episodes, max_episode_steps=100):
        while self.episodes_done < num_episodes:
            print '------Episode {} / {}------'.format(self.episodes_done, num_episodes)
            s = self.env.reset()
            state = torch.tensor(s, device=self.device).unsqueeze(0)
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
                    next_state = torch.tensor(s_, device=self.device).unsqueeze(0)
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
    agent = Agent(DQN, model=DQN(), env=ScoopEnv(port=19997),
                  exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.1), batch_size=128)
    agent.load_checkpoint('20190118201202')
    agent.train(10000, max_episode_steps=200)
