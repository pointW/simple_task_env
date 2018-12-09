import sys
sys.path.append('../..')

from util.utils import LinearSchedule
from agent.dqn import *
from scoop_2d_env import ScoopEnv


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


class HisDQNAgent(DQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=64, target_update_frequency=10):
        saving_dir = '/home/ur5/thesis/simple_task/scoop_grasp_2d/data/history_dqn'
        DQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                          target_update_frequency, saving_dir)

    def encodeState(self, observation, last_s):
        last_s_list = last_s.tolist()[0]
        s_list = last_s_list[4:] + list(observation)
        s_tensor = torch.tensor(s_list, device=self.device).unsqueeze(0)
        return s_tensor

    def initialState(self, observation):
        s_list = [0 for _ in range(36)] + list(observation)
        s_tensor = torch.tensor(s_list, device=self.device).unsqueeze(0)
        return s_tensor


if __name__ == '__main__':
    # agent = HisDQNAgent(HistoryDQN, model=HistoryDQN(), env=ScoopEnv(port=20020),
    #                     exploration=LinearSchedule(100000, initial_p=1.0, final_p=0.1), batch_size=128)
    # agent.load_checkpoint('20181206221209')
    # agent.train(10000)

    agent = HisDQNAgent(HistoryDQN)
    agent.load_checkpoint('20181209152115')
    plotLearningCurve(agent.episode_rewards)
    plt.show()
    plotLearningCurve(agent.episode_lengths, label='length', color='r')
    plt.show()
