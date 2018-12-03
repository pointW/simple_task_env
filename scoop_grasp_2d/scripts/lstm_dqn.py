import sys
sys.path.append('../..')
from agent.drqn import *
from scoop_2d_env import ScoopEnv

class LSTMQNet(torch.nn.Module):
    def __init__(self):
        super(LSTMQNet, self).__init__()

        self.fc1 = nn.Linear(4, 128)
        self.lstm = nn.LSTMCell(128, 64)
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


class LSTMDQNAgent(DRQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=1000, batch_size=10, target_update_frequency=10):
        saving_dir = '/home/ur5/thesis/simple_task/scoop_grasp_2d/data/lstm_dqn'
        DRQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                           target_update_frequency, saving_dir)


if __name__ == '__main__':
    agent = LSTMDQNAgent(LSTMQNet, model=LSTMQNet(), env=ScoopEnv(port=20000),
                         exploration=LinearSchedule(1000, initial_p=1.0, final_p=0.1))
    agent.load_checkpoint('20181203155503')
    agent.train(10000)

