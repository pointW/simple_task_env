from lstm_dqn import *
from history_dqn import *

# plt.style.use('classic')
plt.rcParams['figure.facecolor'] = 'white'

q_learning_reward = np.load('/home/ur5/thesis/simple_task/scoop_1d/data/q_learning/20181207203453reward.npy')
q_learning_step = np.load('/home/ur5/thesis/simple_task/scoop_1d/data/q_learning/20181207203453length.npy')

lstm_agent = LSTMDQNAgent(LSTMQNet)
dqn_agent = HisDQNAgent(HistoryDQN)
lstm_agent.load_checkpoint('20181207161918')
dqn_agent.load_checkpoint('20181201180913')

plotLearningCurve(dqn_agent.episode_rewards[:2000], label='DQN', color='b', shadow=True)
plotLearningCurve(lstm_agent.episode_rewards[:2000], label='DRQN', color='r', shadow=True)
plotLearningCurve(q_learning_reward[:2000], label='Q Learning', color='g', shadow=True)

plt.show()

plotLearningCurve(dqn_agent.episode_lengths[:2000], label='DQN', color='b', shadow=True)
plotLearningCurve(lstm_agent.episode_lengths[:2000], label='DRQN', color='r', shadow=True)
plotLearningCurve(q_learning_step[:2000], label='Q Learning', color='g', shadow=True)

plt.show()
