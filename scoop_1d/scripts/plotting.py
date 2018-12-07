from lstm_dqn import *
from history_dqn import *

lstm_agent = LSTMDQNAgent(LSTMQNet)
dqn_agent = HisDQNAgent(HistoryDQN)
lstm_agent.load_checkpoint('20181207161918')
dqn_agent.load_checkpoint('20181201180913')

plotLearningCurve(dqn_agent.episode_rewards[:6000], label='DQN', color='b')
plotLearningCurve(lstm_agent.episode_rewards[:6000], label='DRQN', color='r')

plt.show()

plotLearningCurve(dqn_agent.episode_lengths[:6000], label='DQN', color='b')
plotLearningCurve(lstm_agent.episode_lengths[:6000], label='DRQN', color='r')

plt.show()
