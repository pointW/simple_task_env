from lstm_dqn import *
from lstm_dqn_action import *
from history_dqn import *
import history_dqn_action
import history_dqn_action_big

# plt.style.use('classic')
plt.rcParams['figure.facecolor'] = 'white'

q_learning_reward = np.load('/home/ur5/thesis/simple_task/scoop_grasp_2d/data/q_learning/20181208154136reward.npy')
q_learning_step = np.load('/home/ur5/thesis/simple_task/scoop_grasp_2d/data/q_learning/20181208154136length.npy')

lstm_agent = LSTMDQNAgent(LSTMQNet)
dqn_agent = HisDQNAgent(HistoryDQN)
lstm_agent.load_checkpoint('20181208153502')
dqn_agent.load_checkpoint('20181208152245')
# dqn_action_agent = history_dqn_action.HisActionDQNAgent(history_dqn_action.HistoryActionDQN)
# dqn_action_agent.load_checkpoint('20181208150213')
# dqn_action_big_agent = history_dqn_action_big.HisActionDQNAgent(history_dqn_action_big.HistoryActionDQN)
# dqn_action_big_agent.load_checkpoint('20181208154423')
# lstm_action_agent = LSTMActionDQNAgent(LSTMActionQNet)
# lstm_action_agent.load_checkpoint('20181208154302')

plotLearningCurve(dqn_agent.episode_rewards[:3000], label='DQN', color='b', shadow=True)
plotLearningCurve(lstm_agent.episode_rewards[:3000], label='DRQN', color='r', shadow=True)
# plotLearningCurve(dqn_action_agent.episode_rewards[:4000], label='DQN_A', color='c', shadow=False)
# plotLearningCurve(dqn_action_big_agent.episode_rewards[:4000], label='DQN_A_B', color='m', shadow=False)
# plotLearningCurve(lstm_action_agent.episode_rewards[:4000], label='DRQN_A', color='y', shadow=False)
plotLearningCurve(q_learning_reward[:3000], label='Q Learning', color='g', shadow=True)

plt.show()

plotLearningCurve(dqn_agent.episode_lengths[:3000], label='DQN', color='b', shadow=True)
plotLearningCurve(lstm_agent.episode_lengths[:3000], label='DRQN', color='r', shadow=True)
plotLearningCurve(q_learning_step[:3000], label='Q Learning', color='g', shadow=True)

plt.show()
