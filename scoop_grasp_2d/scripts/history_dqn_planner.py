from history_dqn import *


class HistoryDQNWithPlannerAgent(HisDQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=64, target_update_frequency=10):
        saving_dir = '/home/ur5/thesis/simple_task/scoop_grasp_2d/data/history_dqn_planner'
        DQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                          target_update_frequency, saving_dir)

    def selectAction(self, state, require_q=False, planner=False):
        e = self.exploration.value(self.steps_done)
        self.steps_done += 1
        with torch.no_grad():
            q_values = self.forwardPolicyNet(state)

        if planner:
            a = self.env.planAction()
            action = torch.tensor([[a]], device=self.device, dtype=torch.long)

        else:
            if random.random() > e:
                action = q_values.max(1)[1].view(1, 1)
            else:
                a = random.randrange(self.env.nA)
                action = torch.tensor([[a]], device=self.device, dtype=torch.long)

        q_value = q_values.gather(1, action).item()
        if require_q:
            return action, q_value
        return action

    def train(self, num_episodes, max_episode_steps=100, planning_episodes=500):
        use_planner = True
        starting_episode = self.episodes_done
        while self.episodes_done < num_episodes:
            if self.episodes_done > starting_episode + planning_episodes:
                use_planner = False
            print '------Episode {} / {}------'.format(self.episodes_done, num_episodes)
            if use_planner:
                print '------Planning episode {} / {}'.format(self.episodes_done - starting_episode, planning_episodes)
            s = self.env.reset()
            state = self.initialState(s)
            r_total = 0
            for step in range(max_episode_steps):
                action, q = self.selectAction(state, require_q=True, planner=use_planner)
                s_, r, done, info = self.env.step(action.item())
                print 'step {}, action: {}, q: {}, reward: {} done: {}'\
                    .format(step, action.item(), q, r, done)
                r_total += r
                s = s_
                if done or step == max_episode_steps - 1:
                    next_state = None
                else:
                    next_state = self.encodeState(s_, state)
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
    # agent = HistoryDQNWithPlannerAgent(HistoryDQN, model=HistoryDQN(), env=ScoopEnv(port=20010),
    #                                    exploration=LinearSchedule(10000, initial_p=1.0, final_p=0.1), batch_size=128)
    # agent.load_checkpoint('20181206220907')
    # agent.train(10000, planning_episodes=0)

    agent = HistoryDQNWithPlannerAgent(HistoryDQN)
    agent.load_checkpoint('20181207165154')
    plotLearningCurve(agent.episode_rewards)
    plt.show()
    plotLearningCurve(agent.episode_lengths, label='length', color='r')
    plt.show()

