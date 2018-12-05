import sys
sys.path.append('../../..')

import torch

from scoop_grasp_2d.scripts.scoop_2d_env import ScoopEnv


def encodeState(observation, last_s):
    last_s_list = last_s.tolist()[0]
    s_list = last_s_list[4:] + list(observation)
    s_tensor = torch.tensor(s_list).unsqueeze(0)
    return s_tensor


def initialState(observation):
    s_list = [0 for _ in range(36)] + list(observation)
    s_tensor = torch.tensor(s_list).unsqueeze(0)
    return s_tensor


def worker(port, child_pipe):
    state = None
    is_done = True
    env = ScoopEnv(port)
    steps = 0
    total_reward = 0
    while True:
        data = child_pipe.recv()
        if data['cmd'] == 'reset':
            observation = env.reset()
            state = initialState(observation)
            is_done = False
            child_pipe.send(state)
            steps = 0
            total_reward = 0
            continue

        if data['cmd'] == 'step':
            if is_done:
                child_pipe.send(False)
                continue
            action = data['action']
            s_, r, done, info = env.step(action.item())
            steps += 1
            total_reward += r
            if done or steps == 100:
                next_state = None
                is_done = True
            else:
                next_state = encodeState(s_, state)
            reward = torch.tensor([r], dtype=torch.float)
            ret_data = {'transition': (state, action, next_state, reward),
                        'log': {'step': steps,
                                'reward': total_reward}}
            child_pipe.send(ret_data)
            continue

        if data['cmd'] == 'close':
            return

