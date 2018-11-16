import sys
import time
from collections import defaultdict

import numpy as np

from vrep_arm_toolkit.simulation import vrep
from vrep_arm_toolkit.robots.ur5 import UR5
from vrep_arm_toolkit.grippers.rdd import RDD
import vrep_arm_toolkit.utils.vrep_utils as utils
from vrep_arm_toolkit.utils import transformations

VREP_BLOCKING = vrep.simx_opmode_blocking


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        '''
        Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Args:
            - schedule_timesteps: Number of timesteps for which to linearly anneal initial_p to final_p
            - initial_p: initial output value
            -final_p: final output value
        '''
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

# class QLearningAgent:
#     def __init__(self, start_eps=1.0, final_eps=0.1, ):
#         self.q_table = defaultdict(lambda: np.zeros(2))
#
#
#
#     def e_greedy(self, s):


def reset(sim_client):
    vrep.simxStopSimulation(sim_client, VREP_BLOCKING)
    time.sleep(1)
    vrep.simxStartSimulation(sim_client, VREP_BLOCKING)
    time.sleep(1)
    # Generate a cube
    position = [-0.2, 0.9, 0.05]
    orientation = [0, 0, 0]
    # orientation = [0, 0, 0]
    size = [0.1, 0.2, 0.05]
    mass = 0.1
    color = [255, 0, 0]
    cube = utils.generateShape(sim_client, 'cube', 0, size, position, orientation, mass, color)
    time.sleep(1)


def main():
    # Attempt to connect to simulator
    sim_client = utils.connectToSimulation('127.0.0.1', 19997)

    # Create UR5 and restart simulator
    gripper = RDD(sim_client)
    ur5 = UR5(sim_client, gripper)
    vrep.simxStopSimulation(sim_client, VREP_BLOCKING)
    time.sleep(1)
    vrep.simxStartSimulation(sim_client, VREP_BLOCKING)
    time.sleep(1)

    reset(sim_client)







