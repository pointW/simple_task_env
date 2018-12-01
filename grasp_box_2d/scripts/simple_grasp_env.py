import time

import numpy as np

from vrep_arm_toolkit.simulation import vrep
from vrep_arm_toolkit.robots.ur5 import UR5
from vrep_arm_toolkit.grippers.rdd import RDD
import vrep_arm_toolkit.utils.vrep_utils as utils

VREP_BLOCKING = vrep.simx_opmode_blocking


class SimpleGraspEnv:
    N = 0
    S = 1
    E = 2
    W = 3
    CLOSE = 4

    def __init__(self, port=19997):
        self.sim_client = utils.connectToSimulation('127.0.0.1', port)

        self.rdd = RDD(self.sim_client, open_force=20, close_force=20)
        self.ur5 = UR5(self.sim_client, self.rdd)
        self.nA = 5

        self.cube = None
        self.cube_start_position = [-0.35, 0.7, 0.03]
        self.cube_size = [0.1, 0.1, 0.1]
        self.open_position = 0.3

    def getState(self):
        sim_ret, narrow_velocity = utils.getJointPosition(self.sim_client, self.rdd.finger_joint_narrow)
        sim_ret, narrow_force = utils.getJointForce(self.sim_client, self.rdd.finger_joint_narrow)
        sim_ret, wide_velocity = utils.getJointPosition(self.sim_client, self.rdd.finger_joint_wide)
        sim_ret, wide_force = utils.getJointForce(self.sim_client, self.rdd.finger_joint_wide)
        return narrow_velocity, narrow_force, wide_velocity, wide_force

    def reset(self):
        vrep.simxStopSimulation(self.sim_client, VREP_BLOCKING)
        time.sleep(1)
        vrep.simxStartSimulation(self.sim_client, VREP_BLOCKING)
        time.sleep(1)
        # Generate a cube
        position = self.cube_start_position
        # orientation = [np.radians(90), 0, np.radians(90)]
        orientation = [0, 0, 0]
        size = self.cube_size
        mass = 10
        color = [255, 0, 0]
        self.cube = utils.generateShape(self.sim_client, 'cube', 0, size, position, orientation, mass, color)

        self.rdd.open(self.open_position)
        utils.setObjectPosition(self.sim_client, self.ur5.UR5_target, [-0.2, 0.5, 0.05])
        # time.sleep(1)

        return self.getState()

    def step(self, a):
        if a in [self.N, self.S, self.W, self.E]:
            current_pose = self.ur5.getEndEffectorPose()
            target_pose = current_pose.copy()
            if a == self.N:
                target_pose[0, 3] += 0.01
            elif a == self.S:
                target_pose[0, 3] -= 0.01
            elif a == self.W:
                target_pose[1, 3] += 0.01
            elif a == self.E:
                target_pose[1, 3] -= 0.01
            utils.setObjectPosition(self.sim_client, self.ur5.UR5_target, target_pose[:3, 3])
        elif a is self.CLOSE:
            closed = self.rdd.close()
            if not closed:
                sim_ret, cube_position = utils.getObjectPosition(self.sim_client, self.cube)
                sim_ret, tip_position = utils.getObjectPosition(self.sim_client, self.ur5.gripper_tip)
                if np.all(tip_position > (np.array(cube_position) - np.array(self.cube_size))) and \
                        np.all(tip_position < (np.array(cube_position) + np.array(self.cube_size))):
                    return None, 1, True, None
            self.rdd.open(self.open_position)
        return self.getState(), 0, False, None


if __name__ == '__main__':
    env = SimpleGraspEnv(port=19998)
    env.reset()
    while True:
        a = input('input action')
        s_, r, done, info = env.step(int(a))
        print s_, r, done
        if done:
            break
