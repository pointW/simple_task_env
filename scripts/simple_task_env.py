import time

import numpy as np

from vrep_arm_toolkit.simulation import vrep
from vrep_arm_toolkit.robots.ur5 import UR5
from vrep_arm_toolkit.grippers.rdd import RDD
import vrep_arm_toolkit.utils.vrep_utils as utils

VREP_BLOCKING = vrep.simx_opmode_blocking


class SimpleTaskEnv:
    def __init__(self):
        self.sim_client = utils.connectToSimulation('127.0.0.1', 19997)

        # Create UR5 and restart simulator
        self.rdd = RDD(self.sim_client, open_force=10)
        self.ur5 = UR5(self.sim_client, self.rdd)
        self.nA = 2

        self.cube = None

    def reset(self):
        vrep.simxStopSimulation(self.sim_client, VREP_BLOCKING)
        time.sleep(1)
        vrep.simxStartSimulation(self.sim_client, VREP_BLOCKING)
        time.sleep(1)
        # Generate a cube
        position = [-0.22, 0.9, 0.05]
        orientation = [0, 0, 0]
        # orientation = [0, 0, 0]
        size = [0.1, 0.2, 0.05]
        mass = 0.2
        color = [255, 0, 0]
        self.cube = utils.generateShape(self.sim_client, 'cube', 0, size, position, orientation, mass, color)
        time.sleep(1)

        # dy = 0.3 * np.random.random()
        dy = 0.2
        current_pose = self.ur5.getEndEffectorPose()
        target_pose = current_pose.copy()
        target_pose[1, 3] += dy

        self.ur5.moveTo(target_pose)
        time.sleep(0.5)

        self.rdd.openFinger(RDD.NARROW)

    def step(self, a):
        current_pose = self.ur5.getEndEffectorPose()
        target_pose = current_pose.copy()
        if a == 0:
            target_pose[1, 3] -= 0.01
        else:
            target_pose[1, 3] += 0.01
        utils.setObjectPosition(self.sim_client, self.ur5.UR5_target, target_pose[:3, 3])
        # utils.setObjectOrientation(self.sim_client, self.ur5.UR5_target, target_pose.flatten()[:-4])
        # self.ur5.moveTo(target_pose)
        time.sleep(0.5)

        # arm is in wrong pose
        sim_ret, tip_position = utils.getObjectPosition(self.sim_client, self.ur5.gripper_tip)
        if np.linalg.norm(tip_position - target_pose[:3, -1]) > 0.03:
            print 'Wrong position, dist: ', np.linalg.norm(tip_position - target_pose[:3, -1])
            return None, 0, True, None
        else:
            # cube is lifted
            sim_ret, cube_orientation = utils.getObjectOrientation(self.sim_client, self.cube)
            print cube_orientation
            if cube_orientation[0] < -0.02:
                return None, 1, True, None
            # sim_ret, cube_pose = utils.getObjectPosition(self.sim_client, self.cube)
            # while any(np.isnan(cube_pose)):
            #     res, cube_pose = utils.getObjectPosition(self.sim_client, self.cube)
            # if cube_pose[2] > 0.03:
            #     return None, 1, True, None

            # cube is not lifted
            else:
                return self.rdd.getFingerPosition(RDD.NARROW), 0, False, None


if __name__ == '__main__':
    env = SimpleTaskEnv()
    env.reset()
    while True:
        a = input('input action')
        s_, r, done, info = env.step(int(a))
        print s_, r, done
        if done:
            break
