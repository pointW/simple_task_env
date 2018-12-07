import time

import numpy as np

from vrep_arm_toolkit.simulation import vrep
from vrep_arm_toolkit.robots.ur5 import UR5
from vrep_arm_toolkit.grippers.rdd import RDD
import vrep_arm_toolkit.utils.vrep_utils as utils

VREP_BLOCKING = vrep.simx_opmode_blocking

CUBE_MESH = '/home/ur5/thesis/simple_task/mesh/block.obj'


class ScoopEnv:
    def __init__(self, port=19997):
        self.sim_client = utils.connectToSimulation('127.0.0.1', port)

        # Create UR5 and restart simulator
        self.rdd = RDD(self.sim_client, open_force=10)
        self.ur5 = UR5(self.sim_client, self.rdd)
        self.nA = 2

        self.cube = None
        self.cube_start_position = [-0.2, 0.9, 0.05]
        self.cube_size = [0.1, 0.2, 0.04]

    def reset(self):
        vrep.simxStopSimulation(self.sim_client, VREP_BLOCKING)
        time.sleep(1)
        vrep.simxStartSimulation(self.sim_client, VREP_BLOCKING)
        time.sleep(1)
        # Generate a cube
        # position = self.cube_start_position
        # orientation = [np.radians(90), 0, np.radians(90)]
        # # orientation = [0, 0, 0]
        # size = self.cube_size
        # mass = 0.1
        # color = [255, 0, 0]
        # self.cube = utils.importShape(self.sim_client, 'cube', CUBE_MESH, position, orientation, color)
        # self.cube = utils.generateShape(self.sim_client, 'cube', 0, size, position, orientation, mass, color)
        # time.sleep(1)

        sim_ret, self.cube = utils.getObjectHandle(self.sim_client, 'cube')

        utils.setObjectPosition(self.sim_client, self.ur5.UR5_target, [-0.2, 0.6, 0.07])

        dy = 0.3 * np.random.random()
        # dy = 0.3
        current_pose = self.ur5.getEndEffectorPose()
        target_pose = current_pose.copy()
        target_pose[1, 3] += dy

        self.ur5.moveTo(target_pose)
        # time.sleep(0.5)

        self.rdd.openFinger(RDD.NARROW)

        return self.rdd.getFingerPosition(RDD.NARROW)

    def step(self, a):
        current_pose = self.ur5.getEndEffectorPose()
        target_pose = current_pose.copy()
        if a == 0:
            target_pose[1, 3] -= 0.02
        else:
            target_pose[1, 3] += 0.02
        utils.setObjectPosition(self.sim_client, self.ur5.UR5_target, target_pose[:3, 3])
        # utils.setObjectOrientation(self.sim_client, self.ur5.UR5_target, target_pose.flatten()[:-4])
        # self.ur5.moveTo(target_pose)
        # time.sleep(0.5)

        # arm is in wrong pose
        sim_ret, tip_position = utils.getObjectPosition(self.sim_client, self.ur5.gripper_tip)
        # if np.linalg.norm(tip_position - target_pose[:3, -1]) > 0.05:
        #     print 'Wrong position, dist: ', np.linalg.norm(tip_position - target_pose[:3, -1])
        #     return None, -1, True, None
        if tip_position[1] < 0.42 or tip_position[1] > 0.95:
            print 'Wrong arm position: ', tip_position
            return None, -1, True, None
        else:
            # cube is lifted
            sim_ret, cube_orientation = utils.getObjectOrientation(self.sim_client, self.cube)
            if cube_orientation[0] < -0.02:
                return None, 1, True, None

            # cube in wrong position
            sim_ret, cube_position = utils.getObjectPosition(self.sim_client, self.cube)
            while any(np.isnan(cube_position)):
                res, cube_position = utils.getObjectPosition(self.sim_client, self.cube)
            # if np.any(np.array(cube_position) < np.array(self.cube_start_position) - 0.5 * np.array(self.cube_size))\
            #         or np.any(np.array(cube_position) > np.array(self.cube_start_position) + 0.5 * np.array(self.cube_size)):
            if cube_position[0] < self.cube_start_position[0] - self.cube_size[0] or \
                    cube_position[0] > self.cube_start_position[0] + self.cube_size[0] or\
                    cube_position[1] < self.cube_start_position[1] - self.cube_size[1] or\
                    cube_position[1] > self.cube_start_position[1] + self.cube_size[1]:
                print 'Wrong cube position: ', cube_position
                return None, 0, True, None

            # cube is not lifted
            return self.rdd.getFingerPosition(RDD.NARROW), 0, False, None


if __name__ == '__main__':
    env = ScoopEnv(port=19997)
    env.reset()
    while True:
        a = input('input action')
        s_, r, done, info = env.step(int(a))
        print s_, r, done
        if done:
            break
