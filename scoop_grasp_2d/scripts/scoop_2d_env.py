import time

import numpy as np

from vrep_arm_toolkit.simulation import vrep
from vrep_arm_toolkit.robots.ur5 import UR5
from vrep_arm_toolkit.grippers.rdd import RDD
import vrep_arm_toolkit.utils.vrep_utils as utils

VREP_BLOCKING = vrep.simx_opmode_blocking

CUBE_MESH = '/home/ur5/thesis/simple_task/mesh/block.obj'


class ScoopEnv:
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3
    CLOSE = 4

    def __init__(self, port=19997):
        self.sim_client = utils.connectToSimulation('127.0.0.1', port)

        # Create UR5 and restart simulator
        self.rdd = RDD(self.sim_client, open_force=20, close_force=20)
        self.ur5 = UR5(self.sim_client, self.rdd)
        self.nA = 5

        self.cube = None
        self.cube_start_position = [-0.2, 0.9, 0.05]
        self.cube_size = [0.1, 0.2, 0.04]

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

        # dy = 0.3 * np.random.random()
        dy = 0
        dz = 0.1 * np.random.random() - 0.05
        current_pose = self.ur5.getEndEffectorPose()
        target_pose = current_pose.copy()
        target_pose[1, 3] += dy
        target_pose[2, 3] += dz

        self.ur5.moveTo(target_pose)

        self.rdd.open(self.open_position)

        return self.getState()

    def step(self, a):
        if a in [self.RIGHT, self.LEFT, self.UP, self.DOWN]:
            current_pose = self.ur5.getEndEffectorPose()
            target_pose = current_pose.copy()
            if a == self.RIGHT:
                target_pose[1, 3] -= 0.01
            elif a == self.LEFT:
                target_pose[1, 3] += 0.01
            elif a == self.UP:
                target_pose[2, 3] += 0.01
            elif a == self.DOWN:
                target_pose[2, 3] -= 0.01
            utils.setObjectPosition(self.sim_client, self.ur5.UR5_target, target_pose[:3, 3])

        elif a == self.CLOSE:
            self.rdd.close()
            sim_ret, cube_orientation = utils.getObjectOrientation(self.sim_client, self.cube)

            sim_ret, cube_position = utils.getObjectPosition(self.sim_client, self.cube)
            sim_ret, tip_position = utils.getObjectPosition(self.sim_client, self.ur5.gripper_tip)

            if np.all(tip_position > (np.array(cube_position) - np.array(self.cube_size))) and \
                    np.all(tip_position < (np.array(cube_position) + np.array(self.cube_size))) and \
                    (cube_orientation[0] < -0.02 or cube_position[2] > self.cube_start_position[2] + 0.005):
                return None, 1, True, None
            self.rdd.open(self.open_position)

        # arm is in wrong pose
        sim_ret, target_position = utils.getObjectPosition(self.sim_client, self.ur5.UR5_target)
        if target_position[1] < 0.42 or target_position[1] > 0.95 or target_position[2] < 0 or target_position[2] > 0.2:
            print 'Wrong arm position: ', target_position
            return None, -1, True, None

        else:
            # cube in wrong position
            sim_ret, cube_position = utils.getObjectPosition(self.sim_client, self.cube)
            while any(np.isnan(cube_position)):
                res, cube_position = utils.getObjectPosition(self.sim_client, self.cube)
            if cube_position[0] < self.cube_start_position[0] - self.cube_size[0] or \
                    cube_position[0] > self.cube_start_position[0] + self.cube_size[0] or\
                    cube_position[1] < self.cube_start_position[1] - self.cube_size[1] or\
                    cube_position[1] > self.cube_start_position[1] + self.cube_size[1]:
                print 'Wrong cube position: ', cube_position
                return None, 0, True, None

            # cube is not lifted
            return self.getState(), 0, False, None

    def planAction(self):
        sim_ret, target_position = utils.getObjectPosition(self.sim_client, self.ur5.UR5_target)
        # step 1: move to bottom
        if target_position[2] > 0.06:
            return self.DOWN

        elif target_position[2] < 0.05:
            return self.UP

        # step 2: move to left
        elif target_position[1] < 0.79:
            return self.LEFT

        elif target_position[1] > 0.8:
            return self.RIGHT

        # step 3: close
        else:
            return self.CLOSE


if __name__ == '__main__':
    env = ScoopEnv(port=20020)
    env.reset()
    while True:
        a = input('input action')
        s_, r, done, info = env.step(int(a))
        sim_ret, tip_position = utils.getObjectPosition(env.sim_client, env.ur5.gripper_tip)
        print s_, r, done
        print tip_position
        if done:
            break

    # while True:
    #     a = env.planAction()
    #     s_, r, done, info = env.step(a)
    #     sim_ret, tip_position = utils.getObjectPosition(env.sim_client, env.ur5.gripper_tip)
    #     print s_, r, done
    #     print tip_position
    #     if done:
    #         break

