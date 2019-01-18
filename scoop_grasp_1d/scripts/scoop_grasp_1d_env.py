import time

import numpy as np

from vrep_arm_toolkit.simulation import vrep
from vrep_arm_toolkit.robots.ur5 import UR5
from vrep_arm_toolkit.grippers.rdd import RDD
import vrep_arm_toolkit.utils.vrep_utils as utils

import rospy
from std_msgs.msg import Float32MultiArray

VREP_BLOCKING = vrep.simx_opmode_blocking


class ScoopEnv:
    RIGHT = 0
    LEFT = 1
    CLOSE = 2

    def __init__(self, port=19997):
        rospy.init_node('env', anonymous=True)

        self.sim_client = utils.connectToSimulation('127.0.0.1', port)

        # Create UR5 and restart simulator
        self.rdd = RDD(self.sim_client, open_force=10, close_force=20)
        self.ur5 = UR5(self.sim_client, self.rdd)
        self.nA = 3

        self.cube = None
        self.cube_start_position = [-0.2, 0.9, 0.05]
        self.cube_size = [0.1, 0.2, 0.04]

        self.open_position = 0.3

        self.state = [0 for _ in range(4 * 60)]

        self.stateSub = rospy.Subscriber('sim/state', Float32MultiArray, self.stateCallback)

    def stateCallback(self, msg):
        data = list(msg.data)
        self.state = self.state[4:] + data

    def reset(self):
        vrep.simxStopSimulation(self.sim_client, VREP_BLOCKING)
        time.sleep(1)
        vrep.simxStartSimulation(self.sim_client, VREP_BLOCKING)
        time.sleep(1)

        sim_ret, self.cube = utils.getObjectHandle(self.sim_client, 'cube')

        utils.setObjectPosition(self.sim_client, self.ur5.UR5_target, [-0.2, 0.6, 0.07])

        dy = 0.3 * np.random.random()
        # dy = 0
        # dz = 0.1 * np.random.random() - 0.05
        current_pose = self.ur5.getEndEffectorPose()
        target_pose = current_pose.copy()
        target_pose[1, 3] += dy
        # target_pose[2, 3] += dz

        self.ur5.moveTo(target_pose)

        self.rdd.open(self.open_position)

        return self.state

    def step(self, a):
        if a in [self.RIGHT, self.LEFT]:
            current_pose = self.ur5.getEndEffectorPose()
            target_pose = current_pose.copy()
            if a == self.RIGHT:
                target_pose[1, 3] -= 0.01
            elif a == self.LEFT:
                target_pose[1, 3] += 0.01
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
        if target_position[1] < 0.42 or target_position[1] > 0.95 or target_position[2] < 0 or target_position[
            2] > 0.2:
            print 'Wrong arm position: ', target_position
            return None, -1, True, None

        else:
            # cube in wrong position
            sim_ret, cube_position = utils.getObjectPosition(self.sim_client, self.cube)
            while any(np.isnan(cube_position)):
                res, cube_position = utils.getObjectPosition(self.sim_client, self.cube)
            if cube_position[0] < self.cube_start_position[0] - self.cube_size[0] or \
                cube_position[0] > self.cube_start_position[0] + self.cube_size[0] or \
                cube_position[1] < self.cube_start_position[1] - self.cube_size[1] or \
                cube_position[1] > self.cube_start_position[1] + self.cube_size[1]:
                print 'Wrong cube position: ', cube_position
                return None, 0, True, None

            # cube is not lifted
            return self.state, 0, False, None


if __name__ == '__main__':
    env = ScoopEnv(port=19997)
    env.reset()
    while True:
        a = input('input action')
        s_, r, done, info = env.step(int(a))
        sim_ret, tip_position = utils.getObjectPosition(env.sim_client, env.ur5.gripper_tip)
        print s_, r, done
        if done:
            break
