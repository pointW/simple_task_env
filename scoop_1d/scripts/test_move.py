import sys
import time
import thread

import numpy as np

from vrep_arm_toolkit.simulation import vrep
# from vrep_arm_toolkit.robots.ur5_joint_control import UR5
from vrep_arm_toolkit.robots.ur5 import UR5
from vrep_arm_toolkit.grippers.rdd import RDD
import vrep_arm_toolkit.utils.vrep_utils as utils
from vrep_arm_toolkit.utils import transformations
from vrep_arm_toolkit.motion_planner.motion_planner import *

import rospy

VREP_BLOCKING = vrep.simx_opmode_blocking


def main():
    rospy.init_node('simple_test')

    # Attempt to connect to simulator
    sim_client = utils.connectToSimulation('127.0.0.1', 19997)

    vrep.simxStopSimulation(sim_client, VREP_BLOCKING)
    time.sleep(1)
    vrep.simxStartSimulation(sim_client, VREP_BLOCKING)
    time.sleep(1)

    rdd = RDD(sim_client)
    ur5 = UR5(sim_client, rdd)

    # Generate a cube
    position = [-0.2, 0.9, 0.05]
    orientation = [0, 0, 0]
    # orientation = [0, 0, 0]
    size = [0.1, 0.2, 0.05]
    mass = 1
    color = [255, 0, 0]
    cube = utils.generateShape(sim_client, 'cube', 0, size, position, orientation, mass, color)
    time.sleep(1)

    pose = ur5.getEndEffectorPose()

    target_pose = pose.copy()

    # rdd.openFinger(RDD.NARROW)

    narrow_finger_joint = rdd.finger_joint_narrow.joint
    def print_joint_pos(joint):
        while True:
            # print utils.getJointPosition(sim_client, joint)

            res, cube_pose = utils.getObjectPosition(sim_client, cube)
            while any(np.isnan(cube_pose)):
                res, cube_pose = utils.getObjectPosition(sim_client, cube)
            print cube_pose

    thread.start_new_thread(print_joint_pos, (narrow_finger_joint,))

    target_pose[1, 3] += 0.3
    rdd.openFinger(RDD.NARROW)

    ur5.moveTo(target_pose)


    ur5.moveTo(pose)

    # dy = 0.01
    # for i in range(30):
    #     target_pose[1, 3] += dy
    #     path = findIkPath(sim_client, target_pose)
    #     ur5.followPath(path)


    # vrep.simxStopSimulation(sim_client, VREP_BLOCKING)


if __name__ == '__main__':
    main()