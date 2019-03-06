#!/usr/bin/env python
import cv2
import math
import threading
import time
import yaml

import rospy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped

import airsim


class IBVS_To_AirSim():
    def __init__(self):
        with open('ibvs_parameter.yaml') as f:
            para = yaml.load(f)

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        rospy.init_node("IBVS_To_AirSim_node", anonymous=True)

        rospy.Subscriber(para['LAND_TOPIC'], Bool, self._land)
        rospy.Subscriber(para['FIX_POSE_TOPIC'], Bool, self._fix_pose)
        rospy.Subscriber(para['CMD_VEL_TOPIC'], TwistStamped, self._set_vel)

    def __call__(self):
        vx, vy, vz, vw = 0, 0, 0, 0

        self.land = False
        self.fix_pose = True

        self.land_done = False
        self.fix_pose_done = False
        self.take_off_done = False

        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('/airsim/image_raw', Image, queue_size=1)

        rate = 5
        r = rospy.Rate(rate)
        #threading.Thread(target=self._camera_thread).start()
        while not rospy.is_shutdown():
            #print(self.land, self.land_done, self.fix_pose, self.fix_pose_done, self.take_off_done)
            self.get_image_and_publish()
            if self.land:
                if not self.land_done:
                    self.client.reset()
                    self.land_done = True
                    print('Landing Done')
                else:
                    pass
                    #print('Landing Has Done')

            else:
                if self.fix_pose:
                    if self.take_off_done:
                        if self.fix_pose_done:
                            pass
                            #print('Fix Pose Has Done')
                        else:
                            self.client.moveToPositionAsync(0, 0, -8, 5).join()
                            self.fix_pose_done = True
                            print('Fix Pose Done')

                    else:
                        self.client.enableApiControl(True)
                        self.client.armDisarm(True)
                        print('Wait For take off')
                        self.client.takeoffAsync().join()
                        self.take_off_done = True
                        print('Take off Done')
                else:
                    self.fix_pose_done = False
                    vx, vy, vz, vw = self.desire_vel
                    #vx, vy, vz, vw = [0, 0, 0, 20]
                    print(vx, vy, vz, vw)
                    self.client.rotateByYawRateAsync(vw, 0.1)
                    self.client.moveByVelocityAsync(vx, vy, vz, 0.1)
            r.sleep()

    def get_image_and_publish(self):
        image = self.client.simGetImage("0", airsim.ImageType.Scene)
        image = cv2.imdecode(
            airsim.string_to_uint8_array(image),
            cv2.IMREAD_UNCHANGED)[:, :, :3]

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))

    def _camera_thread(self):
        rate = 10.
        r = rospy.Rate(rate)
        image_pub = rospy.Publisher('/airsim/image_raw', Image, queue_size=1)
        while not rospy.is_shutdown():
            image = self.client.simGetImage("0", airsim.ImageType.Scene).join()
            image = cv2.imdecode(
                airsim.string_to_uint8_array(image),
                cv2.IMREAD_UNCHANGED)[:, :, :3]

            image_pub.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
            r.sleep()

    def _land(self, land_topic):
        self.land = land_topic.data
        self.land_done = False

        if not land_topic.data:
            self.take_off_done = False

    def _fix_pose(self, fix_pose_topic):
        self.fix_pose = fix_pose_topic.data

    def _set_vel(self, twist_topic):
        self.desire_vel = [
            twist_topic.twist.linear.x,
            twist_topic.twist.linear.y,
            twist_topic.twist.linear.z,
            twist_topic.twist.angular.z * 180 / math.pi]


ibvs2airsim = IBVS_To_AirSim()
ibvs2airsim()
