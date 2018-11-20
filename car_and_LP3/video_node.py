import copy
import cv2
import numpy as np
import sys
import threading
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError

import mxnet
from mxnet import gpu
from mxnet import nd

from gluoncv.model_zoo.yolo.darknet import DarknetBasicBlockV3
from gluoncv.model_zoo.yolo.yolo3 import YOLODetectionBlockV3
from gluoncv.model_zoo.yolo.yolo3 import _upsample

from yolo_modules import basic_yolo
from yolo_modules import yolo_cv
from yolo_modules import licence_plate_render
from yolo_modules import global_variable


class Video():
    def __init__(self, save_frame=False, topic=False, show=True, radar=False, dev=0., pause=0.):
        self.yolo = YOLO()
        self.project_rect_6d = licence_plate_render.ProjectRectangle6D(380, 160)

        self.save_frame = save_frame
        self.topic = topic
        self.show = show
        self.radar = radar
        self.dev = dev
        self.pause = pause

        rospy.init_node("YOLO_ros_node", anonymous=True)
        self.YOLO_img_pub = rospy.Publisher(self.pub_img, Image, queue_size=1)
        self.YOLO_box_pub = rospy.Publisher(self.pub_box, Float32MultiArray, queue_size=1)

        self.bridge = CvBridge()
        self.mat = Float32MultiArray()
        dim = self.mat.layout.dim
        dim.append(MultiArrayDimension())
        dim.append(MultiArrayDimension())
        dim[0].label = "box"
        dim[0].size = self.topk
        dim[0].stride = self.topk * 7

        dim[1].label = "predict"
        dim[1].size = 7
        dim[1].stride = 7

        if save_frame:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # (*'MJPG')#(*'MPEG')
            self.out = cv2.VideoWriter(
                './video/car_rotate.mp4', fourcc, 30, (640, 360))

    def _get_frame(self, dev=0, pause=0., flip=False, clip=None):
        if dev == 'tx2':
            cap = open_cam_onboard(self.cam_w, self.cam_w)
        elif dev == str:
            cap = cv2.VideoCapture('video/GOPR0730.MP4')
            pause = 0.1
        elif dev == int:
            cap = cv2.VideoCapture(dev)
        else:
            print(global_variable.red)
            print('dev should be \
                   tx2(str) or \
                   video_path(str) or \
                   device_index(int)')

        while not rospy.is_shutdown():
            ret, img = cap.read()
            self.img = yolo_cv.cv2_flip_and_clip_frame(
                img, flip=self.flip, clip=self.clip)

        cap.release()

    def _image_callback(self, img):
        img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        self.img = yolo_cv.cv2_flip_and_clip_frame(
            img, flip=self.flip_mode, clip=self.clip)

    def visualize(self, out):
        Cout = out[0][0]
        LP_out = out[1][0]
        img = copy.deepcopy(self.img)
        #self.out.write(img)
        # -------------------- Add Box -------------------- #
        #vec_ang, vec_rad, prob = self.radar_prob.cla2ang(Cout[0], Cout[-self.num_class:])
        if Cout[0] > 0.5:
            yolo_cv.cv2_add_bbox(img, Cout, 4, use_r=False)
            for i in range(6):
                self.mat.data[i] = Cout[i]
        else:
            self.mat.data = [-1]*7

        if LP_out[0] > 0.9:
            print(LP_out)
            img, clipped_LP = self.project_rect_6d.add_edges(img, LP_out[1:])
            if self.show:
                cv2.imshow('Licence Plate', clipped_LP)

        if self.radar:
            self.radar_prob.plot3d(Cout[0], Cout[-self.num_class:])

        self.YOLO_box_pub.publish(self.mat)

        # -------------------- Show Image -------------------- #
        if self.show:
            cv2.imshow('img', img)
            cv2.waitKey(1)
        # -------------------- Publish Image and Box -------------------- #
        self.YOLO_img_pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))

    def video(self, topic=False, show=True, radar=False, ctx=gpu(0)):
        self.radar = radar
        self.show = show

        self.topk = 1
        self._init_ros()
        self.resz = mxnet.image.ForceResizeAug((self.size[1], self.size[0]))

        if radar:
            self.radar_prob = utils_cv.RadarProb(self.num_class, self.classes)

        if not topic:
            threading.Thread(target=self._get_frame).start()
            print(global_variable.blue)
            print('Use USB Camera')

        else:
            rospy.Subscriber(topic, Image, self._image_callback)
            print(global_variable.blue)
            print('Image Topic: %s' % topic)

        #rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if hasattr(self, 'img') and self.img is not None:

                nd_img = self.resz(nd.array(self.img))
                nd_img = nd_img.as_in_context(ctx)
                nd_img = nd_img.transpose((2, 0, 1)).expand_dims(axis=0)/255.
                out = self.predict(nd_img, LP=True)
                self.visualize(out)
                if not topic:
                    rate.sleep()
            # else: print('Wait For Image')