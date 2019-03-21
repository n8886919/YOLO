import copy
import cv2
import threading
import os
import sys

import rospy
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import mxnet
from mxnet import gpu
from mxnet import nd

from yolo_modules import yolo_cv
from yolo_modules import yolo_gluon
from yolo_modules import licence_plate_render
from yolo_modules import global_variable

import utils
from YOLO import *
import numpy as np
from os.path import split

DEPTH_TOPIC = '/zed/depth/depth_registered'
# DEPTH_TOPIC = '/drone/camera/depth/image_raw'
verbose = 0
save_video_size = None

_step = 360 / 24
cos_offset = np.array([math.cos(x*math.pi/180) for x in range(0, 360, _step)])
sin_offset = np.array([math.sin(x*math.pi/180) for x in range(0, 360, _step)])


def main():
    args = utils.video_Parser()
    video = Video(args)
    #video()  # Two thread
    video.run()  # One thread


class Video(object):
    def __init__(self, args=None):
        self.yolo = YOLO(args)
        self.car_threshold = 0.5
        self._init(args)

    def _init(self, args):
        # -------------------- init_args -------------------- #
        self.dev = args.dev
        self.topic = args.topic
        self.show = args.show
        self.radar = args.radar
        self.flip = args.flip
        self.clip = (args.clip_h, args.clip_w)
        self.ctx = yolo_gluon.get_ctx(args.gpu)

        # -------------------- init_ros -------------------- #
        rospy.init_node("YOLO_ros_node", anonymous=True)
        self.bridge = CvBridge()
        self.img_pub = rospy.Publisher(self.yolo.pub_img, Image, queue_size=1)
        self.car_pub = rospy.Publisher(self.yolo.pub_box, Float32MultiArray, queue_size=1)

        self.mat_car = Float32MultiArray()

        # -------------------- init_dev -------------------- #
        if self.dev == 'ros':
            rospy.Subscriber(DEPTH_TOPIC, Image, self._depth_callback)
            rospy.Subscriber(self.topic, Image, self._image_callback)
            print(global_variable.green)
            print('Image Topic: %s' % self.topic)
            print('Depth Topic: %s' % DEPTH_TOPIC)
            print(global_variable.reset_color)

        else:
            #pass
            threading.Thread(target=self._get_frame).start()

        # -------------------- init_radar -------------------- #
        if self.radar:
            self.radar_prob = yolo_cv.RadarProb(
                self.yolo.num_class, self.yolo.classes)

        # -------------------- init_video_saver -------------------- #
        if save_video_size is not None and len(save_video_size) == 2:
            self.save_video = True
            start_time = datetime.datetime.now().strftime("%m-%dx%H-%M")
            out_file = os.path.join('video', 'car_%s.avi' % start_time)

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # (*'MJPG')#(*'MPEG')
            self.out = cv2.VideoWriter(
                out_file, fourcc, 30, (save_video_size))
        else:
            self.save_video = False

    def __call__(self):
        print(global_variable.green)
        print('Start Double Thread Video Node')

        threading.Thread(target=self._net_thread).start()

        shape = (1, 3, self.yolo.size[0], self.yolo.size[1])
        yolo_gluon.test_inference_rate(self.yolo.net, shape, cycles=200)

        while not hasattr(self, 'net_out') or not hasattr(self, 'net_img'):
            time.sleep(0.1)

        #rate = rospy.Rate(30)
        print(global_variable.reset_color)

        while not rospy.is_shutdown():
            if hasattr(self, 'net_dep'):
                net_dep = copy.copy(self.net_dep)
            else:
                net_dep = None

            net_out = copy.copy(self.net_out)  # not sure type(net_out)
            net_img = copy.copy(self.net_img)

            self.net_thread_start = True
            self.process(net_img, net_out, net_dep)
            #rate.sleep()

    def run(self):
        print(global_variable.green)
        print('Start Single Thread Video Node')
        print(global_variable.reset_color)

        shape = (1, 3, self.yolo.size[0], self.yolo.size[1])
        yolo_gluon.test_inference_rate(self.yolo.net, shape, cycles=200)

        mx_resize = mxnet.image.ForceResizeAug(tuple(self.yolo.size[::-1]))
        while not rospy.is_shutdown():
            if not hasattr(self, 'img') or self.img is None:
                print('Wait For Image')
                time.sleep(1.0)
                continue
            self.net_img_time = self.img_cb_time

            # -------------------- image -------------------- #
            if hasattr(self, 'net_dep'):
                net_dep = copy.copy(self.net_dep)
            else:
                net_dep = None

            net_img = copy.copy(self.img)
            nd_img = yolo_gluon.cv_img_2_ndarray(net_img, self.ctx[0], mxnet_resize=mx_resize)
            # nd_img = yolo_gluon.nd_white_balance(nd_img, bgr=[1.0, 1.0, 1.0])
            # if self.yolo.use_fp16:
            #     nd_img = nd_img.astype('float16')
            net_out = self.yolo.net.forward(is_train=False, data=nd_img)
            self.process(net_img, net_out, net_dep)

    def process(self, net_img, net_out, net_dep):
            pred_car = self.yolo.predict(net_out[:3])
            # --------------- data[5] is depth --------------- #
            '''
            if net_dep:  # net_dep != None
                x = int(net_dep.shape[1] * pred_car[0, 2])
                y = int(net_dep.shape[0] * pred_car[0, 1])
                pred_car[0, 5] = net_dep[y, x]
            else:
                pred_car[0, 5] = -1
            '''
            # ---------------- data[5] is azi ---------------- #
            x = pred_car[0, -24:]
            prob = np.exp(x) / np.sum(np.exp(x), axis=0)
            c = sum(cos_offset * prob)
            s = sum(sin_offset * prob)
            vec_ang = math.atan2(s, c)
            pred_car[0, 5] = vec_ang
            # ------------------------------------------------- #

            if verbose:
                print('cam to pub delay: %f' % (
                    rospy.get_rostime() - self.net_img_time).to_sec())

            self.ros_publish_array(self.car_pub, self.mat_car, pred_car[0])
            self.visualize(pred_car, net_img)

    def _get_frame(self):
        print(global_variable.green)
        print('Start OPENCV Video Capture Thread')
        dev = self.dev

        if dev == 'jetson':
            print('Image Source: Jetson OnBoard Camera')
            cap = jetson_onboard_camera(640, 360, dev)

        elif dev.split('.')[-1] in ['mp4', 'avi', 'm2ts']:
            print('Image Source: ' + dev)
            cap = cv2.VideoCapture(dev)
            rate = rospy.Rate(30)

        elif dev.isdigit() and os.path.exists('/dev/video' + dev):
            print('Image Source: /dev/video' + dev)
            cap = cv2.VideoCapture(int(dev))

        else:
            print(global_variable.red)
            print('dev should be jetson / video_path(mp4, avi, m2ts) / device_index')
            sys.exit(0)

        print(global_variable.reset_color)
        seq = 0
        while not rospy.is_shutdown():
            self.img_cb_time = rospy.get_rostime()
            self.img_cb_seq = seq
            self.ret, img = cap.read()
            self.img = yolo_cv.cv2_flip_and_clip_frame(img, self.clip, self.flip)
            seq += 1
            if 'rate' in locals():
                rate.sleep()

        cap.release()

    def _net_thread(self):
        print('Start Net Thread')

        size = tuple(self.yolo.size[::-1])
        ctx = self.ctx[0]
        mx_resize = mxnet.image.ForceResizeAug(size)
        self.net_thread_start = True
        while not rospy.is_shutdown():
            if not hasattr(self, 'img') or self.img is None:
                print('Net thread Wait For Image')
                time.sleep(1.0)
                continue

            if not self.net_thread_start:
                time.sleep(0.01)
                continue
            # -------------------- additional image info-------------------- #
            net_img_time = self.img_cb_time
            net_img_seq = self.img_cb_seq
            now = rospy.get_rostime()
            if verbose:
                print('cam to net: %f' % (now - net_img_time).to_sec())

            if hasattr(self, 'depth_image'):
                net_dep = self.depth_image.copy()
            # -------------------- image -------------------- #
            net_img = self.img.copy()
            nd_img = yolo_gluon.cv_img_2_ndarray(net_img, ctx, mxnet_resize=mx_resize)

            # if self.yolo.use_fp16:
            #     nd_img = nd_img.astype('float16')

            # nd_img = yolo_gluon.nd_white_balance(nd_img, bgr=[1.0, 1.0, 1.0])
            net_out = self.yolo.net.forward(is_train=False, data=nd_img)
            net_out[0].wait_to_read()

            # -------------------- additional image info-------------------- #
            if 'net_dep' in locals():
                self.net_dep = net_dep

            self.net_img_time = net_img_time
            self.net_img_seq = net_img_seq
            now = rospy.get_rostime()
            print('net done time: %f' % (now - net_img_time).to_sec())

            # -------------------- image -------------------- #
            self.net_img = net_img
            self.net_out = net_out
            self.net_thread_start = False

    def _image_callback(self, img):
        self.img_cb_time = img.header.stamp  # .secs + img.header.stamp.nsecs * 10**-6
        self.img_cb_seq = img.header.seq
        img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        self.img = yolo_cv.cv2_flip_and_clip_frame(img, self.clip, self.flip)

    def _depth_callback(self, depth_msgs):
        depth_image = self.bridge.imgmsg_to_cv2(depth_msgs, "32FC1")
        depth_image = np.array(depth_image, dtype=np.float32)
        self.depth_image = yolo_cv.cv2_flip_and_clip_frame(
            depth_image, self.clip, self.flip)

    def visualize(self, pred, img):
        Cout = pred[0]

        if self.radar:
            self.radar_prob.plot3d(
                Cout[0], Cout[-self.yolo.num_class:])

        if Cout[0] > self.car_threshold:
            yolo_cv.cv2_add_bbox(img, Cout, 4, use_r=False)

        if self.save_video:
            self.out.write(img)

        if self.show:
            cv2.imshow('img', img)
            cv2.waitKey(1)

        self.img_pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))

    def ros_publish_array(self, ros_publisher, mat, data):
        mat.data = data
        ros_publisher.publish(mat)


if __name__ == '__main__':
    main()
