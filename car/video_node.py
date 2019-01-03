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


def main():
    args = utils.video_Parser()
    video = Video(args)
    video()


class Video(object):
    def __init__(self, args, save_video_size=None):
        self.yolo = YOLO(args)
        self.car_threshold = 0.9
        self._init(args, save_video_size)

    def _init(self, args, save_video_size):
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
            depth_topic = '/zed/depth/depth_registered'
            rospy.Subscriber(depth_topic, Image, self._depth_callback)
            rospy.Subscriber(self.topic, Image, self._image_callback)
            print('Image Topic: %s' % self.topic)

        else:
            threading.Thread(target=self._get_frame).start()

        threading.Thread(target=self._net_thread).start()

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
        #shape = (1, 3, self.yolo.size[0], self.yolo.size[1])
        #yolo_gluon.test_inference_rate(self.yolo.net, shape)

        while not hasattr(self, 'net_out') or not hasattr(self, 'net_img'):
            time.sleep(0.1)

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            net_dep = copy.copy(self.net_dep)
            net_out = copy.copy(self.net_out)  # not sure type(net_out)
            img = copy.copy(self.net_img)

            pred_car = self.yolo.predict(net_out[:3])
            if hasattr(self, 'net_dep'):
                x = int(net_dep.shape[1] * pred_car[0, 2])
                y = int(net_dep.shape[0] * pred_car[0, 1])
                pred_car[0, 5] = net_dep[y, x]
            else:
                pred_car[0, 5] = -1

            self.ros_publish_array(self.car_pub, self.mat_car, pred_car[0])

            now = rospy.get_rostime()
            print(('zed to pub: ', (now - self.net_img_time).to_sec()))

            self.visualize(pred_car, img)
            rate.sleep()

    def _get_frame(self):
        dev = self.dev
        pause = 0

        if dev == 'jetson':
            print('Image Source: Jetson OnBoard Camera')
            cap = jetson_onboard_camera(640, 360, dev)

        elif dev.split('.')[-1] in ['mp4', 'avi']:
            print('Image Source: ' + dev)
            cap = cv2.VideoCapture(dev)
            pause = 0.03

        elif dev.isdigit() and os.path.exists('/dev/video' + dev):
            print('Image Source: /dev/video' + dev)
            cap = cv2.VideoCapture(int(dev))

        else:
            print(global_variable.red)
            print(('dev should be (jetson) or '
                   '(video_path) or (device_index)'))
            sys.exit(0)

        while not rospy.is_shutdown():
            self.ret, img = cap.read()
            self.img = yolo_cv.cv2_flip_and_clip_frame(img, self.clip, self.flip)

            #if bool(pause):
                #time.sleep(pause)

        cap.release()

    def _net_thread(self):
        size = tuple(self.yolo.size[::-1])
        mx_resize = mxnet.image.ForceResizeAug(size)

        while not rospy.is_shutdown():
            if not hasattr(self, 'img') or self.img is None:
                print('Wait For Image')
                time.sleep(1.0)
                continue
            if hasattr(self, 'depth_image'):
                net_dep = self.depth_image.copy()

            net_img_time = self.img_time
            net_img = self.img.copy()

            nd_img = yolo_gluon.cv_img_2_ndarray(net_img, self.ctx[0], mxnet_resize=mx_resize)
            # nd_img = yolo_gluon.nd_white_balance(nd_img, bgr=[1.0, 1.0, 1.0])

            net_out = self.yolo.net.forward(is_train=False, data=nd_img)
            # [x1, x2, x3]
            net_out[0].wait_to_read()
            '''
            # ----------------------
            now = rospy.get_rostime()
            print(('net time', (now - net_img_time).to_sec()))
            # ----------------------
            '''
            if 'net_dep' in locals():
                self.net_dep = net_dep

            self.net_img_time = net_img_time
            self.net_img = net_img
            self.net_out = net_out

    def _image_callback(self, img):
        self.img_time = img.header.stamp  # .secs + img.header.stamp.nsecs * 10**-6
        img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        self.img = yolo_cv.cv2_flip_and_clip_frame(img, self.clip, self.flip)

    def _depth_callback(self, depth_msgs):
        depth_image = self.bridge.imgmsg_to_cv2(depth_msgs, "32FC1")
        depth_image = np.array(depth_image, dtype=np.float32)
        self.depth_image = yolo_cv.cv2_flip_and_clip_frame(
            depth_image, self.clip, self.flip)

    def visualize(self, pred, img):
        Cout = pred[0]
        #self.out.write(img)

        if self.radar:
            self.radar_prob.plot3d(
                Cout[0], Cout[-self.yolo.num_class:])


        # -------------------- vehicle -------------------- #
        if Cout[0] > self.car_threshold:
            yolo_cv.cv2_add_bbox(img, Cout, 4, use_r=False)

        if self.show:
            cv2.imshow('img', img)
            cv2.waitKey(1)

        self.img_pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))

    def ros_publish_array(self, ros_publisher, mat, data):
        mat.data = data
        ros_publisher.publish(mat)


if __name__ == '__main__':
    main()
