import copy
import cv2
import threading
import sys
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError

import mxnet
from mxnet import gpu
from mxnet import nd

from yolo_modules import yolo_cv
from yolo_modules import licence_plate_render
from yolo_modules import global_variable

from utils import *

if sys.argv[1] == 'v2':
    print(global_variable.cyan)
    print('load car_and_LP v2')
    from car.YOLO import *

else:
    from YOLO import *


def main():
    args = video_Parser()
    video = Video(args)
    rospy.sleep(3)  # camera warm up
    while not rospy.is_shutdown():
        if hasattr(video, 'img') and video.img is not None:

            #nd_img = video.resz(nd.array(video.img))
            nd_img = nd.array(video.img).as_in_context(video.ctx)
            nd_img = nd_img.transpose((2, 0, 1)).expand_dims(axis=0) / 255.
            out = video.yolo.predict(nd_img, LP=True, bind=1)
            video.ros_publish(out)
            video.visualize(out)

        else:
            print('Wait For Image')
            rospy.sleep(0.1)


class Video():
    def __init__(self, args):
        self.yolo = YOLO(args)
        self.project_rect_6d = licence_plate_render.ProjectRectangle6D(380, 160)
        self._init_ros()

        self.dev = args.dev
        self.topic = args.topic
        self.show = args.show
        self.radar = args.radar
        self.LP = args.LP
        self.car = args.car
        self.flip = args.flip
        self.clip = (args.clip_h, args.clip_w)
        self.ctx = [gpu(int(i)) for i in args.gpu][0]


        #self.resz = mxnet.image.ForceResizeAug((self.yolo.size[1], self.yolo.size[0]))
        if self.radar:
            self.radar_prob = yolo_cv.RadarProb(
                self.yolo.num_class, self.yolo.classes)

        print(global_variable.cyan)
        if self.dev == 'ros':
            rospy.Subscriber(self.topic, Image, self._image_callback)
            print('Image Topic: %s' % self.topic)

        else:
            threading.Thread(target=self._get_frame).start()

        save_frame = False
        if save_frame:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # (*'MJPG')#(*'MPEG')
            self.out = cv2.VideoWriter(
                './video/car_rotate.mp4', fourcc, 30, (640, 360))

    def _init_ros(self):
        rospy.init_node("YOLO_ros_node", anonymous=True)
        self.bridge = CvBridge()

        self.img_pub = rospy.Publisher(
            self.yolo.pub_img,
            Image,
            queue_size=1)

        self.clipped_LP_pub = rospy.Publisher(
            self.yolo.pub_clipped_LP,
            Image,
            queue_size=1)

        self.car_pub = rospy.Publisher(
            self.yolo.pub_box,
            Float32MultiArray,
            queue_size=1)

        self.LP_pub = rospy.Publisher(
            self.yolo.pub_LP,
            Float32MultiArray,
            queue_size=1)

        self.topk = 1

        self.mat1 = Float32MultiArray()
        dim = self.mat1.layout.dim
        dim.append(MultiArrayDimension())
        dim.append(MultiArrayDimension())
        dim[0].label = "box"
        dim[0].size = self.topk
        dim[0].stride = self.topk * 7

        dim[1].label = "predict"
        dim[1].size = 7
        dim[1].stride = 7

        self.mat2 = Float32MultiArray()
        dim = self.mat2.layout.dim
        dim.append(MultiArrayDimension())
        dim.append(MultiArrayDimension())
        dim[0].label = "LP"
        dim[0].size = self.topk
        dim[0].stride = self.topk * 8

        dim[1].label = "predict"
        dim[1].size = 8
        dim[1].stride = 8

    def cv2_flip_and_clip_frame(self, img):
        h, w = self.yolo.size

        clip = self.clip
        assert type(clip) == tuple and len(clip) == 2, (
            global_variable.red +
            'clip should be a tuple, (height_ratio, width_ratio')
        if clip[0] < 1:
            top = int((1-clip[0]) * img.shape[0] / 2.)
            bot = img.shape[0] - top
            img = img[top:bot]

        if clip[1] < 1:
            left = int((1-clip[1]) * img.shape[1] / 2.)
            right = img.shape[1] - left
            img = img[:, left:right]

        flip = self.flip
        if flip == 1 or flip == 0 or flip == -1:
            img = cv2.flip(img, flip)
            # flip = 1: left-right
            # flip = 0: top-down
            # flip = -1: 1 && 0

        img = cv2.resize(img, (w, h))
        return img

    def _get_frame(self):
        dev = self.dev

        if dev == 'tx2':
            cap = open_cam_onboard(640, 360)
        elif '.' in dev:
            cap = cv2.VideoCapture(dev)
            pause = 0.1
        elif type(dev) == str:
            print('Image Source: /dev/video' + dev)
            cap = cv2.VideoCapture(int(dev))
        else:
            print(global_variable.red)
            print(('dev should be '
                   'tx2(str) or '
                   'video_path(str) or '
                   'device_index(int)'))

        while not rospy.is_shutdown():
            ret, img = cap.read()
            self.img = self.cv2_flip_and_clip_frame(img)

        cap.release()

    def _image_callback(self, img):
        img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        self.img = self.cv2_flip_and_clip_frame(img)

    def visualize(self, out):
        Cout = out[0][0]
        LP_out = out[1][0]
        img = copy.deepcopy(self.img)
        #self.out.write(img)
        if self.radar:
            self.radar_prob.plot3d(Cout[0], Cout[-self.num_class:])

        # -------------------- Licence Plate -------------------- #
        if LP_out[0] > self.yolo.LP_threshold and self.LP:
            img, clipped_LP = self.project_rect_6d.add_edges(img, LP_out[1:])
            self.clipped_LP_pub.publish(
                self.bridge.cv2_to_imgmsg(clipped_LP, 'bgr8'))

            if self.show:
                cv2.imshow('Licence Plate', clipped_LP)

        # -------------------- vehicle -------------------- #
        if Cout[0] > self.yolo.car_threshold and self.car:
            yolo_cv.cv2_add_bbox(img, Cout, 4, use_r=False)

        if self.show:
            cv2.imshow('img', img)
            cv2.waitKey(1)

        self.img_pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))

    def ros_publish(self, out):
        Cout = out[0][0]
        LP_out = out[1][0]
        self.mat1.data = [-1] * 7
        self.mat2.data = [-1] * 8

        if Cout[0] > self.yolo.car_threshold:
            for i in range(6):
                self.mat1.data[i] = Cout[i]

        if LP_out[0] > self.yolo.LP_threshold:
            for i in range(7):
                self.mat2.data[i] = LP_out[i]
        self.car_pub.publish(self.mat1)
        self.LP_pub.publish(self.mat2)


if __name__ == '__main__':
    main()
