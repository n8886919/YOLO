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

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
DEPTH_TOPIC = '/zed/depth/depth_registered'
# DEPTH_TOPIC = '/drone/camera/depth/image_raw'
verbose = 0
save_video_size = None

# -------------------- radar plot -------------------- #
_step = 360 / 24
_cos_offset = np.array([math.cos(x*math.pi/180) for x in range(0, 360, _step)])
_sin_offset = np.array([math.sin(x*math.pi/180) for x in range(0, 360, _step)])


def main(Video, args):
    if args.trt and args.radar:
        args.trt = False
        print(global_variable.magenta+'--trt, --radar cant use at the same time')
        print('Use Mxnet For Inference')
        print(global_variable.reset_color)

    video = Video(args)
    if video.radar:
        video.run()  # One thread, matplotlib is not thread safty
    else:
        video()  # Two thread


class Video(object):
    def __init__(self, args=None):
        if args.version in['v1', 'v2', 'v3', 'v4']:
            self.yolo = YOLO(args)
        elif args.version in['v11']:
            self.yolo = YOLO_dense(args)
        else:
            print(global_variable.red+'Version Error')
            print(global_variable.reset_color)
            sys.exit(0)

        self.car_threshold = 0.5
        self._init(args)

        if args.trt:
            global do_inference_wrapper
            from yolo_modules.tensorrt_module import do_inference_wrapper
        else:
            shape = (1, 3, self.yolo.size[0], self.yolo.size[1])
            yolo_gluon.test_inference_rate(self.yolo.net, shape, cycles=200)

    def _init(self, args):
        # -------------------- init_args -------------------- #
        self.trt = args.trt
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
        self.depth_image = None
        self.image = None
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

        h, w = self.yolo.size
        threading.Thread(target=self._video_thread).start()
        self.net_thread_start = True

        while not rospy.is_shutdown():
            if self.image is None:
                print('Net thread Wait For Image')
                time.sleep(1.0)
                continue

            if not self.net_thread_start:
                time.sleep(0.01)
                continue
            try:
                # ------------------ additional image info In ------------------ #
                now = rospy.get_rostime()
                net_img_time, net_img_seq = self.img_cb_time, self.img_cb_seq
                net_img, net_dep = copy.copy(self.image), copy.copy(self.depth_image)
                yolo_gluon.switch_print('cam to net: %f' % (now - net_img_time).to_sec(), verbose)

                net_out = self.inference(cv2.resize(net_img, (w, h)))

                # -------------------- additional image info-------------------- #
                now = rospy.get_rostime()
                yolo_gluon.switch_print('net done time: %f' % (now - net_img_time).to_sec(), verbose)

                # ------------------ additional image info Out ------------------ #
                self.net_img_time, self.net_img_seq = net_img_time, net_img_seq
                self.net_dep, self.net_img, self.net_out = net_dep, net_img, net_out
                self.net_thread_start = False

            except Exception as e:
                rospy.signal_shutdown('main_thread Error')
                print(global_variable.red + e + global_variable.reset_color)

        sys.exit(0)

    def _video_thread(self):
        while not hasattr(self, 'net_out') or not hasattr(self, 'net_img'):
            time.sleep(0.1)

        print(global_variable.reset_color)
        while not rospy.is_shutdown():
            net_dep = copy.copy(self.net_dep)
            net_out = copy.copy(self.net_out)
            net_img = copy.copy(self.net_img)

            self.net_thread_start = True
            try:
                self.process(net_img, net_out, net_dep)

            except Exception as e:
                rospy.signal_shutdown(global_variable.red+'_video_thread Error')
                print(e)
                print(global_variable.reset_color)

    def run(self):
        print(global_variable.green)
        print('Start Single Thread Video Node')
        print(global_variable.reset_color)

        h, w = self.yolo.size
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.image is None:
                print('Wait For Image')
                rate.sleep()
                continue
            try:
                # -------------------- additional image info-------------------- #
                now = rospy.get_rostime()
                net_img_time, net_img_seq = self.img_cb_time, self.img_cb_seq
                net_img, net_dep = copy.copy(self.image), copy.copy(self.depth_image)
                yolo_gluon.switch_print('cam to net: %f' % (now - net_img_time).to_sec(), verbose)

                net_out = self.inference(cv2.resize(net_img, (w, h)))

                # -------------------- additional image info-------------------- #
                now = rospy.get_rostime()
                self.net_img_time, self.net_img_seq = net_img_time, net_img_seq
                yolo_gluon.switch_print('net done time: %f' % (now - net_img_time).to_sec(), verbose)

                self.process(net_img, net_out, net_dep)
                rate.sleep()

            except Exception as e:
                rospy.signal_shutdown('main_thread Error')
                print(global_variable.red + e + global_variable.reset_color)

    def inference(self, net_img):
        if self.trt:
            trt_outputs = do_inference_wrapper(self.yolo.net, net_img)
            net_out = nd.array(trt_outputs).as_in_context(self.ctx[0])
            net_out = [net_out.reshape((1, 160, 5, 30))]

        else:
            nd_img = yolo_gluon.cv_img_2_ndarray(net_img, self.ctx[0])
            # if self.yolo.use_fp16:
            #     nd_img = nd_img.astype('float16')
            # nd_img = yolo_gluon.nd_white_balance(nd_img, bgr=[1.0, 1.0, 1.0])
            net_out = self.yolo.net.forward(is_train=False, data=nd_img)
            net_out[0].wait_to_read()

        return net_out

    def process(self, net_img, net_out, net_dep):
        pred_car = self.yolo.predict(net_out[:3])
        # --------------- data[5] is depth --------------- #
        if net_dep:  # net_dep != None
            x = int(net_dep.shape[1] * pred_car[0, 2])
            y = int(net_dep.shape[0] * pred_car[0, 1])
            pred_car[0, 5] = net_dep[y, x]
        else:
            pred_car[0, 5] = -1

        # ---------------- data[5] is azi ---------------- #
        x = pred_car[0, -24:]
        prob = np.exp(x) / np.sum(np.exp(x), axis=0)
        c = sum(_cos_offset * prob)
        s = sum(_sin_offset * prob)
        vec_ang = math.atan2(s, c)
        pred_car[0, 5] = vec_ang
        # ------------------------------------------------- #
        yolo_gluon.switch_print('cam to pub: %f' % (rospy.get_rostime() - self.net_img_time).to_sec(), verbose)
        self.ros_publish_array(self.car_pub, self.mat_car, pred_car[0])
        self.visualize(pred_car, net_img)

    def _get_frame(self):
        print(global_variable.green)
        print('Start OPENCV Video Capture Thread')
        dev = self.dev

        if dev == 'jetson':
            print('Image Source: Jetson OnBoard Camera')
            cap = jetson_onboard_camera(640, 360, dev)

        elif dev.split('.')[-1] in ['mp4', 'avi', 'm2ts'] and \
            os.path.exists(dev):
            print('Image Source: ' + dev)
            cap = cv2.VideoCapture(dev)
            rate = rospy.Rate(30)

        elif dev.isdigit() and os.path.exists('/dev/video' + dev):
            print('Image Source: /dev/video' + dev)
            cap = cv2.VideoCapture(int(dev))

        else:
            print(global_variable.red)
            print('dev should be jetson / video_path(mp4, avi, m2ts) / device_index')
            rospy.signal_shutdown('')
            sys.exit(0)

        print(global_variable.reset_color)
        seq = 0
        while not rospy.is_shutdown():
            ret, img = cap.read()
            if img is None:
                continue
            self.img_cb_time = rospy.get_rostime()
            self.img_cb_seq = seq
            self.image = yolo_cv.cv2_flip_and_clip_frame(img, self.clip, self.flip)
            seq += 1
            if 'rate' in locals():
                rate.sleep()

        cap.release()

    def _image_callback(self, img):
        self.img_cb_time = img.header.stamp  # .secs + img.header.stamp.nsecs * 10**-6
        self.img_cb_seq = img.header.seq
        img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        self.image = yolo_cv.cv2_flip_and_clip_frame(img, self.clip, self.flip)

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
    args = utils.video_Parser()
    main(Video, args)
