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

# DEPTH_TOPIC = '/zed/depth/depth_registered'
DEPTH_TOPIC = '/drone/camera/depth/image_raw'
verbose = 0
save_video_size = None


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
            pass
            #threading.Thread(target=self._get_frame).start()

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

        rate = rospy.Rate(30)
        print(global_variable.reset_color)
        while not rospy.is_shutdown():
            if hasattr(self, 'net_dep'):
                net_dep = copy.copy(self.net_dep)
            else:
                net_dep = None

            net_out = copy.copy(self.net_out)  # not sure type(net_out)
            net_img = copy.copy(self.net_img)

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

    def run2_result(self):
        import matplotlib.pyplot as plt
        path1 = "/media/nolan/SSD1/YOLO_backup/freiburg_static_cars_52_v1.1/result/annotations"
        for annot in os.listdir(path1):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            num = annot.split('_')[0]
            ious = []
            err_squares = []
            x1s = []
            x2s = []
            path2 = os.path.join(path1, annot)
            with open(path2, 'r') as f:
                lines = f.readlines()

            for line in lines:
                iou = float(line.split(' ')[1])
                if iou < 0.5:
                    continue

                ious.append(iou)
                x1 = float(line.split(' ')[2])
                x2 = float(line.split(' ')[3])
                err = x1-x2
                x1s.append(x1)
                x2s.append(x2)

                if err < -180:
                    err += 360
                elif err > 180:
                    err -= 360
                err_squares.append(err**2)

            ax.plot(x1s, 'go-')
            ax.plot(x2s, 'ro-')

            save_path = os.path.join(path1, num + '_pic.png')
            fig.savefig(save_path)
            iou = sum(ious) / float(len(ious))
            azi_L2_err = math.sqrt(sum(err_squares) / float(len(err_squares)))
            print('car' + num + ', iou: %d, mean azi: %d' % (iou, azi_L2_err))

    def run2(self):
        image_w = 960.
        image_h = 540.
        x = np.arange(53)
        index = [0, 6, 20, 23, 31, 36]
        new_x = np.delete(x, index)

        mx_resize = mxnet.image.ForceResizeAug(tuple(self.yolo.size[::-1]))

        path1 = "/media/nolan/SSD1/YOLO_backup/freiburg_static_cars_52_v1.1"
        for i in new_x:
            save_img_path = os.path.join(path1, "result", "car_%d" % i)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)

            save_txt_path = os.path.join(path1, 'result/annotations')
            if not os.path.exists(save_txt_path):
                os.makedirs(save_txt_path)

            txt_path = os.path.join(path1, "annotations", "%d_annot.txt" % i)
            with open(txt_path, 'r') as f:
                all_lines = f.readlines()

            for line_i, line in enumerate(all_lines):
                img_name = line.split('\t')[0].split('.')[0] + '.png'
                img_path = os.path.join(path1, img_name)
                save_img = os.path.join(save_img_path, img_name.split('/')[-1])
                img = cv2.imread(img_path)

                # -------------------- Prediction -------------------- #
                nd_img = yolo_gluon.cv_img_2_ndarray(img, self.ctx[0], mxnet_resize=mx_resize)
                net_out = self.yolo.net.forward(is_train=False, data=nd_img)
                pred_car = self.yolo.predict(net_out[:3])
                pred_car[0, 5] = -1  # (1, 80)
                Cout = pred_car[0]

                left_ = (Cout[2] - 0.5*Cout[4]) * image_w
                up_ = (Cout[1] - 0.5*Cout[3]) * image_h
                right_ = (Cout[2] + 0.5*Cout[4]) * image_w
                down_ = (Cout[1] + 0.5*Cout[3]) * image_h

                #self.radar_prob.plot(Cout[0], Cout[-self.yolo.num_class:])
                vec_ang, vec_rad, prob = self.radar_prob.cls2ang(Cout[0], Cout[-self.yolo.num_class:])

                # -------------------- Ground Truth -------------------- #
                box_para = np.fromstring(line.split('\t')[1], dtype='float32', sep=' ')
                left, up, right, down = box_para
                h = (down - up)/image_h
                w = (right - left)/image_w
                y = (down + up)/2/image_h
                x = (right + left)/2/image_w
                boxA = [0, y, x, h, w, 0]
                # print('boxA=', boxA)

                azi_label = int(line.split('\t')[2]) - 90
                azi_label = azi_label - 360 if azi_label > 180 else azi_label

                inter = (min(right, right_)-max(left, left_)) * (min(down, down_)-max(up, up_))
                a1 = (right-left)*(down-up)
                a2 = (right_-left_)*(down_-up_)
                iou = (inter) / (a1 + a2 - inter)

                save_txt = os.path.join(save_txt_path, '%d_annot' % i)
                with open(save_txt, 'a') as f:
                    f.write('%s %f %f %f %f\n' % (img_name, iou, azi_label, vec_ang*180/math.pi, vec_rad))

                '''
                yolo_cv.cv2_add_bbox(img, Cout, 3, use_r=False)
                yolo_cv.cv2_add_bbox(img, boxA, 4, use_r=False)

                cv2.imshow('img', img)
                cv2.waitKey(1)
                cv2.imwrite(save_img, img)
                a = raw_input()
                '''

    def process(self, net_img, net_out, net_dep):
            pred_car = self.yolo.predict(net_out[:3])

            if net_dep:  # net_dep != None
                x = int(net_dep.shape[1] * pred_car[0, 2])
                y = int(net_dep.shape[0] * pred_car[0, 1])
                pred_car[0, 5] = net_dep[y, x]
            else:
                pred_car[0, 5] = -1

            if hasattr(self, 'net_img_time') and verbose:
                now = rospy.get_rostime()
                print(('cam to pub delay: ', (now - self.net_img_time).to_sec()))

            self.ros_publish_array(self.car_pub, self.mat_car, pred_car[0])
            self.visualize(pred_car, net_img)

    def _get_frame(self):
        dev = self.dev

        print(global_variable.green)
        print('Start OPENCV Video Capture Thread')

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
            print(('dev should be (jetson) or '
                   '(video_path) or (device_index)'))
            sys.exit(0)

        print(global_variable.reset_color)
        while not rospy.is_shutdown():
            self.ret, img = cap.read()
            self.img = yolo_cv.cv2_flip_and_clip_frame(img, self.clip, self.flip)
            if 'rate' in locals():
                rate.sleep()

        cap.release()

    def _net_thread(self):
        print('Start Net Thread')

        size = tuple(self.yolo.size[::-1])
        ctx = self.ctx[0]
        mx_resize = mxnet.image.ForceResizeAug(size)
        while not rospy.is_shutdown():
            if not hasattr(self, 'img') or self.img is None:
                print('Net thread Wait For Image')
                time.sleep(1.0)
                continue
            # -------------------- additional image info-------------------- #
            if hasattr(self, 'depth_image'):
                net_dep = self.depth_image.copy()

            if hasattr(self, 'img_time') and verbose:
                net_img_time = self.img_time
                now = rospy.get_rostime()
                print(('cam to net: ', (now - net_img_time).to_sec()))

            # -------------------- image -------------------- #
            net_img = self.img.copy()
            nd_img = yolo_gluon.cv_img_2_ndarray(net_img, ctx, mxnet_resize=mx_resize)

            # if self.yolo.use_fp16:
            #     nd_img = nd_img.astype('float16')

            # nd_img = yolo_gluon.nd_white_balance(nd_img, bgr=[1.0, 1.0, 1.0])
            net_out = self.yolo.net.forward(is_train=False, data=nd_img)
            net_out[0].wait_to_read()

            if 'net_img_time' in locals():
                self.net_img_time = net_img_time
                now = rospy.get_rostime()
                print(('net done time', (now - net_img_time).to_sec()))

            # -------------------- additional image info-------------------- #
            if 'net_dep' in locals():
                self.net_dep = net_dep

            # -------------------- image -------------------- #
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
