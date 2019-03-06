import copy
import threading

import rospy
from std_msgs.msg import Float32MultiArray

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from yolo_modules import yolo_cv
from yolo_modules import yolo_gluon
from yolo_modules import licence_plate_render
from yolo_modules import global_variable
import time

import mxnet
from mxnet import gpu
from mxnet import nd

import car.utils
from car.video_node import Video
from YOLO import YOLO

verbose = 0


def main():
    args = car.utils.video_Parser()
    video = CarLPVideo(args)
    video()  # Two thread


class CarLPVideo(Video):
    def __init__(self, args):
        self.yolo = YOLO(args)

        self.project_rect_6d = licence_plate_render.ProjectRectangle6D(
            int(380*1.1), int(160*1.1))

        self.car_threshold = 0.95
        self.LP_threshold = 0.1
        self._init(args)

        # -------------------- init LP -------------------- #
        self.LP_pub = rospy.Publisher(self.yolo.pub_LP, Float32MultiArray, queue_size=1)
        self.clipped_LP_pub = rospy.Publisher(self.yolo.pub_clipped_LP, Image, queue_size=1)
        self.mat_LP = Float32MultiArray()

    def __call__(self):
        print(global_variable.green)
        print('Start Double Thread Video Node')

        threading.Thread(target=self._net_thread).start()

        shape = (1, 3, self.yolo.size[0], self.yolo.size[1])
        yolo_gluon.test_inference_rate(self.yolo.net, shape)

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
            now = rospy.get_rostime()
            if verbose:
                print('cam to pub: %f' % (
                    now - self.net_img_time).to_sec())

            self.net_thread_start = True

            pred_car = self.yolo.predict(net_out[:3])

            # --------------- data[5] is depth --------------- #
            if net_dep is not None:
                x = int(net_dep.shape[1] * pred_car[0, 2])
                y = int(net_dep.shape[0] * pred_car[0, 1])
                pred_car[0, 5] = net_dep[y, x]
            else:
                pred_car[0, 5] = -1
            # ------------------------------------------------- #

            pred_LP = self.yolo.predict_LP([net_out[-1]])
            self.ros_publish_array(self.LP_pub, self.mat_LP, pred_LP[0])
            self.ros_publish_array(self.car_pub, self.mat_car, pred_car[0])

            self.visualize_carlp(pred_car, pred_LP, net_img)

            #rate.sleep()

    def visualize_carlp(self, pred_car, pred_LP, img):
        LP_out = pred_LP[0]

        # -------------------- Licence Plate -------------------- #
        if LP_out[0] > self.LP_threshold:
            img, clipped_LP = self.project_rect_6d.add_edges(img,  LP_out[1:])
            clipped_LP_msg = self.bridge.cv2_to_imgmsg(clipped_LP, 'bgr8')
            self.clipped_LP_pub.publish(clipped_LP_msg)

            #if self.show:
                #cv2.imshow('Licence Plate', clipped_LP)

        self.visualize(pred_car, img)


if __name__ == '__main__':
    main()
