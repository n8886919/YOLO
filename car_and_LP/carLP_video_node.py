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
from car.video_node import Video, main
from YOLO import YOLO

verbose = 0


class CarLPVideo(Video):
    def __init__(self, args):
        self.yolo = YOLO(args)

        self.project_rect_6d = licence_plate_render.ProjectRectangle6D(
            int(380*1.1), int(160*1.1))

        self.car_threshold = 0.5
        self.LP_threshold = 0.5
        self._init(args)

        # -------------------- init LP -------------------- #
        self.LP_pub = rospy.Publisher(self.yolo.pub_LP, Float32MultiArray, queue_size=1)
        self.clipped_LP_pub = rospy.Publisher(self.yolo.pub_clipped_LP, Image, queue_size=1)
        self.mat_LP = Float32MultiArray()

    def process(self, net_img, net_out, net_dep):
        pred_LP = self.yolo.predict_LP([net_out[-1]])
        pred_car = self.yolo.predict(net_out[:3])
        # --------------- data[5] is depth --------------- #
        if net_dep:  # net_dep != None
            x = int(net_dep.shape[1] * pred_car[0, 2])
            y = int(net_dep.shape[0] * pred_car[0, 1])
            pred_car[0, 5] = net_dep[y, x]
        else:
            pred_car[0, 5] = -1
        # ---------------- data[5] is azi ---------------- #
        '''
        x = pred_car[0, -24:]
        prob = np.exp(x) / np.sum(np.exp(x), axis=0)
        c = sum(cos_offset * prob)
        s = sum(sin_offset * prob)
        vec_ang = math.atan2(s, c)
        pred_car[0, 5] = vec_ang
        '''
        # ------------------------------------------------- #
        yolo_gluon.switch_print('cam to pub: %f' % (rospy.get_rostime() - self.net_img_time).to_sec(), verbose)

        self.ros_publish_array(self.LP_pub, self.mat_LP, pred_LP[0])
        self.ros_publish_array(self.car_pub, self.mat_car, pred_car[0])
        self.visualize_carlp(pred_car, pred_LP, net_img)

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
    args = car.utils.video_Parser()
    video = CarLPVideo(args)
    main(video)
