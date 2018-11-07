import copy
import cv2
import numpy as np
import sys
import threading
import matplotlib.pyplot as plt
import argparse

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
from yolo_modules import cv


def Parser():
    parser = argparse.ArgumentParser(prog="python YOLO.py")
    parser.add_argument("version", help="v1")
    parser.add_argument("mode", help="train or valid or video")
    parser.add_argument("--topic", "-t", help="ros topic to subscribe", dest="topic", default="")
    parser.add_argument("--radar", help="show radar plot", dest="radar", default=0, type=int)
    parser.add_argument("--show", help="show processed image", dest="show", default=1, type=int)
    parser.add_argument("--gpu", help="gpu index", dest="gpu", default="0")
    parser.add_argument("--weight", help="pretrain weight file", dest="weight", default=None)
    parser.parse_args().show = bool(parser.parse_args().show)
    parser.parse_args().radar = bool(parser.parse_args().radar)
    return parser.parse_args()


class CarLPNet(basic_yolo.BasicYOLONet):
    def __init__(self, spec, num_sync_bn_devices=1, **kwargs):
        super(CarLPNet, self).__init__(spec, num_sync_bn_devices, **kwargs)

        LP_anchor = spec['LP_anchors']
        LP_channel = spec['channels'][-3]
        self.LP_slice_point = spec['LP_slice_point']

        self.LP_branch = mxnet.gluon.nn.HybridSequential()
        self.LP_branch.add(YOLODetectionBlockV3(LP_channel, num_sync_bn_devices))
        self.LP_branch.add(basic_yolo.YOLOOutput(self.LP_slice_point[-1], len(LP_anchor[0])))

    def hybrid_forward(self, F, x, *args):
        routes = []
        all_output = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= len(self.stages) - self.num_pyrmaid_layers:
                routes.append(x)

        # the YOLO output layers are used in reverse order, i.e., from very deep layers to shallow
        end = False
        for i, block, output in zip(range(len(routes)), self.yolo_blocks, self.yolo_outputs):
            if i >= len(routes) - 1:
                _, LP_output = self.LP_branch[0](x)
                LP_output = self.LP_branch[1](LP_output)
                end = True

            x, tip = block(x)
            all_output.append(output(tip))

            if end:
                break
            # add transition layers
            x = self.transitions[i](x)

            # upsample feature map reverse to shallow layers
            upsample = _upsample(x, stride=2)
            x = F.concat(upsample, routes[::-1][i + 1], dim=1)

        out = [self.merge_and_slice(F, all_output[::-1], self.slice_point)] + \
            [self.merge_and_slice(F, LP_output, self.LP_slice_point)]
        return out  # [(score, yxhw, cls_pred), (score, yxhw)]


class Video():
    def _init_ros(self):
        rospy.init_node("YOLO_ros_node", anonymous=True)
        self.YOLO_img_pub = rospy.Publisher(self.pub_img, Image, queue_size=1)
        self.YOLO_box_pub = rospy.Publisher(self.pub_box, Float32MultiArray, queue_size=1)

        self.bridge = CvBridge()
        self.mat = Float32MultiArray()
        self.mat.layout.dim.append(MultiArrayDimension())
        self.mat.layout.dim.append(MultiArrayDimension())
        self.mat.layout.dim[0].label = "box"
        self.mat.layout.dim[1].label = "predict"
        self.mat.layout.dim[0].size = self.topk
        self.mat.layout.dim[1].size = 7
        self.mat.layout.dim[0].stride = self.topk * 7
        self.mat.layout.dim[1].stride = 7
        self.mat.data = [-1] * 7 * self.topk

        #fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # (*'MJPG')#(*'MPEG')
        #self.out = cv2.VideoWriter('./video/car_rotate.mp4', fourcc, 30, (640, 360))

    def _get_frame(self):
        #cap = open_cam_onboard(self.cam_w, self.cam_w)
        #cap = cv2.VideoCapture(1)
        cap = cv2.VideoCapture('video/GOPR0730.MP4')
        while not rospy.is_shutdown():
            ret, img = cap.read()

            self.img = img
            #print(self.img.shape)
            #img = cv2.flip(img, -1)
            #rospy.sleep(0.1)
        cap.release()

    def _image_callback(self, img):
        # -------------------- Convert and Predict -------------------- #
        img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        #self.img = img[60:420]
        self.img = img

    def visualize(self, Cout):
        img = copy.deepcopy(self.img)
        n = np.argmax(Cout[6:])
        #self.out.write(img)
        # -------------------- Add Box -------------------- #
        #vec_ang, vec_rad, prob = self.radar_prob.cla2ang(Cout[0], Cout[-self.num_class:])
        if Cout[0] > 0.5:
            cv.cv2_add_bbox(img, Cout, [0, 255, 0])
            for i in range(6):
                self.mat.data[i] = Cout[i]

            #self.mat.data[6] = vec_ang

        else:
            #self.miss_counter += 1
            #if self.miss_counter > 20:
            self.mat.data = [-1]*7

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
            print('\033[1;33;40mUse USB Camera')
        else:
            rospy.Subscriber(topic, Image, self._image_callback)
            print('\033[1;33;40mImage Topic: %s\033[0m' % topic)

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if hasattr(self, 'img') and self.img is not None:

                nd_img = self.resz(nd.array(self.img))
                nd_img = nd_img.as_in_context(ctx)
                nd_img = nd_img.transpose((2, 0, 1)).expand_dims(axis=0)/255.
                out = self.predict(nd_img)[0]
                self.visualize(out)
                if not topic:
                    rate.sleep()
            # else: print('Wait For Image')


if __name__ == '__main__':
    args = Parser()

    with open(args.version+'/spec.yaml') as f:
        spec = yaml.load(f)

    net = CarLPNet(spec, num_sync_bn_devices=2)
    net.test(64*5, 64*8)

    from mxboard import SummaryWriter
    sw = SummaryWriter(logdir=args.version+'/logs', flush_secs=60)
    sw.add_graph(net)
