#! usr/bin/python
import argparse
import cv2
import datetime
import math
import numpy as np
import os
import sys
import threading
import time
import yaml

import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import mxboard
import mxnet
from mxnet import nd
from mxnet.gluon import nn
from gluoncv.model_zoo.densenet import _make_dense_block, _make_transition

from yolo_modules import global_variable
from yolo_modules import licence_plate_render
from yolo_modules import yolo_cv
from yolo_modules import yolo_gluon
from yolo_modules import tensorrt_module


os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

video_verbose = False
video_threshold = 0.5
video_inference_mode = 'trt'
#video_record_origin = False
#video_record_processed = False


def main():
    args = Parser()
    LP_detection = LicencePlateDetectioin(args)
    available_mode = ['train', 'valid', 'export', 'video', 'trt']
    assert args.mode in available_mode, \
        'Available Modes Are {}'.format(available_mode)

    exec "LP_detection.%s()" % args.mode


def Parser():
    parser = argparse.ArgumentParser(prog="python LP_detection.py")
    parser.add_argument("version", help="v2 is good")
    parser.add_argument("mode", help="train/valid/video")

    parser.add_argument("--gpu", help="gpu index", dest="gpu", default="0")

    # -------------------- Train/Valid -------------------- #
    parser.add_argument(
        "--weight",
        dest="weight", default=None,
        help="pretrain weight file")

    parser.add_argument(
        "--record",
        dest="record", default=1, type=int,
        help="record to tensorboard or not while training")

    # ----------------------- Video ----------------------- #
    parser = yolo_cv.add_video_parser(parser)

    return parser.parse_args()


class LPDenseNet(mxnet.gluon.HybridBlock):
    # https://github.com/dmlc/gluon-cv/blob/3658339acbdfc78c2191c687e4430e3a673
    # 66b7d/gluoncv/model_zoo/densenet.py#L620
    # Densely Connected Convolutional Networks
    # <https://arxiv.org/pdf/1608.06993.pdf>
    def __init__(self, num_init_features, growth_rate, block_config,
                 bn_size=4, dropout=0, classes=1, **kwargs):
        super(LPDenseNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(num_init_features, kernel_size=7,
                                        strides=2, padding=3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            # Add dense blocks
            num_features = num_init_features
            for i, num_layers in enumerate(block_config):
                self.features.add(
                    _make_dense_block(
                        num_layers, bn_size, growth_rate, dropout, i+1))

                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    self.features.add(_make_transition(num_features // 2))
                    num_features = num_features // 2
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(512, (3, 3), padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(7+classes, (1, 1)))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x


class LicencePlateDetectioin():
    def __init__(self, args):
        spec_path = os.path.join(args.version, 'spec.yaml')
        with open(spec_path) as f:
            spec = yaml.load(f)

        for key in spec:
            setattr(self, key, spec[key])

        self.version = args.version
        self.ctx = yolo_gluon.get_ctx(args.gpu)
        self.export_file = args.version + '/export/'

        if args.mode in ['train', 'valid', 'export']:
            self.backup_dir = os.path.join(args.version, 'backup')
            self.net = LPDenseNet(
                self.num_init_features,
                self.growth_rate,
                self.block_config,
                classes=self.LP_num_class)

            if args.weight is None:
                args.weight = yolo_gluon.get_latest_weight_from(
                    self.backup_dir)

            yolo_gluon.init_NN(self.net, args.weight, self.ctx)

            if args.mode == 'train':
                self.record = args.record
                self._init_train()

        else:
            self.trt = args.trt
            self.topic = args.topic
            self.dev = args.dev
            self.flip = args.flip
            self.clip = (args.clip_h, args.clip_w)
            self.show = args.show

    def train(self):

        self._train_or_valid('train')

    def valid(self):

        self._train_or_valid('val')

    def export(self):
        shape = (1, 3, self.size[0], self.size[1])
        yolo_gluon.export(self.net, shape, self.ctx[0], self.export_file, onnx=1, epoch=0)

    def video(self):
        h, w = self.size
        assert video_inference_mode in ['trt', 'mx'], ('mode should be trt/mx')
        if video_inference_mode == 'mx':
            mx_resize = mxnet.image.ForceResizeAug((w, h), interp=2)
            net = yolo_gluon.init_executor(self.export_file, (h, w), self.ctx[0])
            yolo_gluon.test_inference_rate(net, (1, 3, h, w), cycles=100, ctx=self.ctx[0])

        elif video_inference_mode == 'trt':  # tensorRT Inference can't in thread
            engine_wrapper = tensorrt_module.get_engine_wrapper(
                self.version + '/export/onnx/out.onnx',
                self.version + '/export/onnx/out.trt')

        rospy.init_node("LP_Detection", anonymous=True)
        threading.Thread(target=self._video_thread).start()

        while not rospy.is_shutdown():
            if not hasattr(self, 'img') or self.img is None:
                print('Wait For Image')
                time.sleep(1.0)
                continue
            try:
                net_img = self.img.copy()  # (480, 640, 3), np.float32
                net_start_time = time.time()

                if video_inference_mode == 'mx':
                    nd_img = yolo_gluon.cv_img_2_ndarray(net_img, self.ctx[0], mxnet_resize=mx_resize)
                    net_out = net.forward(is_train=False, data=nd_img)[0]
                    net_out[-1].wait_to_read()

                elif video_inference_mode == 'trt':
                    # if cuMemcpyHtoDAsync failed: invalid argument, check input image size
                    trt_img = cv2.resize(net_img, (w, h))  # (320, 512, 3)
                    trt_outputs = tensorrt_module.do_inference_wrapper(engine_wrapper, trt_img)
                    net_out = nd.array(trt_outputs).reshape((1, 10, 10, 16))

                yolo_gluon.switch_print(time.time()-net_start_time, video_verbose)
                self.net_img = net_img
                self.net_out = net_out

            except Exception as e:
                print(e)
                rospy.signal_shutdown(e)

    def predict_LP(self, batch_out):
        batch_out = self.slice_out(batch_out)
        out = nd.concat(nd.sigmoid(batch_out[0]), *batch_out[1:], dim=-1)[0]
        # (10L, 16L, 10L)
        best_index = out[:, :, 0].reshape(-1).argmax(axis=0)
        out = out.reshape((-1, 10))
        pred = out[best_index].reshape(-1)  # best out

        pred[1:4] *= 1000

        for i in range(3):
            p = (nd.sigmoid(pred[i+4]) - 0.5) * 2 * self.LP_r_max[i]
            pred[i+4] = p * math.pi / 180.
        return pred.asnumpy()

    def slice_out(self, x):
            x = x.transpose((0, 2, 3, 1))
            i = 0
            outs = []
            for pt in self.LP_slice_point:
                outs.append(x.slice_axis(begin=i, end=pt, axis=-1))
                i = pt

            return outs

    # -------------------- Train -------------------- #
    def _train_or_valid(self, mode):
        print(global_variable.cyan)
        print(mode)
        print(global_variable.reset_color)
        if mode == 'val':
            self.batch_size = 1
            ax = yolo_cv.init_matplotlib_figure()
            # self.net = yolo_gluon.init_executor(
            #    self.export_file, self.size, self.ctx[0])

        # -------------------- background -------------------- #
        LP_generator = licence_plate_render.LPGenerator(*self.size)
        bg_iter = yolo_gluon.load_background(mode, self.batch_size, *self.size)

        # -------------------- main loop -------------------- #
        self.backward_counter = 0
        while True:
            if (self.backward_counter % 3 == 0 or 'bg' not in locals()):
                bg = yolo_gluon.ImageIter_next_batch(bg_iter)
                bg = bg.as_in_context(self.ctx[0]) / 255.

            # -------------------- render dataset -------------------- #
            imgs, labels = LP_generator.add(bg, self.LP_r_max, add_rate=0.5)

            if mode == 'train':
                batch_xs = yolo_gluon.split_render_data(imgs, self.ctx)
                batch_ys = yolo_gluon.split_render_data(labels, self.ctx)
                self._train_batch_LP(batch_xs, batch_ys)

            elif mode == 'val':
                batch_out = self.net(imgs)
                pred = self.predict_LP(batch_out)

                img = yolo_gluon.batch_ndimg_2_cv2img(imgs)[0]
                labels = labels.asnumpy()

                img, _ = LP_generator.project_rect_6d.add_edges(
                    img, labels[0, 0, 1:7])

                img, clipped_LP = LP_generator.project_rect_6d.add_edges(
                    img, pred[1:])

                yolo_cv.matplotlib_show_img(ax, img)
                print('Labels: {}'.format(labels))
                print('predictions: {}'.format(pred))
                raw_input('--------------------------------------------------')

    def _init_train(self):
        self.exp = self.exp + datetime.datetime.now().strftime("%m-%dx%H-%M")

        self.batch_size *= len(self.ctx)

        self.num_downsample = len(self.block_config) + 1

        print(global_variable.yellow)
        print('Batch Size = {}'.format(self.batch_size))
        print('Record Step = {}'.format(self.record_step))

        #self.L1_loss = mxnet.gluon.loss.L1Loss()
        #self.L2_loss = mxnet.gluon.loss.L2Loss()
        self.HB_loss = mxnet.gluon.loss.HuberLoss()
        self.LG_loss = mxnet.gluon.loss.LogisticLoss(label_format='binary')
        self.CE_loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss(
            from_logits=False, sparse_label=False)

        # -------------------- init trainer -------------------- #
        optimizer = mxnet.optimizer.create(
            'adam',
            learning_rate=self.learning_rate,
            multi_precision=False)

        self.trainer = mxnet.gluon.Trainer(
            self.net.collect_params(),
            optimizer=optimizer)

        logdir = os.path.join(self.version, 'logs')
        self.sw = mxboard.SummaryWriter(logdir=logdir, verbose=True)

        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def _find_best_LP(self, L, gpu_index):
        step = 2**self.num_downsample
        h_feature_max = self.size[0] / 2**self.num_downsample - 1
        w_feature_max = self.size[1] / 2**self.num_downsample - 1
        h_feature = np.clip(int(L[8].asnumpy()/step), 0, h_feature_max)
        w_feature = np.clip(int(L[7].asnumpy()/step), 0, w_feature_max)
        #print(h_feature_max, w_feature_max)
        #print(h_feature, w_feature)
        t_X = L[1] / 1000.
        t_Y = L[2] / 1000.
        t_Z = L[3] / 1000.

        r1_max = self.LP_r_max[0] * math.pi / 180.
        r2_max = self.LP_r_max[1] * math.pi / 180.
        r3_max = self.LP_r_max[2] * math.pi / 180.  # 2 * theta (rad)

        t_r1 = yolo_gluon.nd_inv_sigmoid(L[4] / r1_max / 2. + 0.5)
        t_r2 = yolo_gluon.nd_inv_sigmoid(L[5] / r2_max / 2. + 0.5)
        t_r3 = yolo_gluon.nd_inv_sigmoid(L[6] / r3_max / 2. + 0.5)

        label = nd.concat(t_X, t_Y, t_Z, t_r1, t_r2, t_r3, dim=-1)

        return (h_feature, w_feature), label

    def _loss_mask_LP(self, label_batch, gpu_index):
        """Generate training targets given predictions and label_batch.
        label_batch: bs*object*[class, cent_y, cent_x, box_h, box_w, rotate]
        """
        bs = label_batch.shape[0]
        ctx = self.ctx[gpu_index]
        h_ = self.size[0] / 2**self.num_downsample
        w_ = self.size[1] / 2**self.num_downsample

        score = nd.zeros((bs, h_, w_, 1), ctx=ctx)
        mask = nd.zeros((bs, h_, w_, 1), ctx=ctx)
        pose_xy = nd.zeros((bs, h_, w_, 2), ctx=ctx)
        pose_z = nd.zeros((bs, h_, w_, 1), ctx=ctx)
        pose_r = nd.zeros((bs, h_, w_, 3), ctx=ctx)
        LP_class = nd.zeros((bs, h_, w_, self.LP_num_class), ctx=ctx)

        for b in range(bs):
            for L in label_batch[b]:  # all object in the image
                if L[0] < 0:
                    continue

                else:
                    (h_f, w_f), p_6D = self._find_best_LP(L, gpu_index)
                    score[b, h_f, w_f, :] = 1.0  # others are zero
                    mask[b, h_f, w_f, :] = 1.0  # others are zero
                    pose_xy[b, h_f, w_f, :] = p_6D[:2]
                    pose_z[b, h_f, w_f, :] = p_6D[2]
                    pose_r[b, h_f, w_f, :] = p_6D[3:]
                    LP_class[b, h_f, w_f, L[-1]] = 1

        return [score, pose_xy, pose_z, pose_r, LP_class], mask

    def _train_batch_LP(self, bxs, bys):
        all_gpu_loss = [None] * len(self.ctx)
        with mxnet.autograd.record():
            for gpu_i, (bx, by) in enumerate(zip(bxs, bys)):
                # ---------- Forward ---------- #
                by_ = self.net(bx)
                by_ = self.slice_out(by_)

                with mxnet.autograd.pause():
                    by, mask = self._loss_mask_LP(by, gpu_i)
                    ones = nd.ones_like(mask)
                    s_weight = nd.where(
                        mask > 0,
                        ones * self.LP_positive_weight,
                        ones * self.LP_negative_weight,
                        ctx=self.ctx[gpu_i])

                loss = self._get_loss_LP(by_, by, s_weight, mask)

                # ---------- Backward ---------- #
                sum(loss).backward()

        # ---------- Update Weights ---------- #
        self.trainer.step(self.batch_size)
        self.backward_counter += 1  # do not save at first step

        # ---------- Record Loss ---------- #
        if self.record and self.backward_counter % 10 == 0:
            # only record last gpu loss
            yolo_gluon.record_loss(loss, self.loss_name, self.sw,
                                   step=self.backward_counter,
                                   exp=self.exp)

        # ---------- Save Weights ---------- #
        if self.backward_counter % self.record_step == 0:
            idx = self.backward_counter // self.record_step
            path = os.path.join(self.backup_dir, self.exp + '_%d' % idx)
            self.net.collect_params().save(path)

    def _get_loss_LP(self, by_, by, s_weight, mask):
        s = self.LG_loss(by_[0], by[0], s_weight * self.scale['LP_score'])
        xy = self.HB_loss(by_[1], by[1], mask * self.scale['LP_xy'])
        z = self.HB_loss(by_[2], by[2], mask * self.scale['LP_z'])
        r = self.HB_loss(by_[3], by[3], mask * self.scale['LP_r'])
        c = self.CE_loss(by_[4], by[4], mask * self.scale['LP_class'])
        return (s, xy, z, r, c)

    # -------------------- Video -------------------- #
    def _video_thread(self, record=False):
        pjct_6d = licence_plate_render.ProjectRectangle6D(int(380*1.05), int(160*1.05))

        self.bridge = CvBridge()
        LP_pub = rospy.Publisher(self.pub_clipped_LP, Image, queue_size=0)
        ps_pub = rospy.Publisher(self.pub_LP, Float32MultiArray, queue_size=0)

        if self.dev == 'ros':
            rospy.Subscriber(self.topic, Image, self._image_callback)
            print('Image Topic: %s' % self.topic)
        else:
            threading.Thread(target=self._get_frame).start()

        pose_msg = Float32MultiArray()

        # -------------------- video record -------------------- #
        '''
        if record:
            start_time = datetime.datetime.now().strftime("%m-%dx%H-%M")
            out_file = os.path.join('video', 'LPD_ % s.avi' % start_time)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            v_size = (640, 480)
            video_out = cv2.VideoWriter(out_file, fourcc, 30, v_size)
        '''
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if not hasattr(self, 'net_out') or not hasattr(self, 'net_img'):
                time.sleep(1)
                print('Wait For Net')
                continue

            img = self.net_img.copy()
            net_out = self.net_out.copy()
            pred = self.predict_LP(self.net_out)

            ps_pub.publish(pose_msg)

            if pred[0] > video_threshold:
                img, clipped_LP = pjct_6d.add_edges(img, pred[1:])
                clipped_LP = self.bridge.cv2_to_imgmsg(clipped_LP, 'bgr8')
                LP_pub.publish(clipped_LP)

            if self.show:
                cv2.imshow('img', img)
                cv2.waitKey(1)
            #video_out.write(ori_img)
            #rate.sleep()

    def _image_callback(self, img):

        self.img = self.bridge.imgmsg_to_cv2(img, "bgr8")

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


if __name__ == '__main__':
    main()
