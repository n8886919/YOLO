#! usr/bin/python
import argparse
import cv2
import datetime
import math
import numpy as np
import os
import time
import yaml

import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import mxboard
import mxnet
from mxnet import nd, cpu

from LP_densenet import LPDenseNet
from yolo_modules import global_variable
from yolo_modules import licence_plate_render
from yolo_modules import yolo_cv
from yolo_modules import yolo_gluon


def main():
    args = Parser()
    LP_detection = LicencePlateDetectioin(args)
    exec "LP_detection.%s()" % args.mode


def Parser():
    parser = argparse.ArgumentParser(prog="python YOLO.py")
    parser.add_argument("version", help="v1")
    parser.add_argument("mode", help="train or valid or video")

    parser.add_argument("--gpu", help="gpu index", dest="gpu", default="0")

    parser.add_argument(
        "--record",
        dest="record", default=1, type=int,
        help="record to tensorboard or not")

    parser.add_argument(
        "--tensorrt",
        dest="tensorrt", default=0, type=int,
        help="use Tensor_RT or not")

    parser.add_argument(
        "--topic",
        dest="topic", default="/usb_cam/image_raw",
        help="ros topic to subscribe")

    parser.add_argument(
        "--weight",
        dest="weight", default=None,
        help="pretrain weight file")

    parser.add_argument(
        "--show",
        dest="show", default=1, type=int,
        help="show processed image")

    return parser.parse_args()


def slice_out(x):
    x = x.transpose((0, 2, 3, 1))
    score_x = x.slice_axis(begin=0, end=1, axis=-1)
    xy = x.slice_axis(begin=1, end=3, axis=-1)
    z = x.slice_axis(begin=3, end=4, axis=-1)
    r = x.slice_axis(begin=4, end=7, axis=-1)
    class_x = x.slice_axis(begin=7, end=10, axis=-1)
    return [score_x, xy, z, r, class_x]


class LicencePlateDetectioin():
    def __init__(self, args):
        spec_path = os.path.join(args.version, 'spec.yaml')
        with open(spec_path) as f:
            spec = yaml.load(f)

        for key in spec:
            setattr(self, key, spec[key])

        self.ctx = yolo_gluon.get_ctx(args.gpu)
        self.export_file = args.version + '/export/'
        if args.mode != 'video':  # train/valid/export
            self.backup_dir = os.path.join(args.version, 'backup')
            self.net = LPDenseNet(
                self.num_init_features,
                self.growth_rate,
                self.block_config,
                classes=self.num_class)

            if args.weight is None:
                args.weight = yolo_gluon.get_latest_weight_from(
                    self.backup_dir)

            yolo_gluon.init_NN(self.net, args.weight, self.ctx)

            if args.mode == 'train':
                self.version = args.version
                self.record = args.record
                self._init_train()

        else:
            self.tensorrt = args.tensorrt
            self.topic = args.topic
            self.show = args.show

    def _init_train(self):
        self.exp = self.exp + datetime.datetime.now().strftime("%m-%dx%H-%M")

        self.batch_size *= len(self.ctx)

        self.num_downsample = len(self.block_config) + 1
        self.area = int(self.size[0] * self.size[1] / 4**self.num_downsample)

        print(global_variable.yellow)
        print('Batch Size = {}'.format(self.batch_size))
        print('Record Step = {}'.format(self.record_step))
        print('Area = {}'.format(self.area))

        self.L1_loss = mxnet.gluon.loss.L1Loss()
        self.L2_loss = mxnet.gluon.loss.L2Loss()
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

    def _find_best(self, L, gpu_index):

        w_feature = np.clip(int(L[7].asnumpy()/32), 0, 31)
        h_feature = np.clip(int(L[8].asnumpy()/32), 0, 19)

        t_X = L[1] / 1000.
        t_Y = L[2] / 1000.
        t_Z = nd.log(L[3]/1000.)
        r1_max = self.r_max[0] * 2 * math.pi / 180.
        r2_max = self.r_max[1] * 2 * math.pi / 180.
        r3_max = self.r_max[2] * 2 * math.pi / 180.

        t_r1 = yolo_gluon.nd_inv_sigmoid(L[4] / r1_max + 0.5)
        t_r2 = yolo_gluon.nd_inv_sigmoid(L[5] / r2_max + 0.5)
        t_r3 = yolo_gluon.nd_inv_sigmoid(L[6] / r3_max + 0.5)

        label = nd.concat(t_X, t_Y, t_Z, t_r1, t_r2, t_r3, dim=-1)
        return (h_feature, w_feature), label

    def _loss_mask(self, label_batch, gpu_index):
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
        LP_class = nd.zeros((bs, h_, w_, self.num_class), ctx=ctx)

        for b in range(bs):
            for L in label_batch[b]:  # all object in the image
                if L[0] < 0:
                    continue

                else:
                    (h_f, w_f), p_6D = self._find_best(L, gpu_index)
                    score[b, h_f, w_f, :] = 1.0  # others are zero
                    mask[b, h_f, w_f, :] = 1.0  # others are zero
                    pose_xy[b, h_f, w_f, :] = p_6D[:2]
                    pose_z[b, h_f, w_f, :] = p_6D[2]
                    pose_r[b, h_f, w_f, :] = p_6D[3:]
                    LP_class[b, h_f, w_f, L[-1]] = 1

        return [score, pose_xy, pose_z, pose_r, LP_class], mask

    def _train_batch(self, bxs, bys):
        all_gpu_loss = [None] * len(self.ctx)
        with mxnet.autograd.record():
            for gpu_i, (bx, by) in enumerate(zip(bxs, bys)):
                # ---------- Forward ---------- #
                by_ = self.net(bx)
                by_ = slice_out(bx)
                loss = self._get_loss(by_, by, gpu_i)

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
            path = self._gen_save_path()
            self.net.collect_params().save(path)

    def _get_loss(self, by_, by, gpu_i):
        with mxnet.autograd.pause():
            by, mask = self._loss_mask(by, gpu_i)
            ones = nd.ones_like(mask)
            s_weight = nd.where(
                mask > 0,
                ones * self.positive_weight,
                ones * self.negative_weight,
                ctx=self.ctx[gpu_i])

        s = self.LG_loss(by_[0], by[0], s_weight * self.scale['score'])
        xy = self.L2_loss(by_[1], by[1], mask * self.scale['xy'])
        z = self.L2_loss(by_[2], by[2], mask * self.scale['z'])
        r = self.L1_loss(by_[3], by[3], mask * self.scale['r'])
        c = self.CE_loss(by_[4], by[4], mask * self.scale['class'])
        return (s, xy, z, r, c)

    def _gen_save_path(self):
        idx = self.backward_counter // self.record_step
        return os.path.join(self.backup_dir, self.exp + '_%d' % idx)

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
            if (self.backward_counter % 10 == 0 or 'bg' not in locals()):
                bg = yolo_gluon.ImageIter_next_batch(bg_iter)
                bg = bg.as_in_context(self.ctx[0]) / 255.

            # -------------------- render dataset -------------------- #
            imgs, labels = LP_generator.add(bg, self.r_max, add_rate=0.5)

            if mode == 'train':
                batch_xs = yolo_gluon.split_render_data(imgs, self.ctx)
                batch_ys = yolo_gluon.split_render_data(labels, self.ctx)
                self._train_batch(batch_xs, batch_ys)

            elif mode == 'val':
                # batch_out = self.net.forward(is_train=False, data=imgs)
                batch_out = self.net(imgs)
                pred = self.predict(batch_out)

                img = yolo_gluon.batch_ndimg_2_cv2img(imgs)[0]
                img, clipped_LP = LP_generator.project_rect_6d.add_edges(
                    img, pred[1:])

                yolo_cv.matplotlib_show_img(ax, img)

                print(nd.concat(
                    nd.array(pred[:7]).reshape(-1, 1),
                    labels[0, 0, :7].reshape(-1, 1).as_in_context(cpu(0)),
                    dim=-1))
                raw_input('--------------------------------------------------')

    def predict(self, batch_out):
        batch_out = slice_out(batch_out)
        out = nd.concat(nd.sigmoid(batch_out[0]), *batch_out[1:], dim=-1)[0]
        # (10L, 16L, 10L)
        best_index = out[:, :, 0].reshape(-1).argmax(axis=0)
        out = out.reshape((-1, 10))

        pred = out[best_index].reshape(-1)  # best out

        pred[1:3] *= 1000
        pred[3] = nd.exp(pred[3]) * 1000
        for i in range(3):
            p = (nd.sigmoid(pred[i+4]) - 0.5) * 2 * self.r_max[i]
            pred[i+4] = p * math.pi / 180.
        return pred.asnumpy()

    def train(self):

        self._train_or_valid('train')

    def valid(self):

        self._train_or_valid('val')

    def video(self, record=False):
        LP_size = int(380*1.05), int(160*1.05)
        pjct_6d = licence_plate_render.ProjectRectangle6D(LP_size)
        net = yolo_gluon.init_executor(
            self.export_file,
            self.size,
            self.ctx[0],
            use_tensor_rt=self.tensorrt)

        # -------------------- ROS -------------------- #
        self.bridge = CvBridge()
        rospy.init_node("LP_Detection", anonymous=True)
        LP_pub = rospy.Publisher(self.pub_clipped_LP, Image, queue_size=0)
        ps_pub = rospy.Publisher(self.pub_LP, Float32MultiArray, queue_size=0)

        rospy.Subscriber(self.topic, Image, self._image_callback)

        print('Image Topic: /YOLO/clipped_LP')
        print('checkpoint file: %s' % self.export_file)

        # -------------------- video record -------------------- #
        if record:
            time = datetime.datetime.now().strftime("%m-%dx%H-%M")
            out_file = os.path.join('video', 'LPD_ % s.avi' % time)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            v_size = (640, 480)
            video_out = cv2.VideoWriter(out_file, fourcc, 30, v_size)

        r = rospy.Rate(30)
        while not rospy.is_shutdown():
            if hasattr(self, 'img') and self.img is not None:
                ori_img = self.img.copy()
                img = self.img.copy()

                resized_img = cv2.resize(img, tuple(self.size[::-1]))
                nd_img = yolo_gluon.cv_img_2_ndarray(resized_img, self.ctx[0])

                out = net.forward(is_train=False, data=nd_img)
                pred = self.predict(out[0])

                if pred[0] > 0.9:
                    img, clipped_LP = pjct_6d.add_edges(img, pred[1:])
                    clipped_LP = self.bridge.cv2_to_imgmsg(clipped_LP, 'bgr8')
                    LP_pub.publish(clipped_LP)

                if self.show:
                    cv2.imshow('img', img)
                    cv2.waitKey(1)

                if record:
                    video_out.write(ori_img)
            else:
                print('Wait For Image')
            r.sleep()

    def export(self):
        shape = (1, 3, self.size[0], self.size[1])
        yolo_gluon.export(self.net,
                          shape,
                          self.export_file,
                          onnx=True)

    def _image_callback(self, img):
        self.img = self.bridge.imgmsg_to_cv2(img, "bgr8")


if __name__ == '__main__':
    main()
