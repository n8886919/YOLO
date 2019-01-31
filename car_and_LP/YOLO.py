#!/usr/bin/env python
import datetime
import math
import os
import sys
import threading
import time
import yaml

import numpy as np
from pprint import pprint

import mxnet
from mxnet import gluon, nd, gpu
from mxboard import SummaryWriter

from gluoncv.model_zoo.yolo.yolo3 import YOLODetectionBlockV3
from gluoncv.model_zoo.yolo.yolo3 import _upsample

from yolo_modules import basic_yolo

# self define modules
from yolo_modules import yolo_gluon
from yolo_modules import yolo_cv
from yolo_modules import licence_plate_render
from yolo_modules import global_variable

from car import utils
from car.render_car import RenderCar
import car.YOLO as car_YOLO
import licence_plate.LP_detection as LP_detection

render_thread_pre_load = True
export_onnx = False


def main():
    args = utils.yolo_Parser()
    yolo = YOLO(args)

    assert args.mode in car_YOLO.available_mode, \
        'Available Modes Are {}'.format(available_mode)

    exec "yolo.%s()" % args.mode


class CarLPNet(basic_yolo.BasicYOLONet):
    def __init__(self, spec, num_sync_bn_devices=-1, **kwargs):
        super(CarLPNet, self).__init__(spec, num_sync_bn_devices, **kwargs)

        LP_channel = spec['channels'][-3]
        LP_slice_point = spec['LP_slice_point']

        self.LP_branch = mxnet.gluon.nn.HybridSequential()
        self.LP_branch.add(YOLODetectionBlockV3(LP_channel, num_sync_bn_devices))
        self.LP_branch.add(YOLODetectionBlockV3(LP_channel, num_sync_bn_devices))
        self.LP_branch.add(YOLODetectionBlockV3(LP_channel, num_sync_bn_devices))
        self.LP_branch.add(gluon.nn.Conv2D(LP_slice_point[-1], kernel_size=1))

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
                _, LP_output = self.LP_branch[1](LP_output)
                _, LP_output = self.LP_branch[2](LP_output)
                LP_output = self.LP_branch[3](LP_output)
                LP_output = LP_output.transpose((0, 2, 3, 1))
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

        return all_output[::-1], [LP_output]


class YOLO(car_YOLO.YOLO, LP_detection.LicencePlateDetectioin):
    def __init__(self, args):
        car_YOLO.YOLO.__init__(self, args)
        # pprint(self.classes)

    def _init_net(self, spec, weight):
        # Do not set num_sync_bn_devices=len(self.ctx)
        # because No conversion function for contrib_SyncBatchNorm yet.
        # (ONNX)
        # self.net = CarLPNet(spec, num_sync_bn_devices=len(self.cx))

        self.net = CarLPNet(spec, num_sync_bn_devices=-1)
        self.num_downsample = len(spec['layers']) - 2  # For LP training

        # self.backup_dir = os.path.join(self.version, 'backup')
        self.backup_dir = os.path.join(
            '/media/nolan/SSD1/YOLO_backup/car_and_LP',
            self.version,
            'backup')

        if weight is None:
            weight = yolo_gluon.get_latest_weight_from(self.backup_dir)

        yolo_gluon.init_NN(self.net, weight, self.ctx)

    # -------------------- LP -------------------- #
    def _score_weight_LP(self, mask, ctx):
        n = self.LP_negative_weight
        p = self.LP_positive_weight

        ones = nd.ones_like(mask)
        score_weight = nd.where(mask > 0, ones*p, ones*n, ctx=ctx)

        return score_weight

    def predict_LP(self, LP_batch_out):
        # LP_batch_out = self.fp16_2_fp32(LP_batch_out)
        LP_batch_out = self.merge_and_slice(LP_batch_out, self.LP_slice_point)

        LP_score = nd.sigmoid(LP_batch_out[0])
        LP_pose_xy = LP_batch_out[1]
        LP_pose_z = LP_batch_out[2]
        LP_pose_r = LP_batch_out[3]
        LP_batch_out = nd.concat(
            LP_score, LP_pose_xy, LP_pose_z, LP_pose_r, dim=-1)

        LP_batch_out = nd.split(LP_batch_out, axis=0, num_outputs=len(LP_batch_out))

        LP_batch_pred = []
        for i, out in enumerate(LP_batch_out):
            best_index = LP_score[i].reshape(-1).argmax(axis=0)
            out = out.reshape((-1, 7))

            pred = out[best_index][0]  # best out
            pred[1:7] = self.LP_pose_activation(pred[1:7])
            LP_batch_pred.append(nd.expand_dims(pred, axis=0))

        LP_batch_pred = nd.concat(*LP_batch_pred, dim=0)

        return LP_batch_pred.asnumpy()

    def LP_pose_activation(self, data_in):
        data_out = nd.zeros(6)

        data_out[0] = data_in[0] * 1000
        data_out[1] = data_in[1] * 1000
        data_out[2] = data_in[2] * 1000
        for i in range(3):
            data = (nd.sigmoid(data_in[i+3]) - 0.5) * 2 * self.LP_r_max[i]
            data_out[i+3] = data * math.pi / 180.

        return data_out

    # -------------------- Training Main -------------------- #
    def render_and_train(self):
        print(global_variable.green)
        print('Render And Train')
        print(global_variable.reset_color)
        # -------------------- show training image # --------------------
        '''
        self.batch_size = 1
        ax = yolo_cv.init_matplotlib_figure()
        '''
        h, w = self.size
        bs = self.batch_size
        # -------------------- background -------------------- #
        self.bg_iter_valid = yolo_gluon.load_background('val', bs, h, w)
        self.bg_iter_train = yolo_gluon.load_background('train', bs, h, w)

        self.car_renderer = RenderCar(h, w, self.classes, self.ctx[0], pre_load=False)
        LP_generator = licence_plate_render.LPGenerator(h, w)

        # -------------------- main loop -------------------- #
        while True:
            if (self.backward_counter % 10 == 0 or 'bg' not in locals()):
                bg = yolo_gluon.ImageIter_next_batch(self.bg_iter_train)
                bg = bg.as_in_context(self.ctx[0])

            # -------------------- render dataset -------------------- #
            imgs, labels = self.car_renderer.render(
                bg, 'train', render_rate=0.5, pascal_rate=0.1)

            imgs, LP_labels = LP_generator.add(imgs, self.LP_r_max, add_rate=0.5)

            batch_xs = yolo_gluon.split_render_data(imgs, self.ctx)
            car_batch_ys = yolo_gluon.split_render_data(labels, self.ctx)
            LP_batch_ys = yolo_gluon.split_render_data(LP_labels, self.ctx)

            self._train_batch(batch_xs, car_batch_ys, LP_batch_ys)

            # -------------------- show training image # --------------------
            '''
            img = yolo_gluon.batch_ndimg_2_cv2img(batch_xs[0])[0]
            img = yolo_cv.cv2_add_bbox(img, car_batch_ys[0][0, 0].asnumpy(), 4, use_r=0)
            yolo_cv.matplotlib_show_img(ax, img)
            print(car_batch_ys[0][0])
            raw_input()
            '''

    def _render_thread(self):
        h, w = self.size
        bs = self.batch_size

        self.bg_iter_valid = yolo_gluon.load_background('val', bs, h, w)
        bg_iter_train = yolo_gluon.load_background('train', bs, h, w)
        LP_generator = licence_plate_render.LPGenerator(h, w)

        self.car_renderer = RenderCar(
            h, w, self.classes, self.ctx[0], pre_load=render_thread_pre_load)

        while not self.shutdown_training:
            if self.rendering_done:
                #print('render done')
                time.sleep(0.01)
                continue

            # ready to render new images
            if (self.backward_counter % 10 == 0 or 'bg' not in locals()):
                bg = yolo_gluon.ImageIter_next_batch(bg_iter_train)
                bg = bg.as_in_context(self.ctx[0])

            # change an other batch of background
            imgs, self.labels = self.car_renderer.render(
                bg, 'train', render_rate=0.5, pascal_rate=0.2)

            self.imgs, self.LP_labels = LP_generator.add(
                imgs, self.LP_r_max, add_rate=0.5)
            self.rendering_done = True

    def _train_thread(self):
        while not self.shutdown_training:
            if not self.rendering_done:
                # training images are not ready
                #print('rendering')
                time.sleep(0.01)
                continue

            batch_xs = self.imgs.copy()
            car_batch_ys = self.labels.copy()
            LP_batch_ys = self.LP_labels.copy()
            batch_xs = yolo_gluon.split_render_data(batch_xs, self.ctx)
            car_batch_ys = yolo_gluon.split_render_data(car_batch_ys, self.ctx)
            LP_batch_ys = yolo_gluon.split_render_data(LP_batch_ys, self.ctx)

            self.rendering_done = False
            self._train_batch(batch_xs, car_batch_ys, LP_batch_ys)

    def _train_batch(self, bxs, car_bys, LP_bys, car_rotate=False):
        all_gpu_loss = []
        with mxnet.autograd.record():
            for gpu_i in range(len(bxs)):
                all_gpu_loss.append([])  # new loss list for gpu_i
                ctx = self.ctx[gpu_i]  # gpu_i = GPU index

                bx = bxs[gpu_i]  # .astype('float16', copy=False)

                car_by = car_bys[gpu_i]
                LP_by = LP_bys[gpu_i]

                x, LP_x = self.net(bx)  # [x, x, x], [x]
                x = self.merge_and_slice(x, self.slice_point)  # from car_YOLO
                LP_x = self.merge_and_slice(LP_x, self.LP_slice_point)

                with mxnet.autograd.pause():

                    y, mask = self._loss_mask(car_by, gpu_i)
                    s_weight = self._score_weight(mask, ctx)
                    # y = self.fp32_2_fp16(y)
                    # s_weight, mask = self.fp32_2_fp16([s_weight, mask])

                    LP_y, LP_mask = self._loss_mask_LP(LP_by, gpu_i)
                    LP_s_weight = self._score_weight_LP(LP_mask, ctx)
                    # LP_y = self.fp32_2_fp16(LP_y)
                    # LP_s_weight, LP_mask = self.fp32_2_fp16([LP_s_weight, LP_mask])

                car_loss = self._get_loss(x, y, s_weight, mask, car_rotate=car_rotate)
                all_gpu_loss[gpu_i].extend(car_loss)

                LP_loss = self._get_loss_LP(LP_x, LP_y, LP_s_weight, LP_mask)
                all_gpu_loss[gpu_i].extend(LP_loss)

                sum(all_gpu_loss[gpu_i]).backward()

        self.trainer.step(self.batch_size)

        if self.record:
            self._record_to_tensorboard_and_save(all_gpu_loss[0])

    def _valid_iou(self):
        for pascal_rate in [1, 0]:
            iou_sum = 0
            c = 0
            for bg in self.bg_iter_valid:
                c += 1
                bg = bg.data[0].as_in_context(self.ctx[0])
                imgs, labels = self.car_renderer.render(bg, 'valid', pascal_rate=pascal_rate)

                # imgs = imgs.astype('float16', copy=False)
                x, LP_x = self.net(imgs)
                outs = self.predict(x)

                pred = nd.zeros((self.batch_size, 4))
                pred[:, 0] = outs[:, 2] - outs[:, 4] / 2
                pred[:, 1] = outs[:, 1] - outs[:, 3] / 2
                pred[:, 2] = outs[:, 2] + outs[:, 4] / 2
                pred[:, 3] = outs[:, 1] + outs[:, 3] / 2
                pred = pred.as_in_context(self.ctx[0])

                for i in range(self.batch_size):
                    label = labels[i, 0, 0:5]
                    iou_sum += yolo_gluon.get_iou(pred[i], label, mode=2)

            mean_iou = iou_sum.asnumpy() / float(self.batch_size * c)
            self.sw.add_scalar(
                'Mean_IOU',
                (self.exp + 'PASCAL %r' % pascal_rate, mean_iou),
                self.backward_counter)

            self.bg_iter_valid.reset()

    # -------------------- Validation Part -------------------- #
    def valid(self):
        print(global_variable.cyan)
        print('Valid')

        bs = 1  # batch size = 1
        h, w = self.size

        # init two matplotlib figures
        ax1 = yolo_cv.init_matplotlib_figure()
        ax2 = yolo_cv.init_matplotlib_figure()

        # init radar figure for vizualizing class distribution
        radar_prob = yolo_cv.RadarProb(self.num_class, self.classes)

        # init background, LP adder, car renderer
        BG_iter = yolo_gluon.load_background('val', bs, h, w)
        LP_generator = licence_plate_render.LPGenerator(h, w)
        car_renderer = RenderCar(h, w, self.classes, self.ctx[0], pre_load=False)

        for bg in BG_iter:
            # select background
            bg = bg.data[0].as_in_context(self.ctx[0])  # b*RGB*w*h
            # render images, type(imgs) = mxnet.ndarray
            imgs, labels = car_renderer.render(bg, 'valid', pascal_rate=0.5, render_rate=0.9)
            imgs, LP_labels = LP_generator.add(imgs, self.LP_r_max, add_rate=0.8)

            # return all_output[::-1], [LP_output]
            x1, x2, x3, LP_x = self.net.forward(is_train=False, data=imgs)
            outs = self.predict([x1, x2, x3])
            LP_outs = self.predict_LP([LP_x])

            # convert ndarray to np.array
            img = yolo_gluon.batch_ndimg_2_cv2img(imgs)[0]

            # draw licence plate border
            img, clipped_LP = LP_generator.project_rect_6d.add_edges(img, LP_outs[0, 1:])
            yolo_cv.matplotlib_show_img(ax2, clipped_LP)

            # draw car border
            img = yolo_cv.cv2_add_bbox(img, labels[0, 0].asnumpy(), 4, use_r=0)
            img = yolo_cv.cv2_add_bbox(img, outs[0], 5, use_r=0)
            yolo_cv.matplotlib_show_img(ax1, img)

            # vizualize class distribution
            radar_prob.plot(outs[0, 0], outs[0, -self.num_class:])
            raw_input('next')

    def export(self):
        yolo_gluon.export(
            self.net,
            (1, 3, self.size[0], self.size[1]),
            self.ctx[0],
            self.export_folder,
            onnx=False,
            fp16=False)


# -------------------- Main -------------------- #
if __name__ == '__main__':
    main()
