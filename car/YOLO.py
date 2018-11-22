#!/usr/bin/env python
import datetime
import glob
import sys
import threading
import time
import yaml

import mxnet as mx
from mxnet import gluon
from mxboard import SummaryWriter

# self define modules
from yolo_modules import yolo_gluon
from yolo_modules import yolo_cv
from yolo_modules import licence_plate_render
from yolo_modules import global_variable


from utils import *
from car_and_LP3.utils import yolo_Parser
from car_and_LP3.YOLO import YOLO as car_and_LP3_YOLO
from render_car import *

#os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
#os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


def main():
    args = yolo_Parser()
    yolo = YOLO(args)

    if args.mode == 'train':
        yolo.render_and_train2()

    elif args.mode == 'valid':
        yolo.valid()

    elif args.mode == 'export':
        yolo.export()

    elif args.mode == 'kmean':
        yolo.get_default_anchors()

    elif args.mode == 'PR':
        yolo.pr_curve()

    else:
        print('args 2 should be train or valid')


class YOLO(car_and_LP3_YOLO):
    def __init__(self, args):
        super(YOLO, self).__init__(args)
        self.car_threshold = 0.5
        self.LP_threshold = 1.0
        self.export_file = '../car/v2/export/YOLO_export'

        if args.mode == 'export':
            self.net = CarLPNet(self.spec, num_sync_bn_devices=len(self.ctx))
            yolo_gluon.init_NN(
            self.net, '/home/nolan/Desktop/YOLO/car/v2/backup/iter_107000', self.ctx)
        elif args.mode == 'valid':
            self._init_executor(use_tensor_rt=args.tensorrt)

        
    def predict(self, x, LP=False, bind=0):
        if not bind:
            batch_out, LP_batch_out = self.net(x)

        else:
            batch_out = self.executor.forward(is_train=False, data=x)
        
        batch_score = nd.sigmoid(batch_out[0])
        batch_box = self._yxhw_to_ltrb(batch_out[1])
        batch_out = nd.concat(
            batch_score, batch_box, batch_out[2], batch_out[3], dim=-1)
        # (1L, 840L, 3L, 30L)

        batch_out = nd.split(batch_out, axis=0, num_outputs=len(batch_out))

        batch_pred = []
        for i, out in enumerate(batch_out):
            best_anchor_index = batch_score[i].reshape(-1).argmax(axis=0)
            out = out.reshape((-1, 6+self.num_class))

            pred = out[best_anchor_index][0]  # best out
            y = (pred[2] + pred[4])/2
            x = (pred[1] + pred[3])/2
            h = (pred[4] - pred[2])
            w = (pred[3] - pred[1])
            pred[1:5] = nd.concat(y, x, h, w, dim=-1)
            batch_pred.append(nd.expand_dims(pred, axis=0))

        batch_pred = nd.concat(*batch_pred, dim=0)
        return batch_pred.asnumpy(), nd.zeros((1, 7)).asnumpy()

    def valid(self):
        print(global_variable.cyan)
        print('Valid')

        bs = 1
        h, w = self.size
        ax1 = yolo_cv.init_matplotlib_figure()
        #ax2 = yolo_cv.init_matplotlib_figure()
        #radar_prob = yolo_cv.RadarProb(self.num_class, self.classes)

        BG_iter = yolo_gluon.load_background('val', bs, h, w)
        LP_generator = licence_plate_render.LPGenerator(h, w)
        car_renderer = RenderCar(1, h, w, self.ctx[0])

        for bg in BG_iter:
            bg = bg.data[0].as_in_context(self.ctx[0])  # b*RGB*w*h

            imgs, labels = car_renderer.render(bg, 'valid')

            outs = self.predict(imgs, bind=1)
            # outs[car or LP][batch]
            img = yolo_gluon.batch_ndimg_2_cv2img(imgs)[0]

            #img = yolo_cv.cv2_add_bbox(img, labels[0, 0].asnumpy(), 4, use_r=1)  # Green
            img = yolo_cv.cv2_add_bbox(img, outs[0][0], 5, use_r=1)  # Red box
            #radar_prob.plot(outs[0][0, 0], outs[0][0, -self.num_class:])

            yolo_cv.matplotlib_show_img(ax1, img)
            raw_input('next')

# -------------------- Main -------------------- #
if __name__ == '__main__':
    main()
