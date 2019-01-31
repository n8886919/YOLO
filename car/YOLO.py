#!/usr/bin/env python
import datetime
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
from yolo_modules import global_variable
from utils import *
from render_car import *


render_thread_pre_load = False
export_onnx = False
freiburg_path = ('/media/nolan/SSD1/YOLO_backup/'
                 'freiburg_static_cars_52_v1.1')

available_mode = ['train',
                  'valid',
                  'export',
                  'render_and_train',
                  'kmean',
                  # 'pr',
                  'valid_Nima',
                  'valid_Nima_plot'
                  ]


def main():
    args = yolo_Parser()
    yolo = YOLO(args)

    assert args.mode in available_mode, \
        'Available Modes Are {}'.format(available_mode)

    exec "yolo.%s()" % args.mode


class YOLO(object):
    def __init__(self, args):
        # -------------------- Load args -------------------- #
        self.ctx = yolo_gluon.get_ctx(args.gpu)
        self.export_folder = os.path.join(args.version, 'export')

        # -------------------- Load network sprc. -------------------- #
        spec_path = os.path.join(args.version, 'spec.yaml')
        with open(spec_path) as f:
            spec = yaml.load(f)

        for key in spec:
            setattr(self, key, spec[key])

        self.all_anchors = nd.array(self.all_anchors)
        self.num_class = len(self.classes)

        self._init_step()
        self._init_area()
        self._init_syxhw()

        self.version = args.version
        # -------------------- Load "Executor" !!! -------------------- #
        if (args.mode == 'video' or
           args.mode == 'valid' or
           args.mode == 'valid_Nima'):

            self.net = yolo_gluon.init_executor(
                self.export_folder,
                self.size, self.ctx[0],
                use_tensor_rt=args.tensorrt,
                fp16=self.use_fp16)

        # -------------------- Use gluon Block !!! -------------------- #
        elif (args.mode == 'export' or
              args.mode == 'train' or
              args.mode == 'render_and_train'):

            self.record = args.record
            self._init_net(spec, args.weight)

            if args.mode != 'export':
                self._init_train()

    # -------------------- initialization Part -------------------- #
    def _init_net(self, spec, weight):
        # (ONNX Error) Do not set num_sync_bn_devices=len(self.ctx)
        # because No conversion function for contrib_SyncBatchNorm yet.
        self.net = CarNet(spec, num_sync_bn_devices=-1)

        if self.use_fp16:
            print('cast net to FP16')
            self.net.cast('float16')

        # self.backup_dir = os.path.join(self.version, 'backup')
        self.backup_dir = os.path.join(
            '/media/nolan/SSD1/YOLO_backup/car',
            self.version,
            'backup')

        if weight is None:
            weight = yolo_gluon.get_latest_weight_from(self.backup_dir)

        yolo_gluon.init_NN(self.net, weight, self.ctx)

    def _init_step(self):
        num_downsample = len(self.layers)  # number of downsample
        num_prymaid_layers = len(self.all_anchors)  # number of pyrmaid layers
        prymaid_start = num_downsample - num_prymaid_layers + 1
        self.steps = [2**(prymaid_start+i) for i in range(num_prymaid_layers)]

    def _init_area(self):
        h = self.size[0]
        w = self.size[1]
        self.area = [int(h*w/step**2) for step in self.steps]

    def _init_syxhw(self):
        size = self.size

        n = len(self.all_anchors[0])  # anchor per sub_map
        ctx = self.ctx[0]
        self.s = nd.zeros((1, sum(self.area), n, 1), ctx=ctx)
        self.y = nd.zeros((1, sum(self.area), n, 1), ctx=ctx)
        self.x = nd.zeros((1, sum(self.area), n, 1), ctx=ctx)
        self.h = nd.zeros((1, sum(self.area), n, 1), ctx=ctx)
        self.w = nd.zeros((1, sum(self.area), n, 1), ctx=ctx)

        a_start = 0
        for i, anchors in enumerate(self.all_anchors):  # [12*16,6*8,3*4]
            a = self.area[i]
            step = self.steps[i]
            s = nd.repeat(nd.array([step], ctx=ctx), repeats=a*n)

            x_num = int(size[1]/step)
            y = nd.arange(0, size[0], step=step, repeat=n*x_num, ctx=ctx)

            x = nd.arange(0, size[1], step=step, repeat=n, ctx=ctx)
            x = nd.tile(x, int(size[0]/step))

            hw = nd.tile(self.all_anchors[i], (a, 1))
            h, w = hw.split(num_outputs=2, axis=-1)

            self.s[0, a_start:a_start+a] = s.reshape(a, n, 1)
            self.y[0, a_start:a_start+a] = y.reshape(a, n, 1)
            self.x[0, a_start:a_start+a] = x.reshape(a, n, 1)
            self.h[0, a_start:a_start+a] = h.reshape(a, n, 1)
            self.w[0, a_start:a_start+a] = w.reshape(a, n, 1)

            a_start += a

    def _init_train(self):
        self.exp = datetime.datetime.now().strftime("%m-%dx%H-%M")
        self.exp = self.exp + '_' + self.dataset
        self.batch_size *= len(self.ctx)

        print(global_variable.yellow)
        print('Training Title = {}'.format(self.exp))
        print('Batch Size = {}'.format(self.batch_size))
        print('Record Step = {}'.format(self.record_step))
        print('Step = {}'.format(self.steps))
        print('Area = {}'.format(self.area))
        '''
        for k in self.loss_name:
            print('%s%s: 10^%s%d' % (
                global_variable.blue, k, global_variable.yellow,
                math.log10(self.scale[k])))
        '''
        self.backward_counter = self.train_counter_start

        self.nd_all_anchors = [
            self.all_anchors.copyto(dev) for dev in self.ctx]

        self._get_default_ltrb()

        if self.use_fp16:
            self.nd_all_anchors = self.fp32_2_fp16(self.nd_all_anchors)
            self.all_anchors_ltrb = self.fp32_2_fp16(self.all_anchors_ltrb)

        self.HB_loss = gluon.loss.HuberLoss()
        # self.L1_loss = gluon.loss.L1Loss()
        # self.L2_loss = gluon.loss.L2Loss()
        self.LG_loss = gluon.loss.LogisticLoss(label_format='binary')
        self.CE_loss = gluon.loss.SoftmaxCrossEntropyLoss(
            from_logits=False, sparse_label=False)

        # -------------------- init trainer -------------------- #
        optimizer = mx.optimizer.create(
            'adam',
            learning_rate=self.learning_rate,
            multi_precision=self.use_fp16)

        self.trainer = gluon.Trainer(
            self.net.collect_params(),
            optimizer=optimizer)

        # -------------------- init tensorboard -------------------- #
        logdir = self.version + '/logs'
        self.sw = SummaryWriter(logdir=logdir, verbose=False)

        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def _get_default_ltrb(self):
        hw = self.size
        LTRB = []  # nd.zeros((sum(self.area),n,4))
        a_start = 0

        for i, anchors in enumerate(self.all_anchors):  # [12*16,6*8,3*4]
            n = len(self.all_anchors[i])
            a = self.area[i]
            step = float(self.steps[i])
            h, w = anchors.split(num_outputs=2, axis=-1)

            x_num = int(hw[1]/step)

            y = nd.arange(step/hw[0]/2., 1, step=step/hw[0], repeat=n*x_num)
            # [[.16, .16, .16, .16],
            #  [.50, .50, .50, .50],
            #  [.83, .83, .83, .83]]
            h = nd.tile(h.reshape(-1), a)  # same shape as y
            top = (y - 0.5*h).reshape(a, n, 1)
            bot = (y + 0.5*h).reshape(a, n, 1)

            x = nd.arange(step/hw[1]/2., 1, step=step/hw[1], repeat=n)
            # [1/8, 3/8, 5/8, 7/8]
            w = nd.tile(w.reshape(-1), int(hw[1]/step))
            left = nd.tile(x - 0.5*w, int(hw[0]/step)).reshape(a, n, 1)
            right = nd.tile(x + 0.5*w, int(hw[0]/step)).reshape(a, n, 1)

            LTRB.append(nd.concat(left, top, right, bot, dim=-1))
            a_start += a

        LTRB = nd.concat(*LTRB, dim=0)
        self.all_anchors_ltrb = [LTRB.copyto(device) for device in self.ctx]

    # -------------------- Training Main -------------------- #
    def render_and_train(self):
        print(global_variable.green)
        print('Render And Train')
        print(global_variable.reset_color)
        # -------------------- show training image # --------------------

        self.batch_size = 1
        ax = yolo_cv.init_matplotlib_figure()

        h, w = self.size
        # -------------------- background -------------------- #
        self.bg_iter_valid = yolo_gluon.load_background('val', self.batch_size, h, w)
        self.bg_iter_train = yolo_gluon.load_background('train', self.batch_size, h, w)
        self.car_renderer = RenderCar(h, w, self.classes, self.ctx[0], pre_load=False)

        # -------------------- main loop -------------------- #
        while True:
            if (self.backward_counter % 10 == 0 or 'bg' not in locals()):
                bg = yolo_gluon.ImageIter_next_batch(self.bg_iter_train)
                bg = bg.as_in_context(self.ctx[0])

            # -------------------- render dataset -------------------- #
            imgs, labels = self.car_renderer.render(
                bg, 'train', render_rate=0.5, pascal_rate=0.1)

            batch_xs = yolo_gluon.split_render_data(imgs, self.ctx)
            car_batch_ys = yolo_gluon.split_render_data(labels, self.ctx)

            self._train_batch(batch_xs, car_bys=car_batch_ys)

            # -------------------- show training image # --------------------
            if self.use_fp16:
                img = img.astype('float32')

            img = yolo_gluon.batch_ndimg_2_cv2img(batch_xs[0])[0]
            img = yolo_cv.cv2_add_bbox(img, car_batch_ys[0][0, 0].asnumpy(), 4, use_r=0)
            yolo_cv.matplotlib_show_img(ax, img)
            print(car_batch_ys[0][0])
            raw_input()

    def train(self):
        print(global_variable.cyan)
        print('Render And Train (Double Threads)')

        self.rendering_done = False
        self.training_done = True
        self.shutdown_training = False

        threading.Thread(target=self._render_thread).start()
        threading.Thread(target=self._train_thread).start()

        while not self.shutdown_training:
            try:
                time.sleep(0.01)

            except KeyboardInterrupt:
                self.shutdown_training = True

        print('Shutdown Training !!!')

    def _render_thread(self):
        h, w = self.size
        self.car_renderer = RenderCar(
            h, w, self.classes, self.ctx[0], pre_load=render_thread_pre_load)

        self.bg_iter_valid = yolo_gluon.load_background(
            'val', self.batch_size, h, w)

        bg_iter_train = yolo_gluon.load_background(
            'train', self.batch_size, h, w)

        self.labels = nd.array([0])

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
            self.imgs, self.labels = self.car_renderer.render(
                bg, 'train', render_rate=0.5, pascal_rate=0.2)

            self.rendering_done = True

    def _train_thread(self):
        while not self.shutdown_training:
            if not self.rendering_done:
                # training images are not ready
                # print('rendering')
                time.sleep(0.01)
                continue

            batch_xs = self.imgs.copy()
            car_batch_ys = self.labels.copy()

            batch_xs = yolo_gluon.split_render_data(batch_xs, self.ctx)
            car_batch_ys = yolo_gluon.split_render_data(car_batch_ys, self.ctx)

            self.rendering_done = False
            self._train_batch(batch_xs, car_batch_ys)

    def _train_batch(self, bxs, car_bys, car_rotate=False):
        '''
        Parameter:
        ----------
        bxs: list
          list of mx.ndarray, len(bxs) = len(ctx),
          shape(ndarray) = (bs, 3, self.h, self.w)
        car_bys: mxnet.ndarray
          list of mx.ndarray, len(bxs) = len(ctx),
          shape(ndarray) = (bs, num_object, num_object_label)
        car_rotate: bool
          0: do not train image rotation
          1: train image rotation

        Returns
        ----------
        None
        '''
        if self.use_fp16:
            bxs = self.fp32_2_fp16(bxs)
            car_by = self.fp32_2_fp16(car_bys)

        all_gpu_loss = []
        with mxnet.autograd.record():
            for gpu_i in range(len(bxs)):
                all_gpu_loss.append([])  # new loss list for gpu_i

                ctx = self.ctx[gpu_i]  # gpu_i = GPU index
                car_by = car_bys[gpu_i]
                bx = bxs[gpu_i]  # .astype('float16', copy=False)

                y_ = self.net(bx)
                y_ = self.merge_and_slice(y_, self.slice_point)

                with mxnet.autograd.pause():
                    y, mask = self._loss_mask(car_by, gpu_i)
                    s_weight = self._score_weight(mask, ctx)
                    if self.use_fp16:
                        y = self.fp32_2_fp16(y)
                        s_weight, mask = self.fp32_2_fp16([s_weight, mask])

                car_loss = self._get_loss(y_, y, s_weight, mask)

                all_gpu_loss[gpu_i].extend(car_loss)
                sum(all_gpu_loss[gpu_i]).backward()

        self.trainer.step(self.batch_size)

        if self.record:
            self._record_to_tensorboard_and_save(all_gpu_loss[0])

    def _find_best(self, L, gpu_index):
        gi = gpu_index
        IOUs = yolo_gluon.get_iou(self.all_anchors_ltrb[gi], L, mode=2)
        best_match = int(IOUs.reshape(-1).argmax(axis=0).asnumpy()[0])
        # print(best_pixel, best_anchor)
        best_pixel = int(best_match // len(self.all_anchors[0]))
        best_anchor = int(best_match % len(self.all_anchors[0]))
        '''
        assert best_pixel < (self.area[0] + self.area[1] + self.area[2]), (
            "best_pixel < sum(area), given {} vs {}".format(
                best_pixel, sum(self.area)))
        '''
        if best_pixel >= sum(self.area):
            best_pixel = sum(self.area) - 1

        best_ltrb = self.all_anchors_ltrb[gi][best_pixel, best_anchor]
        best_ltrb = best_ltrb.reshape(-1)

        a0 = 0
        for i, a in enumerate(self.area):
            a0 += a
            if best_pixel < a0:
                pyramid_layer = i
                break
        '''
        print('best_pixel = %d' % best_pixel)
        print('best_anchor = %d' % best_anchor)
        print('pyramid_layer = %d' % pyramid_layer)
        '''
        step = self.steps[pyramid_layer]

        by_minus_cy = L[1] - (best_ltrb[3] + best_ltrb[1]) / 2
        sigmoid_ty = by_minus_cy * self.size[0] / step + 0.5
        sigmoid_ty = nd.clip(sigmoid_ty, 0.0001, 0.9999)
        ty = yolo_gluon.nd_inv_sigmoid(sigmoid_ty)

        bx_minus_cx = L[2] - (best_ltrb[2] + best_ltrb[0]) / 2
        sigmoid_tx = bx_minus_cx*self.size[1]/step + 0.5
        sigmoid_tx = nd.clip(sigmoid_tx, 0.0001, 0.9999)
        tx = yolo_gluon.nd_inv_sigmoid(sigmoid_tx)

        bh = (L[3]) / self.nd_all_anchors[gi][pyramid_layer, best_anchor, 0]
        th = nd.log(bh)

        bw = (L[4]) / self.nd_all_anchors[gi][pyramid_layer, best_anchor, 1]
        tw = nd.log(bw)

        return best_pixel, best_anchor, nd.concat(ty, tx, th, tw, dim=-1)

    def _loss_mask(self, label_batch, gpu_index):
        """Generate training targets given predictions and label_batch.
        label_batch: bs*object*[class, cent_y, cent_x, box_h, box_w, rotate]
        """
        bs = label_batch.shape[0]
        a = sum(self.area)
        n = len(self.all_anchors[0])
        ctx = self.ctx[gpu_index]

        C_mask = nd.zeros((bs, a, n, 1), ctx=ctx)
        C_score = nd.zeros((bs, a, n, 1), ctx=ctx)
        C_box_yx = nd.zeros((bs, a, n, 2), ctx=ctx)
        C_box_hw = nd.zeros((bs, a, n, 2), ctx=ctx)
        C_rotate = nd.zeros((bs, a, n, 1), ctx=ctx)
        C_class = nd.zeros((bs, a, n, self.num_class), ctx=ctx)

        for b in range(bs):
            for L in label_batch[b]:  # all object in the image
                if L[0] < 0:
                    continue
                else:
                    px, anc, box = self._find_best(L, gpu_index)
                    C_mask[b, px, anc, :] = 1.0  # others are zero
                    C_score[b, px, anc, :] = 1.0  # others are zero
                    C_box_yx[b, px, anc, :] = box[:2]
                    C_box_hw[b, px, anc, :] = box[2:]
                    C_rotate[b, px, anc, :] = L[5]

                    C_class[b, px, anc, :] = L[6:]

        return [C_score, C_box_yx, C_box_hw, C_rotate, C_class], C_mask

    def _score_weight(self, mask, ctx):
        n = self.negative_weight
        p = self.positive_weight

        ones = nd.ones_like(mask)
        score_weight = nd.where(mask > 0, ones*p, ones*n, ctx=ctx)

        return score_weight

    def _get_loss(self, x, y, s_weight, mask, car_rotate=False):
        rotate_lr = self.scale['rotate'] if car_rotate else 0
        s = self.LG_loss(x[0], y[0], s_weight * self.scale['score'])
        yx = self.HB_loss(x[1], y[1], mask * self.scale['box_yx'])  # L2
        hw = self.HB_loss(x[2], y[2], mask * self.scale['box_hw'])  # L1
        r = self.HB_loss(x[3], y[3], mask * rotate_lr)  # L1
        c = self.CE_loss(x[4], y[4], mask * self.scale['class'])
        return (s, yx, hw, r, c)

    # -------------------- Tensor Board -------------------- #
    def _valid_iou(self):
        for pascal_rate in [1, 0]:
            iou_sum = 0
            c = 0
            for bg in self.bg_iter_valid:
                c += 1
                bg = bg.data[0].as_in_context(self.ctx[0])
                imgs, labels = self.car_renderer.render(
                    bg, 'valid', pascal_rate=pascal_rate)

                if self.use_fp16:
                    imgs = self.fp32_2_fp16([imgs])

                x = self.net(imgs[0])
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

    def _record_to_tensorboard_and_save(self, loss):
        self.backward_counter += 1  # do not save at first step
        if self.backward_counter % 10 == 0:
            yolo_gluon.record_loss(
                loss, self.loss_name, self.sw,
                step=self.backward_counter, exp=self.exp)

        if self.backward_counter % self.valid_step == 0:
            self._valid_iou()

        if self.backward_counter % self.record_step == 0:
            idx = self.backward_counter // self.record_step
            path = os.path.join(self.backup_dir, self.exp + '_%d' % idx)
            self.net.collect_params().save(path)

    # -------------------- Validation Part -------------------- #
    def _yxhw_to_ltrb(self, yxhw):
        ty, tx, th, tw = yxhw.split(num_outputs=4, axis=-1)
        by = (nd.sigmoid(ty)*self.s + self.y) / self.size[0]
        bx = (nd.sigmoid(tx)*self.s + self.x) / self.size[1]

        bh = nd.exp(th) * self.h
        bw = nd.exp(tw) * self.w

        bh2 = bh / 2
        bw2 = bw / 2
        l = bx - bw2
        r = bx + bw2
        t = by - bh2
        b = by + bh2
        return nd.concat(l, t, r, b, dim=-1)

    def predict(self, batch_out):
        if self.use_fp16:
            batch_out = self.fp16_2_fp32(batch_out)

        batch_out = self.merge_and_slice(batch_out, self.slice_point)

        batch_score = nd.sigmoid(batch_out[0])
        batch_box = nd.concat(batch_out[1], batch_out[2], dim=-1)
        batch_box = self._yxhw_to_ltrb(batch_box)
        batch_out = nd.concat(batch_score, batch_box,
                              batch_out[3], batch_out[4], dim=-1)

        batch_out = nd.split(batch_out, axis=0, num_outputs=len(batch_out))

        batch_pred = []
        for i, out in enumerate(batch_out):
            best_anchor_index = batch_score[i].reshape(-1).argmax(axis=0)
            out = out.reshape((-1, 6+self.num_class))

            pred = out[best_anchor_index][0]  # best out
            y = (pred[2] + pred[4]) / 2
            x = (pred[1] + pred[3]) / 2
            h = (pred[4] - pred[2])
            w = (pred[3] - pred[1])
            pred[1:5] = nd.concat(y, x, h, w, dim=-1)
            batch_pred.append(nd.expand_dims(pred, axis=0))

        batch_pred = nd.concat(*batch_pred, dim=0)

        return batch_pred.asnumpy()

    def kmean(self):
        import yolo_modules.iou_kmeans as kmeans
        print(global_variable.cyan)
        print('KMeans Get Default Anchors')
        bs = 1000
        k = 9
        h, w = self.size
        car_renderer = RenderCar(
            h, w, self.classes,
            self.ctx[0], pre_load=False)

        labels = nd.zeros((bs, 2))
        for i in range(bs):
            bg = nd.zeros((1, 3, h, w), ctx=self.ctx[0])  # b*RGB*h*w
            img, label = car_renderer.render(
                bg, 'train', pascal_rate=0.2, render_rate=1.0)

            labels[i] = label[0, 0, 3:5]

            if i % (bs/10) == 0:
                print(i/float(bs))

        ans = kmeans.main(labels, k, dis_method='iou')

        all_means_cent = [0, 0]
        for a in ans:

            a = a.asnumpy()
            all_means_cent[0] += a[0]
            all_means_cent[1] += a[1]
            # anchor areas, for sort
            print('[h, w] = [%.4f, %.4f], area = %.2f' % (
                a[0], a[1], a[0]*a[1]))

        all_means_cent[0] /= float(k)
        all_means_cent[1] /= float(k)
        print('all_means_cent = {}'.format(all_means_cent))

        while 1:
            time.sleep(0.1)

    def valid(self):
        print(global_variable.cyan)
        print('Valid')

        bs = 1
        h, w = self.size
        ax1 = yolo_cv.init_matplotlib_figure()
        radar_prob = yolo_cv.RadarProb(self.num_class, self.classes)

        BG_iter = yolo_gluon.load_background('val', bs, h, w)
        car_renderer = RenderCar(h, w, self.classes,
                                 self.ctx[0], pre_load=False)

        for bg in BG_iter:
            # -------------------- get image -------------------- #
            bg = bg.data[0].as_in_context(self.ctx[0])  # b*RGB*w*h
            imgs, labels = car_renderer.render(
                bg, 'valid', pascal_rate=0.5, render_rate=0.9)

            # -------------------- predict -------------------- #
            x1, x2, x3 = self.net.forward(is_train=False, data=imgs)
            outs = self.predict([x1, x2, x3])

            # -------------------- show -------------------- #
            img = yolo_gluon.batch_ndimg_2_cv2img(imgs)[0]

            img = yolo_cv.cv2_add_bbox(img, labels[0, 0].asnumpy(), 4, use_r=0)
            img = yolo_cv.cv2_add_bbox(img, outs[0], 5, use_r=0)
            yolo_cv.matplotlib_show_img(ax1, img)

            radar_prob.plot3d(outs[0, 0], outs[0, -self.num_class:])
            raw_input('next')

    def export(self):
        yolo_gluon.export(
            self.net,
            (1, 3, self.size[0], self.size[1]),
            self.ctx[0],
            self.export_folder,
            onnx=export_onnx,
            fp16=self.use_fp16)

    # -------------------- utils Part -------------------- #
    def merge_and_slice(self, all_output, points):
        output = nd.concat(*all_output, dim=1)
        i = 0
        x = []
        for pt in points:
            y = output.slice_axis(begin=i, end=pt, axis=-1)
            x.append(y)
            i = pt
        return x

    def fp32_2_fp16(self, x):
        for i, data in enumerate(x):
            assert type(data) == mx.ndarray.ndarray.NDArray, (
                'list  element should be mxnet.ndarray')
            x[i] = x[i].astype('float16', copy=False)
        return x

    def fp16_2_fp32(self, x):
        for i, data in enumerate(x):
            assert type(data) == mx.ndarray.ndarray.NDArray, (
                'list  element should be mxnet.ndarray')
            x[i] = x[i].astype('float32', copy=False)
        return x

    def valid_Nima(self):
        '''
        compare with "Unsupervised Generation of a Viewpoint"
        Annotated Car Dataset from Videos
        -path: /media/nolan/SSD1/YOLO_backup/freiburg_static_cars_52_v1.1
          |-annotations (d)
            |-txt_path (d)
          |-all car folders (d)
          |-result_path (v)
            |-all car folders with box (v)
              |-img_path (v)
            |-annotations (v)
              |-save_txt_path (v)
              |-plot (p)
                |-all car azi compare (p)

        d: download from "Unsupervised Generation of a Viewpoint"
        v: generate from valid_Nima
        o: generate from valid_Nima_plot
        '''
        import cv2

        image_w = 960.
        image_h = 540.

        mx_resize = mxnet.image.ForceResizeAug(tuple(self.size[::-1]))
        radar_prob = yolo_cv.RadarProb(self.num_class, self.classes)

        x = np.arange(53)
        x = np.delete(x, [0, 6, 20, 23, 31, 36])
        for i in x:
            result_path = os.path.join(freiburg_path, "result_%s" % self.version)
            save_img_path = os.path.join(result_path, "car_%d" % i)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)

            save_txt_path = os.path.join(result_path, 'annotations')
            if not os.path.exists(save_txt_path):
                os.makedirs(save_txt_path)

            txt_path = os.path.join(freiburg_path, "annotations", "%d_annot.txt" % i)
            with open(txt_path, 'r') as f:
                all_lines = f.readlines()

            for line_i, line in enumerate(all_lines):
                img_name = line.split('\t')[0].split('.')[0] + '.png'
                img_path = os.path.join(freiburg_path, img_name)
                save_img = os.path.join(save_img_path, img_name.split('/')[-1])
                img = cv2.imread(img_path)

                # -------------------- Prediction -------------------- #
                nd_img = yolo_gluon.cv_img_2_ndarray(img, self.ctx[0], mxnet_resize=mx_resize)
                net_out = self.net.forward(is_train=False, data=nd_img)
                pred_car = self.predict(net_out[:3])
                pred_car[0, 5] = -1  # (1, 80)
                Cout = pred_car[0]

                left_ = (Cout[2] - 0.5*Cout[4]) * image_w
                up_ = (Cout[1] - 0.5*Cout[3]) * image_h
                right_ = (Cout[2] + 0.5*Cout[4]) * image_w
                down_ = (Cout[1] + 0.5*Cout[3]) * image_h

                vec_ang, vec_rad, prob = radar_prob.cls2ang(Cout[0], Cout[-self.num_class:])

                # -------------------- Ground Truth -------------------- #
                box_para = np.fromstring(line.split('\t')[1], dtype='float32', sep=' ')
                left, up, right, down = box_para
                h = (down - up) / image_h
                w = (right - left) / image_w
                y = (down + up) / 2 / image_h
                x = (right + left) / 2 / image_w
                boxA = [0, y, x, h, w, 0]

                azi_label = int(line.split('\t')[2]) - 90
                azi_label = azi_label - 360 if azi_label > 180 else azi_label

                inter = (min(right, right_)-max(left, left_)) * (min(down, down_)-max(up, up_))
                a1 = (right-left)*(down-up)
                a2 = (right_-left_)*(down_-up_)
                iou = (inter) / (a1 + a2 - inter)

                save_txt = os.path.join(save_txt_path, '%d_annot' % i)
                with open(save_txt, 'a') as f:
                    f.write('%s %f %f %f %f\n' % (img_name, iou, azi_label, vec_ang*180/math.pi, vec_rad))

                yolo_cv.cv2_add_bbox(img, Cout, 3, use_r=False)
                yolo_cv.cv2_add_bbox(img, boxA, 4, use_r=False)

                cv2.imshow('img', img)
                cv2.waitKey(1)
                cv2.imwrite(save_img, img)
                #a = raw_input()

    def valid_Nima_plot(self):
        import matplotlib.pyplot as plt
        filter_index = [14, 17]
        path = (freiburg_path + '/result_v2/annotations')
        plot_path = os.path.join(path, 'plot')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        all_car_iou = []
        all_car_azi = []
        for annot in os.listdir(path):  # all car
            if annot == 'plot':
                continue

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            num = annot.split('_')[0]
            if num in filter_index:
                continue

            ious, err_squares = [], []
            x1s, x2s = [], []

            path2 = os.path.join(path, annot)
            with open(path2, 'r') as f:
                lines = f.readlines()

            for line in lines:  # all images of the car
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

            save_path = os.path.join(plot_path, num + '.png')
            fig.savefig(save_path)

            iou = sum(ious) / float(len(ious))
            all_car_iou.append(iou)
            azi_L2_err = math.sqrt(sum(err_squares) / float(len(err_squares)))
            all_car_azi.append(azi_L2_err)
            print('car' + num + ', iou: %f, mean azi: %f' % (iou, azi_L2_err))
            #break

        print(sum(all_car_iou)/float(len(all_car_iou)))
        print(sum(all_car_azi)/float(len(all_car_azi)))


'''
class Benchmark():
    def __init__(self, logdir, car_renderer):
        self.logdir = logdir
        self.car_renderer = car_renderer
        self.iou_step = 0

    def pr_curve(self):

        from mxboard import SummaryWriter
        sw = SummaryWriter(logdir=NN + '/logs/PR_Curve', flush_secs=5)

        path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/'
        BG_iter = image.ImageIter(100, (3, self.size[0], self.size[1]),
            path_imgrec=path+'sun2012_train.rec',
            path_imgidx=path+'sun2012_train.idx',
            shuffle=True, pca_noise=0,
            brightness=0.5, saturation=0.5, contrast=0.5, hue=1.0,
            rand_crop=True, rand_resize=True, rand_mirror=True, inter_method=10)

        car_renderer = RenderCar(100, self.size[0], self.size[1], ctx[0])
        predictions = [] #[[]*24,[]*24,[]*24],............]
        labels = [] #[1,5,25,3,5,12,22,.............]

        for BG in BG_iter:
            BG = BG.data[0].as_in_context(ctx[0])
            img_batch, label_batch = car_renderer.render_pascal(BG, 'valid')

            C_pred = self.get_feature(img_batch)


            for i in range(100):
                C_score = C_pred[0][i]
                C_1 = C_score.reshape(-1).argmax(axis=0).reshape(-1)

                Cout = C_pred[3][i].reshape((-1, self.num_class))
                Cout = softmax(Cout[C_1][0].asnumpy())

                predictions.append(Cout)
                labels.append(int(label_batch[i,0,0].asnumpy()))

            if len(labels)>(3000-1): break

        labels = np.array(labels)
        predictions = np.array(predictions)


        for i in range(self.num_class):
            if i == 0:
                j = 23
                k = 1
            elif i == 23:
                j = 22
                k = 0
            else:
                j = i - 1
                k = i + 1
            label = ((labels==i)+(labels==j)+(labels==k)).astype(int)

            predict = predictions[:,i] + predictions[:,j] + predictions[:,k]
            label = nd.array(label)
            predict = nd.array(predict)

            sw.add_pr_curve('%d'%i, label, predict, 100, global_step=0)

        predictions = nd.uniform(low=0, high=1, shape=(100,), dtype=np.float32)
        labels = nd.uniform(low=0, high=2, shape=(100,), dtype=np.float32).astype(np.int32)
        print(labels)
        print(predictions)
        sw1.add_pr_curve(tag='pseudo_pr_curve', predictions=predictions, labels=labels, num_thresholds=120)
'''

# -------------------- Main -------------------- #
if __name__ == '__main__':
    main()
