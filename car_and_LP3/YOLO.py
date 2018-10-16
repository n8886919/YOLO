#!/usr/bin/env python
from __future__ import print_function
import glob
import sys
import yaml
import datetime

from mxnet import gluon
from mxboard import SummaryWriter

from utils import *
from render_car import *

sys.path.append('../')

from modules.utils_gluon import *
from modules.utils_cv import *
from modules.licence_plate_render import *

# -------------------- Global variables -------------------- #
args = Parser()

if args.mode == 'train':
    ctx = [gpu(int(i)) for i in args.gpu]
else:
    ctx = [gpu(int(args.gpu[0]))]
# -------------------- Global variables # -------------------- #
#scale = {'score': 0.1, 'rotate': 10.0, 'class': 0.01, 'box': 1.0}
scale = {'score': 0.1, 'rotate': 10.0, 'class': 0.1, 'box': 1.0}
label_mode = ['Dense', 'Sparse']
exp = datetime.datetime.now().strftime("%m-%dx%H-%M") + '_100' + label_mode[1]
train_counter = 18010


def main():
    yolo = YOLO()

    if args.mode == 'train':
        yolo.render_and_train()

    elif args.mode == 'valid':
        yolo.valid()

    elif args.mode == 'video':
        yolo.run(ctx=ctx[0], topic=args.topic, radar=args.radar, show=args.show)

    elif args.mode == 'kmean':
        yolo.get_default_anchors()

    elif args.mode == 'PR':
        yolo.pr_curve()

    else:
        print('Parser 2 should be train or valid or video')


class YOLO(Video):
    def __init__(self, pretrain=None):
        with open(os.path.join(args.version, 'spec.yaml')) as f:
            spec = yaml.load(f)

        self.net = CarLPNet(spec, num_sync_bn_devices=len(ctx))

        self.size = spec['size']
        self.num_class = len(spec['classes'])
        self.classes = spec['classes']
        self.all_anchors = nd.array(spec['anchors'])
        self.record_step = spec['record_step']
        self.batch_size = spec['batch_size'] * len(ctx)
        # ------------------------------------------------------------ #

        #self.loss_name = ['score', 'box', 'class']
        self.loss_name = ['score', 'box', 'rotate', 'class']
        # ------------------------------------------------------------ #

        num_downsample = len(spec['layers'])  # number of downsample
        num_prymaid_layers = len(self.all_anchors)  # number of pyrmaid layers
        prymaid_start = num_downsample - num_prymaid_layers + 1

        self.steps = [2**(prymaid_start+i) for i in range(num_prymaid_layers)]
        h = self.size[0]
        w = self.size[1]
        self.area = [int(h*w/step**2) for step in self.steps]
        print('\033[1;33m')
        print(exp)
        print('Device = {}'.format(ctx))
        print('scale = {}'.format(scale))
        print('Loss = {}'.format(self.loss_name))
        print('Step = {}'.format(self.steps))
        print('Area = {}'.format(self.area))

        self.backup_dir = os.path.join(args.version, 'backup')
        backup_list = glob.glob(self.backup_dir+'/*')


        if pretrain is None:
            if len(backup_list) != 0:
                print('Use latest weight')
                pretrain = max(backup_list, key=os.path.getctime)
            else:
                pretrain = 'no pretrain weight'

        init_NN(self.net, pretrain, ctx)
        self.pretrain = pretrain

        self._init_valid()
        if args.mode == 'train':
            print('\033[1;33mBatch Size = {}'.format(self.batch_size))
            print('Record Step = {}\033[0m'.format(self.record_step))
            self._init_train()

    def _init_train(self):

        self.backward_counter = train_counter

        self.nd_all_anchors = [self.all_anchors.copyto(dev) for dev in ctx]
        self.get_default_ltrb()

        self.L1_loss = gluon.loss.L1Loss()
        self.L2_loss = gluon.loss.L2Loss()
        self.LG_loss = gluon.loss.LogisticLoss(label_format='binary')
        self.CE_loss = gluon.loss.SoftmaxCrossEntropyLoss(
            from_logits=False, sparse_label=False)

        self.trainer = gluon.Trainer(
            self.net.collect_params(),
            'adam',
            {'learning_rate': 0.0001})

        logdir = args.version+'/logs'
        self.sw = SummaryWriter(logdir=logdir, verbose=False)
        self.sw.add_text(tag=logdir, text=exp)
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def get_default_ltrb(self):
        LTRB = []  # nd.zeros((sum(self.area),n,4))
        size = self.size
        a_start = 0
        for i, anchors in enumerate(self.all_anchors):  # [12*16,6*8,3*4]
            n = len(self.all_anchors[i])
            a = self.area[i]
            step = float(self.steps[i])
            h, w = anchors.split(num_outputs=2, axis=-1)

            x_num = int(size[1]/step)

            y = nd.arange(step/size[0]/2., 1, step=step/size[0], repeat=n*x_num)
            # [[.16, .16, .16, .16],
            #  [.50, .50, .50, .50],
            #  [.83, .83, .83, .83]]
            h = nd.tile(h.reshape(-1), a)  # same shape as y
            t = (y - 0.5*h).reshape(a, n, 1)
            b = (y + 0.5*h).reshape(a, n, 1)

            x = nd.arange(step/size[1]/2., 1, step=step/size[1], repeat=n)
            # [1/8, 3/8, 5/8, 7/8]
            w = nd.tile(w.reshape(-1), int(size[1]/step))
            l = nd.tile(x - 0.5*w, int(size[0]/step)).reshape(a, n, 1)
            r = nd.tile(x + 0.5*w, int(size[0]/step)).reshape(a, n, 1)

            LTRB.append(nd.concat(l, t, r, b, dim=-1))
            #LTRB[a_start:a_start+a] = nd.concat(l,t,r,b, dim=-1)
            a_start += a

        LTRB = nd.concat(*LTRB, dim=0)
        self.all_anchors_ltrb = [LTRB.copyto(device) for device in ctx]

    def find_best(self, L, gpu_index):
        anc_ltrb = self.all_anchors_ltrb[gpu_index][:]
        IOUs = get_iou(anc_ltrb, L, mode=2)
        best_match = int(IOUs.reshape(-1).argmax(axis=0).asnumpy()[0])
        # print(best_match)
        best_pixel = int(best_match // len(self.all_anchors[0]))
        best_anchor = int(best_match % len(self.all_anchors[0]))

        best_ltrb = self.all_anchors_ltrb[gpu_index][best_pixel, best_anchor]
        best_ltrb = best_ltrb.reshape(-1)

        assert best_pixel < self.area[0] + self.area[1] + self.area[2], (
            "best_pixel < sum(area), given {} vs {}".format(
                best_pixel, sum(self.area)))

        a0 = 0
        for i, a in enumerate(self.area):
            a0 += a
            if best_pixel < a0:
                pyramid_layer = i
                break
        '''
        print('best_pixel = %d'%best_pixel)
        print('best_anchor = %d'%best_anchor)
        print('pyramid_layer = %d'%pyramid_layer)
        '''
        step = self.steps[pyramid_layer]

        by_minus_cy = L[1] - (best_ltrb[3] + best_ltrb[1]) / 2
        sigmoid_ty = by_minus_cy * self.size[0] / step + 0.5
        sigmoid_ty = nd.clip(sigmoid_ty, 0.0001, 0.9999)
        ty = nd_inv_sigmoid(sigmoid_ty)

        bx_minus_cx = L[2] - (best_ltrb[2] + best_ltrb[0]) / 2
        sigmoid_tx = bx_minus_cx*self.size[1]/step + 0.5
        sigmoid_tx = nd.clip(sigmoid_tx, 0.0001, 0.9999)
        tx = nd_inv_sigmoid(sigmoid_tx)
        th = nd.log((L[3]) / self.nd_all_anchors[gpu_index][pyramid_layer, best_anchor, 0])
        tw = nd.log((L[4]) / self.nd_all_anchors[gpu_index][pyramid_layer, best_anchor, 1])
        return best_pixel, best_anchor, nd.concat(ty, tx, th, tw, dim=-1)

    def loss_mask(self, label_batch, gpu_index):
        """Generate training targets given predictions and label_batch.
        label_batch: bs*object*[class, cent_y, cent_x, box_h, box_w, rotate]
        """
        bs = label_batch.shape[0]
        a = sum(self.area)
        n = len(self.all_anchors[0])

        C_class = nd.zeros((bs, a, n, self.num_class), ctx=ctx[gpu_index])
        C_score = nd.zeros((bs, a, n, 1), ctx=ctx[gpu_index])
        C_rotate = nd.zeros((bs, a, n, 1), ctx=ctx[gpu_index])
        C_box = nd.zeros((bs, a, n, 4), ctx=ctx[gpu_index])
        C_mask = nd.zeros((bs, a, n, 1), ctx=ctx[gpu_index])

        for b in range(bs):
            for L in label_batch[b]:  # all object in the image
                if L[0] < 0:
                    continue
                else:
                    px, anc, box = self.find_best(L, gpu_index)
                    C_mask[b, px, anc, :] = 1.0  # others are zero
                    C_score[b, px, anc, :] = 1.0  # others are zero
                    C_box[b, px, anc, :] = box
                    C_rotate[b, px, anc, :] = L[5]
                    if 'Sparse' in exp:
                        a = nd.argmax(L[6:], axis=0)
                        C_class[b, px, anc, a] = 1

                    elif 'Dense' in exp:
                        C_class[b, px, anc, :] = L[6:]

                    else:
                        print('Dense Or Sparse?')

        return [C_score, C_box, C_rotate, C_class], C_mask

    def train_the(self, batch_xs, batch_ys, rotate_lr=scale['rotate']):

        all_gpu_loss = []
        with mxnet.autograd.record():
            for gpu_i, (bx, by) in enumerate(zip(batch_xs, batch_ys)):
                all_gpu_loss.append([])

                x, L_pred = self.net(bx)
                with mxnet.autograd.pause():
                    y, mask = self.loss_mask(by, gpu_i)

                    score_weight = nd.where(
                        mask > 0,
                        nd.ones_like(mask)*10.0,
                        nd.ones_like(mask)*0.1,
                        ctx=ctx[gpu_i])

                if 'score' in self.loss_name:
                    s = self.LG_loss(x[0], y[0], score_weight * scale['score'])
                    all_gpu_loss[gpu_i].append(s)

                if 'box' in self.loss_name:
                    b = self.L2_loss(x[1], y[1], mask * scale['box'])
                    all_gpu_loss[gpu_i].append(b)

                if 'rotate' in self.loss_name:
                    r = self.L2_loss(x[2], y[2], mask * rotate_lr)
                    all_gpu_loss[gpu_i].append(r)

                if 'class' in self.loss_name:
                    c = self.CE_loss(x[3], y[3], mask * scale['class'])
                    all_gpu_loss[gpu_i].append(c)

                sum(all_gpu_loss[gpu_i]).backward()

        self.trainer.step(self.batch_size)
        self.record_to_tensorboard_and_save(all_gpu_loss[0])

    def render_and_train(self):
        print('\033[1;33;40mRender And Train\033[0m')
        # -------------------- show training image # --------------------
        '''
        self.batch_size = 1
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        '''
        # -------------------- load data -------------------- #
        h, w = self.size

        # -------------------- valid -------------------- #
        self.iou_bs = 30
        self.bg_iter_valid = load_background('val', self.iou_bs, h, w)

        # -------------------- train -------------------- #
        self.bg_iter_train = load_background('train', self.batch_size, h, w)
        self.car_renderer = RenderCar(h, w, self.classes, ctx[0])
        '''addLP = AddLP(h, w, self.num_class)'''

        # -------------------- main loop -------------------- #
        self.time_recorder = np.zeros(5)
        while True:
            t = time.time()
            #print('\r%d'%self.backward_counter, end='')
            if self.backward_counter % 10 == 0:
                bg = ImageIter_next_batch(self.bg_iter_train)
                bg = bg.as_in_context(ctx[0])
            self.time_recorder[0] += (time.time() - t)
            # -------------------- render dataset -------------------- #
            imgs, labels = self.car_renderer.render(bg, 'train', render_rate=0.5, pascal=False)
            batch_xs, batch_ys = split_render_data(imgs, labels, ctx)

            '''if np.random.rand() > 0.2:
                batch_x[0], batch_y[0] = addLP.add(batch_x[0], batch_y[0])'''
            self.time_recorder[1] += (time.time() - t)
            # -------------------- training -------------------- #
            for _ in range(1):
                self.train_the(batch_xs, batch_ys)
            self.time_recorder[2] += (time.time() - t)
            # -------------------- render dataset -------------------- #
            imgs, labels = self.car_renderer.render(bg, 'train', render_rate=0.5, pascal=True)
            batch_xs, batch_ys = split_render_data(imgs, labels, ctx)
            self.time_recorder[3] += (time.time() - t)
            # -------------------- training -------------------- #
            for _ in range(1):
                self.train_the(batch_xs, batch_ys, rotate_lr=0.)
            self.time_recorder[4] += (time.time() - t)

            # -------------------- show training image # --------------------
            '''
            img = batch_ndimg_2_cv2img(batch_xs[0])[0]
            img = cv2_add_bbox(img, batch_ys[0][0, 0].asnumpy(), [0, 1, 0])
            ax.imshow(img)
            #print(batch_ys[0][0])
            raw_input()
            '''

    def valid_iou(self):
        for pascal in [True, False]:
            iou_sum = 0
            c = 0
            for bg in self.bg_iter_valid:
                c += 1
                bg = bg.data[0].as_in_context(ctx[0])
                imgs, labels = self.car_renderer.render(bg, 'valid', pascal=pascal)
                outs = self.predict(imgs)

                pred = nd.zeros((self.iou_bs, 4))
                pred[:, 0] = outs[:, 2] - outs[:, 4] / 2
                pred[:, 1] = outs[:, 1] - outs[:, 3] / 2
                pred[:, 2] = outs[:, 2] + outs[:, 4] / 2
                pred[:, 3] = outs[:, 1] + outs[:, 3] / 2
                pred = pred.as_in_context(ctx[0])

                for i in range(self.iou_bs):
                    iou_sum += get_iou(pred[i], labels[i, 0, 0:5], mode=2)

            mean_iou = iou_sum.asnumpy() / float(self.iou_bs * c)
            self.sw.add_scalar(
                'Mean_IOU',
                (exp + 'PASCAL %r' % pascal , mean_iou),
                self.backward_counter)

            self.bg_iter_valid.reset()

    def record_to_tensorboard_and_save(self, loss):
        for i, L in enumerate(loss):
            loss_name = self.loss_name[i]
            self.sw.add_scalar(
                exp + 'Scaled_Loss',
                (loss_name, nd.mean(L).asnumpy()),
                self.backward_counter)

            self.sw.add_scalar(
                loss_name,
                (exp, nd.mean(L).asnumpy()/scale[loss_name]),
                self.backward_counter)

        if self.backward_counter % 1000 == 0:
            self.valid_iou()

            for i, L in enumerate(self.time_recorder):
                self.sw.add_scalar('time', (str(i), L/3600.), self.backward_counter)

        self.backward_counter += 1

        if self.backward_counter % self.record_step == 0:
            idx = self.backward_counter//self.record_step
            save_model = os.path.join(self.backup_dir, exp + 'iter' + '_%d' % idx)
            self.net.collect_params().save(save_model)

    def _init_valid(self):
        size = self.size
        n = len(self.all_anchors[0])  # anchor per sub_map
        self.s = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        self.y = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        self.x = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        self.h = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        self.w = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])

        a_start = 0
        for i, anchors in enumerate(self.all_anchors):  # [12*16,6*8,3*4]
            a = self.area[i]
            step = self.steps[i]
            s = nd.repeat(nd.array([step], ctx=ctx[0]), repeats=a*n)

            x_num = int(size[1]/step)
            y = nd.arange(0, size[0], step=step, repeat=n*x_num, ctx=ctx[0])

            x = nd.arange(0, size[1], step=step, repeat=n, ctx=ctx[0])
            x = nd.tile(x, int(size[0]/step))

            hw = nd.tile(self.all_anchors[i], (a, 1))
            h, w = hw.split(num_outputs=2, axis=-1)

            self.s[0, a_start:a_start+a] = s.reshape(a, n, 1)
            self.y[0, a_start:a_start+a] = y.reshape(a, n, 1)
            self.x[0, a_start:a_start+a] = x.reshape(a, n, 1)
            self.h[0, a_start:a_start+a] = h.reshape(a, n, 1)
            self.w[0, a_start:a_start+a] = w.reshape(a, n, 1)

            a_start += a

    def yxhw_to_ltrb(self, yxhw):
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

    def predict(self, x):
        batch_out, _ = self.net(x)

        batch_score = nd.sigmoid(batch_out[0])
        batch_box = self.yxhw_to_ltrb(batch_out[1])

        batch_out = nd.concat(batch_score, batch_box, batch_out[2], batch_out[3], dim=-1)
        batch_out = nd.split(batch_out, axis=0, num_outputs=len(batch_out))
        # bs * (len(anchors) +)

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

        return batch_pred.asnumpy()  # [score,y,x,h,w,r,........]

    def get_default_anchors(self):
        import module.IOU_Kmeans as kmeans
        bs = 2000
        addLP = AddLP(self.size[0], self.size[1], self.num_class)
        car_renderer = RenderCar(
            bs, self.size[0], self.size[1], self.classes, ctx[0])

        BG = nd.zeros((bs, 3, 320, 512), ctx=gpu(0))  # b*RGB*h*w
        img, label = car_renderer.render(BG, prob=1.0)
        #img, label = addLP.add(img, label)
        label = label.reshape(-1, 6)[:, 3:5]

        ans = kmeans.main(label, 9)
        print(ans)
        for a in ans:
            b = a.asnumpy()
            print(b[0]*b[1])
        while 1:
            time.sleep(0.1)

    def valid(self):
        print('\033[7;33mValid\033[0m')
        bs = 1
        BG_iter = load_background('val', bs, self.size[0], self.size[1])
        car_renderer = RenderCar(
            self.size[0], self.size[1], self.classes, ctx[0])
        #addLP = AddLP(self.size[0], self.size[1], self.num_class)

        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        self.radar_prob = utils_cv.RadarProb(self.num_class, self.classes)

        for bg in BG_iter:
            bg = bg.data[0].as_in_context(ctx[0])  # b*RGB*w*h

            if np.random.rand() > 0.5:
                imgs, labels = car_renderer.render(
                    bg, 'train', pascal=False, render_rate=0.7)
            else:
                imgs, labels = car_renderer.render(
                    bg, 'train', pascal=True, render_rate=0.7)
            #img, label = addLP.add(img, label)
            imgs = nd.clip(imgs, 0, 1)

            outs = self.predict(imgs)
            #LP_label = label[0,1].asnumpy()

            img = batch_ndimg_2_cv2img(imgs)[0]
            img = cv2_add_bbox(img, labels[0, 0].asnumpy(), [0, 1, 0])
            # Green box
            img = cv2_add_bbox(img, outs[0], [1, 0, 0])
            # Red box
            #im = cv2_add_bbox(im, LP_label, [0,0,1]) # Blue box

            ax1.clear()
            ax1.imshow(img)
            ax1.axis('off')

            #vec_ang, vec_rad, prob = cls2ang(Cout[0], Cout[-self.num_class:])
            self.radar_prob.plot3d(outs[0, 0], outs[0, -self.num_class:])

            raw_input('next')


'''
class Benchmark():
    def __init__(self, logdir, car_renderer):
        self.logdir = logdir
        self.car_renderer = car_renderer
        self.iou_step = 0

    def mean_iou(self, mode):

        bg = load_background(mode)
        sum_iou = 0
        for i in range(iters):
            bg = load_background(self, train_or_val, bs, w, h)
            imgs, labels = self.car_renderer.render(bg, True, 'valid', 1.0)
            Cout = self.predict(img)
            Couts = net(imgs)
            for j in range(batch_size):
                iou = get_iou(labels[j], couts[j])
                sum_iou += iou

        mean_iou = sum_iou / (iters * batch_size)
        self.sw.add_scalar('iou', mean_iou, self.iou_step)
        self.iou_step += 1

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
'''
class LP():
    def __init__(self):
        self.all_anchors = nd.array(spec['LP_anchors'])
        self.step = 16jhjjjj
        self.area = int(self.size[0] * self.size[1] / 16**2)

        if args.mode == 'train':
            self.loss_name = ['LP_score', 'LP_box']
            self.nd_all_anchors = [self.all_anchors.copyto(device) for device in ctx]

    def get_default_ltrb(self):
        n = len(self.all_anchors[0])
        LTRB = nd.zeros((sum(self.area),n,4))
        size = self.size
        a_start = 0
        #for i, anchors in enumerate(self.all_anchors): # [12*16,6*8,3*4]
        a = self.area
        step = float(self.step)
        h, w = self.all_anchors.split(num_outputs=2, axis=-1)

        x_num = int(size[1]/step)

        y = nd.arange(step/size[0]/2., 1, step=step/size[0], repeat=n*x_num)
        #[[.16, .16, .16, .16],
        # [.50, .50, .50, .50],
        # [.83, .83, .83, .83]]
        h = nd.tile(h.reshape(-1), a) # same shape as y
        t = (y - 0.5*h).reshape(a, n, 1)
        b = (y + 0.5*h).reshape(a, n, 1)

        x = nd.arange(step/size[1]/2., 1, step=step/size[1], repeat=n)
        #[1/8, 3/8, 5/8, 7/8]
        w = nd.tile(w.reshape(-1), int(size[1]/step))
        l = nd.tile(x - 0.5*w, int(size[0]/step)).reshape(a, n, 1)
        r = nd.tile(x + 0.5*w, int(size[0]/step)).reshape(a, n, 1)

        LTRB = nd.concat(l,t,r,b, dim=-1)

        self.all_anchors_ltrb = [LTRB.copyto(device) for device in ctx]

    def find_best(self, L, gpu_index):
        anc_ltrb = self.all_anchors_ltrb[gpu_index][:]
        IOUs = get_iou(anc_ltrb, L, mode=2)
        best_match = int(IOUs.reshape(-1).argmax(axis=0).asnumpy()[0]) #print(best_match)
        best_pixel = int(best_match//len(self.all_anchors[0])) 
        best_anchor = int(best_match%len(self.all_anchors[0]))

        best_ltrb = self.all_anchors_ltrb[gpu_index][best_pixel, best_anchor].reshape(-1)

        by_minus_cy = L[1] - (best_ltrb[3]+best_ltrb[1])/2
        sigmoid_ty = by_minus_cy*self.size[0]/self.step + 0.5
        sigmoid_ty = nd.clip(sigmoid_ty, 0.0001, 0.9999)
        ty = nd_inv_sigmoid(sigmoid_ty)

        bx_minus_cx = L[2] - (best_ltrb[2]+best_ltrb[0])/2
        sigmoid_tx = bx_minus_cx*self.size[1]/self.step + 0.5
        sigmoid_tx = nd.clip(sigmoid_tx, 0.0001, 0.9999)
        tx = nd_inv_sigmoid(sigmoid_tx)
        th = nd.log((L[3]) / self.nd_all_anchors[gpu_index][0, best_anchor, 0])
        tw = nd.log((L[4]) / self.nd_all_anchors[gpu_index][0, best_anchor, 1])
        return best_pixel, best_anchor, nd.concat(ty, tx, th, tw, dim=-1)
        def loss_mask(self, labels, gpu_index):
        """Generate training targets given predictions and labels.
        labels: bs*object*[class, cent_y, cent_x, box_h, box_w, rotate]
        """
        a1 = sum(self.area)
        bs = labels.shape[0]
        n = len(self.all_anchors[0])

        L_score = nd.zeros((bs, a0, n, 1), ctx=ctx[gpu_index])
        L_box = nd.zeros((bs, a0, n, 4), ctx=ctx[gpu_index])
        L_mask = nd.zeros((bs, a0, n, 1), ctx=ctx[gpu_index])

        for b in range(bs): 
            label = labels[b]
            #nd.random.shuffle(label)
            for L in label: # all object in the image
                if L[0] < 0: continue
                elif L[0] == self.num_class: # LP
                    px, anc, box = self.LP.find_best(L, gpu_index)
                    L_mask[b, px, anc, :] = 1.0 # others are zero
                    L_score[b, px, anc, :] = 1.0 # others are zero
                    L_box[b, px, anc, :] = box

                    
        return [C_score, C_box, C_class], C_mask

'''

# -------------------- Main -------------------- #
if __name__ == '__main__':
    main()
