#!/usr/bin/env python
from time import gmtime, strftime
import sys
import copy
from mxnet import autograd

from utils import *

# -------------------- Data --------------------
args = Parser()
NN = args.dataset
topk = 1

if args.mode == 'train':
    ctx = [gpu(0), gpu(1)]
    batch_size = 20 * len(ctx)
else:
    ctx = [gpu(1)]


class Valid(Video):
    def __init__(self, pretrain):
        self.size = all_size[3]
        self.steps = [16, 16, 32]
        self.cls_names = all_cls_names[3]
        self.all_anchors = nd.array(
            [[[0.100, 0.200], [0.140, 0.160]],
             [[0.300, 0.500], [0.200, 0.400]],
             [[0.500, 0.700], [0.400, 0.600]]])

        self.get_feature = HP_32(self.all_anchors, len(self.cls_names))
        init_NN(self.get_feature, pretrain, ctx)

        n = len(self.all_anchors[0])  # anchor per sub_map

        self.area = [int(self.size[0]*self.size[1]/step**2) for step in self.steps]
        self.s = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        self.y = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        self.x = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        self.h = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        self.w = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        a_start = 0
        for i, anchors in enumerate(self.all_anchors):  # [12*16,6*8,3*4]
            step = self.steps[i]
            a = self.area[i]

            s = nd.repeat(nd.array([step], ctx=ctx[0]), repeats=a*n)

            x_num = int(self.size[1]/step)
            y = nd.arange(0, self.size[0], step=step, repeat=n*x_num, ctx=ctx[0])

            x = nd.arange(0, self.size[1], step=step, repeat=n, ctx=ctx[0])
            x = nd.tile(x, int(self.size[0]/step))

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
        L_pred, C_pred = self.get_feature(x)
        box = self.yxhw_to_ltrb(nd.concat(L_pred[1], C_pred[1], dim=1))
        L_box = box[:, :self.area[0]]
        C_box = box[:, self.area[0]:]

        L_score = nd.sigmoid(L_pred[0])  # 1,a,2,1
        C_score = nd.sigmoid(C_pred[0])

        L_best = L_score.reshape(-1).argmax(axis=0)
        Lout = nd.concat(L_score, L_box, dim=-1).reshape((-1, 5))
        Lout = Lout[L_best][0].asnumpy()

        Cout = nd.concat(C_score, C_box, C_pred[2], dim=-1).reshape((-1, 5+len(self.cls_names)))

        C_score = C_score.reshape(-1)
        C_1 = C_score.argmax(axis=0)
        C_score[C_1] = 0.0
        C_2 = C_score.argmax(axis=0)
        C_score[C_2] = 0.0
        C_3 = C_score.argmax(axis=0)

        Cout1 = Cout[C_1][0].asnumpy()
        Cout2 = Cout[C_2][0].asnumpy()
        Cout3 = Cout[C_3][0].asnumpy()
        return Lout, Cout1, Cout2, Cout3

    def test_speed(self):
        path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/'
        BG_iter = image.ImageIter(1, (3, size[0], size[1]),
            path_imgrec=path+'sun2012_train.rec',
            path_imgidx=path+'sun2012_train.idx',
            shuffle=True, pca_noise=0, 
            brightness=0.5, saturation=0.5, contrast=0.5, hue=1.0,
            rand_crop=True, rand_resize=True, rand_mirror=True, inter_method=10)
        BG = BG_iter.next().data[0].as_in_context(ctx[0]) # b*RGB*w*h
        render_car = RenderCar(1, size[0], size[1])
        img, label = render_car.render(BG)
        tt=0
        t = time.time()
        while True:
            tt+=1
            L_pred, Cout1, Cout2, Cout3 = self.predict(img)
            print((time.time()-t)/tt)

    def visualize(self, out):
        img = self.img
        L_pred = out[0]
        C_pred = out[1]
        n = np.argmax(C_pred[5:])
        if self.show:
            # -------------------- Add Box --------------------
            if C_pred[0]>0.9:
                cv2_add_bbox_text(img, C_pred, 'Car', 2)
                plt_radar_prob(self.ax, self.cls_names, C_pred)
            '''
            if L_pred[0]>0.9:
                cv2_add_bbox_text(img, L_pred, 'LP', [self.cam_h,self.cam_w], 2)
            '''
            # -------------------- Save Image --------------------
            # self.out.write(img)
            # -------------------- Show Image --------------------
            cv2.imshow('img', img)
            cv2.waitKey(1)
        # -------------------- Publish Image and Box --------------------
        self.YOLO_img_pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))

        self.mat.data = [-1]*6*topk
        if C_pred[0] > 0.95:  # [class, score, l, t, r, b]
            self.mat.data[0] = n
            for i in range(5): self.mat.data[i+1] = C_pred[i]
        self.YOLO_box_pub.publish(self.mat)

    def valid_HP32(self):
        print('\033[1;33;40mValid\033[0;37;40m')
        path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/PASCAL_valid'
        batch_iter = load_ImageDetIter(path, 1, size[0], size[1])
        addLP = AddLP(size[0], size[1], len(self.cls_names))

        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, polar=True)
        for batch in batch_iter:
            img = batch.data[0].as_in_context(ctx[0])/255.  # b*RGB*w*h
            img = nd.clip(img, 0, 1)
            label = batch.label[0].as_in_context(ctx[0])  # b*L*5
            # img, label = addLP.add(img, label)

            L_pred, C_pred = self.predict(img)
            ax1.clear()
            ax1.imshow(img[0].transpose((1, 2, 0)).asnumpy())
            plt_add_bbox(ax1, size, C_pred, 'green')
            plt_radar_prob(ax2, self.cls_names, C_pred)
            raw_input('next')

    def valid_HP33(self):
        print('\033[1;33;40mValid\033[0;37;40m')
        path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/'

        # PASCAL_iter = load_ImageDetIter(path + 'PASCAL_train', batch_size, size[0], size[1])
        BG_iter = image.ImageIter(1, (3, size[0], size[1]),
                                  path_imgrec=path+'sun2012_train.rec',
                                  path_imgidx=path+'sun2012_train.idx',
                                  shuffle=True, pca_noise=0,
                                  brightness=0.5, saturation=0.5, contrast=0.5, hue=1.0,
                                  rand_crop=True, rand_resize=True, rand_mirror=True, inter_method=10)
                                  BG_iter.reset()

        render_car = RenderCar(1, size[0], size[1])
        addLP = AddLP(size[0], size[1], len(self.cls_names))

        plt.ion()
        fig = plt.figure()
        ax = []
        for i in range(1):
            ax.append(fig.add_subplot(211))
            ax.append(fig.add_subplot(212, polar=True))
        for BG in BG_iter:
            t = time.time()

            BG = BG.data[0].as_in_context(ctx[0])  # b*RGB*w*h
            img, label = render_car.render(BG)
            img, label = addLP.add(img, label)
            #  print(time.time()-t)
            img = nd.clip(img, 0, 1)

            L_pred, Cout1, Cout2, Cout3 = self.predict(img)
            for i in range(1):
                ax[i].clear()
                ax[i].imshow(img[0].transpose((1, 2, 0)).asnumpy())
                print(Cout1)
                print(label)
                plt_add_bbox(ax[0], size, Cout1, 'blue')
                plt_add_bbox(ax[0], size, label[i, 0].asnumpy(), 'red')
                #  plt_add_bbox(ax[i], size, Cout2, 'green')
                #  plt_add_bbox(ax[i], size, Cout3, 'red')
                plt_add_bbox(ax[i], size, L_pred, 'black')
                plt_radar_prob(ax[1], self.cls_names, Cout1)
                #  plt_radar_prob(ax[i], self.cls_names, Cout2)
                #  plt_radar_prob(ax[i], self.cls_names, Cout3)
            #  print(time.time()-t)
            raw_input('next')


class Train():
    def __init__(self):
        from mxboard import SummaryWriter
        self.all_anchors = [all_anchors.copyto(device) for device in ctx]
        self.area = [int(size[0]*size[1]/step**2) for step in steps]
        self.get_default_ltrb()

        self.CE_loss = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)
        self.LG_loss = gluon.loss.LogisticLoss(label_format='binary')
        self.L1_loss = gluon.loss.L1Loss()
        self.L2_loss = gluon.loss.L2Loss()

        self.trainer = gluon.Trainer(get_feature.collect_params(),
                                     'adam',
                                     {'learning_rate': LR})

        self.sw = SummaryWriter(logdir=NN + '/logs')  #, flush_secs=30)
        a = get_feature(nd.zeros((1, 3, size[0], size[1]), ctx=ctx[0]))
        #  print(a[0].shape, a[1].shape)
        #  self.sw.add_graph(get_feature)

        self.backup_dir = NN + '/backup/'
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def get_default_ltrb(self):

        n = len(all_anchors[0])
        LTRB = nd.zeros((sum(self.area), n, 4))

        a_start = 0
        for i, anchors in enumerate(all_anchors):  # [12*16,6*8,3*4]
            a = self.area[i]

            h, w = anchors.split(num_outputs=2, axis=-1)

            x_num = int(size[1]/steps[i])
            y = nd.arange(steps[i]/size[0]/2., 1, step=steps[i]/size[0], repeat=n*x_num)
            # [[.16, .16, .16, .16],
            #  [.50, .50, .50, .50],
            #  [.83, .83, .83, .83]]
            h = nd.tile(h.reshape(-1), a)  # same shape as y
            t = (y - 0.5*h).reshape(a, n, 1)
            b = (y + 0.5*h).reshape(a, n, 1)

            x = nd.arange(steps[i]/size[1]/2., 1, step=steps[i]/size[1], repeat=n)
            # [1/8, 3/8, 5/8, 7/8]
            w = nd.tile(w.reshape(-1), int(size[1]/steps[i]))
            l = nd.tile(x - 0.5*w, int(size[0]/steps[i])).reshape(a, n, 1)
            r = nd.tile(x + 0.5*w, int(size[0]/steps[i])).reshape(a, n, 1)

            LTRB[a_start:a_start+a] = nd.concat(l, t, r, b, dim=-1)
            a_start += a

        self.all_anchors_ltrb = [LTRB.copyto(device) for device in ctx]

    def find_best_LP(self, L, gpu_index):
        IOUs = get_iou(self.all_anchors_ltrb[gpu_index][:self.area[0]], L)
        # best_match = int(np.argmax(IOUs)), Global reduction not supported yet
        best_match = int(IOUs.reshape(-1).argmax(axis=0).asnumpy()[0])
        best_pixel = int(best_match//len(all_anchors[0]))
        best_anchor = int(best_match % len(all_anchors[0]))
        best_ltrb = self.all_anchors_ltrb[gpu_index][best_pixel, best_anchor].reshape(-1)

        by_minus_cy = (L[2]+L[4])/2 - (best_ltrb[3]+best_ltrb[1])/2
        sigmoid_ty = by_minus_cy*size[0]/steps[0] + 0.5
        sigmoid_ty = nd.clip(sigmoid_ty, 0.001, 0.999)
        ty = -nd.log(1/sigmoid_ty - 1)

        bx_minus_cx = (L[1]+L[3])/2 - (best_ltrb[2]+best_ltrb[0])/2
        sigmoid_tx = bx_minus_cx*size[1]/steps[0] + 0.5
        sigmoid_tx = nd.clip(sigmoid_tx, 0.001, 0.999)
        tx = -nd.log(1/sigmoid_tx - 1)

        th = nd.log((L[4]-L[2]) / self.all_anchors[gpu_index][0, best_anchor, 0])
        tw = nd.log((L[3]-L[1]) / self.all_anchors[gpu_index][0, best_anchor, 1])
        return best_pixel, best_anchor, nd.concat(ty, tx, th, tw, dim=-1)

    def find_best_car(self, L, gpu_index):
        IOUs = get_iou(self.all_anchors_ltrb[gpu_index][self.area[0]:], L)  # L is (f,l,t,r,b)
        # best_match = int(np.argmax(IOUs)), Global reduction not supported yet ...
        best_match = int(IOUs.reshape(-1).argmax(axis=0).asnumpy()[0])  # print(best_match)
        best_pixel = int(best_match//len(all_anchors[0]))
        best_anchor = int(best_match % len(all_anchors[0]))
        best_ltrb = self.all_anchors_ltrb[gpu_index][best_pixel, best_anchor].reshape(-1)
        # print('best match anchor', best_ltrb)

        if (best_pixel < self.area[1]):
            pyramid_layer = 1
            step = steps[1]
        else:
            pyramid_layer = 2
            step = steps[2]

        by_minus_cy = (L[2]+L[4])/2 - (best_ltrb[3]+best_ltrb[1])/2
        sigmoid_ty = by_minus_cy*size[0]/step + 0.5
        sigmoid_ty = nd.clip(sigmoid_ty, 0.001, 0.999)
        ty = nd_inv_sigmoid(sigmoid_ty)

        bx_minus_cx = (L[1]+L[3])/2 - (best_ltrb[2]+best_ltrb[0])/2
        sigmoid_tx = bx_minus_cx*size[1]/step + 0.5
        sigmoid_tx = nd.clip(sigmoid_tx, 0.001, 0.999)
        tx = nd_inv_sigmoid(sigmoid_tx)

        th = nd.log((L[4]-L[2]) / self.all_anchors[gpu_index][pyramid_layer, best_anchor, 0])
        tw = nd.log((L[3]-L[1]) / self.all_anchors[gpu_index][pyramid_layer, best_anchor, 1])

        return best_pixel, best_anchor, nd.concat(ty, tx, th, tw, dim=-1)

    def loss_mask(self, labels, gpu_index):
        """Generate training targets given predictions and labels."""
        a0 = self.area[0]
        a1 = self.area[1] + self.area[2]
        bs = labels.shape[0]
        n = len(all_anchors[0])
        L_score = nd.zeros((bs, a0, n, 1), ctx=ctx[gpu_index])
        L_box = nd.zeros((bs, a0, n, 4), ctx=ctx[gpu_index])
        L_mask = nd.zeros((bs, a0, n, 1), ctx=ctx[gpu_index])
        C_class = nd.ones((bs, a1, n, 1), ctx=ctx[gpu_index]) * (-1)
        C_score = nd.zeros((bs, a1, n, 1), ctx=ctx[gpu_index])
        C_box = nd.zeros((bs, a1, n, 4), ctx=ctx[gpu_index])
        C_mask = nd.zeros((bs, a1, n, 1), ctx=ctx[gpu_index])

        for b in range(bs):
            label = labels[b]
            # nd.random.shuffle(label)
            for L in label:  # all object in the image
                if L[0] < 0:
                    continue
                elif L[0] == len(cls_names):  # LP
                    px, anc, box = self.find_best_LP(L, gpu_index)
                    L_mask[b, px, anc, :] = 1.0  # others are zero
                    L_score[b, px, anc, :] = 1.0  # others are zero
                    L_box[b, px, anc, :] = box
                else:  # 0~7
                    px, anc, box = self.find_best_car(L, gpu_index)
                    C_mask[b, px, anc, :] = 1.0  # others are zero
                    C_score[b, px, anc, :] = 1.0  # others are zero
                    C_box[b, px, anc, :] = box
                    C_class[b, px, anc, :] = L[0]  # others are ignore_label=-1

        return [L_score, L_box], L_mask, [C_score, C_box, C_class], C_mask

    def train_the(self, batch_xs, batch_ys):
        loss = []
        with autograd.record():
            for gpu_index, (batch_x, batch_y) in enumerate(zip(batch_xs, batch_ys)):
                L_pred, C_pred = get_feature(batch_x)
                with autograd.pause():
                    L_label, L_mask, C_label, C_mask = self.loss_mask(batch_y, gpu_index)

                    L_score_weight = nd.where(L_mask > 0,
                                              nd.ones_like(L_mask)*10.0,
                                              nd.ones_like(L_mask)*0.1,
                                              ctx=ctx[gpu_index])

                    C_score_weight = nd.where(C_mask > 0,
                                              nd.ones_like(C_mask)*10.0,
                                              nd.ones_like(C_mask)*0.1,
                                              ctx=ctx[gpu_index])

                Lsl = self.LG_loss(L_pred[0], L_label[0], L_score_weight)
                Lbl = self.L1_loss(L_pred[1], L_label[1], L_mask)

                Csl = self.LG_loss(C_pred[0], C_label[0], C_score_weight)
                Cbl = self.L1_loss(C_pred[1], C_label[1], C_mask*1.0)
                Ccl = self.CE_loss(C_pred[2], C_label[2], C_mask*0.1)

                # sparse_label=True, so shape(cls_pred, tid) are different
                # loss.append(Lsl + Lbl)
                loss.append(Lsl + Lbl + Csl + Cbl + Ccl)
        for l in loss:
            l.backward()
        self.trainer.step(batch_size)
        self.record_to_tensorboard([Lsl, Lbl, Csl, Cbl, Ccl])
        # self.record_to_tensorboard([Lsl, Lbl])

    def render_and_train_pascal(self):
        print('\033[1;33;40mRender And Train\033[0m')
        # -------------------- load data --------------------
        path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/'
        BG_iter = image.ImageIter(batch_size, (3, size[0], size[1]),
                                  path_imgrec=path+'sun2012_train.rec',
                                  path_imgidx=path+'sun2012_train.idx',
                                  shuffle=True, pca_noise=0,
                                  brightness=0.5, saturation=0.5, contrast=0.5, hue=1.0,
                                  rand_crop=True, rand_resize=True, rand_mirror=True, inter_method=10)
        '''
        iter1 = image.ImageDetIter(batch_size, (3, size[0], size[1]),
            path_imgrec=path+'pascal24_train.rec',
            path_imgidx=path+'pascal24_train.idx',
            shuffle=True, pca_noise=0.1,
            brightness=0.5, saturation=0.5, contrast=0.5, hue=1.0
            #rand_crop=0.2, #rand_pad=0.2, #area_range=(0.8, 1.2)
            )
        '''
        # -------------------- load render tools --------------------
        render_car = RenderCar(batch_size, size[0], size[1])
        # addLP = AddLP(size[0], size[1], len(cls_names))
        '''###########
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        '''##########
        self.ct = 0
        while True:
            ################## get dataset 0 ##################
            try:
                BG = BG_iter.next().data[0].as_in_context(ctx[0])
            except:
                BG_iter.reset()
                BG = BG_iter.next().data[0].as_in_context(ctx[0])
            ################## render dataset 0 ##################
            batch_xs, batch_ys = [], []
            img_batch, label_batch = render_car.render(BG)
            for i, dev in enumerate(ctx):
                start = int(i*batch_size/len(ctx))
                end = int((i+1)*batch_size/len(ctx))
                batch_x = img_batch[start:end].as_in_context(dev)
                batch_y = label_batch[start:end].as_in_context(dev)
                '''
                if np.random.rand() > 0.2:
                    batch_x, batch_y = addLP.add(batch_x, batch_y)
                '''
                batch_xs.append(batch_x)
                batch_ys.append(batch_y)
            #ax1.imshow(batch_xs[0][0].transpose((1,2,0)).asnumpy())
            #ax2.imshow(batch_xs[1][0].transpose((1,2,0)).asnumpy())
            ################## train dataset 0 ##################
            for _ in range(1):
                self.train_the(batch_xs, batch_ys)
            ################## get dataset 1 ##################
            '''
            try:
                batch = iter1.next()
            except:
                iter1.reset()
                batch = iter1.next()
            batch.data[0] = nd.clip(batch.data[0]/255., 0, 1)
            ################## render dataset 1 ##################
            batch_xs, batch_ys = assign_batch(batch, ctx)
            batch_xs[0], batch_ys[0] = addLP.add(batch_xs[0], batch_ys[0])

            #ax3.imshow(batch_xs[0][0].transpose((1,2,0)).asnumpy())
            #ax4.imshow(batch_xs[1][0].transpose((1,2,0)).asnumpy())
            ################## train dataset 1 ##################
            for _ in range(1):
                self.train_the(batch_xs, batch_ys)
            '''
            #input()    
    def record_to_tensorboard(self, loss):
        for i, L in enumerate(loss):
            self.sw.add_scalar(tag='Loss'+str(i), value=nd.mean(L).asnumpy(), global_step=self.ct)
        self.ct += 1
        if self.ct%record_step==0:
            save_model = os.path.join(self.backup_dir, NN + '_%d'%self.ct)
            get_feature.collect_params().save(save_model)

class Train34(Video):
    def __init__(self, pretrain):
        self.steps = [16, 32]
        self.size = all_size[3]
        self.cls_names = all_cls_names[4]
        self.area = [int(self.size[0]*self.size[1]/step**2) for step in self.steps]
        
        self.all_anchors = nd.array(
            [[[0.300, 0.500], [0.200, 0.400]],
            [[0.500, 0.700], [0.400, 0.600]]])

        self.get_feature = HP_34(len(self.cls_names))
        init_NN(self.get_feature, pretrain, ctx)
        if args.mode == 'train':
            from mxboard import SummaryWriter
            self.nd_all_anchors = [self.all_anchors.copyto(device) for device in ctx]
            self.get_default_ltrb()

            self.CE_loss = gluon.loss.SoftmaxCrossEntropyLoss(
                from_logits=False, sparse_label=False)
            self.LG_loss = gluon.loss.LogisticLoss(label_format='binary')
            self.L1_loss = gluon.loss.L1Loss()
            self.L2_loss = gluon.loss.L2Loss()
            self.trainer = gluon.Trainer(self.get_feature.collect_params(), 
                'adam', {'learning_rate': 0.0001})

            self.sw = SummaryWriter(logdir=NN + '/logs')#, flush_secs=30)
            #a = self.get_feature(nd.zeros((1,3,self.size[0],self.size[1]), ctx=ctx[0]))
            #self.sw.add_graph(self.get_feature)
            self.record_step = 5000
            self.backup_dir = NN + '/backup/'
            if not os.path.exists(self.backup_dir): os.makedirs(self.backup_dir)
            self.loss_name = ['score', 'box', 'class', 'rotate']
        else:
            self.init_valid()       
    def get_default_ltrb(self): 
        n = len(self.all_anchors[0])
        LTRB = nd.zeros((sum(self.area),n,4))
        size = self.size
        a_start = 0
        for i, anchors in enumerate(self.all_anchors): # [12*16,6*8,3*4]
            a = self.area[i]

            h, w = anchors.split(num_outputs=2, axis=-1)    

            x_num = int(size[1]/self.steps[i])
            y = nd.arange(self.steps[i]/size[0]/2., 1, step=self.steps[i]/size[0], repeat=n*x_num)
            #[[.16, .16, .16, .16],
            # [.50, .50, .50, .50],
            # [.83, .83, .83, .83]]
            h = nd.tile(h.reshape(-1), a) # same shape as y
            t = (y - 0.5*h).reshape(a, n, 1)
            b = (y + 0.5*h).reshape(a, n, 1)

            x = nd.arange(self.steps[i]/size[1]/2., 1, step=self.steps[i]/size[1], repeat=n)
            #[1/8, 3/8, 5/8, 7/8]
            w = nd.tile(w.reshape(-1), int(size[1]/self.steps[i]))
            l = nd.tile(x - 0.5*w, int(size[0]/self.steps[i])).reshape(a, n, 1)
            r = nd.tile(x + 0.5*w, int(size[0]/self.steps[i])).reshape(a, n, 1)

            LTRB[a_start:a_start+a] = nd.concat(l,t,r,b, dim=-1)
            a_start += a

        self.all_anchors_ltrb = [LTRB.copyto(device) for device in ctx]
    def find_best(self, L, gpu_index):
        anc_ltrb = self.all_anchors_ltrb[gpu_index][:]
        IOUs = get_iou(anc_ltrb, L, mode=2)
        best_match = int(IOUs.reshape(-1).argmax(axis=0).asnumpy()[0]) #print(best_match)
        best_pixel = int(best_match//len(self.all_anchors[0])) 
        best_anchor = int(best_match%len(self.all_anchors[0]))

        best_ltrb = self.all_anchors_ltrb[gpu_index][best_pixel, best_anchor].reshape(-1)

        if (best_pixel < self.area[0]):
            pyramid_layer = 0
            step = self.steps[0]
            #print((best_pixel//32)*16, (best_pixel%32)*16)
        else:
            pyramid_layer = 1
            step = self.steps[1]
            #print(((best_pixel-640)//16)*32, ((best_pixel-640)%16)*32)

        by_minus_cy = L[1] - (best_ltrb[3]+best_ltrb[1])/2
        sigmoid_ty = by_minus_cy*self.size[0]/step + 0.5
        sigmoid_ty = nd.clip(sigmoid_ty, 0.001, 0.999)
        ty = nd_inv_sigmoid(sigmoid_ty)

        bx_minus_cx = L[2] - (best_ltrb[2]+best_ltrb[0])/2
        sigmoid_tx = bx_minus_cx*self.size[1]/step + 0.5
        sigmoid_tx = nd.clip(sigmoid_tx, 0.001, 0.999)
        tx = nd_inv_sigmoid(sigmoid_tx)
        th = nd.log((L[3]) / self.nd_all_anchors[gpu_index][pyramid_layer, best_anchor, 0])
        tw = nd.log((L[4]) / self.nd_all_anchors[gpu_index][pyramid_layer, best_anchor, 1])
        return best_pixel, best_anchor, nd.concat(ty, tx, th, tw, dim=-1)
    def loss_mask(self, labels, gpu_index):
        """Generate training targets given predictions and labels.
        labels: bs*object*[class, cent_y, cent_x, box_h, box_w, rotate]
        """
        a1 = self.area[0] + self.area[1]
        bs = labels.shape[0]
        n = len(self.all_anchors[0])
        C_class = nd.zeros((bs, a1, n, len(self.cls_names)), ctx=ctx[gpu_index])
        #C_class = nd.ones((bs, a1, n, 1), ctx=ctx[gpu_index]) * (-1)
        C_score = nd.zeros((bs, a1, n, 1), ctx=ctx[gpu_index])
        C_rotate = nd.zeros((bs, a1, n, 1), ctx=ctx[gpu_index])
        C_box = nd.zeros((bs, a1, n, 4), ctx=ctx[gpu_index])
        C_mask = nd.zeros((bs, a1, n, 1), ctx=ctx[gpu_index])

        for b in range(bs): 
            label = labels[b]
            #nd.random.shuffle(label)
            for L in label: # all object in the image
                if L[0] < 0: continue
                else: # 0~7
                    px, anc, box = self.find_best(L, gpu_index)
                    C_mask[b, px, anc, :] = 1.0 # others are zero
                    C_score[b, px, anc, :] = 1.0 # others are zero
                    C_rotate[b, px, anc, :] = L[5] #+-pi -> +-inf
                    C_box[b, px, anc, :] = box
                    C_class[b, px, anc, L[0]] = 1.0
                    
                    C_class[b, px, anc, L[0]] = 0.8
                    if L[0] == 0:
                        C_class[b, px, anc, len(self.cls_names)-1] = 0.1
                        C_class[b, px, anc, 1] = 0.1


                    elif L[0] == len(self.cls_names)-1:
                        C_class[b, px, anc, len(self.cls_names)-2] = 0.1
                        C_class[b, px, anc, 0] = 0.1

                    else:
                        C_class[b, px, anc, L[0]+1] = 0.1
                        C_class[b, px, anc, L[0]-1] = 0.1
                    
        return [C_score, C_rotate, C_box, C_class], C_mask
    def train_the(self, batch_xs, batch_ys):
        loss = []
        with autograd.record():
            for gpu_index, (batch_x, batch_y) in enumerate(zip(batch_xs, batch_ys)):
                C_pred = self.get_feature(batch_x)
                with autograd.pause():
                    C_label, C_mask = self.loss_mask(batch_y, gpu_index)    

                    C_score_weight = nd.where(C_mask>0,
                        nd.ones_like(C_mask)*10.0,
                        nd.ones_like(C_mask)*0.1,
                        ctx=ctx[gpu_index])

                Csl = self.LG_loss(C_pred[0], C_label[0], C_score_weight*0.1)
                Crl = self.L2_loss(C_pred[1], C_label[1], C_mask*50.0)
                Cbl = self.L2_loss(C_pred[2], C_label[2], C_mask*1.0)
                Ccl = self.CE_loss(C_pred[3], C_label[3], C_mask*0.1) #0.1 after 1day:1.0
                loss.append(Csl + Crl + Cbl + Ccl)
                #loss.append(Csl + Crl + Cbl)
        for l in loss: l.backward()
        self.trainer.step(batch_size)
        
        self.record_to_tensorboard([Csl, Cbl, Ccl, Crl])
    def render_and_train(self):
        print('\033[1;33;40mRender And Train\033[0m')
        ################## load data ##################
        path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/'
        BG_iter = image.ImageIter(batch_size, (3, self.size[0], self.size[1]),
            path_imgrec=path+'sun2012_train.rec',
            path_imgidx=path+'sun2012_train.idx',
            shuffle=True, pca_noise=0, 
            brightness=0.5, saturation=0.5, contrast=0.5, hue=1.0,
            rand_crop=True, rand_resize=True, rand_mirror=True, inter_method=10)

        PS_iter = image.ImageDetIter(batch_size, (3, self.size[0], self.size[1]),
            path_imgrec=path+'pascal24_train.rec', 
            path_imgidx=path+'pascal24_train.idx',
            shuffle=True, pca_noise=0.1, 
            brightness=0.5, saturation=0.5, contrast=0.5, hue=1.0
            #rand_crop=0.2, #rand_pad=0.2, #area_range=(0.8, 1.2)
            )

        ################## load render tools ##################
        render_car = RenderCar(batch_size, self.size[0], self.size[1], ctx[0])
        self.ct = 0
        '''
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        '''
        while True:
            ##################### dataset 0 #####################
            try:
                BG = BG_iter.next().data[0].as_in_context(ctx[0])
            except:
                BG_iter.reset()
                BG = BG_iter.next().data[0].as_in_context(ctx[0])
            ################## render dataset 0 ##################
            batch_xs, batch_ys = [], []
            img_batch, label_batch = render_car.render(BG)
            for i, dev in enumerate(ctx):
                start = int(i*batch_size/len(ctx))
                end = int((i+1)*batch_size/len(ctx))
                batch_x = img_batch[start:end].as_in_context(dev)
                batch_y = label_batch[start:end].as_in_context(dev)

                batch_xs.append(batch_x)
                batch_ys.append(batch_y)
            ################## train dataset 0 ##################
            #ax1.imshow(batch_xs[0][0].transpose((1,2,0)).asnumpy())
            #ax2.imshow(batch_xs[1][0].transpose((1,2,0)).asnumpy())
            for _ in range(3):
                self.train_the(batch_xs, batch_ys)

            ##################### dataset 1 #####################
            ################## render dataset 1 ##################
            batch_xs, batch_ys = [], []
            img_batch, label_batch = render_car.render_pascal(BG, 'train')
            for i, dev in enumerate(ctx):
                start = int(i*batch_size/len(ctx))
                end = int((i+1)*batch_size/len(ctx))
                batch_x = img_batch[start:end].as_in_context(dev)
                batch_y = label_batch[start:end].as_in_context(dev)

                batch_xs.append(batch_x)
                batch_ys.append(batch_y)
            
            #ax3.imshow(batch_xs[0][0].transpose((1,2,0)).asnumpy())
            #ax4.imshow(batch_xs[1][0].transpose((1,2,0)).asnumpy())
            #input()
            ################## train dataset 1 ##################
            for _ in range(3):
                self.train_the(batch_xs, batch_ys)
    
    def record_to_tensorboard(self, loss):
        for i, L in enumerate(loss):
            #self.sw.add_scalar(tag='Loss'+str(i), value=nd.mean(L).asnumpy(), global_step=self.ct)
            self.sw.add_scalar('Loss', (self.loss_name[i],nd.mean(L).asnumpy()), self.ct)
        self.ct += 1
        if self.ct%self.record_step==0:
            save_model = os.path.join(self.backup_dir, NN + '_%d'%self.ct)
            self.get_feature.collect_params().save(save_model)  
    
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
        C_pred = self.get_feature(x)

        C_score = nd.sigmoid(C_pred[0])
        C_box = self.yxhw_to_ltrb(C_pred[2])
        
        Cout = nd.concat(C_score, C_box, C_pred[1], C_pred[3], dim=-1)
        Cout = Cout.reshape((-1, 6+len(self.cls_names)))

        C_1 = C_score.reshape(-1).argmax(axis=0).reshape(-1)

        Cout = Cout[C_1][0].asnumpy()
        
        y = (Cout[2] + Cout[4])/2
        x = (Cout[1] + Cout[3])/2
        h = (Cout[4] - Cout[2])
        w = (Cout[3] - Cout[1])
        
        Cout[1:5] = [y,x,h,w]
        
        return Cout #[score,y,x,h,w,r,........]
    def init_valid(self):
        size = self.size
        n = len(self.all_anchors[0]) # anchor per sub_map
        self.s = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        self.y = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        self.x = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        self.h = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        self.w = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
        a_start = 0
        for i, anchors in enumerate(self.all_anchors): # [12*16,6*8,3*4]
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
    def valid(self):
        self.init_valid()
        print('\033[1;33;40m Valid \033[0;37;40m')
        path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/'

        #PASCAL_iter = load_ImageDetIter(path + 'PASCAL_train', batch_size, size[0], size[1])
        BG_iter = image.ImageIter(1, (3, self.size[0], self.size[1]),
            path_imgrec=path+'sun2012_val.rec',
            path_imgidx=path+'sun2012_val.idx',
            shuffle=True, pca_noise=0, 
            brightness=0.5, saturation=0.5, contrast=0.5, hue=1.0,
            rand_crop=True, rand_resize=True, rand_mirror=True, inter_method=10)
        BG_iter.reset()

        render_car = RenderCar(1, self.size[0], self.size[1], ctx[0])

        plt.ion()
        fig = plt.figure()
        ax = []
        for i in range(1):
            ax.append(fig.add_subplot(211))
            ax.append(fig.add_subplot(212, polar=True))
        for BG in BG_iter:
            BG = BG.data[0].as_in_context(ctx[0]) # b*RGB*w*h
            img, label = render_car.render(BG)
            img = nd.clip(img, 0, 1)
            Cout = self.predict(img)

            for i in range(1):
                ax[0].clear()
                im = img[0].transpose((1,2,0)).asnumpy()
                label = label[0,0].asnumpy()
                im = add_bbox(im, label, [1,0,0])

                im = add_bbox(im, Cout, [0,1,0])
                print(Cout[0])
                #plt_radar_prob(ax[1], cls_names, Cout)
                ax[0].imshow(im)
                ax[0].axis('off')

            raw_input('next')
    def visualize(self, Cout):
        img = copy.deepcopy(self.img)
        n = np.argmax(Cout[6:])
        #self.out.write(img)
        ########################### Add Box ###########################
        vec_ang, vec_rad, prob = cls2ang(Cout[0], Cout[-len(self.cls_names):])
        if Cout[0]>0.5:
            #self.miss_counter = 0
            add_bbox(img, Cout, [0,0,255])
            for i in range(6): 
                self.mat.data[i] = Cout[i]

            self.mat.data[6] = vec_ang

        else:
            #self.miss_counter += 1
            #if self.miss_counter > 20:
            self.mat.data = [-1]*7*self.topk

        if self.radar: 
            plt_radar_prob(self.ax, vec_ang, vec_rad, prob)
            plt.pause(0.01)

        self.YOLO_box_pub.publish(self.mat) 

        ########################## Show Image ##########################    
        if self.show:
            cv2.imshow('img', img)
            cv2.waitKey(1)

        ########################## Save Image ##########################
        
        ##################### Publish Image and Box #####################
        self.YOLO_img_pub.publish(self.bridge.cv2_to_imgmsg(img,'bgr8'))
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

        render_car = RenderCar(100, self.size[0], self.size[1], ctx[0])
        predictions = [] #[[]*24,[]*24,[]*24],............]
        labels = [] #[1,5,25,3,5,12,22,.............]
        
        for BG in BG_iter:
            BG = BG.data[0].as_in_context(ctx[0])
            img_batch, label_batch = render_car.render_pascal(BG, 'valid')

            C_pred = self.get_feature(img_batch)

            
            for i in range(100):
                C_score = C_pred[0][i]
                C_1 = C_score.reshape(-1).argmax(axis=0).reshape(-1)

                Cout = C_pred[3][i].reshape((-1, len(self.cls_names)))
                Cout = softmax(Cout[C_1][0].asnumpy())

                predictions.append(Cout)
                labels.append(int(label_batch[i,0,0].asnumpy()))

            if len(labels)>(3000-1): break
        
        labels = np.array(labels)
        predictions = np.array(predictions)


        for i, name in enumerate(self.cls_names):
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
            
            sw.add_pr_curve(name, label, predict, 100, global_step=0)
            
        '''
        predictions = nd.uniform(low=0, high=1, shape=(100,), dtype=np.float32)
        labels = nd.uniform(low=0, high=2, shape=(100,), dtype=np.float32).astype(np.int32)
        print(labels)
        print(predictions)
        sw1.add_pr_curve(tag='pseudo_pr_curve', predictions=predictions, labels=labels, num_thresholds=120)
        '''

if NN == 'HP_31':
    get_feature = HP_31(all_anchors, len(cls_names))
    pretrain = 'HP_31/backup/weight_128000'
elif NN == 'HP_32':
    t = Valid('HP_32/backup/HP_32_75000')
elif NN == 'HP_33':
    t = Valid('HP_33/backup/HP_33_52000')
elif NN == 'HP_34':
    #t = Train34('HP_34/backup/HP_34_60000')
    t = Train34('HP_34/backup/HP_34_105k181')

######################### Main #########################
if args.mode == 'train':
    t.render_and_train()
elif args.mode == 'valid':
    #t.valid()
    t.run(ctx=ctx[0],topic='/drone/camera/image_raw', radar=True, show=True)
    #'/drone/camera/image_raw'
    #'/usb_cam/image_decompressed'
    #'/ardrone/front/image_raw'
    #'/usb_cam/image_raw'
    #'/zed/left/image_raw_color'

    
    #v.valid_HP33()
elif args.mode == 'PR':
    t.pr_curve()
elif args.mode == 'cal':
    calibration()
