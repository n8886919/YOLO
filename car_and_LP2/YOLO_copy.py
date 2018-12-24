#!/usr/bin/env python
import argparse
import copy
import sys
import yaml

from mxnet import gluon
from utils import *
from render_car import *

from yolo_modules.gluon import *
from yolo_modules.cv import *
from yolo_modules.licence_plate_render import *

#################### Global variables ####################
parser = argparse.ArgumentParser(prog="python YOLO.py")
parser.add_argument("version", help="v2")
parser.add_argument("mode", help="train or valid or video")
parser.add_argument("-t", "--topic", help="ros topic to subscribe", dest="topic", default="")
parser.add_argument("--radar", help="show radar plot", dest="radar", default=0, type=int)
parser.add_argument("--show", help="show processed image", dest="show", default=1, type=int)
parser.add_argument("--gpu", help="gpu index", dest="gpu", default="0")
args = parser.parse_args()
args.show = bool(args.show)
args.radar = bool(args.radar)

if args.mode == 'train':
    ctx = [gpu(int(i)) for i in args.gpu]
else:
    ctx = [gpu(int(args.gpu[0]))]

#################### Global variables ####################

def main():
    yolo = YOLO('iter_107000')

    if args.mode == 'train':
        yolo.render_and_train()

    elif args.mode == 'valid':
        yolo.valid()
        #yolo.get_default_anchors()
    elif args.mode == 'video':
        
        yolo.run(ctx=ctx[0], topic=args.topic, radar=args.radar, show=args.show)

    elif args.mode == 'PR':
        yolo.pr_curve()

    else: print('Parser 2 should be train or valid or video')

class YOLO(Video):
    def __init__(self, pretrain):
        self.version = args.version
        with open(os.path.join(self.version, 'spec.yaml')) as f:
            spec = yaml.load(f)

        self.size = spec['size']
        self.num_class = len(spec['classes'])
        self.all_anchors = nd.array(spec['all_anchors'])

        ################################################################
        #self.loss_name = ['score', 'box', 'class']
        self.loss_name = ['score', 'box', 'rotate', 'class']

        ################################################################
        num_downsample = len(spec['layers']) # number of downsample
        num_prymaid_layers = len(self.all_anchors)# number of pyrmaid layers
        prymaid_start = num_downsample - num_prymaid_layers + 1
        self.steps = [2**(prymaid_start+i) for i in range(num_prymaid_layers)]
        self.area = [int(self.size[0]*self.size[1]/step**2) for step in self.steps]

        print('\033[1;33;40mCTX = {}\033[0m'.format(ctx))
        print('\033[1;33;40mLoss = {}\033[0m'.format(self.loss_name))
        print('\033[1;33;40mStep = {}\033[0m'.format(self.steps))
        print('\033[1;33;40mArea = {}\033[0m'.format(self.area))

        self.backup_dir = os.path.join(self.version, 'backup')
        self.net = CarNet(spec, num_sync_bn_devices=len(ctx))
        #init_NN(self.net, os.path.join(self.backup_dir, pretrain), ctx)
        init_NN(self.net, '/media/nolan/SSD1/car_and_LP2_backup/iter_107000', ctx)
        if args.mode == 'train':
            self.record_step = spec['record_step']
            self.batch_size = spec['batch_size'] * len(ctx)
            print('\033[1;33;40mBatch Size = {}\033[0m'.format(self.batch_size))
            print('\033[1;33;40mRecord Step = {}\033[0m'.format(self.record_step))
            self._init_train()  
        else:
            self._init_valid()      

    def _init_train(self):
        from mxboard import SummaryWriter

        self.backward_counter = 0
        self.record_step = spec['record_step']
        self.batch_size = spec['batch_size'] * len(ctx)
        print('\033[1;33;40mBatch Size = {}\033[0m'.format(self.batch_size))
        print('\033[1;33;40mRecord Step = {}\033[0m'.format(self.record_step))
        
        self.nd_all_anchors = [self.all_anchors.copyto(device) for device in ctx]
        self.get_default_ltrb()

        self.L1_loss = gluon.loss.L1Loss()
        self.L2_loss = gluon.loss.L2Loss()
        self.LG_loss = gluon.loss.LogisticLoss(label_format='binary')
        self.CE_loss = gluon.loss.SoftmaxCrossEntropyLoss(
            from_logits=False, sparse_label=False)

        self.trainer = gluon.Trainer(
            self.net.collect_params(), 
            'adam', 
            {'learning_rate': 0.0001}
        )

        self.sw = SummaryWriter(logdir=self.version+'/logs')#, flush_secs=30)

        if not os.path.exists(self.backup_dir): 
            os.makedirs(self.backup_dir)

    def get_default_ltrb(self): 
        LTRB = []  #nd.zeros((sum(self.area),n,4))
        size = self.size
        a_start = 0
        for i, anchors in enumerate(self.all_anchors): # [12*16,6*8,3*4]
            n = len(self.all_anchors[i])
            a = self.area[i]
            step = float(self.steps[i])
            h, w = anchors.split(num_outputs=2, axis=-1)    

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

            LTRB.append(nd.concat(l,t,r,b, dim=-1))
            #LTRB[a_start:a_start+a] = nd.concat(l,t,r,b, dim=-1)
            a_start += a

        LTRB = nd.concat(*LTRB, dim=0)
        self.all_anchors_ltrb = [LTRB.copyto(device) for device in ctx]

    def find_best(self, L, gpu_index):
        anc_ltrb = self.all_anchors_ltrb[gpu_index][:]
        IOUs = get_iou(anc_ltrb, L, mode=2)
        best_match = int(IOUs.reshape(-1).argmax(axis=0).asnumpy()[0])
        best_pixel = int(best_match//len(self.all_anchors[0]))
        best_anchor = int(best_match % len(self.all_anchors[0]))

        best_ltrb = self.all_anchors_ltrb[gpu_index][best_pixel, best_anchor].reshape(-1)

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
            for L in label_batch[b]: # all object in the image
                if L[0] < 0: continue
                else:
                    px, anc, box = self.find_best(L, gpu_index)
                
                    C_mask[b, px, anc, :] = 1.0 # others are zero
                    C_score[b, px, anc, :] = 1.0 # others are zero
                    C_box[b, px, anc, :] = box
                    C_rotate[b, px, anc, :] = L[5]
                    C_class[b, px, anc, L[0]] = 1.0
                    '''
                    C_class[b, px, anc, L[0]] = 0.8
                    if L[0] == 0:
                        C_class[b, px, anc, self.num_class-1] = 0.1
                        C_class[b, px, anc, 1] = 0.1


                    elif L[0] == self.num_class-1:
                        C_class[b, px, anc, self.num_class-2] = 0.1
                        C_class[b, px, anc, 0] = 0.1

                    else:
                        C_class[b, px, anc, L[0]+1] = 0.1
                        C_class[b, px, anc, L[0]-1] = 0.1
                    '''
        return [C_score, C_box, C_rotate, C_class], C_mask  

    def train_the(self, batch_xs, batch_ys, r=50.0):
        all_gpu_loss = []
        with mxnet.autograd.record():
            for gpu_i, (batch_x, batch_y) in enumerate(zip(batch_xs, batch_ys)):
                all_gpu_loss.append([])

                C_pred, L_pred = self.net(batch_x)
                with mxnet.autograd.pause():
                    C_label, C_mask = self.loss_mask(batch_y, gpu_i)

                    C_score_weight = nd.where(C_mask>0,
                        nd.ones_like(C_mask)*10.0,
                        nd.ones_like(C_mask)*0.1,
                        ctx=ctx[gpu_i])

                if 'score' in self.loss_name:
                    all_gpu_loss[gpu_i].append(self.LG_loss(C_pred[0], C_label[0], C_score_weight * 0.1))
                if 'box' in self.loss_name: 
                    all_gpu_loss[gpu_i].append(self.L2_loss(C_pred[1], C_label[1], C_mask * 1.0))
                if 'rotate' in self.loss_name:  
                    all_gpu_loss[gpu_i].append(self.L2_loss(C_pred[2], C_label[2], C_mask * r))
                if 'class' in self.loss_name:
                    all_gpu_loss[gpu_i].append(self.CE_loss(C_pred[3], C_label[3], C_mask * 0.1)) 

                sum(all_gpu_loss[gpu_i]).backward()

        self.trainer.step(self.batch_size)
        self.record_to_tensorboard_and_save(all_gpu_loss[0])

    def render_and_train(self):
        print('\033[1;33;40mRender And Train\033[0m')
        '''
        #################### show training image ####################
        ax = []
        plt.ion()
        fig = plt.figure()
        for i in range(self.batch_size):
            ax.append(fig.add_subplot(2,self.batch_size//2+2,i+1))
        '''

        ################## load data ##################
        BG_iter = self.load_BG('train', self.batch_size)
        car_renderer = RenderCar(self.batch_size, self.size[0], self.size[1], ctx[0])
        #addLP = AddLP(self.size[0], self.size[1], self.num_class)

        while True:
            if self.backward_counter % 10 == 0:
                background = ImageIter_next_batch(BG_iter).as_in_context(ctx[0])
            ################## render dataset ##################
            img_batch, label_batch = renderer.render(background)
            batch_xs, batch_ys = split_render_data(img_batch, label_batch, ctx)
            #if np.random.rand() > 0.2:
            #   batch_x[0], batch_y[0] = addLP.add(batch_x[0], batch_y[0])

            ################## training ##################
            for _ in range(3):
                self.train_the(batch_xs, batch_ys)

            '''
            #################### show training image ####################
            ax[0].imshow(batch_xs[0][0].transpose((1,2,0)).asnumpy())
            print(batch_ys[0][0])
            raw_input()
            '''
            '''
            ################## render dataset ##################
            img_batch, label_batch = renderer.render(background, pascal=True, mode='train')
            batch_xs, batch_ys = split_render_data(img_batch, label_batch, ctx)
            ################## training ##################
            for _ in range(1):
                self.train_the(batch_xs, batch_ys, r=0)
            '''
    
    def record_to_tensorboard_and_save(self, loss):
        for i, L in enumerate(loss):
            #self.sw.add_scalar(tag='Loss'+str(i), value=nd.mean(L).asnumpy(), global_step=self.backward_counter)
            self.sw.add_scalar('Loss', (self.loss_name[i],nd.mean(L).asnumpy()), self.backward_counter)
        self.backward_counter += 1
        if self.backward_counter%self.record_step==0:
            save_model = os.path.join(self.backup_dir, 'iter'+'_%d'%self.backward_counter)
            self.net.collect_params().save(save_model)  
    
    def load_BG(self, train_or_val, bs, **kargs):
        path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/'

        BG_iter = mxnet.image.ImageIter(bs, (3, self.size[0], self.size[1]),
            path_imgrec=path+'sun2012_' + train_or_val + '.rec',
            path_imgidx=path+'sun2012_' + train_or_val + '.idx',
            shuffle=True, pca_noise=0, 
            brightness=0.5, saturation=0.5, contrast=0.5, hue=1.0,
            rand_crop=True, 
            rand_resize=True, 
            rand_mirror=True, 
            inter_method=10, 
            **kargs
        )

        BG_iter.reset()
        return BG_iter          
    
    def _init_valid(self):
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
        C_pred, L_pred = self.net(x)

        C_score = nd.sigmoid(C_pred[0])
        C_box = self.yxhw_to_ltrb(C_pred[1])
        
        Cout = nd.concat(C_score, C_box, C_pred[2], C_pred[3], dim=-1)
        Cout = Cout.reshape((-1, 6+self.num_class))
        C_1 = C_score.reshape(-1).argmax(axis=0).reshape(-1)
        Cout = Cout[C_1][0].asnumpy()

        y = (Cout[2] + Cout[4])/2
        x = (Cout[1] + Cout[3])/2
        h = (Cout[4] - Cout[2])
        w = (Cout[3] - Cout[1])
        
        Cout[1:5] = [y,x,h,w]
        
        return Cout #[score,y,x,h,w,r,........]

    def get_default_anchors(self):
        import module.IOU_Kmeans as kmeans 
        bs = 2000
        car_renderer = RenderCar(bs, self.size[0], self.size[1], ctx[0])
        addLP = AddLP(self.size[0], self.size[1], self.num_class)

        BG = nd.zeros((bs,3,320,512), ctx=gpu(0)) # b*RGB*h*w
        img, label = car_renderer.render(BG, prob=1.0)
        #img, label = addLP.add(img, label)
        label = label.reshape(-1,6)[:,3:5]

        ans = kmeans.main(label, 9)
        print(ans)
        for a in ans:
            b = a.asnumpy()
            print(b[0]*b[1])
        while 1:
            time.sleep(0.1)

    def valid(self, bs=1):
        print('\033[1;33;40m Valid \033[0;37;40m')
        bs = 1
        BG_iter = self.load_BG('val', bs)
        render_car = RenderCar(bs, self.size[0], self.size[1], ctx[0])
        #addLP = AddLP(self.size[0], self.size[1], self.num_class)

        plt.ion()
        fig = plt.figure()
        ax = [None] * bs * 2
        for i in range(bs):
            ax[i] = fig.add_subplot(2,bs,i+1)
            ax[i+bs] = fig.add_subplot(2,bs,i+1+bs, polar=True)

        for BG in BG_iter:
            BG = BG.data[0].as_in_context(ctx[0]) # b*RGB*w*h
            t = time.time()
            img, label = car_renderer.render(BG, pascal=False, prob=1.0)

            #img, label = addLP.add(img, label)

            img = nd.clip(img, 0, 1)
            Cout = self.predict(img)

            for i in range(bs):
                ax[i].clear()
                im = img[i].transpose((1,2,0)).asnumpy()
                car_label = label[i,0].asnumpy()
                #LP_label = label[i,1].asnumpy()
                #im = cv2_add_bbox(im, LP_label, [0,0,1]) # Blue box
                im = cv2_add_bbox(im, car_label, [0,1,0]) # Green box
                im = cv2_add_bbox(im, Cout, [1,0,0]) # Red box
                vec_ang, vec_rad, prob = cls2ang(Cout[0], Cout[-self.num_class:])
                plt_radar_prob(ax[i+bs], vec_ang, vec_rad, prob)
                ax[i].imshow(im)
                ax[i].axis('off')

            raw_input('next')

    
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
            
        '''
        predictions = nd.uniform(low=0, high=1, shape=(100,), dtype=np.float32)
        labels = nd.uniform(low=0, high=2, shape=(100,), dtype=np.float32).astype(np.int32)
        print(labels)
        print(predictions)
        sw1.add_pr_curve(tag='pseudo_pr_curve', predictions=predictions, labels=labels, num_thresholds=120)
        '''


######################### Main #########################
if __name__ == '__main__':
    main()


