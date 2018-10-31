#!/usr/bin/env python
import copy
import sys
import yaml

from mxnet import autograd

from utils import *

sys.path.append('../')
from module.net import *

args = Parser()
ctx = [gpu(int(i)) for i in args.gpu]

topk = 1
version = args.version
batch_size = 300 * len(ctx)
record_step = 5000


def main():
    yolo = YOLOForInslator('iter_10000')

    if args.mode == 'train':
        yolo.render_and_train()

    elif args.mode == 'valid':
        yolo.valid()

    elif args.mode == 'video':
        yolo.run(ctx=ctx[0], topic=args.topic, radar=args.radar, show=args.show)

    elif args.mode == 'PR':
        yolo.pr_curve()

    else:
        print('Parser 2 should be train or valid or video')


class YOLOForInslator(Video):
    def __init__(self, pretrain):
        self.size = [64*3, 64*4]

        with open(os.path.join(args.version, 'spec.yaml')) as f:
            spec = yaml.load(f)

        num_ds = len(spec['layers']) # number of downsample

        self.steps = [2**(num_ds-2), 2**(num_ds-1), 2**num_ds] # [16, 32, 64]
        self.all_anchors = nd.array(spec['anchors'])
        self.cls_names = spec['classes']
    
        self.area = [int(self.size[0]*self.size[1]/step**2) for step in self.steps]
        
        self.net = Net(spec, 2)

        self.backup_dir = os.path.join(version, 'backup')

        pretrain_path = os.path.join(self.backup_dir, pretrain)

        init_NN(self.net, pretrain_path, ctx)

        if args.mode == 'train':
            self._init_train()

        else:
            self._init_valid()

    def _init_train(self):
        from mxboard import SummaryWriter
        self.record_step = record_step
        self.loss_name = ['score', 'box', 'class']

        self.nd_all_anchors = [self.all_anchors.copyto(device) for device in ctx]
        self.get_default_ltrb()

        self.L1_loss = gluon.loss.L1Loss()
        self.L2_loss = gluon.loss.L2Loss()
        self.LG_loss = gluon.loss.LogisticLoss(label_format='binary')
        self.CE_loss = gluon.loss.SoftmaxCrossEntropyLoss(
            from_logits=False, sparse_label=False)

        self.trainer = gluon.Trainer(self.net.collect_params(),
            'adam', {'learning_rate': 0.0001})

        self.sw = SummaryWriter(logdir=version+'/logs')#, flush_secs=30)
        #a = self.net(nd.zeros((1,3,self.size[0],self.size[1]), ctx=ctx[0]))
        #self.sw.add_graph(self.net)

        if not os.path.exists(self.backup_dir): 
            os.makedirs(self.backup_dir)

    def get_default_ltrb(self):
        n = len(self.all_anchors[0])
        LTRB = nd.zeros((sum(self.area),n,4))
        size = self.size
        a_start = 0
        for i, anchors in enumerate(self.all_anchors): # [12*16,6*8,3*4]
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

            LTRB[a_start:a_start+a] = nd.concat(l,t,r,b, dim=-1)
            a_start += a

        self.all_anchors_ltrb = [LTRB.copyto(device) for device in ctx]
    
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
        sigmoid_ty = nd.clip(sigmoid_ty, 0.0001, 0.9999)
        ty = nd_inv_sigmoid(sigmoid_ty)

        bx_minus_cx = L[2] - (best_ltrb[2]+best_ltrb[0])/2
        sigmoid_tx = bx_minus_cx*self.size[1]/step + 0.5
        sigmoid_tx = nd.clip(sigmoid_tx, 0.0001, 0.9999)
        tx = nd_inv_sigmoid(sigmoid_tx)
        th = nd.log((L[3]) / self.nd_all_anchors[gpu_index][pyramid_layer, best_anchor, 0])
        tw = nd.log((L[4]) / self.nd_all_anchors[gpu_index][pyramid_layer, best_anchor, 1])
        return best_pixel, best_anchor, nd.concat(ty, tx, th, tw, dim=-1)
    
    def loss_mask(self, labels, gpu_index):
        """Generate training targets given predictions and labels.
        labels: bs*object*[class, cent_y, cent_x, box_h, box_w, rotate]
        """
        a1 = sum(self.area)
        bs = labels.shape[0]
        n = len(self.all_anchors[0])
        C_class = nd.zeros((bs, a1, n, len(self.cls_names)), ctx=ctx[gpu_index])
        #C_class = nd.ones((bs, a1, n, 1), ctx=ctx[gpu_index]) * (-1)
        C_score = nd.zeros((bs, a1, n, 1), ctx=ctx[gpu_index])
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
                    C_box[b, px, anc, :] = box
                    C_class[b, px, anc, L[0]] = 1.0
                    '''
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
                    '''
        return [C_score, C_box, C_class], C_mask

    def train_the(self, batch_xs, batch_ys):
        loss = []
        with autograd.record():
            for gpu_index, (batch_x, batch_y) in enumerate(zip(batch_xs, batch_ys)):
                C_pred = self.net(batch_x)
                with autograd.pause():
                    C_label, C_mask = self.loss_mask(batch_y, gpu_index)    

                    C_score_weight = nd.where(C_mask>0,
                        nd.ones_like(C_mask)*10.0,
                        nd.ones_like(C_mask)*0.1,
                        ctx=ctx[gpu_index])

                Csl = self.LG_loss(C_pred[0], C_label[0], C_score_weight * 0.1)
                Cbl = self.L2_loss(C_pred[1], C_label[1], C_mask * 1.0)
                Ccl = self.CE_loss(C_pred[2], C_label[2], C_mask * 0.1) #0.1 after 1day:1.0
                loss.append(Csl + Cbl + Ccl)

        for l in loss: l.backward()
        self.trainer.step(batch_size)
        
        self.record_to_tensorboard_and_save([Csl, Cbl, Ccl])

    def render_and_train(self):
        print('\033[1;33;40mRender And Train\033[0m')
        ################## load data ##################
        BG_iter = self.load_BG('train')

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
            if self.ct % 10 == 0:
                try:
                    BG = BG_iter.next().data[0].as_in_context(ctx[0])
                except:
                    BG_iter.reset()
                    BG = BG_iter.next().data[0].as_in_context(ctx[0])#
                #print(time.time() - t)
            
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
            for _ in range(2):
                self.train_the(batch_xs, batch_ys)

    def record_to_tensorboard_and_save(self, loss):
        for i, L in enumerate(loss):
            #self.sw.add_scalar(tag='Loss'+str(i), value=nd.mean(L).asnumpy(), global_step=self.ct)
            self.sw.add_scalar('Loss', (self.loss_name[i],nd.mean(L).asnumpy()), self.ct)
        self.ct += 1
        if self.ct%self.record_step==0:
            save_model = os.path.join(self.backup_dir, 'iter'+'_%d'%self.ct)
            self.net.collect_params().save(save_model)  
    
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
        C_pred = self.net(x)

        C_score = nd.sigmoid(C_pred[0])
        C_box = self.yxhw_to_ltrb(C_pred[1])
        
        Cout = nd.concat(C_score, C_box, C_pred[2], dim=-1)
        Cout = Cout.reshape((-1, 5+len(self.cls_names)))
        C_1 = C_score.reshape(-1).argmax(axis=0).reshape(-1)
        
        Cout = Cout[C_1][0].asnumpy()

        y = (Cout[2] + Cout[4])/2
        x = (Cout[1] + Cout[3])/2
        h = (Cout[4] - Cout[2])
        w = (Cout[3] - Cout[1])
        
        Cout[1:5] = [y,x,h,w]
        
        return Cout #[score,y,x,h,w,r,........]
    
    def load_BG(self, train_or_val, **kargs):

        path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/'
        if train_or_val == 'train':
            bs = batch_size
        elif train_or_val == 'val':
            bs = 1
        else:
            return 0
        BG_iter = image.ImageIter(bs, (3, self.size[0], self.size[1]),
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

    def valid(self):
        print('\033[1;33;40m Valid \033[0;37;40m')

        BG_iter = self.load_BG('val')
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
            plt.pause(0.001)

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


######################### Main #########################
if __name__ == '__main__':
    main()


