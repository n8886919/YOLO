from mxnet import autograd
from mxnet import init
from utils import *

ctx=[gpu(0), gpu(1)]
LR = 0.0001
topk = 7
batch_size = 100*len(ctx)
size = [160, 384]
cls_names = ['0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','J','K','L','M',
    'N','P','Q','R','S','T','U','V','W','X','Y','Z']
NN_name = 'OCR'
get_feature = OCR(len(cls_names))
pretrain = 'OCR/backup/OCR_100'
record_step = 500
try:
    get_feature.collect_params().load(pretrain, ctx=ctx)
except:
    print('\033[1;31mLoad Pretrain Fail\033[0m')
    get_feature.initialize(init=init.Xavier(), ctx=ctx)

get_feature.hybridize()
def psr():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="train or valid or val")
    args = parser.parse_args()
    return args

class Train():
    def __init__(self):
        from mxboard import SummaryWriter

        self.CE_loss = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)
        self.LG_loss = gluon.loss.LogisticLoss(label_format='binary')
        self.L2_loss = gluon.loss.L2Loss()
        self.trainer = gluon.Trainer(get_feature.collect_params(), 
            'adam', {'learning_rate': LR})

        self.sw = SummaryWriter(logdir=NN_name + '/logs')#, flush_secs=30)
        #prob, score  = get_feature(nd.zeros((1,3,160,384), ctx=ctx[0]))
        #print(prob.shape, score.shape)
        #self.sw.add_graph(get_feature)
        self.backup_dir = NN_name + '/backup/'
        if not os.path.exists(self.backup_dir): os.makedirs(self.backup_dir)
    def loss_mask(self, labels, gpu_index):
        a = 24
        bs = labels.shape[0]
        T_class = nd.ones((bs, a, 1), ctx=ctx[gpu_index]) * (-1)
        T_score = nd.zeros((bs, a, 1), ctx=ctx[gpu_index])
        T_mask = nd.zeros((bs, a, 1), ctx=ctx[gpu_index])
        for b in range(bs): 
            label = labels[b] 
            #nd.random.shuffle(label)
            for L in label: # all object in the image
                if L[0] < 0: continue
                left = int(round(L[1].asnumpy()[0]*a))
                right = int(round(L[3].asnumpy()[0]*a))
                for i in range(left, right):
                    text_cent = (L[3] + L[1]) / 2
                    box_cent = (i + 0.5) / a
                    score = 1-nd.abs(box_cent-text_cent)/(L[3]-L[1])
                    T_score[b, i, :] = score # others are zero
                    if score.asnumpy()[0] > 0.7:
                        T_class[b, i, 0] = L[0] # others are ignore_label=-1
                        T_mask[b, i, 0] = 1.0

        return T_score, T_class, T_mask
    def train_the(self, batch_xs, batch_ys):
        loss = []
        with autograd.record():
            for gpu_index, (batch_x, batch_y) in enumerate(zip(batch_xs, batch_ys)):
                prob, score = get_feature(batch_x) 
                with autograd.pause():
                    score_label, class_label, mask = self.loss_mask(batch_y, gpu_index)

                #score_loss = self.LG_loss(score, score_label*0.01)
                score_loss = self.L2_loss(score, score_label)
                class_loss = self.CE_loss(prob, class_label, mask)      
                loss.append(score_loss + class_loss)

        for l in loss: l.backward()
        self.trainer.step(batch_size)
        self.record_to_tensorboard([score_loss, class_loss])
    def record_to_tensorboard(self, loss):
        for i, L in enumerate(loss):
            self.sw.add_scalar(tag=str(i), value=nd.mean(L).asnumpy(), global_step=self.ct)
        self.ct += 1
        if self.ct%record_step==0:
            save_model = os.path.join(self.backup_dir, NN_name + '_%d'%self.ct)
            get_feature.collect_params().save(save_model)
    def render_and_train(self):
        print('\033[1;33;40mRender And Train\033[0m')
        addLP = AddLP(160, 384, 0)
        self.ct = 0
        while True:
            batch_xs, batch_ys = [], []
            for i, dev in enumerate(ctx):
                batch_x, batch_y = addLP.render(int(batch_size/len(ctx)), dev)
                batch_xs.append(batch_x)
                batch_ys.append(batch_y)
            for i in range(10):
                self.train_the(batch_xs, batch_ys)
            #input('next')
    def valid(self, n):
        print('\033[1;33;40mRender And Valid\033[0m')
        addLP = AddLP(160, 384, 0)
        plt.ion()
        fig = plt.figure()
        ax = []
        for i in range(20):
            ax.append(fig.add_subplot(5,4,1+i))

        batch_x = nd.zeros((20, 3, 160, 384), ctx=gpu(0))
        for i, im in enumerate(os.listdir('testimg')):
            im = os.path.join('testimg', im)
            im = PILIMG.open(im)
            im = im.resize((384, 160), PILIMG.BILINEAR)
            batch_x[i] = nd.array(im).transpose((2,0,1))/255.

        while True:
            #batch_x, batch_y = addLP.render(n, ctx[0])
            prob, score = get_feature(batch_x)
            for xi, x in enumerate(ax):
                #################
                #y = batch_y[xi]
                #s_label = addLP.label2nparray(y)

                s = score[xi].reshape(-1).asnumpy()
                p = nd.argmax(prob[xi], axis=-1).reshape(-1).asnumpy()
                x.clear()       
                x.imshow(batch_x[xi].transpose((1,2,0)).asnumpy())  
                x.plot(range(8,384,16),(1-s)*160)
                #x.plot(range(8,384,16),(1-s_label)*160, '-r')
                x.axis('off')
                t = ''
                s = np.concatenate(([0],s,[0]))
                for i in range(24):
                    if s[i+1] > 0.2 and s[i+1] > s[i+2] and s[i+1] > s[i]:
                        c = int(p[i])               
                        t = t + cls_names[c]
                print(t)

            input('next')

args = psr()
t = Train()
if args.mode == 'train':
    t.render_and_train()
elif args.mode == 'valid':
    t.valid(16)
    #t.real()
else: print('invalid args')