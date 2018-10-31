import argparse
import math
import numpy as np
import os
import threading
import time
# ros 
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError
# MXNet
from mxnet import gluon
from mxnet import gpu, cpu
from mxnet import image
from mxnet import init
from mxnet import nd
# cv packages
import cv2
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import PIL.Image as PILIMG
from PIL import ImageFilter, ImageEnhance


cos_offset = np.array([math.cos(x*math.pi/180) for x in range(0,360,30)])
sin_offset = np.array([math.sin(x*math.pi/180) for x in range(0,360,30)])

softmax = lambda x: np.exp(x)/np.sum(np.exp(x),axis=0)
nd_inv_sigmoid = lambda x: -nd.log(1/x - 1)


def Parser():
    parser = argparse.ArgumentParser(prog="python YOLO.py")
    parser.add_argument("version", help="v1")
    parser.add_argument("mode", help="train or valid or video")

    parser.add_argument("-t", "--topic", help="ros topic to subscribe", dest="topic", default="")
    parser.add_argument("--radar", help="show radar plot", dest="radar", default=False, type=bool)
    parser.add_argument("--show", help="show processed image", dest="show", default=True, type=bool)
    parser.add_argument("--gpu", help="gpu index", dest="gpu", default="0")

    args = parser.parse_args()
    return args


def batch_ys_ltrb2yxhw(batch_ys):
    new_batch_ys = []
    for batch_y in batch_ys:
        dim = batch_y.shape
        new_batch_y = nd.zeros((dim[0], dim[1], dim[2]+1), ctx=batch_y.context)
        new_batch_y[:,:,0] = batch_y[:,:,0]
        new_batch_y[:,:,1] = (batch_y[:,:,2] + batch_y[:,:,4])/2 #y
        new_batch_y[:,:,2] = (batch_y[:,:,1] + batch_y[:,:,3])/2 #x
        new_batch_y[:,:,3] = batch_y[:,:,4] - batch_y[:,:,2] #h
        new_batch_y[:,:,4] = batch_y[:,:,3] - batch_y[:,:,1] #w
        new_batch_ys.append(new_batch_y)
    return new_batch_ys

def get_iou(predict, target, mode=1):
    '''
    @input:
        predict: m*n*4, 
        target :(cltrb), 
        mode   :1:target is cltrb 
                2:target is cyxhw 
    @return
        (m*n*1) ndarray
    '''
    l, t, r, b = predict.split(num_outputs=4, axis=-1)
    if mode == 1:
        l2 = target[1]
        t2 = target[2]
        r2 = target[3]
        b2 = target[4]
    elif mode == 2:
        l2 = target[2] - target[4]/2
        t2 = target[1] - target[3]/2
        r2 = target[2] + target[4]/2
        b2 = target[1] + target[3]/2
    else: print('mode should be int 1 or 2')

    i_left = nd.maximum(l2, l)
    i_top = nd.maximum(t2, t)
    i_right = nd.minimum(r2, r)
    i_bottom = nd.minimum(b2, b)
    iw = nd.maximum(i_right - i_left, 0.)
    ih = nd.maximum(i_bottom - i_top, 0.)
    inters = iw * ih
    predict_area = (r-l)*(b-t)
    target_area = target[3] * target[4]
    ious = inters/(predict_area + target_area - inters) 
    return ious # 1344x3x1
def init_NN(target, pretrain, ctx):
    try:
        target.collect_params().load(pretrain, ctx=ctx)
    except:
        print('\033[1;31mLoad Pretrain Fail\033[0m')
        target.initialize(init=init.Xavier(), ctx=ctx)

    target.hybridize()
def assign_batch(batch, ctx):
    if len(ctx) > 1:
        batch_xs = gluon.utils.split_and_load(batch.data[0], ctx)
        batch_ys = gluon.utils.split_and_load(batch.label[0], ctx)
    else:
        batch_xs = [batch.data[0].as_in_context(ctx[0])] # b*RGB*w*h
        batch_ys = [batch.label[0].as_in_context(ctx[0])] # b*L*5   
    return batch_xs, batch_ys
def load_ImageDetIter(path, batch_size, h, w):
    print('Loading ImageDetIter ' + path)
    batch_iter = image.ImageDetIter(batch_size, (3, h, w),
        path_imgrec=path+'.rec',
        path_imgidx=path+'.idx',
        shuffle=True,
        pca_noise=0.1, 
        brightness=0.5,
        saturation=0.5, 
        contrast=0.5, 
        hue=1.0
        #rand_crop=0.2,
        #rand_pad=0.2,
        #area_range=(0.8, 1.2),
        )
    return batch_iter
def get_iterators(data_root, data_shape, batch_size, TorV):
    print('Loading Data....')
    if TorV == 'train':
        batch_iter = image.ImageDetIter(
            batch_size=batch_size,
            data_shape=(3, data_shape[0], data_shape[1]),
            path_imgrec=os.path.join(data_root, TorV+'.rec'),
            path_imgidx=os.path.join(data_root, TorV+'.idx'),
            shuffle=True,
            brightness=0.5, 
            contrast=0.2, 
            saturation=0.5, 
            hue=1.0,
            )
    elif TorV == 'valid':
        batch_iter = image.ImageDetIter(
            batch_size=1,
            data_shape=(3, data_shape[0], data_shape[1]),
            path_imgrec=os.path.join(data_root, 'train.rec'),
            path_imgidx=os.path.join(data_root, 'train.idx'),
            shuffle=True,
            #rand_crop=0.2,
            #rand_pad=0.2,
            #area_range=(0.8, 1.2),
            brightness=0.2, 
            #contrast=0.2, 
            saturation=0.5, 
            #hue=1.0,
            )
    else:
        batch_iter = None
    return batch_iter
    
def add_bbox(im, b, c):
    # Parameters:
    #### ax  : plt.ax
    #### size: [h,w]
    #### pred: numpy.array [score, l, t, r, b, prob1, prob2, ...]
    r = 0
    h = b[3]*im.shape[0]
    w = b[4]*im.shape[1]
    a = np.array([[
        [ w*math.cos(r)/2 - h*math.sin(r)/2,  w*math.sin(r)/2 + h*math.cos(r)/2],
        [-w*math.cos(r)/2 - h*math.sin(r)/2, -w*math.sin(r)/2 + h*math.cos(r)/2],
        [-w*math.cos(r)/2 + h*math.sin(r)/2, -w*math.sin(r)/2 - h*math.cos(r)/2],
        [ w*math.cos(r)/2 + h*math.sin(r)/2,  w*math.sin(r)/2 - h*math.cos(r)/2]]])
    s = np.array([b[2], b[1]])*[im.shape[1],im.shape[0]]
    a = (a + s).astype(int)
    cv2.polylines(im, a, 1, c, 2)
    #cv2.putText(img, '%s %.3f'%(text, p[0]), (l, t-10), 2, 1, c, 2)
    return im

def plt_radar_prob(ax, vec_ang, vec_rad, prob):
    ax.clear()
    cls_num = len(prob)
    ang = np.linspace(0, 2*np.pi, cls_num, endpoint=False)
    ang = np.concatenate((ang, [ang[0]]))

    prob = np.concatenate((prob, [prob[0]]))
    
    ax.plot([0,vec_ang], [0,vec_rad], 'r-', linewidth=3)
    ax.plot(ang, prob, 'b-', linewidth=1)
    ax.set_ylim(0,1)
    ax.set_thetagrids(ang*180/np.pi)

def cls2ang(confidence, prob):
    prob = softmax(prob)
    c = sum(cos_offset*prob)
    s = sum(sin_offset*prob)
    vec_ang = math.atan2(s,c)
    vec_rad = confidence*(s**2+c**2)**0.5

    prob = confidence * prob
    return vec_ang, vec_rad, prob
    
def calibration():
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError

    rospy.init_node('image_converter', anonymous=True)
    image_pub = rospy.Publisher("/img",Image)
    bridge = CvBridge()
    w, h = 480, 270
    #cap = open_cam_onboard(w, h)
    cap = cv2.VideoCapture(0)
    while not rospy.is_shutdown():
        _, img = cap.read()
        image_pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))


class PILImageEnhance():
    def __init__(self, M=0, N=0, R=0, G=1, noise_var=50):
        self.M = M
        self.N = N
        self.R = R
        self.G = G
        self.noise_var = noise_var

    def __call__(self, img):
        r = 0
        if self.M>0 or self.N>0:
            img = self.random_shearing(img)
        if self.R != 0:
            img, r = self.random_rotate(img)
        if self.G != 0:
            img = self.random_blur(img)
        if self.noise_var != 0:
            img = self.random_noise(img)
        return img, r

    def random_shearing(self, img):
        #https://stackoverflow.com/questions/14177744/
        #how-does-perspective-transformation-work-in-pil
        M, N = self.M, self.N
        w, h = img.size

        m, n = np.random.random()*M*2-M, np.random.random()*N*2-N # +-M or N
        xshift, yshift = abs(m)*h, abs(n)*w

        w, h = w + int(round(xshift)), h + int(round(yshift))
        img = img.transform((w, h), PILIMG.AFFINE, 
            (1, m, -xshift if m > 0 else 0, n, 1, -yshift if n > 0 else 0), 
            PILIMG.BILINEAR)

        return img

    def random_noise(self, img):
        np_img = np.array(img)
        noise = np.random.normal(0., self.noise_var, np_img.shape)
        np_img = np_img + noise 
        np_img = np.clip(np_img, 0, 255)
        img = PILIMG.fromarray(np.uint8(np_img))
        return img

    def random_rotate(self, img):
        r = np.random.random() * self.R * 2 - self.R
        img = img.rotate(r, PILIMG.BILINEAR, expand=1)
        r = r*np.pi/180
        return img, r
    
    def random_blur(self, img):
        img = img.filter(ImageFilter.GaussianBlur(radius=np.random.rand()*self.G))
        return img

class Video():              
    def _init_ros(self):
        rospy.init_node("YOLO_ros_node", anonymous = True)
        self.YOLO_img_pub = rospy.Publisher('/YOLO/img', Image, queue_size=1)
        self.YOLO_box_pub = rospy.Publisher('/YOLO/box', Float32MultiArray, queue_size=1)

        self.bridge = CvBridge()
        self.mat = Float32MultiArray()
        self.mat.layout.dim.append(MultiArrayDimension())
        self.mat.layout.dim.append(MultiArrayDimension())
        self.mat.layout.dim[0].label = "box"
        self.mat.layout.dim[1].label = "predict"
        self.mat.layout.dim[0].size = self.topk
        self.mat.layout.dim[1].size = 7
        self.mat.layout.dim[0].stride = self.topk*7
        self.mat.layout.dim[1].stride = 7
        self.mat.data = [-1]*7*self.topk

        self.miss_counter = 0
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')#(*'MJPG')#(*'MPEG')
        #self.out = cv2.VideoWriter('./video/car_rotate.mp4', fourcc, 30, (640, 360))
    
    def _get_frame(self):
        #cap = open_cam_onboard(self.cam_w, self.cam_w)
        cap = cv2.VideoCapture(0)
        #cap = cv2.VideoCapture('/home/nolan/Desktop/mxnet/video/DJI_0048.MP4')
        while not rospy.is_shutdown():
            ret, self.img = cap.read()
            #img = cv2.flip(img, -1)
        cap.release()
    
    def _image_callback(self, img):
        ##################### Convert and Predict #####################
        img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        self.img = img
    
    def run(self, topic=False, show=True, radar=False, ctx=gpu(0)):
        self.radar = radar
        self.show = show

        self.topk = 1
        self._init_ros()
        self.resz = image.ForceResizeAug((self.size[1], self.size[0]))

        if radar:
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111, polar=True)
            
            self.ax.grid(True)

        if not topic:
            threading.Thread(target=self._get_frame).start()
            print('\033[1;33;40m Use USB Camera')
        else: 
            rospy.Subscriber(topic, Image, self._image_callback)
            print('\033[1;33;40m Image Topic: %s\033[0m'%topic)
        
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if hasattr(self, 'img'):
                nd_img = nd.array(self.img)
                nd_img = self.resz(nd_img).as_in_context(ctx)

                nd_img =  nd_img.transpose((2,0,1)).expand_dims(axis=0)/255.        
                out = self.predict(nd_img)
                self.visualize(out)
                #rate.sleep()
            #else: print('Wait For Image')      
        
class RenderCar():
    def __init__(self, batch_size, img_h, img_w, ctx):
        self.h = img_h
        self.w = img_w  
        self.bs = batch_size
        self.ctx = ctx
        self.BIL = PILIMG.BILINEAR

        self.all_img = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        #png_img_path = os.path.join(base_dir, 'syn_images')
        png_img_path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/insulator_raw_images'

        for img in os.listdir(png_img_path):
            img_path = os.path.join(png_img_path, img)
            self.all_img.append(img_path)

        self.pil_image_enhance = PILImageEnhance(G=0.1, noise_var=2)
        self.augs = image.CreateAugmenter(data_shape=(3, img_h, img_w), 
            inter_method=10, brightness=0.1, contrast=0.3, saturation=0.3, hue=1.0, pca_noise=0.1)
        print('Loading background!')
        
    def render(self, bg):
        '''
        input:
            bg: background ndarray, bs*channel*h*w
        output:
            img_batch: bg add car
            label_batch: bs*object*[cls, y(0~1), x(0~1), h(0~1), w(0~1), r(+-pi)]
        '''
        ctx = self.ctx
        label_batch = nd.ones((self.bs,1,5), ctx=ctx) * (-1) 
        img_batch = nd.zeros((self.bs,3,self.h,self.w), ctx=ctx)
        mask = nd.zeros((self.bs,3,self.h,self.w), ctx=ctx)
        selected = np.random.randint(len(self.all_img), size=self.bs)

        for i in range(self.bs):
            if np.random.rand() > 0.8: continue

            img_path = self.all_img[selected[i]]
            pil_img = PILIMG.open(img_path)
            #################### resize ####################
            

            box_l, box_t, box_r, box_b = pil_img.getbbox()
            box_w = box_r - box_l
            box_h = box_b - box_t

            paste_x = np.random.randint(low=int(-0.2*box_w-box_l), high=int(self.w-0.8*box_w-box_l))
            paste_y = np.random.randint(low=int(-0.2*box_h-box_t), high=int(self.h-0.8*box_h-box_t))
            box_x = (box_r + box_l)/2 + paste_x #+ (box_w*math.cos(r) + abs(box_w*math.sin(r)) - box_w)/2
            box_y = (box_b + box_t)/2 + paste_y #+ (abs(box_w*math.sin(r)) + box_w*math.cos(r) - box_h)/2

            tmp = PILIMG.new('RGBA', (self.w, self.h))
            tmp.paste(pil_img, (paste_x, paste_y))
            m = nd.array(tmp.split()[-1], ctx=ctx).reshape(1, self.h, self.w)
            mask[i] = nd.tile(m, (3,1,1))/255.

            tmp, _ = self.pil_image_enhance(tmp)

            fg = PILIMG.merge("RGB", (tmp.split()[:3]))
            
            fg = nd.array(fg)

            #for aug in self.augs: fg = aug(fg)

            img_batch[i] = fg.as_in_context(ctx).transpose((2,0,1))

            img_cls = int(img_path.split('_')[-1].split('.png')[0])
            #label_batch[i] = nd.array([[img_cls, box_l/self.w, box_t/self.h, box_r/self.w, box_b/self.h]])
            label_batch[i] = nd.array([[
                img_cls, 
                float(box_y)/self.h, 
                float(box_x)/self.w, 
                float(box_h)/self.h, 
                float(box_w)/self.w
            ]])


        img_batch = (bg * (1-mask) + img_batch * mask)/255.
        #img_batch = nd.where(mask<200, bg, img_batch)/255.
        img_batch = nd.clip(img_batch, 0, 1) # 0~1 (batch_size, channels, h, w)
        return img_batch, label_batch

    def test(self):
        ctx=[gpu(0)]

        plt.ion()
        fig = plt.figure()
        ax = []
        for i in range(self.bs):
            ax.append(fig.add_subplot(1,1,1+i)) 
            #ax.append(fig.add_subplot(4,4,1+i)) 
        t=time.time()

        background_iter = image.ImageIter(self.bs, (3, self.h, self.w),
            path_imgrec='/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/sun2012_val.rec',
            path_imgidx='/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/sun2012_val.idx',
            shuffle=True, pca_noise=0, 
            brightness=0.5, saturation=0.5, contrast=0.5, hue=1.0,
            rand_crop=True, rand_resize=True, rand_mirror=True, inter_method=10
            )
        while True: # bg:0~255
            bg = background_iter.next().data[0].as_in_context(ctx[0])
            
            #img_batch, label_batch = self.render_pascal(bg, 'train')
            img_batch, label_batch = self.render(bg)
            #img_batch, label_batch = add_LP.add(img_batch, label_batch)
            for i in range(self.bs):
                ax[i].clear()

                im = img_batch[i].transpose((1,2,0)).asnumpy()
                b = label_batch[i,0].asnumpy()
                print(b)
                im = add_bbox(im,b,[0,0,1])

                ax[i].imshow(im)
                ax[i].axis('off')

            raw_input('next')
    
if __name__ == '__main__':
    B = RenderCar(1, 64*3, 64*4, gpu(0))
    B.test()
