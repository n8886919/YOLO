import math
import numpy as np
import os
import scipy.io as sio
import sys

import mxnet
from mxnet import gpu
from mxnet import nd

import PIL

from yolo_modules import pil_image_enhancement
from yolo_modules import global_variable

rad_2_deg = lambda rad: rad * 180. / math.pi
deg_2_rad = lambda deg: deg * math.pi / 180.


class RenderCar():
    def __init__(self, batch_size, img_h, img_w, ctx):
        self.bs = batch_size
        self.h = img_h
        self.w = img_w

        self.ctx = ctx
        self.BIL = PIL.Image.BILINEAR

        self.pil_image_enhance = pil_image_enhancement.PILImageEnhance(M=0, N=0, R=30.0, G=0.3, noise_var=0)
        self.augs = mxnet.image.CreateAugmenter(data_shape=(3, img_h, img_w),   
            inter_method=10, brightness=0.5, contrast=0.5, saturation=0.5, hue=1.0, pca_noise=0.1)

        self.rawcar_dataset = {'train':[], 'valid':[]}
        raw_car_train_path = global_variable.training_data_path + '/HP_31/rawcar24/rawcar_train'
        raw_car_valid_path = global_variable.training_data_path + '/HP_31/rawcar24/rawcar_valid'
        for file in os.listdir(raw_car_train_path):
            for img in os.listdir(os.path.join(raw_car_train_path, file)):
                img_path = os.path.join(raw_car_train_path, file, img)
                self.rawcar_dataset['train'].append(img_path)

        for file in os.listdir(raw_car_valid_path):
            for img in os.listdir(os.path.join(raw_car_valid_path, file)):
                img_path = os.path.join(raw_car_valid_path, file, img)
                self.rawcar_dataset['valid'].append(img_path)
        '''
        self.pascal_dataset = {'train':[], 'valid':[]}
        self.pascal3d_anno = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/pascal_image_and_label/car_imagenet'
        pascal3d_train_path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/pascal_image_and_label/pascal_train'
        pascal3d_valid_path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/pascal_image_and_label/pascal_valid'
        for img in os.listdir(pascal3d_train_path):
            img_path = os.path.join(pascal3d_train_path, img)
            self.pascal_dataset['train'].append(img_path)

        for img in os.listdir(pascal3d_valid_path):
            img_path = os.path.join(pascal3d_valid_path, img)
            self.pascal_dataset['valid'].append(img_path)
        '''
        print('Loading background!')

    def render(self, bg, mode='valid', prob=0.8, pascal=False, num_cls=24):
        '''
        Parameters
        ----------
        bg: mxnet.ndarray(4D) 
          background array, 
          dimension = bs * channel * h * w
        mode: {'train', 'valid'}

        Returns
        ----------
        img_batch: mxnet.ndarray(4D) 

        label_batch: mxnet.ndarray(3D) 
          bs * object * [cls, y(0~1), x(0~1), h(0~1), w(0~1), r(+-pi)]
        '''
        dataset = self.pascal_dataset[mode] if pascal else self.rawcar_dataset[mode]

        ctx = self.ctx
        label_batch = nd.ones((self.bs,1,6), ctx=ctx) * (-1)
        img_batch = nd.zeros((self.bs,3,self.h,self.w), ctx=ctx)
        mask = nd.zeros((self.bs,3,self.h,self.w), ctx=ctx)
        selected = np.random.randint(len(dataset), size=self.bs)

        for i in range(self.bs):
            if np.random.rand() > prob: 
                continue

            img_path = dataset[selected[i]]

            if pascal:
                img_cls, box_l, box_t, box_r, box_b, skip = self.get_pascal3d_label(img_path, num_cls)
                if skip: continue
            else:
                img_cls = int(img_path.split('_no')[0].split('abel')[1])

            pil_img = PIL.Image.open(img_path).convert('RGBA')
            r1 = np.random.uniform(low=0.9, high=1.1)

            if pascal:
                box_w = box_r - box_l
                box_h = (box_b - box_t) * r1

                w_max_scale = 0.9*self.w / box_w
                h_max_scale = 0.9*self.h / box_h
                max_scale = min(w_max_scale, h_max_scale)

                w_min_scale = 0.2*self.w / float(box_w)
                h_min_scale = 0.2*self.h / float(box_h)
                min_scale = max(w_min_scale, h_min_scale)


            else:
                min_scale = 0.25
                max_scale = 1.0
            #################### resize ####################
            resize = np.random.uniform(low=min_scale, high=max_scale)
            resize_w = resize * pil_img.size[0]
            resize_h = resize * pil_img.size[1] * r1
            pil_img = pil_img.resize((int(resize_w), int(resize_h)), self.BIL)
            #################### resize ####################
            if pascal:
                box_w = resize * box_w
                box_h = resize * box_h * r1

                pil_img, r = self.pil_image_enhance(pil_img, R=0)

                box_l2 = box_l * resize - 0.5 * resize_w
                box_r2 = box_r * resize - 0.5 * resize_w
                box_t2 = box_t * resize * r1 - 0.5 * resize_h
                box_b2 = box_b * resize * r1 - 0.5 * resize_h
                # box_x2 means origin at image center

                new_corner = []
                for x in [box_l2, box_r2]:
                    for y in [box_t2, box_b2]:
                        rotated_corner = [x*math.cos(r)-y*math.sin(r), y*math.cos(r)+x*math.sin(r)]
                        new_corner.append(rotated_corner)
                
                r_resize_w = abs(resize_h * math.sin(r)) + abs(resize_w * math.cos(r))
                r_resize_h = abs(resize_h * math.cos(r)) + abs(resize_w * math.sin(r))

                offset = np.array([r_resize_w, r_resize_h]) * 0.5
                r_box_l, r_box_t = np.amin(new_corner, axis=0) + offset
                r_box_r, r_box_b = np.amax(new_corner, axis=0) + offset

            else:
                box_l, box_t, box_r, box_b = pil_img.getbbox()
                box_w = box_r - box_l
                box_h = box_b - box_t
                pil_img, r = self.pil_image_enhance(pil_img)
                r_box_l, r_box_t, r_box_r, r_box_b = pil_img.getbbox()

            r_box_w = r_box_r - r_box_l # r_box_xx means after rotate       
            r_box_h = r_box_b - r_box_t # r_box_xx means after rotate       
                    

            ##################### move #####################
            paste_x = np.random.randint(
                low=int(-r_box_l-0.2*r_box_w), 
                high=int(self.w-r_box_l-0.8*r_box_w)
            )
            paste_y = np.random.randint(
                low=int(-r_box_t-0.2*r_box_h), 
                high=int(self.h-r_box_t-0.8*r_box_h)
            )
            
            box_x = (r_box_r + r_box_l)/2. + paste_x
            box_y = (r_box_b + r_box_t)/2. + paste_y 
            ####################################################################
            tmp = PIL.Image.new('RGBA', (self.w, self.h))
            tmp.paste(pil_img, (paste_x, paste_y))

            m = nd.array(tmp.split()[-1], ctx=ctx).reshape(1, self.h, self.w)
            mask[i] = nd.tile(m, (3,1,1)) / 255.

            fg = PIL.Image.merge("RGB", (tmp.split()[:3]))
            fg = nd.array(fg)####
            for aug in self.augs: fg = aug(fg)

            
            ####################################################################
            img_batch[i] = fg.as_in_context(ctx).transpose((2,0,1))
            label_batch[i] = nd.array([[
                img_cls,
                float(box_y)/self.h,
                float(box_x)/self.w,
                float(box_h)/self.h,
                float(box_w)/self.w,
                r
            ]])

        ####################################################################
        img_batch = (bg * (1 - mask) + img_batch * mask) / 255.####
        img_batch = nd.clip(img_batch, 0, 1) # 0~1 (batch_size, channels, h, w)####
        
        return img_batch, label_batch
        ####################################################################
        #return 0, label_batch
    def get_pascal3d_label(self, img_path, num_cls):
        f = img_path.split('/')[-1].split('.')[0]+'.mat'
        mat = sio.loadmat(os.path.join(self.pascal3d_anno, f))
        mat = mat['record'][0][0]

        skip = False
        for mi, m in enumerate(mat):
            if mi == 1:
                label = [[],[]]
                for ni, n in enumerate(m[0]):
                    for pi, p in enumerate(n):      
                        if pi == 1: 
                            box = [int(i) for i in p[0]] 
                            label[0].append(box)    
                            #print('\t\t\t{}\t{}'.format(p[0][2]-p[0][0],p[0][3]-p[0][1]), end='')
                            #print('\t\t\t{}\t{}\t{}\t{}'.format(*p[0]), end='')
                        if pi == 3: 
                            for qi, q in enumerate(p[0][0]):
                                if qi == 2:
                                    img_cls = int((q[0][0]+float(360/num_cls)/2)/15)
                                    if img_cls>num_cls-1:
                                        img_cls=num_cls-1
                                    #print('\t{}'.format(img_cls))
                                    label[1].append(img_cls)

        # if more then one car in an image, do not use it, so skip 
        if len(label[1]) != 1: 
            skip = True

        box_l = label[0][0][0]
        box_t = label[0][0][1]
        box_r = label[0][0][2]
        box_b = label[0][0][3]

        return label[1][0], box_l, box_t, box_r, box_b, skip
    
    def test(self):
        ctx=[gpu(0)]
        add_LP = AddLP(self.h, self.w, -1)

        plt.ion()
        fig = plt.figure()
        ax = []
        for i in range(self.bs):
            ax.append(fig.add_subplot(1,1,1+i)) 
            #ax.append(fig.add_subplot(4,4,1+i)) 
        t = time.time()
        background_iter = mxnet.image.ImageIter(self.bs, (3, self.h, self.w),
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
                im = cv2_add_bbox(im,b,[0,0,1])

                ax[i].imshow(im)
                ax[i].axis('off')

            raw_input('next')

if __name__ == '__main__':
    pass
