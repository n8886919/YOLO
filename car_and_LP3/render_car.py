import math
import numpy as np
import os
import scipy.io as sio
import sys
from pprint import pprint

import mxnet
from mxnet import gpu
from mxnet import nd

import PIL

from yolo_modules import pil_image_enhancement

join = os.path.join


class RenderCar():
    def __init__(self, img_h, img_w, classes, ctx, pre_load=False):
        self.h = img_h
        self.w = img_w
        self.num_cls = len(classes)
        self.ele_label = np.array(classes)[:, 1]
        self.azi_label = np.array(classes)[:, 0]
        self.ctx = ctx
        self.pre_load = pre_load

        self.BIL = PIL.Image.BILINEAR
        self.disk = '/media/nolan/SSD1'
        # -------------------- init image enhencement -------------------- #
        self.pil_image_enhance = pil_image_enhancement.PILImageEnhance(
            M=0, N=0, R=30.0, G=0.3, noise_var=0)
        self.augs = mxnet.image.CreateAugmenter(
            data_shape=(3, img_h, img_w), inter_method=10, pca_noise=0.1,
            brightness=0.5, contrast=0.5, saturation=0.5, hue=1.0)

        self.load_png_images()
        self.load_mtv_images()
        self.load_pascal_images()

    def render(self, bg, mode, pascal=True, render_rate=1.0):
        '''
        Parameters
        ----------
        bg: mxnet.ndarray(4D)
          background array,
          dimension = bs * channel * h * w
        mode: str, {'train', 'valid'}
          use training dataset or not
        pascal: boolean
          use pascal_3D dataset or not
        render_rate: float
          probability of image contain a car

        Returns
        ----------
        img_batch: mxnet.ndarray(4D)
          same as bg input
        label_batch: mxnet.ndarray(3D)
          bs * object * [cls, y(0~1), x(0~1), h(0~1), w(0~1), r(+-pi), all labels prob]
        '''
        bs = len(bg)
        ctx = self.ctx
        label_batch = nd.ones((bs, 1, 6+self.num_cls), ctx=ctx) * (-1)
        img_batch = nd.zeros((bs, 3, self.h, self.w), ctx=ctx)
        mask = nd.zeros((bs, 3, self.h, self.w), ctx=ctx)

        for i in range(bs):
            if np.random.rand() > render_rate:
                continue

            r1 = np.random.uniform(low=0.9, high=1.1)
            if pascal:
                pil_img, r_box_l, r_box_t, r_box_r, r_box_b, r, \
                    img_cls, label_distribution = self._render_pascal(mode, r1)

            else:
                pil_img, r_box_l, r_box_t, r_box_r, r_box_b, r, \
                    img_cls, label_distribution = self._render_png(mode, r1)

            r_box_w = r_box_r - r_box_l  # r_box_xx means after rotate
            r_box_h = r_box_b - r_box_t  # r_box_xx means after rotate

            # -------------------- move -------------------- #
            paste_x = np.random.randint(
                low=int(-r_box_l-0.3*r_box_w),
                high=int(self.w-r_box_l-0.7*r_box_w))

            paste_y = np.random.randint(
                low=int(-r_box_t-0.3*r_box_h),
                high=int(self.h-r_box_t-0.7*r_box_h))

            box_y = (r_box_b + r_box_t)/2. + paste_y
            box_x = (r_box_r + r_box_l)/2. + paste_x
            box_h = float(r_box_b - r_box_t)
            box_w = float(r_box_r - r_box_l)

            # -------------------- -------------------- #
            tmp = PIL.Image.new('RGBA', (self.w, self.h))
            tmp.paste(pil_img, (paste_x, paste_y))
            #tmp.show()
            m = nd.array(tmp.split()[-1], ctx=ctx).reshape(1, self.h, self.w)
            mask[i] = nd.tile(m, (3, 1, 1)) / 255.

            fg = PIL.Image.merge("RGB", (tmp.split()[:3]))
            fg = nd.array(fg)

            for aug in self.augs:
                fg = aug(fg)

            # -------------------- -------------------- #
            img_batch[i] = fg.as_in_context(ctx).transpose((2, 0, 1))

            label = nd.array([[
                img_cls,
                box_y/self.h, box_x/self.w,
                box_h/self.h, box_w/self.w, r]])

            label = nd.concat(label, label_distribution, dim=-1)
            label_batch[i] = label
        ####################################################################
        img_batch = (bg * (1 - mask) + img_batch * mask) / 255.
        img_batch = nd.clip(img_batch, 0, 1)
        # 0~1 (batch_size, channels, h, w)
        return img_batch, label_batch

    def load_mtv_images(self):
        self.mtv_dataset = []
        mtv_txt = []
        path = join(self.disk, 'muti_view_car')

        with open(join(path, 'tripod-seq.txt'), 'r') as f:
            mtv_txt = f.readlines()

        with open(join(path, 'times.txt'), 'r') as f:
            times = f.readlines()

        for i, seq_times in enumerate(times):
            seq_times = seq_times.split(' ')[:-1]  # [-1] is \n
            times[i] = map(int, seq_times)

        cycle_frame = map(int, mtv_txt[4].split(' '))

        num_seqs = len(mtv_txt[1].split(' '))
        for i in range(num_seqs):
            d_ang = 360. / times[i][cycle_frame[i]-1]  # d_ang / d_t
            d_ang *= int(mtv_txt[6].split(' ')[i])  # rotattion direction
            # cycle_frame[i]-1 to match list index
            # times[0][3] correspond to  time of tripod_seq_1_4.jpg

            front_frame_idx = int(mtv_txt[5].split(' ')[i])
            front_time = times[i][front_frame_idx - 1]
            ang_start = - d_ang * front_time

            box_file = join(path, mtv_txt[3] % (i+1))[:-1]
            # because mtv_txt[3][-1] is \n, remove!
            box_file = np.loadtxt(box_file)
            num_images = int(mtv_txt[4].split(' ')[i])
            for j in range(num_images):
                img_name = 'tripod_seq_%02d_%03d.jpg' % (i+1, j+1)
                # seq and frame number are start from 1, so (i, j) += 1
                img_path = join(path, img_name)
                box = box_file[j]
                frame_time = times[i][j]
                azi = ang_start + frame_time * d_ang
                img_cls, label_distribution = self.get_label_dist(0, azi)
                pil_img = PIL.Image.open(img_path).convert('RGBA')
                self.mtv_dataset.append([
                    pil_img,
                    box,
                    img_cls,
                    label_distribution])
            break

    def load_png_images(self):
        # -------------------- load png image and path-------------------- #
        #path = join(self.disk, 'color_material')
        path = join(self.disk, 'no_label_car_raw_images/100')
        cad_path = {
            'train': join(path, 'train'),
            'valid': join(path, 'valid')}
        self.rawcar_dataset = {'train': [], 'valid': []}

        if self.pre_load:
            print('\033[1;34mLoading png images to RAM')

        for mode in self.rawcar_dataset:
            for cad in os.listdir(cad_path[mode]):
                for img in os.listdir(join(cad_path[mode], cad)):
                    img_path = join(cad_path[mode], cad, img)
                    if self.pre_load:
                        ele = (float(img_path.split('ele')[1].split('.')[0]) * math.pi) / (100 * 180)
                        azi = (float(img_path.split('azi')[1].split('_')[0]) * math.pi) / (100 * 180)
                        img_cls, label_distribution = self.get_label_dist(ele, azi)
                        with PIL.Image.open(img_path).convert('RGBA') as f:
                            pil_img = np.array(f, dtype="uint8")
                        self.rawcar_dataset[mode].append([pil_img, img_cls, label_distribution])

                    else:
                        self.rawcar_dataset[mode].append(img_path)

    def load_pascal_images(self):
        # -------------------- set pascal path-------------------- #
        path = join(self.disk, 'HP_31/pascal3d_image_and_label')
        label_path = join(path, 'car_imagenet_label')
        pascal_path = {
            'train': join(path, 'car_imagenet_train'),
            'valid': join(path, 'car_imagenet_valid')}

        # -------------------- load pascal label -------------------- #
        self.pascal3d_anno = {}
        for f in os.listdir(label_path):
            self.pascal3d_anno[f] = sio.loadmat(join(label_path, f))

        if self.pre_load:
            print('\033[1;34mLoading pascal images to RAM')
        # -------------------- load pascal image -------------------- #
        self.pascal_dataset = {'train': [], 'valid': []}
        for mode in self.pascal_dataset:
            for img in os.listdir(pascal_path[mode]):
                img_path = join(pascal_path[mode], img)
                if self.pre_load:
                    ele, azi, box, skip = self.get_pascal3d_azi_ele(img_path)
                    if skip:
                        continue

                    img_cls, label_distribution = self.get_label_dist(ele, azi)
                    self.pascal_dataset[mode].append(
                        [PIL.Image.open(img_path).convert('RGBA'),
                         box,
                         img_cls,
                         label_distribution])
                else:
                    self.pascal_dataset[mode].append(img_path)
        '''
        if self.pre_load:
            print('\033[1;34mLoading pascal images to RAM')
            
            self.pascal_dataset = {'train': {}, 'valid': {}}
            for data in self.pascal_dataset:
                for img in os.listdir(pascal_path[data]):
                    img_path = join(pascal_path[data], img)
                    ele, azi, box, skip = self.get_pascal3d_azi_ele(img_path)
                    if skip:
                        continue

                    img_cls, label_distribution = self.get_label_dist(ele, azi)
                    self.pascal_dataset[data][img] = [
                        PIL.Image.open(img_path).convert('RGBA'),
                        box,
                        img_cls,
                        label_distribution]

            print('\033[1;34mDone')

        else:
            self.pascal_dataset = {'train': [], 'valid': []}
            for data in self.pascal_dataset:
                for img in os.listdir(pascal_path[data]):
                    img_path = join(pascal_path[data], img)
                    self.pascal_dataset[data].append(img_path)
        '''

    def _render_pascal(self, mode, r1=1.0, pre_load=False):
        n = np.random.randint(len(self.pascal_dataset[mode]))
        if self.pre_load:
            #n = self.pascal_dataset[mode].keys()[n]
            pil_img, box, img_cls, \
                label_distribution = self.pascal_dataset[mode][n]

        else:
            skip = True
            while skip:
                img_path = self.pascal_dataset[mode][n]
                ele, azi, box, skip = self.get_pascal3d_azi_ele(img_path)
                if not skip:
                    break

            img_cls, label_distribution = self.get_label_dist(ele, azi)
            pil_img = PIL.Image.open(img_path).convert('RGBA')

        box_l, box_t, box_r, box_b = box

        box_w = box_r - box_l
        box_h = (box_b - box_t) * r1

        w_max_scale = 0.9 * self.w / box_w
        h_max_scale = 0.9 * self.h / box_h

        w_min_scale = 0.3 * self.w / float(box_w)
        h_min_scale = 0.3 * self.h / float(box_h)

        max_scale = min(w_max_scale, h_max_scale)
        min_scale = max(w_min_scale, h_min_scale)

        resize, resize_w, resize_h, pil_img = self._resize(
            pil_img, min_scale, max_scale, r1)

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

        return pil_img, r_box_l, r_box_t, r_box_r, r_box_b, r, img_cls, label_distribution

    def _render_png(self, mode, r1=1.0):
        n = np.random.randint(len(self.rawcar_dataset[mode]))
        #n = str(n)

        if self.pre_load:
            #n = self.pascal_dataset[mode].keys()[n]
            pil_img, img_cls, label_distribution = self.png_dataset[mode][n]

        else:
            img_path = self.rawcar_dataset[mode][n]
            ele = (float(img_path.split('ele')[1].split('.')[0]) * math.pi) / (100 * 180)
            azi = (float(img_path.split('azi')[1].split('_')[0]) * math.pi) / (100 * 180)
            img_cls, label_distribution = self.get_label_dist(ele, azi)
            pil_img = PIL.Image.open(img_path).convert('RGBA')

        min_scale = 0.3
        max_scale = 1.0

        resize, resize_w, resize_h, pil_img = self._resize(
            pil_img, min_scale, max_scale, r1)

        box_l, box_t, box_r, box_b = pil_img.getbbox()
        box_w = box_r - box_l
        box_h = box_b - box_t
        pil_img, r = self.pil_image_enhance(pil_img)
        r_box_l, r_box_t, r_box_r, r_box_b = pil_img.getbbox()

        return (pil_img, r_box_l, r_box_t, r_box_r, r_box_b,
                r, img_cls, label_distribution)

    def _resize(self, pil_img, min_scale, max_scale, r1):
        resize = np.random.uniform(low=min_scale, high=max_scale)
        resize_w = resize * pil_img.size[0]
        resize_h = resize * pil_img.size[1] * r1
        pil_img = pil_img.resize((int(resize_w), int(resize_h)), self.BIL)

        return resize, resize_w, resize_h, pil_img

    def get_label_dist(self, ele, azi, sigma=0.1):
        ''' Reference: https://en.wikipedia.org/wiki/Great-circle_distance
        Parameters
        ----------
        ele: float
          angle of elevation in rad
        azi: float
          angle of azimuth in rad

        Returns
        ----------
        class_label: int
          Maximum likelihood of classes
        class_label_distribution: mxnet.ndarray
          probability of each class
        '''

        cos_ang = np.arccos(
            math.sin(ele) * np.sin(deg_2_rad(self.ele_label)) + \
            math.cos(ele) * np.cos(deg_2_rad(self.ele_label)) * np.cos(azi-deg_2_rad(self.azi_label)))

        cos_ang = np.expand_dims(cos_ang, axis=0)
        cos_ang_gaussion = nd.exp(-nd.array(cos_ang)**2/sigma)

        cos_ang_gaussion_softmax = cos_ang_gaussion / sum(cos_ang_gaussion[0])
        class_label = np.argmin(cos_ang)

        return class_label, cos_ang_gaussion_softmax

    def get_pascal3d_label(self, img_path, num_cls):
        f = img_path.split('/')[-1].split('.')[0]+'.mat'
        #mat = sio.loadmat(join(self.pascal3d_anno, f))
        mat = self.pascal3d_anno[f]
        mat = mat['record'][0][0]

        skip = False
        for mi, m in enumerate(mat):
            if mi == 1:
                label = [[], []]
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
                                    print(q[0][0])
                                    img_cls = int((q[0][0]+float(360/num_cls)/2)/15)
                                    if img_cls > num_cls-1:
                                        img_cls = num_cls-1
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

    def get_pascal3d_azi_ele(self, img_path):
        f = img_path.split('/')[-1].split('.')[0]+'.mat'
        # mat = sio.loadmat('/media/nolan/SSD1/
        #                   HP_31/pascal3d_image_and_label/car_imagenet_label/n03770085_6172.mat')
        #mat = sio.loadmat(join(self.pascal3d_anno, f))
        mat = self.pascal3d_anno[f]
        mat = mat['record'][0][0][1][0]

        # if more then one car in an image, do not use it, so skip
        if len(mat) > 1:
            return 0, 0, 0, True

        box = [int(i) for i in mat[0][1][0]]
        # mat[0][3][0][0]: [azi_coarse, ele_coarse, azi, ele, distance, focal,
        #                   px, py, theta, error, interval_azi, interval_ele,
        #                   num_anchor, viewport]
        ele = mat[0][3][0][0][3][0]
        azi = mat[0][3][0][0][2][0]

        return ele, azi, box, False

    def test(self):
        ctx = [gpu(0)]
        add_LP = AddLP(self.h, self.w, -1)

        plt.ion()
        fig = plt.figure()
        ax = []
        for i in range(self.bs):
            ax.append(fig.add_subplot(1, 1, 1+i))
            #ax.append(fig.add_subplot(4,4,1+i))
        t = time.time()
        background_iter = mxnet.image.ImageIter(
            self.bs, (3, self.h, self.w),
            path_imgrec='/media/nolan/SSD1/HP_31/sun2012_val.rec',
            path_imgidx='/media/nolan/SSD1/HP_31/sun2012_val.idx',
            shuffle=True, pca_noise=0,
            brightness=0.5, saturation=0.5, contrast=0.5, hue=1.0,
            rand_crop=True, rand_resize=True, rand_mirror=True, inter_method=10
            )
        while True:  # bg:0~255
            bg = background_iter.next().data[0].as_in_context(ctx[0])

            #img_batch, label_batch = self.render_pascal(bg, 'train')
            img_batch, label_batch = self.render(bg)
            #img_batch, label_batch = add_LP.add(img_batch, label_batch)
            for i in range(self.bs):
                ax[i].clear()

                im = img_batch[i].transpose((1, 2, 0)).asnumpy()
                b = label_batch[i, 0].asnumpy()
                print(b)
                im = cv2_add_bbox(im, b, [0, 0, 1])

                ax[i].imshow(im)
                ax[i].axis('off')

            raw_input('next')


def rad_2_deg(rad):
    return rad * 180. / math.pi


def deg_2_rad(deg):
    return deg * math.pi / 180.
