import cv2
import math
import os
import sympy
import sys
import time
import PIL
import yaml
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')

import mxnet
from mxnet import gpu
from mxnet import nd

from yolo_modules import yolo_cv
from yolo_modules import yolo_gluon
from yolo_modules import global_variable


class LPGenerator():
    def __init__(self, img_h, img_w, class_index=1):
        self.class_index = int(class_index)
        self.h = img_h
        self.w = img_w
        self.LP_WH = [[380, 160], [320, 150], [320, 150]]
        self.x = [np.array([7, 56, 106, 158, 175, 225, 274, 324]),
                  np.array([7, 57, 109, 130, 177, 223, 269])]

        self.font0 = [None] * 35
        self.font1 = [None] * 35

        module_dir = os.path.dirname(os.path.abspath(__file__))
        fonts_dir = os.path.join(module_dir, 'fonts')

        self.dot = PIL.Image.open(fonts_dir+"/34.png")
        self.dot = self.dot.resize((10, 70), PIL.Image.BILINEAR)

        for font_name in range(0, 34):
            f = PIL.Image.open(fonts_dir+'/'+str(font_name)+".png")
            self.font0[font_name] = f.resize((45, 90), PIL.Image.BILINEAR)
            self.font1[font_name] = f.resize((40, 80), PIL.Image.BILINEAR)

        self.project_rect_6d = ProjectRectangle6D(*self.LP_WH[0])

        self.pil_image_enhance = yolo_cv.PILImageEnhance(
            M=0., N=0., R=0., G=1.0, noise_var=10.)
        '''
        self.augs = mxnet.image.CreateAugmenter(
            data_shape=(3, self.h, self.w), inter_method=10,
            brightness=0.5, contrast=0.5, saturation=0.5, pca_noise=1.0)
        '''
        self.augs = mxnet.image.CreateAugmenter(
            data_shape=(3, self.h, self.w),
            inter_method=10, pca_noise=1.0,
            brightness=0.5, contrast=0.5, saturation=0.3, hue=1.0)

    def draw_LP(self):
        LP_type = 0  # np.random.randint(2)
        LP_w, LP_h = self.LP_WH[LP_type]
        x = self.x[LP_type]
        label = []
        if LP_type == 0:  # ABC-1234
            LP = PIL.Image.new('RGBA', (LP_w, LP_h), yolo_cv._color[6])
            abc = np.random.randint(10, 34, size=3)
            for i, j in enumerate(abc):
                LP.paste(self.font0[j], (x[i], 35))
                label.append([j, float(x[i])/LP_w, float(x[i]+45)/LP_w])

            LP.paste(self.dot, (x[3], 45))

            num = np.random.randint(0, 10, size=4)
            for i, j in enumerate(num):
                LP.paste(self.font0[j], (x[i+4], 35))
                label.append([j, float(x[i+4])/LP_w, float(x[i+4]+45)/LP_w])
        '''
        if LP_type == 1:  # AB-1234
            LP = PIL.Image.new('RGBA', (LP_w, LP_h), yolo_cv._color[7])
            abc = np.random.randint(10, 34, size=2)
            for i, j in enumerate(abc):
                LP.paste(self.font1[j], (x[i], 40))
                label.append([j, float(x[i])/LP_w, float(x[i]+40)/LP_w])

            LP.paste(self.dot, (x[2], 45))

            num = np.random.randint(0, 10, size=4)
            for i, j in enumerate(num):
                LP.paste(self.font1[j], (x[i+3], 40))
                label.append([j, float(x[i+3])/LP_w, float(x[i+3]+40)/LP_w])
        '''
        return LP, LP_type, label

    def resize_and_paste_LP(self, LP, OCR_labels=None):
        # print(LP.size[0], LP.size[1]) # (320, 150)
        resize_w = int((np.random.rand()*0.15+0.15)*LP.size[0])
        resize_h = int((np.random.rand()*0.1+0.2)*LP.size[1])
        LP = LP.resize((resize_w, resize_h), PIL.Image.BILINEAR)

        LP, r = self.pil_image_enhance(LP)

        paste_x = int(np.random.rand() * (self.w-120))
        paste_y = int(np.random.rand() * (self.h-120))

        tmp = PIL.Image.new('RGBA', (self.w, self.h))
        tmp.paste(LP, (paste_x, paste_y))

        m = nd.array(tmp.split()[-1]).reshape(1, self.h, self.w)
        mask = nd.tile(m, (3, 1, 1))

        LP = PIL.Image.merge("RGB", (tmp.split()[:3]))
        LP = nd.array(LP)
        LP_ltrb = tmp.getbbox()
        LP_label = nd.array(LP_ltrb)

        if OCR_labels is not None:
            print('TODO')
            # TODO
        else:
            for aug in self.augs:
                LP = aug(LP)
            LP_label = nd.array([[
                self.class_index,
                float(LP_ltrb[0])/self.w,
                float(LP_ltrb[1])/self.h,
                float(LP_ltrb[2])/self.w,
                float(LP_ltrb[3])/self.h,
                0  # i dot car Licence plate rotating
            ]])

        return LP.transpose((2, 0, 1)), mask, LP_label

    def random_projection_LP_6D(self, LP, in_size, out_size, r_max):
        Z = np.random.uniform(low=1000., high=3000.)
        X = (Z * 8 / 30.) * np.random.uniform(low=-1, high=1)
        Y = (Z * 6 / 30.) * np.random.uniform(low=-1, high=1)
        r1 = np.random.uniform(low=-1, high=1) * r_max[0] * math.pi / 180.
        r2 = np.random.uniform(low=-1, high=1) * r_max[1] * math.pi / 180.
        r3 = np.random.uniform(low=-1, high=1) * r_max[2] * math.pi / 180.

        pose_6d = [X, Y, Z, r1, r2, r3]
        projected_points = self.project_rect_6d(pose_6d)

        M = cv2.getPerspectiveTransform(
            projected_points,
            np.float32([[380, 160], [0, 160], [0, 0], [380, 0]]))

        LP = LP.transform(
            in_size[::-1],
            PIL.Image.PERSPECTIVE,
            M.reshape(-1),
            PIL.Image.BILINEAR)

        LP = LP.resize((out_size[1], out_size[0]), PIL.Image.BILINEAR)
        LP, _ = self.pil_image_enhance(LP)

        mask = yolo_gluon.pil_mask_2_rgb_ndarray(LP.split()[-1])
        image = yolo_gluon.pil_rgb_2_rgb_ndarray(LP, augs=self.augs)

        x = X * self.project_rect_6d.fx / Z + self.project_rect_6d.cx
        x = x * out_size[1] / float(self.project_rect_6d.camera_w)

        y = Y * self.project_rect_6d.fy / Z + self.project_rect_6d.cy
        y = y * out_size[0] / float(self.project_rect_6d.camera_h)

        label = nd.array([[1] + [X, Y, Z, r1, r2, r3, x, y]])

        return mask, image, label

    def add(self, bg_batch, r_max, add_rate=1.0):
        ctx = bg_batch.context
        bs = bg_batch.shape[0]
        h = bg_batch.shape[2]
        w = bg_batch.shape[3]

        mask_batch = nd.zeros_like(bg_batch)
        image_batch = nd.zeros_like(bg_batch)
        label_batch = nd.ones((bs, 1, 10), ctx=ctx) * (-1)

        for i in range(bs):
            if np.random.rand() > add_rate:
                continue

            LP, LP_type, _ = self.draw_LP()

            output_size = (h, w)
            input_size = (
                self.project_rect_6d.camera_h,
                self.project_rect_6d.camera_w)

            mask, image, label = self.random_projection_LP_6D(
                LP, input_size, output_size, r_max)

            mask_batch[i] = mask.as_in_context(ctx)
            image_batch[i] = image.as_in_context(ctx)
            label_batch[i, :, :-1] = label
            label_batch[i, :, -1] = LP_type

        img_batch = bg_batch * (1 - mask_batch) + image_batch * mask_batch
        img_batch = nd.clip(img_batch, 0, 1)

        return img_batch, label_batch

    def render(self, bs, ctx):
        LP_label = nd.ones((bs, 7, 5), ctx=ctx) * -1
        LP_batch = nd.zeros((bs, 3, self.h, self.w), ctx=ctx)
        mask = nd.zeros((bs, 3, self.h, self.w), ctx=ctx)

        for i in range(bs):
            LP, LP_type, labels = self.draw_LP()
            LP_w, LP_h = LP.size

            resize = np.random.rand() * 0.1 + 0.9
            LP_w = int(resize * self.w)
            LP_h = int((np.random.rand()*0.1+0.9) * resize * self.h)
            LP = LP.resize((LP_w, LP_h), PIL.Image.BILINEAR)

            LP, r = self.pil_image_enhance(LP, M=.0, N=.0, R=5.0, G=8.0)
            #LP = LP.filter(ImageFilter.GaussianBlur(radius=np.random.rand()*8.))

            paste_x = np.random.randint(int(-0.0*LP_w), int(self.w-LP_w))
            paste_y = np.random.randint(int(-0.0*LP_h), int(self.h-LP_h))

            tmp = PIL.Image.new('RGBA', (self.w, self.h))
            tmp.paste(LP, (paste_x, paste_y))
            bg = PIL.Image.new('RGBA', (self.w, self.h), tuple(np.random.randint(255, size=3)))
            LP = PIL.Image.composite(tmp, bg, tmp)

            LP = nd.array(PIL.Image.merge("RGB", (LP.split()[:3])))
            for aug in self.augs2:
                LP = aug(LP)

            LP_batch[i] = LP.as_in_context(ctx).transpose((2, 0, 1))/255.

            r = r * np.pi / 180
            offset = paste_x + abs(LP_h*math.sin(r)/2)
            for j, c in enumerate(labels):

                LP_label[i, j, 0] = c[0]
                LP_label[i, j, 1] = (offset + c[1]*LP_w*math.cos(r))/self.w
                LP_label[i, j, 3] = (offset + c[2]*LP_w*math.cos(r))/self.w
                #LP_label[i,j,1] = (c[1]*LP_w*math.cos(r) - 40*math.sin(r) + paste_x)/self.w
                #LP_label[i,j,3] = (c[2]*LP_w*math.cos(r) + 40*math.sin(r) + paste_x)/self.w
        LP_batch = nd.clip(LP_batch, 0, 1)

        return LP_batch, LP_label

    def test_render(self, n):
        plt.ion()
        fig = plt.figure()
        ax = []
        for i in range(n):
            ax.append(fig.add_subplot(321+i))
        while True:
            img_batch, label_batch = self.render(n, gpu(0))
            for i in range(n):
                label = label_batch[i]
                s = self.label2nparray(label)
                ax[i].clear()
                ax[i].plot(range(8, 384, 16), (1-s)*160, 'r-')
                ax[i].imshow(img_batch[i].transpose((1, 2, 0)).asnumpy())

            raw_input('next')

    def label2nparray(self, label):
        score = nd.zeros((24))
        for L in label:  # all object in the image
            if L[0] < 0:
                continue
            text_cent = ((L[3] + L[1])/2.)
            left = int(round((text_cent.asnumpy()[0]-15./self.w)*24))
            right = int(round((text_cent.asnumpy()[0]+15./self.w)*24))
            #left = int(round(L[1].asnumpy()[0]*24))
            #right = int(round(L[3].asnumpy()[0]*24))
            for ii in range(left, right):
                box_cent = (ii + 0.5) / 24.
                score[ii] = 1-nd.abs(box_cent-text_cent)/(L[3]-L[1])
        return score.asnumpy()

    def test_add(self, b):
        #while True:
        batch_iter = load(b, h, w)
        for batch in batch_iter:
            imgs = batch.data[0].as_in_context(ctx[0])  # b*RGB*w*h
            labels = batch.label[0].as_in_context(ctx[0])  # b*L*5
            #imgs = nd.zeros((b, 3, self.h, self.w), ctx=gpu(0))*0.5
            tic = time.time()
            imgs, labels = self.add(imgs/255, labels)
            #print(time.time()-tic)
            for i, img in enumerate(imgs):
                R, G, B = img.transpose((1, 2, 0)).split(num_outputs=3, axis=-1)
                img = nd.concat(B, G, R, dim=-1).asnumpy()
                print(labels[i])
                cv2.imshow('%d' % i, img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break


class ProjectRectangle6D():
    def __init__(self, w, h):
        h /= 2.
        w /= 2.
        path = global_variable.camera_parameter_path
        with open(path) as f:
            spec = yaml.load(f)

        self.camera_w = spec['image_width']
        self.camera_h = spec['image_height']
        self.fx = spec['projection_matrix']['data'][0]
        self.fy = spec['projection_matrix']['data'][5]
        self.cx = spec['projection_matrix']['data'][2]
        self.cy = spec['projection_matrix']['data'][6]

        self.X = sympy.Symbol('X')
        self.Y = sympy.Symbol('Y')
        self.Z = sympy.Symbol('Z')
        self.r1 = sympy.Symbol('r1')
        self.r2 = sympy.Symbol('r2')
        self.r3 = sympy.Symbol('r3')

        P_3d = sympy.Matrix(
            [[w, -w, -w, w],
             [h, h, -h, -h],
             [0, 0, 0, 0]])

        R1 = sympy.Matrix(
            [[1, 0, 0],
             [0, sympy.cos(self.r1), -sympy.sin(self.r1)],
             [0, sympy.sin(self.r1), sympy.cos(self.r1)]])

        R2 = sympy.Matrix(
            [[sympy.cos(self.r2), 0, sympy.sin(self.r2)],
             [0, 1, 0],
             [-sympy.sin(self.r2), 0, sympy.cos(self.r2)]])

        R3 = sympy.Matrix(
            [[sympy.cos(self.r3), -sympy.sin(self.r3), 0],
             [sympy.sin(self.r3), sympy.cos(self.r3), 0],
             [0, 0, 1]])

        T_matrix = sympy.Matrix(
            [[self.X]*4,
             [self.Y]*4,
             [self.Z]*4])

        intrinsic_matrix = sympy.Matrix(
            [[self.fx, 0, self.cx],
             [0, self.fy, self.cy],
             [0, 0, 1]])

        extrinsic_matrix = R3 * R2 * R1 * P_3d + T_matrix
        self.projection_matrix = intrinsic_matrix * extrinsic_matrix

    def __call__(self, pose_6d):
        # [mm, mm, mm, rad, rad, rad]
        points = np.zeros((4, 2))
        subs = {
            self.X: pose_6d[0], self.Y: pose_6d[1], self.Z: pose_6d[2],
            self.r1: pose_6d[3], self.r2: pose_6d[4], self.r3: pose_6d[5]}

        ans = self.projection_matrix.evalf(subs=subs)
        for i in range(4):
            points[i, 0] = ans[0, i] / ans[2, i]
            points[i, 1] = ans[1, i] / ans[2, i]

        return points.astype(np.float32)

    def add_edges(self, img, pose, LP_size=(160, 380)):
        corner_pts = self.__call__(pose)

        x_scale = img.shape[1] / float(self.camera_w)
        y_scale = img.shape[0] / float(self.camera_h)

        corner_pts[:, 0] = corner_pts[:, 0] * x_scale
        corner_pts[:, 1] = corner_pts[:, 1] * y_scale
        # 2----------->3
        # ^            |
        # |  ABC-1234  |
        # |            |
        # 1<-----------0
        LP_corner = np.float32([[LP_size[1], LP_size[0]],
                                [0, LP_size[0]],
                                [0, 0],
                                [LP_size[1], 0]])
        M = cv2.getPerspectiveTransform(corner_pts, LP_corner)

        clipped_LP = cv2.warpPerspective(img, M, (LP_size[1], LP_size[0]))

        p = np.expand_dims(corner_pts, axis=0).astype(np.int32)
        img = cv2.polylines(img, p, 1, (0, 0, 1), 2)

        return img, clipped_LP


if __name__ == '__main__':
    g = LPGenerator(640, 480, 0)
    g.test_render(4)
