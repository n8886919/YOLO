import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from mxnet import gluon
from mxnet import gpu, cpu
from mxnet import image
from mxnet import metric
from mxnet import nd, sym
from mxnet.gluon import nn

import PIL.Image as PILIMG
from PIL import ImageFilter, ImageEnhance
Leaky_alpha = 0.1
####################### Parser #######################
color = [(0,0,255), (0,255,255), (255,255,0),(0,255,0), (0,255,255),
	(255,0,255),(0,0,0),(255,255,255)]
def add_noise(img, noise_var):
	np_img = np.array(img)
	noise = np.random.normal(0., noise_var, np_img.shape)
	np_img = np_img + noise 
	np_img = np.clip(np_img, 0, 255)
	img = PILIMG.fromarray(np.uint8(np_img))
	return img
def get_iou(predict, target):
	l, t, r, b = predict.split(num_outputs=4, axis=-1)
	i_left = nd.maximum(target[1], l)
	i_top = nd.maximum(target[2], t)
	i_right = nd.minimum(target[3], r)
	i_bottom = nd.minimum(target[4], b)
	iw = nd.maximum(i_right -i_left, 0.)
	ih = nd.maximum(i_bottom-i_top, 0.)
	inters = iw * ih
	predict_area = (r-l)*(b-t)
	target_area = (target[3] - target[1]) * (target[4] - target[2])
	ious = inters/(predict_area + target_area - inters) 
	return ious # 1344x3x1
class Res(gluon.HybridBlock):
	def __init__(self, channels, **kwargs):
		super(Res, self).__init__(**kwargs)
		
		self.conv1 = nn.Conv2D(channels[0], kernel_size=1)
		self.conv2 = nn.Conv2D(channels[1], kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm()
		self.bn2 = nn.BatchNorm()
		self.act = nn.LeakyReLU(Leaky_alpha)
	def hybrid_forward(self, F, x, *args):
		out = self.act(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		return self.act((out + x))
class ConvBNLrelu(gluon.HybridBlock):
	def __init__(self, channels, kl, stride=1, **kwargs):
		super(ConvBNLrelu, self).__init__(**kwargs)
		if kl==3: pad = 1 
		if kl==1: pad = 0 
		with self.name_scope():
			self.seq = nn.HybridSequential()
			self.seq.add(
				nn.Conv2D(channels, kernel_size=kl, strides=stride, padding=pad), 
				nn.BatchNorm(), 
				nn.LeakyReLU(Leaky_alpha), 
				)
	def hybrid_forward(self, F, x, *args):
		return self.seq(x)
class LossRecorder(metric.EvalMetric):
	def __init__(self, name):
		super(LossRecorder, self).__init__(name)
	def update(self, labels, preds=0):
		"""Update metric with pure loss"""
		for loss in labels:
			if isinstance(loss, nd.NDArray):
				loss = loss.asnumpy()
			self.sum_metric += loss.sum()
			self.num_inst += 1
class GetFeature16(gluon.HybridBlock):
	def __init__(self, all_anchors, cls_num, **kwargs):
		super(GetFeature16, self).__init__(**kwargs)
		self.act = nn.LeakyReLU(0.1)

		self.all_acrs = all_anchors
		self.clsn = cls_num

		with self.name_scope():
			# add name_scope on the outermost Sequential
			self.trunk = nn.HybridSequential()
			self.trunk.add(
				ConvBNLrelu(32, 3),
				ConvBNLrelu(64, 3, stride=2),
				Res([32,64]),
				ConvBNLrelu(128, 3, stride=2),
				Res([64,128]), Res([64,128]),  
				ConvBNLrelu(256, 3, stride=2),
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256]),
				ConvBNLrelu(512, 3, stride=2),
				Res([256, 512]),Res([256, 512]),
				Res([256, 512]), Res([256, 512]),
				Res([256, 512]), Res([256, 512]),
				Res([256, 512]), Res([256, 512]),
				ConvBNLrelu(512, 1),
				nn.Conv2D((self.clsn+5)*len(self.all_acrs[0]), kernel_size=1)
			)
	def hybrid_forward(self, F, x, *args):
		fm = self.trunk(x)
		fm = fm.transpose((0, 2, 3, 1)).reshape(
			(0, -1, len(self.all_acrs[0]), self.clsn+5)
			)
		cls_pred = fm.slice_axis(begin=0, end=self.clsn, axis=-1) 
		score = fm.slice_axis(begin=self.clsn, end=self.clsn+1, axis=-1)
		yxhw = fm.slice_axis(begin=self.clsn+1, end=self.clsn+5, axis=-1)
		return cls_pred, score, yxhw
def cv2_add_bbox_text(img, p, class_names, shape, pi):
	c = color[p[0].astype(int)]
	l = int(p[2] * shape[1])
	t = int(p[3] * shape[0])
	r = int(p[4] * shape[1])
	b = int(p[5] * shape[0])
	cv2.rectangle(img, (l ,t), (r ,b), c, 2)
	cv2.putText(img, '%s %.2f'%(class_names[p[0].astype(int)], p[1]), 
		(l+5, b+25), 6, 1, c, 2)
	return [l, t, r, b] 
class AddMonster():
	def __init__(self):
		self.img_path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/home/nolan/Desktop/RenderForDarknet/datasets/sun2012pascalformat/JPEGImages'
		self.pic = [None]*4
		for pic_type in range(4):
			self.pic[pic_type] = PILIMG.open('pic/'+str(pic_type)+".png").convert("RGBA")  		
		self.augs = image.CreateAugmenter(data_shape=(3, 288, 480),inter_method=10,
                                    brightness=0.1, contrast=0.1, saturation=0.1,
                                    pca_noise=0.01)
		self.augs1 = image.CreateAugmenter(data_shape=(3, 288, 480),inter_method=10,
                                    brightness=0.1, contrast=0.1, saturation=0.1,
                                    pca_noise=0.01)
		self.resz=image.ForceResizeAug((480,288))
			
	def add(self, batch_size, device):
		LP_labels = nd.zeros((batch_size,1,5), ctx=device)
		
		imgs = nd.random.uniform(0,1,(batch_size, 3, 288, 480), ctx=device)

		for b in range(batch_size):
			imgname = np.random.choice(os.listdir(self.img_path))
			imgname = os.path.join(self.img_path, imgname)
			with open(imgname, 'rb') as fp:
				str_image = fp.read()

			img = image.imdecode(str_image)
			img = nd.array(img, dtype='float32')
			img = self.resz(img)/255.

			monster_type = np.random.randint(4)

			monster = self.pic[monster_type]
			if np.random.randint(2): monster = monster.transpose(PILIMG.FLIP_LEFT_RIGHT)

			pic_h = np.random.randint(50, 250)
			pic_w = int(pic_h*monster.size[0]*(np.random.rand()*0.6+0.7)/monster.size[1])

			monster = monster.resize((pic_w, pic_h))
			#monster = monster.rotate(np.random.random()*360, PILIMG.BILINEAR, expand=1)
			monster = monster.filter(ImageFilter.GaussianBlur(radius=np.random.rand()*1.5))	
			w, h = 480, 288

			paste_x = int(np.random.rand()*(w-monster.size[0]))
			paste_y = int(np.random.rand()*(h-monster.size[1]))

			tmp = PILIMG.new('RGBA', (w, h))
			tmp.paste(monster, (paste_x, paste_y))

			LP_mask = nd.array(tmp.split()[-1], ctx=device).reshape(1,288,480) 
			LP_mask = nd.tile(LP_mask, (3,1,1))

			monster = PILIMG.merge("RGB", (tmp.split()[:3]))
			monster = nd.array(monster)/255.
			for aug in self.augs: monster = aug(monster)
			for aug in self.augs1: img = aug(img)
			monster = monster.as_in_context(gpu(0)).transpose((2,0,1))
			img = img.as_in_context(gpu(0)).transpose((2,0,1))

			img = nd.where(LP_mask<200, img, monster)
			
			LPl, LPt, LPr, LPb = tmp.getbbox()
			LP_labels[b] = nd.array([[monster_type, LPl/float(w), LPt/float(h), 
				LPr/float(w), LPb/float(h)]])
			imgs[b] = img


		return imgs, LP_labels
	def test_add_LP(self):
		while True:
			imgs, labels = self.add(1, gpu(0)) 

			for i, img in enumerate(imgs):
				img = img.transpose((1,2,0))
				B, G, R = img.split(num_outputs=3, axis=-1)
				img = nd.concat(R, G, B, dim= -1).asnumpy()
				print(labels[i])
				cv2.imshow('%d'%i, img)
			if cv2.waitKey(0) & 0xFF == ord('q'): break

if __name__ == '__main__':
	add_LP = AddMonster()
	add_LP.test_add_LP()
