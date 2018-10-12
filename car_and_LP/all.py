#!/usr/bin/env python
import time
from time import gmtime, strftime
import threading
import sys

from mxnet import autograd
from mxnet import init

from utils import *

ctx = [gpu(0)]

args, size ,cls_names = Parser()

steps = [16, 16, 32]
all_anchors = nd.array(
	[[[0.100, 0.200], [0.140, 0.160]], 
	 [[0.300, 0.500], [0.200, 0.400]], 
 	 [[0.500, 0.700], [0.400, 0.600]]])
pretrain = 'HP_33/backup/HP_33_52000'
get_feature = HP_32(all_anchors, len(cls_names))
get_feature.collect_params().load(pretrain, ctx=ctx)
get_feature.hybridize()

cls_names2 = ['0','1','2','3','4','5','6','7','8','9',
	'A','B','C','D','E','F','G','H','J','K','L','M',
	'N','P','Q','R','S','T','U','V','W','X','Y','Z']
get_feature2 = OCR(len(cls_names2))
get_feature2.collect_params().load('OCR/backup/OCR_100', ctx=ctx)

class Valid():
	def __init__(self):
		import cv2
		n = len(all_anchors[0]) # anchor per sub_map
		self.area = [int(size[0]*size[1]/step**2) for step in steps]
		self.s = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
		self.y = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
		self.x = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
		self.h = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
		self.w = nd.zeros((1, sum(self.area), n, 1), ctx=ctx[0])
		a_start = 0
		for i, anchors in enumerate(all_anchors): # [12*16,6*8,3*4]
			a = self.area[i]

			s = nd.repeat(nd.array([steps[i]], ctx=ctx[0]), repeats=a*n)

			x_num = int(size[1]/steps[i])
			y = nd.arange(0, size[0], step=steps[i], repeat=n*x_num, ctx=ctx[0])
			
			x = nd.arange(0, size[1], step=steps[i], repeat=n, ctx=ctx[0])
			x = nd.tile(x, int(size[0]/steps[i]))

			hw = nd.tile(all_anchors[i], (a, 1))
			h, w = hw.split(num_outputs=2, axis=-1)

			self.s[0, a_start:a_start+a] = s.reshape(a, n, 1)
			self.y[0, a_start:a_start+a] = y.reshape(a, n, 1)
			self.x[0, a_start:a_start+a] = x.reshape(a, n, 1)
			self.h[0, a_start:a_start+a] = h.reshape(a, n, 1)
			self.w[0, a_start:a_start+a] = w.reshape(a, n, 1)
			
			a_start += a
	def yxhw_to_ltrb(self, yxhw):
		ty, tx, th, tw = yxhw.split(num_outputs=4, axis=-1)	
		by = (nd.sigmoid(ty)*self.s + self.y) / size[0]
		bx = (nd.sigmoid(tx)*self.s + self.x) / size[1]

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
		L_pred, C_pred = get_feature(x)
		box = self.yxhw_to_ltrb(nd.concat(L_pred[1], C_pred[1], dim=1))
		L_box = box[:,:self.area[0]] 
		C_box = box[:,self.area[0]:] 

		L_score = nd.sigmoid(L_pred[0]) # 1,a,2,1
		C_score = nd.sigmoid(C_pred[0])

		L_best = L_score.reshape(-1).argmax(axis=0)
		Lout = nd.concat(L_score, L_box, dim=-1).reshape((-1, 5))
		Lout = Lout[L_best][0].asnumpy()

		Cout = nd.concat(C_score, C_box, C_pred[2], dim=-1).reshape((-1, 5+len(cls_names)))

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

v = Valid()
resz1 = image.ForceResizeAug((size[1], size[0]))
resz2 = image.ForceResizeAug((384, 160))
cap = cv2.VideoCapture('video/4.avi')
fig = plt.figure()
x = fig.add_subplot(1,1,1)
while True:
	ret, img = cap.read()
	img2 = img
	nd_img = resz1(nd.array(img)).as_in_context(ctx[0])
	nd_img = nd_img.transpose((2,0,1)).expand_dims(axis=0)/255.
	
	L_pred, C_pred, _, _ = v.predict(nd_img)
 
	n = np.argmax(C_pred[5:])

	########################### Add Box ###########################
	if C_pred[0]>0.9:
		cv2_add_bbox_text(img, C_pred, cls_names[n], [480,640], 1)

	if L_pred[0]>0.9:
		cv2_add_bbox_text(img, L_pred, 'LP', [480,640], 2)
		batch_x = img2[int(480*L_pred[2]):int(480*L_pred[4]),
			int(640*L_pred[1]):int(640*L_pred[3])]
		batch_x = resz2(nd.array(batch_x)).as_in_context(ctx[0])
		batch_x = batch_x.transpose((2,0,1)).expand_dims(axis=0)/255.
		prob, score = get_feature2(batch_x)

		s = score[0].reshape(-1).asnumpy()
		p = nd.argmax(prob[0], axis=-1).reshape(-1).asnumpy()
		x.clear()		
		x.imshow(batch_x[0].transpose((1,2,0)).asnumpy())	
		x.plot(range(8,384,16),(1-s)*160)
		x.axis('off')

		t = ''
		s = np.concatenate(([0],s,[0]))
		for i in range(24):
			if s[i+1] > 0.2	and s[i+1] > s[i+2] and s[i+1] > s[i]:
				c = int(p[i])				
				t = t + cls_names2[c]
		print(t)
		x.set_title(t)
		plt.pause(0.01)		
	########################## Save Image ##########################
	#self.out.write(img)	
	########################## Show Image ##########################
	cv2.imshow('img', img)
	cv2.waitKey(1)

cap.release()

