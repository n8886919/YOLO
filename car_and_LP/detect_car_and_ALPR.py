# -*- coding: utf-8 -*-
import argparse
import copy
import cv2
import os
import time
from time import gmtime, strftime

from mxnet import autograd
from mxnet import gluon
from mxnet import gpu
from mxnet import init
from mxnet import nd
from mxnet import sym
from mxnet.gluon import nn

from utils import *

class Valid():
	def __init__(self):
		n = [len(nn) for nn in all_anchors] # anchor per sub_map
		area = [None]*all_anchors.shape[0]
		step = max_step
		for i, a in enumerate(area):
			area[i] = data_shape[0]*data_shape[1]/(step**2)

			scale_map = nd.repeat(nd.array([step], ctx=gpu(0)), repeats=area[i]*n[i]).reshape(1, area[i], n[i], 1)
			
			offset_y = nd.arange(0, data_shape[0], step=step, repeat=(n[i]*data_shape[1]/step), ctx=gpu(0))
			offset_y = offset_y.reshape(1, area[i], n[i], 1) #self.offset_y = nd.tile(offset_y1, (b, 1, 1, 1))

			offset_x = nd.tile(nd.arange(0, data_shape[1], step=step, repeat=n[i], ctx=gpu(0)), data_shape[0]/step)
			offset_x = offset_x.reshape(1, area[i], n[i], 1) #self.offset_x = nd.tile(offset_x1, (b, 1, 1, 1))
			ahw1 = nd.tile(all_anchors[i], (area[i], 1)).reshape(1, area[i], n[i], 2)
			if i == 0:
				self.scale_map = scale_map
				self.offset_y  = offset_y
				self.offset_x  = offset_x
				ahw  = ahw1
			else:
				self.scale_map = nd.concat(self.scale_map, scale_map, dim=1)
				self.offset_y = nd.concat(self.offset_y, offset_y, dim=1)
				self.offset_x = nd.concat(self.offset_x, offset_x, dim=1)
				ahw = nd.concat(ahw, ahw1, dim=1)
			step /= 2
		self.ah, self.aw = ahw.split(num_outputs=2, axis=-1)

		step = LP_step
		n = len(LP_anchors) # anchor per sub_map
		area = LP_shape[0]*LP_shape[1]/(step**2)

		self.LP_scale_map = nd.repeat(nd.array([step], ctx=gpu(0)), repeats=area*n).reshape(1, area, n, 1)
		self.LP_y = nd.arange(0, LP_shape[0], step=step, repeat=(n*LP_shape[1]/step), ctx=gpu(0))
		self.LP_y = offset_y.reshape(1, area, n, 1)

		self.LP_x = nd.tile(nd.arange(0, LP_shape[1], step=step, repeat=n, ctx=gpu(0)), LP_shape[0]/step)
		self.LP_x = offset_x.reshape(1, area, n, 1)
		ahw = nd.tile(LP_anchors, (area, 1)).reshape(1, area, n, 2)


		self.LP_ah, self.LP_aw = ahw.split(num_outputs=2, axis=-1)
	def yxhw2ltrb(self, yxhw):
		ty, tx, th, tw = yxhw.split(num_outputs=4, axis=-1)
		by = (nd.sigmoid(ty)*self.scale_map + self.offset_y) / data_shape[0]
		bx = (nd.sigmoid(tx)*self.scale_map + self.offset_x) / data_shape[1]

		bh = nd.exp(th) * self.ah
		bw = nd.exp(tw) * self.aw
		bh2 = bh / 2
		bw2 = bw / 2
		left = bx - bw2
		right = bx + bw2
		top = by - bh2
		bottom = by + bh2
		return nd.concat(left, top, right, bottom, dim=-1)
	def LP_yxhw2ltrb(self, yxhw):
		ty, tx, th, tw = yxhw.split(num_outputs=4, axis=-1)
		by = (nd.sigmoid(ty)*self.LP_scale_map + self.LP_y) / LP_shape[0]
		bx = (nd.sigmoid(tx)*self.LP_scale_map + self.LP_x) / LP_shape[1]
		bh = nd.exp(th) * self.LP_ah
		bw = nd.exp(tw) * self.LP_aw
		bh2 = bh / 2
		bw2 = bw / 2
		left = bx - bw2
		right = bx + bw2
		top = by - bh2
		bottom = by + bh2
		return nd.concat(left, top, right, bottom, dim=-1)
	def predict(self, x, predictor, LP):
		cls_pred, score, yxhw = predictor(x/255.)
		score = nd.sigmoid(score)
		cid = nd.argmax(cls_pred, axis=-1, keepdims=True)
		predict_ltrb = self.LP_yxhw2ltrb(yxhw) if LP else self.yxhw2ltrb(yxhw)
		output = nd.concat(cid, score, predict_ltrb, dim=-1)#.reshape(-1,6)
		nms = nd.contrib.box_nms(
			output.reshape((0, -1, 6)), 
			force_suppress=True,
			overlap_thresh=0.3, 
			coord_start=2, 
			score_index=1, 
			id_index=0,
			topk=7)
		return nms
	def valid(self):
		for b, batch in enumerate(batch_iter):
			x = batch.data[0].as_in_context(gpu(0)) # b*RGB*w*h
			R,G,B = x[0].transpose((1,2,0)).split(num_outputs=3, axis=-1)
			img = nd.concat(B,G,R, dim=-1).asnumpy()/255.
			self.detect_and_draw_with_LP(img, x)
			if cv2.waitKey(0) & 0xFF == ord('q'): break		
	def demo(self):
		#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		#out = cv2.VideoWriter('111.avi', fourcc, 30,(480,288))
		cap = cv2.VideoCapture('/home/nolan/Desktop/output.avi')
		while(True):
			ret, img = cap.read()
			B,G,R = nd.array(img, gpu(0)).split(num_outputs=3, axis=-1)
			x = nd.concat(R,G,B, dim=-1).transpose((2,0,1)).expand_dims(axis=0)
			self.detect_and_draw_with_LP(img, x)
			if cv2.waitKey(0) & 0xFF == ord('q'): break
	def detect_and_draw_with_LP(self, img, x):
		img_copy = copy.deepcopy(img)
		predict = self.predict(x, get_feature, 0)
		for pred in predict[0].asnumpy():
			if pred[0] > -0.5 and pred[1] > 0.9:
				ltrb = cv2_add_bbox_text(img_copy, pred, cls_names, data_shape)
			else: continue
			if pred[0] == 8:
				LP = cv2.resize(
					img[ltrb[1]:ltrb[3], ltrb[0]:ltrb[2]], 
					(LP_shape[1],LP_shape[0]), 
					cv2.INTER_CUBIC)
				LP_x = nd.array(LP, ctx=gpu(0)).transpose((2,0,1)).expand_dims(axis=0)
				LP_predict = self.predict(LP_x, get_LP_feature, 1)
				for LP_pred in LP_predict[0].asnumpy():
					if LP_pred[0] > -0.5 and LP_pred[1] > 0.5:
						print(LP_pred)
						cv2_add_bbox_text(LP, LP_pred, LP_names, LP_shape)
				cv2.imshow('LP', LP)					
		cv2.imshow('data', img_copy)
		#out.write(frame)

ctx = [gpu(0)]
######################## Data ########################
max_step = 32
data_shape = [288, 480]
all_anchors = nd.array(
	[[[0.552, 0.704], [0.441, 0.606], [0.281, 0.368]], 
	 [[0.065, 0.095], [0.080, 0.120], [0.105, 0.145]]], ctx=gpu(0))
cls_names = ['0','45','90','135','180','225','270','315','LP']

get_feature = GetFeaturePrymaid(all_anchors, len(cls_names))
get_feature.hybridize()
get_feature.initialize(ctx=ctx)
pretrain = os.path.join('realcar', 'backup', 'weight_%s'%args.PretrainIndex)
get_feature.collect_params().load(pretrain)	
#####################################################################
LP_step = 16
LP_shape = [160, 384]
LP_anchors = nd.array([[[0.5800, 0.1650]]], ctx=gpu(0))
LP_names = ['1','2','3','4','5','6','7','8','9',
	'A','B','C','D','E','F','G','H','I','J','K','L',
	'M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

get_LP_feature = GetFeature16(LP_anchors, len(LP_names))
get_LP_feature.hybridize()
get_LP_feature.initialize(ctx=ctx, init=init.Xavier())
get_LP_feature.collect_params().load('LV4/backup/weight_20')

######################## Main ########################
v = Valid()

if args.TrainOrValid == 'valid': 
	batch_iter = get_iterators('realcar', data_shape, 1, 'valid')
	v.valid()

if args.TrainOrValid == 'demo':	
	v.demo()

