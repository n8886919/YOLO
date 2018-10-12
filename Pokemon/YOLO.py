# -*- coding: utf-8 -*-
import time
from time import gmtime, strftime

from mxnet import autograd
from mxnet import init
from mxboard import SummaryWriter
'''
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError
'''
from utils import *

class Valid():
	def __init__(self):
		self.addLP = AddMonster()
		n = [len(nn) for nn in all_anchors] # anchor per sub_map
		area = [None]*all_anchors.shape[0]
		step = max_step
		for i, a in enumerate(area):
			area[i] = int(data_shape[0]*data_shape[1]/(step**2))

			scale_map = nd.repeat(nd.array([step], ctx=ctx[0]), repeats=area[i]*n[i]).reshape(1, area[i], n[i], 1)
			
			offset_y = nd.arange(0, data_shape[0], step=step, repeat=int(n[i]*data_shape[1]/step), ctx=ctx[0])
			offset_y = offset_y.reshape(1, area[i], n[i], 1) #self.offset_y = nd.tile(offset_y1, (b, 1, 1, 1))

			offset_x = nd.tile(nd.arange(0, data_shape[1], step=step, repeat=n[i], ctx=ctx[0]), int(data_shape[0]/step))
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
	def yxhw_to_ltrb(self, yxhw):
		#left/top/right/bottom
		#print(self.offset_y)
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
	def predict(self, x):
		cls_pred, score, yxhw = get_feature(x)
		score = nd.sigmoid(score)
		cid = nd.argmax(cls_pred, axis=-1, keepdims=True)
		predict_ltrb = self.yxhw_to_ltrb(yxhw)
		output = nd.concat(cid, score, predict_ltrb, dim=-1)#.reshape(-1,6)
		nms = nd.contrib.box_nms(
			output.reshape((0, -1, 6)), 
			force_suppress=True,
			overlap_thresh=0.3, 
			coord_start=2, 
			score_index=1, 
			id_index=0,
			topk=topk)
		return nms		
	def valid(self):
		print('\033[1;33;40mValid\033[0;37;40m')
		while True:
			x, labels = self.addLP.add(1, ctx[0])
			pred = self.predict(x)[0].asnumpy()
			img = x.transpose((0,2,3,1))[0]
			B, G, R = img.split(num_outputs=3, axis=-1)
			img = nd.concat(R, G, B, dim= -1).asnumpy()
			print(labels[0])
			print(pred[:topk])
			for pi,p in enumerate(pred):
				if p[0] < -0.5 or p[1] < 0.8: continue
				cv2_add_bbox_text(img, p, cls_names, [288,480], pi)
			cv2.imshow('frame1', img)
			if cv2.waitKey(0) & 0xFF == ord('q'): break
	def video(self):
		cap = cv2.VideoCapture(-1)
		while(True):
			ret, frame = cap.read()
			frame1 = cv2.resize(frame, (data_shape[1],data_shape[0]))
			B,G,R = nd.array(frame1).split(num_outputs=3, axis=-1)
			x = nd.concat(R,G,B, dim=-1)
			x = nd.transpose(x, (2,0,1)).expand_dims(axis=0)/255.
			pred = self.predict(x.as_in_context(gpu(0)))[0].asnumpy()
			#print(pred)
			for pi,p in enumerate(pred):
				if p[0] < -0.5 or p[1] < 0.8: continue
				cv2_add_bbox_text(frame1, p, cls_names, [288,480], pi)
			#out.write(frame)
			cv2.imshow('frame1', frame1)
			if cv2.waitKey(1) & 0xFF == ord('q'): break
	
class Train():
	def __init__(self):
		self.add_monster = AddMonster()
		self.get_default_ltrb()

		self.SCE_loss = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)
		self.LGS_loss = gluon.loss.LogisticLoss(label_format='binary')
		self.L1_loss = gluon.loss.L1Loss()
		self.trainer = gluon.Trainer(
			get_feature.collect_params(), 
			'adam', {'learning_rate': LR})
		self.backup_dir = 'backup/'
		if not os.path.exists(self.backup_dir): os.makedirs(self.backup_dir)
		self.sw = SummaryWriter(logdir='./logs', flush_secs=5)
	def get_default_ltrb(self):
		area = [None] * all_anchors.shape[0] # 3 sub_map
		ltrb = [None] * all_anchors.shape[0]
		self.all_anchors = [all_anchors.copyto(device) for device in ctx]

		n = [len(anchors) for anchors in all_anchors]
		step = max_step
		for i in range(all_anchors.shape[0]):
			area[i] = int(data_shape[0]*data_shape[1]/(step**2))
			h, w = all_anchors[i].split(num_outputs=2, axis=-1)	

			y = nd.arange(step/2, data_shape[0], step=step, repeat=int(n[i]*data_shape[1]/step), ctx=ctx[0])
			h = nd.tile(h.reshape(-1), area[i]) * data_shape[0]
			t = (y - 0.5*h).reshape(area[i], n[i], 1) / data_shape[0]
			b = (y + 0.5*h).reshape(area[i], n[i], 1) / data_shape[0]

			x = nd.arange(step/2, data_shape[1], step=step, repeat=n[i], ctx=ctx[0])	
			w = nd.tile(w.reshape(-1), int(data_shape[1]/step)) * data_shape[1]
			l = nd.tile(x - 0.5*w, int(data_shape[0]/step)).reshape(area[i], n[i], 1) / data_shape[1]
			r = nd.tile(x + 0.5*w, int(data_shape[0]/step)).reshape(area[i], n[i], 1) / data_shape[1]

			ltrb[i] = nd.concat(l,t,r,b, dim=-1)
			if i ==0: LTRB = ltrb[0]
			else: LTRB = nd.concat(LTRB, ltrb[i], dim=0)
				
			step /= 2

		area = nd.array(area, ctx=ctx[0])
		self.area = [area.copyto(device) for device in ctx]
		self.all_anchors_ltrb = [LTRB.copyto(device) for device in ctx]
	def find_best(self, b, L, gpu_index):
		IOUs = get_iou(self.all_anchors_ltrb[gpu_index], L) # L is (f,l,t,r,b)
		#best_match = int(np.argmax(IOUs)), Global reduction not supported yet ...
		IOUs = IOUs.reshape(-1)
		best_match = IOUs.argmax(axis=0) #print(best_match)
		best_pixel = nd.floor(best_match/all_anchors.shape[1])
		best_anchor = nd.modulo(best_match, all_anchors.shape[1]) 
		best_ltrb = self.all_anchors_ltrb[gpu_index][best_pixel, best_anchor].reshape(-1)
		#print('best match anchor', best_ltrb)
		stride = float(max_step)	

		if best_pixel < self.area[gpu_index][0]: 
			pyramid_layer = 0
		elif (best_pixel >= self.area[gpu_index][0]) \
			and (best_pixel < self.area[gpu_index][1]+self.area[gpu_index][0]): 
			pyramid_layer = 1
			stride = stride/2
		elif (best_pixel >= self.area[gpu_index][1]+self.area[gpu_index][0]) \
			and (best_pixel < self.area[gpu_index][2]+self.area[gpu_index][1]+self.area[gpu_index][0]): 
			pyramid_layer = 2
			stride = stride/4
		else: print('Best_pixel Error')

		by_minus_cy = (L[2]+L[4])/2 - (best_ltrb[3]+best_ltrb[1])/2
		sigmoid_ty = by_minus_cy*data_shape[0]/stride + 0.5
		sigmoid_ty = nd.clip(sigmoid_ty, 0.001, 0.999)
		ty = -nd.log(1/sigmoid_ty - 1)

		bx_minus_cx = (L[1]+L[3])/2 - (best_ltrb[2]+best_ltrb[0])/2
		sigmoid_tx = bx_minus_cx*data_shape[1]/stride + 0.5		
		sigmoid_tx = nd.clip(sigmoid_tx, 0.001, 0.999)
		tx = -nd.log(1/sigmoid_tx - 1)	

		th = nd.log((L[4]-L[2]) / self.all_anchors[gpu_index][pyramid_layer, best_anchor, 0])
		tw = nd.log((L[3]-L[1]) / self.all_anchors[gpu_index][pyramid_layer, best_anchor, 1])
		return best_pixel, best_anchor, nd.concat(ty, tx, th, tw, dim=-1) 
	def loss_mask(self, scores, labels, gpu_index):
		"""Generate training targets given predictions and labels."""
		b, a, n, _ = scores.shape
		t_id = nd.ones((b, a, n, 1), ctx=ctx[gpu_index]) * (-1)
		t_score = nd.zeros((b, a, n, 1), ctx=ctx[gpu_index])
		t_box = nd.zeros((b, a, n, 4), ctx=ctx[gpu_index])
		t_mask = nd.zeros((b, a, n, 1), ctx=ctx[gpu_index])
		for b in range(scores.shape[0]): 
			label = labels[b] #print(label)
			nd.random.shuffle(label)
			for L in label: # all object in the image
				if L[0] > -0.5: #print('label', L)
					best_pixel, best_anchor, best_box = self.find_best(b, L, gpu_index)
					t_box[b, best_pixel, best_anchor, :] = best_box
					t_id[b, best_pixel, best_anchor, :] = L[0] # others are ignore_label=-1
					t_score[b, best_pixel, best_anchor, :] = 1.0 # others are zero
					t_mask[b, best_pixel, best_anchor, :] = 1.0 # others are zero
		return t_id, t_score, t_box, t_mask
	def train(self):
		print('\033[1;33;40mTrain Pocket Monsters\033[0;37;40m')
		c = 0
		while True:
			c += 1
			x, y = self.add_monster.add(batch_size, ctx[0]) 
			with autograd.record():
				cls_pred, score, yxhw = get_feature(x) 
				with autograd.pause():
					t_id, t_score, t_box, t_mask = self.loss_mask(score, y, 0)	

					score_weight = nd.where(t_mask>0, nd.ones_like(t_mask)*10,
						nd.ones_like(t_mask)*0.1, ctx=ctx[0])

				s_loss = self.LGS_loss(score, t_score, score_weight)
				c_loss = self.SCE_loss(cls_pred, t_id, t_mask * 10.)
				b_loss = self.L1_loss(yxhw, t_box, t_mask * 1.)

				loss = c_loss + s_loss + b_loss
			loss.backward()
			self.trainer.step(batch_size) #backpropagation, normalize to 1/batch_size
			#if c%100==0:
			self.sw.add_scalar(tag='score_loss', value=np.mean(s_loss.asnumpy()), global_step=c)
			self.sw.add_scalar(tag='box_loss', value=np.mean(b_loss.asnumpy()), global_step=c)
			self.sw.add_scalar(tag='class_loss', value=np.mean(c_loss.asnumpy()), global_step=c)
			if c%1000==0:
				save_model = os.path.join(self.backup_dir, 'weight_%d'%int(c/1000))
				print(save_model)	
				get_feature.collect_params().save(save_model)

ctx = [gpu(0)]
topk = 1
######################## Data ########################
data_shape = [288,480] 
cls_names = ['Charmander','Pikachu','Squirtle', 'Bulbasau']

#################### NN Initialize ####################

all_anchors = nd.array([[[0.2, 0.2]]], ctx=ctx[0])
max_step = 16
get_feature = GetFeature16(all_anchors, len(cls_names))

get_feature.initialize(ctx=ctx, init=init.Xavier())
get_feature.hybridize()

######################## Main ########################
pretrain = os.path.join('backup', 'weight_0')
get_feature.collect_params().load(pretrain)	

####### Hyper Parameter #######
batch_size = 25
LR = 0.0001
####### Hyper Parameter #######
t = Train()
#t.train()

v = Valid()
v.valid()
v.video()

