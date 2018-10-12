import argparse

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import scipy.io as sio 
import threading
import time

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError


from mxnet import gluon
from mxnet import gpu, cpu
from mxnet import image
from mxnet import init
from mxnet import metric
from mxnet import nd, sym
from mxnet.gluon import nn

import PIL.Image as PILIMG
from PIL import ImageFilter, ImageEnhance
'''
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
'''
import cv2

cos_offset = np.array([math.cos(x*math.pi/180) for x in range(0,360,15)])
sin_offset = np.array([math.sin(x*math.pi/180) for x in range(0,360,15)])
Leaky_alpha = 0.1
numeral = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['A','B','C','D','E','F','G','H','J','K','L',
		'M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
color = [(255,255,0), (255,0,255), (0,255,255),(0,0,255), 
	(0,255,0), (255,0,0), (0,0,0), (255,255,255)]
'''
shape_clsnames = {
	'LV0': ([96,192], ['0','1']),
	'LV1': ([96,192], ['0','1']), 
	'LV2': ([96,192], numeral), 
	'LV3': ([160,384], numeral+alphabet), 
	'LV4': ([160,384], numeral+alphabet),
	'LP' : ([160,384], numeral+alphabet),
	'LV2_34class': ([192,384], numeral+alphabet),
	'azimuth': ([288, 480], ['0','45','90','135','180','225','270','315','LP']),
	'realcar': ([288, 480], ['0','45','90','135','180','225','270','315','LP']),
	'gazebo': ([288, 480], ['0','45','90','135','180','225','270','315','LP']),
	'HP_31': ([320, 512], ['0','45','90','135','180','225','270','315']),
	'HP_32': ([320, 512], ['0','45','90','135','180','225','270','315']),
	'HP_33': ([320, 512], ['0','15','30','45','60','75','90','105',
		'120','135','150','165','180','195','210','225',
		'240','255','270','285','300','315','330','345']),
	'HP_34': ([320, 512], ['0','15','30','45','60','75','90','105',
		'120','135','150','165','180','195','210','225',
		'240','255','270','285','300','315','330','345'])
	}
'''
all_cls_names = [
	['0','1'], 
	numeral, 
	numeral+alphabet, 
	['0','45','90','135','180','225','270','315'], 
	['0','15','30','45','60','75','90','105',
	'120','135','150','165','180','195','210','225',
	'240','255','270','285','300','315','330','345']
	]
all_size = [
	[96,192],
	[160,384],
	[288, 480],
	[320, 512]
	]
'''
def Parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", 
		help="LV0=find9, LV1=find2&9, LV2=find1~9, LV3=find all, LV4=add BG, azimuth")
	parser.add_argument("TrainOrValid", help="train or valid")
	parser.add_argument("PretrainIndex", help="zero means don't use pretrain")
	args = parser.parse_args()
	data_shape, cls_names = shape_clsnames[args.dataset]
	return args, data_shape, cls_names
'''
def Parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", help="31 32 33 34")
	parser.add_argument("mode", help="train or valid or video")
	args = parser.parse_args()
	#data_shape, cls_names = shape_clsnames[args.dataset]
	return args#, data_shape, cls_names

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
def softmax(x):return np.exp(x)/np.sum(np.exp(x),axis=0)
def nd_inv_sigmoid(x): return -nd.log(1/x - 1)	
	
def add_bbox(im, b, c):
	# Parameters:
	#### ax  : plt.ax
	#### size: [h,w]
	#### pred: numpy.array [score, l, t, r, b, prob1, prob2, ...]
	r = -b[5]
	im_w = im.shape[1]
	im_h = im.shape[0]
	h = b[3] * im_h
	w = b[4] * im_w
	a = np.array([[
		[ w*math.cos(r)/2 - h*math.sin(r)/2,  w*math.sin(r)/2 + h*math.cos(r)/2],
		[-w*math.cos(r)/2 - h*math.sin(r)/2, -w*math.sin(r)/2 + h*math.cos(r)/2],
		[-w*math.cos(r)/2 + h*math.sin(r)/2, -w*math.sin(r)/2 - h*math.cos(r)/2],
		[ w*math.cos(r)/2 + h*math.sin(r)/2,  w*math.sin(r)/2 - h*math.cos(r)/2]]])
	s = np.array([b[2], b[1]])*[im_w,im_h]
	a = (a + s).astype(int)
	cv2.polylines(im, a, 1, c, 2)
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

def cv2_add_bbox_text(img, p, text, c):
	size = img.shape
	c = color[c%len(color)]
	l = min(max(int(p[1] * size[1]), 0),size[1])
	t = min(max(int(p[2] * size[0]), 0),size[0])
	r = min(max(int(p[3] * size[1]), 0),size[1])
	b = min(max(int(p[4] * size[0]), 0),size[0])
	cv2.rectangle(img, (l ,t), (r ,b), c, 2)
	cv2.putText(img, '%s %.3f'%(text, p[0]), 
		(l, t-10), 2, 1, c, 2)

def add_noise(img, noise_var):
	np_img = np.array(img)
	noise = np.random.normal(0., noise_var, np_img.shape)
	np_img = np_img + noise 
	np_img = np.clip(np_img, 0, 255)
	img = PILIMG.fromarray(np.uint8(np_img))
	return img
def img_enhance(img, M=0, N=1, R=10, G=1):
	#img = add_noise(img, 50)
	w, h = img.size
	rd = np.random
	#https://stackoverflow.com/questions/14177744/
	#how-does-perspective-transformation-work-in-pil
	if M>0 or N>0:
		m, n = rd.random()*M*2-M, rd.random()*N*2-N # +-M or N
		xshift, yshift = abs(m)*h, abs(n)*w

		w, h = w + int(round(xshift)), h + int(round(yshift))
		img = img.transform((w, h), PILIMG.AFFINE, 
			(1, m, -xshift if m > 0 else 0, n, 1, -yshift if n > 0 else 0), 
			PILIMG.BILINEAR)

	r = rd.random()*R*2-R
	img = img.rotate(r, PILIMG.BILINEAR, expand=1)
	if G != 0:
		img = img.filter(ImageFilter.GaussianBlur(radius=rd.rand()*G))
	r = r*np.pi/180
	return img, r

def remove_car_from_sun2012():
	from shutil import copyfile
	import xml.etree.cElementTree as ET
	bg_root = '/media/nolan/HDD1/sun2012pascalformat'
	sun_img_path = os.path.join(bg_root, 'JPEGImages')
	sun_anno_path = os.path.join(bg_root, 'Annotations')
	counter = 0
	for img in os.listdir(sun_img_path):
		detected = False
		img_name = (img.split('.')[0]).split('/')[-1]


		img_xml_path = os.path.join(sun_anno_path, (img_name+'.xml'))
		try:
			img_xml = ET.ElementTree(file=img_xml_path)
			root = img_xml.getroot()
			for child in root:
				if child.tag == 'object':
					for sub_child in child:
						if sub_child.tag == 'name':
							text = sub_child.text
							if ('car' in text or 'van' in text or 'truck' in text):
								detected = True
								break
				if detected == True:
					break
		except: pass
		if detected == False:
			counter += 1
			src = os.path.join(sun_img_path, img)
			dst = os.path.join('/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/sun2012', img)
			copyfile(src, dst)
	print(counter)
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
def open_cam_onboard(width, height):
    # On versions of L4T previous to L4T 28.1, flip-method=2
    # Use Jetson onboard camera
    gst_str = ("nvcamerasrc ! "
               "video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458, format=(string)I420, framerate=(fraction)30/1 ! "
               "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
               "videoconvert ! appsink").format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

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
		#cap = cv2.VideoCapture(1)
		cap = cv2.VideoCapture('video/GOPR0730.MP4')
		while not rospy.is_shutdown():
			ret, img = cap.read()
			
			self.img = img
			#print(self.img.shape)
			#img = cv2.flip(img, -1)

			#rospy.sleep(0.1)
		cap.release()
	def _image_callback(self, img):
		##################### Convert and Predict #####################
		img = self.bridge.imgmsg_to_cv2(img, "bgr8")
		#self.img = img[60:420]
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
		
		rate = rospy.Rate(10)
		while not rospy.is_shutdown():
			if hasattr(self, 'img') and self.img is not None:

				nd_img = self.resz(nd.array(self.img))
				nd_img = nd_img.as_in_context(ctx)
				nd_img =  nd_img.transpose((2,0,1)).expand_dims(axis=0)/255.		
				out = self.predict(nd_img)
				self.visualize(out)
				rate.sleep()
			#else: print('Wait For Image')		
		
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
class GetFeaturePrymaid(gluon.HybridBlock):
	def __init__(self, all_anchors, cls_num, **kwargs):
		super(GetFeaturePrymaid, self).__init__(**kwargs)
		self.act = nn.LeakyReLU(Leaky_alpha)

		self.all_acrs = all_anchors
		self.clsn = cls_num

		with self.name_scope():
			# add name_scope on the outermost Sequential
			self.layer36_seq = nn.HybridSequential()
			self.layer36_seq.add(
				ConvBNLrelu(32, 3),
				ConvBNLrelu(64, 3, stride=2),
				Res([32,64]),
				ConvBNLrelu(128, 3, stride=2),
				Res([64,128]), Res([64,128]),  
				ConvBNLrelu(256, 3, stride=2),
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256])
			)
			self.layer61_seq = nn.HybridSequential()
			self.layer61_seq.add(
				ConvBNLrelu(512, 3, stride=2),
				Res([256, 512]),Res([256, 512]),
				Res([256, 512]), Res([256, 512]),
				Res([256, 512]), Res([256, 512]),
				Res([256, 512]), Res([256, 512])
			)			
			self.layer79_seq = nn.HybridSequential()
			self.layer79_seq.add(
				ConvBNLrelu(1024, 3, stride=2),
				Res([512, 1024]), Res([512, 1024]), 
				Res([512, 1024]), Res([512, 1024]),
				ConvBNLrelu(512, 1), ConvBNLrelu(1024, 3),
				ConvBNLrelu(512, 1), ConvBNLrelu(1024, 3),
				ConvBNLrelu(512, 1),				
			)
			self.branch_1 = nn.HybridSequential()
			self.branch_1.add(
				ConvBNLrelu(512, 1),
				nn.Conv2D((self.clsn+5)*len(self.all_acrs[0]), kernel_size=1)
			)
			if self.all_acrs.shape[0] > 1:
				self.conv2 = nn.Conv2D((self.clsn+5)*len(self.all_acrs[1]), kernel_size=1)
				self.conv2_1 = ConvBNLrelu(256, 1)
				self.conv2_2 = ConvBNLrelu(512, 3)

				self.branch_2 = nn.HybridSequential()
				self.branch_2.add(
					ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
					ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
					ConvBNLrelu(256, 1),
				)
			if self.all_acrs.shape[0] > 2:
				self.conv3_1 = nn.Conv2D(128, kernel_size=1)

				self.branch_3 = nn.HybridSequential()
				self.branch_3.add(
					ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
					ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),		
					ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),	
					nn.Conv2D((self.clsn+5)*len(self.all_acrs[2]), kernel_size=1)
					)
	def hybrid_forward(self, F, x, *args):
		#print(x.shape)
		L36 = self.layer36_seq(x)
		L61 = self.layer61_seq(L36)
		L79 = self.layer79_seq(L61)

		fm1 = self.branch_1(L79)
		fm = fm1.transpose((0, 2, 3, 1)).reshape(
			(0, -1, len(self.all_acrs[0]), self.clsn+5)
			)
		if self.all_acrs.shape[0] > 1:
			fm2 = self.act(self.conv2_1(L79))
			fm2 = F.UpSampling(fm2, scale=2, sample_type='nearest')
			fm2 = F.concat(fm2, L61, dim=1)
			fm2 = self.branch_2(fm2)
			
			node = fm2
			fm2 = self.act(self.conv2_2(fm2))
			fm2 = self.conv2(fm2)
			fm2 = fm2.transpose((0, 2, 3, 1)).reshape(
				(0, -1, len(self.all_acrs[1]), self.clsn+5)
				)
			fm=F.concat(fm, fm2, dim=1)
		if self.all_acrs.shape[0] > 2:
			fm3 = self.act(self.conv3_1(node))
			fm3 = F.UpSampling(fm3, scale=2, sample_type='nearest')
			fm3 = F.concat(fm3, L36, dim=1)
			fm3 = self.branch_3(fm3)
			fm3 = fm3.transpose((0, 2, 3, 1)).reshape(
				(0, -1, len(self.all_acrs[2]), self.clsn+5)
				)
			fm=F.concat(fm, fm3, dim=1)	
		cls_pred = fm.slice_axis(begin=0, end=self.clsn, axis=-1) 
		score = fm.slice_axis(begin=self.clsn, end=self.clsn+1, axis=-1)
		yxhw = fm.slice_axis(begin=self.clsn+1, end=self.clsn+5, axis=-1)
		return cls_pred, score, yxhw
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
class L1(gluon.HybridBlock):
	def __init__(self, **kwargs):
		super(L1, self).__init__(**kwargs)
		with self.name_scope():
			self.seq = nn.HybridSequential()
			self.seq.add(
				ConvBNLrelu(16, 3),
				ConvBNLrelu(32, 3, stride=2),
				Res([16,32]),Res([16,32]),
				Res([16,32]),Res([16,32]),
				ConvBNLrelu(64, 3, stride=2),
				Res([32,64]), Res([32,64]),
				Res([32,64]), Res([32,64]),
				ConvBNLrelu(128, 3, stride=2),
				Res([64, 128]), Res([64, 128]),
				Res([64, 128]), Res([64, 128]),
				ConvBNLrelu(256, 3, stride=2),
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256]))
	def hybrid_forward(self, F, x, *args):
		return self.seq(x)
class HP_31(gluon.HybridBlock):
	def __init__(self, all_anchors, cls_num, **kwargs):
		super(HP_31, self).__init__(**kwargs)
		self.act = nn.LeakyReLU(Leaky_alpha)
		self.n = len(all_anchors[0])
		self.c = cls_num
		with self.name_scope():
			self.L1 = L1()
			#self.L2 = L2()
			#self.L3 = L3()
			self.B16_LP = nn.HybridSequential()
			self.B16_LP.add(
				ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				nn.Conv2D(5*self.n, kernel_size=1))	
			self.B16_car = nn.HybridSequential()
			self.B16_car.add(
				ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				nn.Conv2D((cls_num+5)*self.n, kernel_size=1))	
			self.B32_car = nn.HybridSequential()
			self.B32_car.add(
				ConvBNLrelu(512, 3, stride=2),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				nn.Conv2D((cls_num+5)*self.n, kernel_size=1))
	def hybrid_forward(self, F, x, *args):
		#print(x.shape)
		N1 = self.L1(x)
		#N2 = self.L2(N1)
		#N3 = self.L3(N2)
		LP = self.B16_LP(N1).transpose((0, 2, 3, 1)).reshape((0, -1, 2, 5))
		
		LP_s = LP.slice_axis(begin=0, end=1, axis=-1)
		LP_b= LP.slice_axis(begin=1, end=5, axis=-1)
		
		N2 = self.B16_car(N1).transpose((0, 2, 3, 1)).reshape((0, -1, 2, 5+self.c))
		N3 = self.B32_car(N1).transpose((0, 2, 3, 1)).reshape((0, -1, 2, 5+self.c))

		car = F.concat(N2, N3, dim=1)	
		car_s = car.slice_axis(begin=0, end=1, axis=-1) 
		car_b = car.slice_axis(begin=1, end=5, axis=-1)
		car_c = car.slice_axis(begin=5, end=5+self.c, axis=-1)
		
		return [LP_s, LP_b], [car_s, car_b, car_c]
class HP_32(gluon.HybridBlock):
	'''
	NO Feature Prymaid
	'''
	def __init__(self, all_anchors, cls_num, **kwargs):
		super(HP_32, self).__init__(**kwargs)
		self.act = nn.LeakyReLU(Leaky_alpha)
		self.n = len(all_anchors[0])
		self.c = cls_num
		with self.name_scope():
			self.seq = nn.HybridSequential()
			self.seq.add(
				ConvBNLrelu(16, 3),
				ConvBNLrelu(32, 3, stride=2),
				Res([16,32]),Res([16,32]),
				Res([16,32]),Res([16,32]),
				ConvBNLrelu(64, 3, stride=2),
				Res([32,64]), Res([32,64]),
				Res([32,64]), Res([32,64]),
				ConvBNLrelu(128, 3, stride=2),
				Res([64, 128]), Res([64, 128]),
				Res([64, 128]), Res([64, 128]),
				Res([64, 128]), Res([64, 128]),
				ConvBNLrelu(256, 3, stride=2),
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256]))
			self.B16_LP = nn.HybridSequential()
			self.B16_LP.add(
				ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				#ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				nn.Conv2D(5*self.n, kernel_size=1))	
			self.B16_car = nn.HybridSequential()
			self.B16_car.add(
				ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				ConvBNLrelu(128, 1), ConvBNLrelu(256, 3),
				nn.Conv2D((cls_num+5)*self.n, kernel_size=1))	
			self.B32_car = nn.HybridSequential()
			self.B32_car.add(
				ConvBNLrelu(512, 3, stride=2),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				nn.Conv2D((cls_num+5)*self.n, kernel_size=1))
	def hybrid_forward(self, F, x, *args):
		N1 = self.seq(x)
		LP = self.B16_LP(N1).transpose((0, 2, 3, 1)).reshape((0, -1, 2, 5))
		LP_s = LP.slice_axis(begin=0, end=1, axis=-1)
		LP_b= LP.slice_axis(begin=1, end=5, axis=-1)
		
		N2 = self.B16_car(N1).transpose((0, 2, 3, 1)).reshape((0, -1, 2, 5+self.c))
		N3 = self.B32_car(N1).transpose((0, 2, 3, 1)).reshape((0, -1, 2, 5+self.c))

		car = F.concat(N2, N3, dim=1)	
		car_s = car.slice_axis(begin=0, end=1, axis=-1) 
		car_b = car.slice_axis(begin=1, end=5, axis=-1)
		car_c = car.slice_axis(begin=5, end=5+self.c, axis=-1)
		
		return [LP_s, LP_b], [car_s, car_b, car_c]
class HP_34(gluon.HybridBlock):
	def __init__(self, cls_num, **kwargs):
		'''
		box rotate
		'''
		super(HP_34, self).__init__(**kwargs)
		self.act = nn.LeakyReLU(Leaky_alpha)
		self.n = 2
		self.c = cls_num
		with self.name_scope():
			self.seq = nn.HybridSequential()
			self.seq.add(
				ConvBNLrelu(16, 3),
				ConvBNLrelu(32, 3, stride=2),
				Res([16,32]),Res([16,32]),
				Res([16,32]),Res([16,32]),
				#Res([16,32]),Res([16,32]),#
				#Res([16,32]),Res([16,32]),#
				ConvBNLrelu(64, 3, stride=2),
				Res([32,64]), Res([32,64]),
				Res([32,64]), Res([32,64]),
				#Res([32,64]), Res([32,64]),#
				#Res([32,64]), Res([32,64]),#
				ConvBNLrelu(128, 3, stride=2),
				#Res([64, 128]), Res([64, 128]),#
				Res([64, 128]), Res([64, 128]),
				Res([64, 128]), Res([64, 128]),
				Res([64, 128]), Res([64, 128]),
				ConvBNLrelu(256, 3, stride=2),
				#Res([128, 256]), Res([128, 256]),#
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256]))
			self.B16_car = nn.HybridSequential()
			self.B16_car.add(
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				#ConvBNLrelu(512, 1), ConvBNLrelu(1024, 3),#
				#ConvBNLrelu(512, 1), ConvBNLrelu(1024, 3),#

				nn.Conv2D((cls_num+6)*self.n, kernel_size=1))	
			self.B32_car = nn.HybridSequential()
			self.B32_car.add(
				ConvBNLrelu(512, 3, stride=2),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				ConvBNLrelu(256, 1), ConvBNLrelu(512, 3),
				#ConvBNLrelu(512, 1), ConvBNLrelu(1024, 3),#
				#ConvBNLrelu(512, 1), ConvBNLrelu(1024, 3),#

				nn.Conv2D((cls_num+6)*self.n, kernel_size=1))
	def hybrid_forward(self, F, x, *args):
		N1 = self.seq(x)
		N2 = self.B16_car(N1).transpose((0, 2, 3, 1)).reshape((0, -1, 2, 6+self.c))
		N3 = self.B32_car(N1).transpose((0, 2, 3, 1)).reshape((0, -1, 2, 6+self.c))

		car = F.concat(N2, N3, dim=1)	
		car_s = car.slice_axis(begin=0, end=1, axis=-1)
		car_r = car.slice_axis(begin=1, end=2, axis=-1)
		car_r = F.tanh(car_r)
		car_b = car.slice_axis(begin=2, end=6, axis=-1)
		car_c = car.slice_axis(begin=6, end=6+self.c, axis=-1)
		
		return [car_s, car_r, car_b, car_c]
class OCR(gluon.HybridBlock):
	def __init__(self, cls_num, **kwargs):
		super(OCR, self).__init__(**kwargs)
		self.n = cls_num
		with self.name_scope():
			self.seq = nn.HybridSequential()
			self.seq.add(
				ConvBNLrelu(16, 3),
				ConvBNLrelu(16, 3),
				ConvBNLrelu(32, 3, stride=2),
				Res([16,32]),Res([16,32]),
				ConvBNLrelu(64, 3, stride=2),
				Res([32,64]), Res([32,64]), 
				ConvBNLrelu(128, 3, stride=2),
				Res([64,128]), Res([64,128]),
				Res([64,128]), Res([64,128]),
				#Res([64,128]), Res([64,128]),
				ConvBNLrelu(256, 3, stride=2),
				Res([128, 256]), Res([128, 256]),
				Res([128, 256]), Res([128, 256]),
				#Res([128, 256]), Res([128, 256]),

				nn.Conv2D(256, kernel_size=(3, 1), padding=(1, 0), strides=(2,1)),
				nn.BatchNorm(),
				nn.LeakyReLU(Leaky_alpha),
				#nn.Conv2D(512, kernel_size=(3, 1), padding=(1, 0)),
				#nn.BatchNorm(),
				#nn.LeakyReLU(Leaky_alpha), 
				nn.Conv2D(256, kernel_size=(5, 1)),
				nn.BatchNorm(),
				nn.LeakyReLU(Leaky_alpha), 
				#nn.Conv2D(1024, kernel_size=1),
				#nn.BatchNorm(),
				#nn.LeakyReLU(Leaky_alpha),
				nn.Conv2D(cls_num+1, kernel_size=1))	

	def hybrid_forward(self, F, x, *args):
		N = self.seq(x).transpose((0, 2, 3, 1)).reshape((0, -1, self.n+1))
		prob = N.slice_axis(begin=0, end=self.n, axis=-1)
		score = N.slice_axis(begin=self.n, end=self.n+1, axis=-1)
		score = F.sigmoid(score)
		return prob, score

class AddLP():
	def __init__(self, img_h, img_w, class_index):
		self.class_index = int(class_index)
		self.h = img_h
		self.w = img_w
		self.BIL = PILIMG.BILINEAR
		self.LP_WH = [[380, 160], [320, 150], [320, 150]]
		self.x = [np.array([7, 56, 106, 158, 175, 225, 274, 324]), 
				  np.array([7, 57, 109, 130, 177, 223, 269])]

		self.font0 = [None] * 35
		self.font1 = [None] * 35

		self.dot = PILIMG.open('fonts/'+"34.png").resize((10, 70), self.BIL)
		for font_name in range(0, 34):
			f = PILIMG.open('fonts/'+str(font_name)+".png")
			self.font0[font_name] = f.resize((45, 90), self.BIL)
			self.font1[font_name] = f.resize((40, 80), self.BIL)
		

		self.augs = image.CreateAugmenter(data_shape=(3, self.h, self.w),	
			inter_method=10, brightness=0.5, contrast=0.5, saturation=0.5, pca_noise=1.0)
		self.augs2 = image.CreateAugmenter(data_shape=(3, self.h, self.w),	
			inter_method=10, brightness=0.5, contrast=0.5, saturation=0.5, pca_noise=1.0)
	def draw_LP(self, LP_type):
		LP_w, LP_h = self.LP_WH[LP_type]
		x = self.x[LP_type]
		label = []
		if LP_type == 0: # ABC-1234
			LP = PILIMG.new('RGBA', (LP_w, LP_h), color[7])
			abc = np.random.randint(10, 34, size=3)
			for i, j in enumerate(abc): 
				LP.paste(self.font0[j], (x[i], 35))
				label.append([j, float(x[i])/LP_w, float(x[i]+45)/LP_w])

			LP.paste(self.dot,(x[3], 45))

			num = np.random.randint(0, 10, size=4)
			for i, j in  enumerate(num): 
				LP.paste(self.font0[j], (x[i+4], 35)) 
				label.append([j, float(x[i+4])/LP_w, float(x[i+4]+45)/LP_w])

		if LP_type == 1: # AB-1234
			LP = PILIMG.new('RGBA', (LP_w, LP_h), color[7])
			abc = np.random.randint(10, 34, size=2)
			for i, j in enumerate(abc): 
				LP.paste(self.font1[j], (x[i], 40))
				label.append([j, float(x[i])/LP_w, float(x[i]+40)/LP_w])

			LP.paste(self.dot,(x[2], 45))

			num = np.random.randint(0, 10, size=4)
			for i, j in  enumerate(num): 
				LP.paste(self.font1[j], (x[i+3], 40))
				label.append([j, float(x[i+3])/LP_w, float(x[i+3]+40)/LP_w])

		return LP, LP_w, LP_h, label
	def add(self, img_batch, label_batch):
		ctx = label_batch.context

		bs = label_batch.shape[0]
		h = img_batch.shape[2]
		w = img_batch.shape[3]
		
		LP_label = nd.zeros((bs,1,5), ctx=ctx)
		LP_batch = nd.zeros((bs,3,h,w), ctx=ctx)
		mask = nd.zeros((bs,3,h,w), ctx=ctx)

		for i in  range(bs):
			LP_type = np.random.randint(2)
			LP, LP_w, LP_h, _ = self.draw_LP(LP_type)

			resize_w = int((np.random.rand()*0.15+0.15)*LP_w)
			resize_h = int((np.random.rand()*0.1+0.2)*LP_h)
			LP = LP.resize((resize_w, resize_h), self.BIL)

			LP = add_noise(LP, 20.)
			LP, r = img_enhance(LP, M=.1, N=.2, R=20.)
			

			paste_x = int(np.random.rand() * (self.w-120))
			paste_y = int(np.random.rand() * (self.h-120))
			
			tmp = PILIMG.new('RGBA', (self.w, self.h))
			tmp.paste(LP, (paste_x, paste_y))
			m = nd.array(tmp.split()[-1], ctx=ctx).reshape(1, self.h, self.w)
			mask[i] = nd.tile(m, (3,1,1))

			LP = PILIMG.merge("RGB", (tmp.split()[:3]))
			LP = nd.array(LP)

			for aug in self.augs: 
				LP = aug(LP)
			LP_batch[i] = LP.as_in_context(ctx).transpose((2,0,1))/255.

			LPl, LPt, LPr, LPb = tmp.getbbox()
			LP_label[i] = nd.array([[self.class_index, 
				float(LPl)/self.w, float(LPt)/self.h, 
				float(LPr)/self.w, float(LPb)/self.h]])
			
		img_batch = nd.where(mask<200, img_batch, LP_batch)
		img_batch = nd.clip(img_batch, 0, 1)

		label_batch = nd.concat(label_batch, LP_label, dim=1)
		return img_batch, label_batch

	def render(self, bs, ctx):
		LP_label = nd.ones((bs,7,5), ctx=ctx) * -1
		LP_batch = nd.zeros((bs,3,self.h,self.w), ctx=ctx)
		mask = nd.zeros((bs,3,self.h,self.w), ctx=ctx)
		
		for i in range(bs):
			LP_type = np.random.randint(2) # 0,1
			LP, LP_w, LP_h, labels = self.draw_LP(LP_type)

			resize = np.random.rand() * 0.1 + 0.9
			LP_w = int(resize * self.w)
			LP_h = int((np.random.rand()*0.1+0.9) * resize * self.h)
			LP = LP.resize((LP_w, LP_h), self.BIL)

			LP, r = img_enhance(LP, M=.0, N=.0,R=5.0, G=8.0)
			#LP = LP.filter(ImageFilter.GaussianBlur(radius=np.random.rand()*8.))

			paste_x = np.random.randint(int(-0.0*LP_w), int(self.w-LP_w))
			paste_y = np.random.randint(int(-0.0*LP_h), int(self.h-LP_h))

			tmp = PILIMG.new('RGBA', (self.w, self.h))
			tmp.paste(LP, (paste_x, paste_y))
			bg = PILIMG.new('RGBA', (self.w, self.h), tuple(np.random.randint(255,size=3)))
			LP = PILIMG.composite(tmp, bg, tmp)

			LP = nd.array(PILIMG.merge("RGB", (LP.split()[:3])))
			for aug in self.augs2: LP = aug(LP)

			LP_batch[i] = LP.as_in_context(ctx).transpose((2,0,1))/255.

			r = r*np.pi/180
			offset = paste_x + abs(LP_h*math.sin(r)/2)
			for j,c in enumerate(labels):

				LP_label[i,j,0] = c[0]
				LP_label[i,j,1] = (offset + c[1]*LP_w*math.cos(r))/self.w
				LP_label[i,j,3] = (offset + c[2]*LP_w*math.cos(r))/self.w
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
				ax[i].plot(range(8,384,16),(1-s)*160, 'r-')
				ax[i].imshow(img_batch[i].transpose((1,2,0)).asnumpy())

			raw_input('next')
	def label2nparray(self, label):
		score = nd.zeros((24))
		for L in label: # all object in the image
			if L[0] < 0: continue
			text_cent = ((L[3] + L[1])/2.)
			left = int(round((text_cent.asnumpy()[0]-15./self.w)*24))
			right = int(round((text_cent.asnumpy()[0]+15./self.w)*24))
			#left = int(round(L[1].asnumpy()[0]*24))
			#right = int(round(L[3].asnumpy()[0]*24))
			for ii in range(left, right):
				box_cent = (ii + 0.5) / 24.
				score[ii] = 1-nd.abs(box_cent-text_cent)/(L[3]-L[1])
		return score.asnumpy()
	def test_add(self,b):
		#while True:
		batch_iter = load(b, h, w)
		for batch in batch_iter:
			imgs = batch.data[0].as_in_context(ctx[0]) # b*RGB*w*h
			labels = batch.label[0].as_in_context(ctx[0]) # b*L*5
			#imgs = nd.zeros((b, 3, self.h, self.w), ctx=gpu(0))*0.5
			tic = time.time()
			imgs, labels = self.add(imgs/255, labels)
			#print(time.time()-tic)
			for i, img in enumerate(imgs):
				R,G,B = img.transpose((1,2,0)).split(num_outputs=3, axis=-1)
				img = nd.concat(B,G,R, dim=-1).asnumpy()
				print(labels[i])
				cv2.imshow('%d'%i, img)
			if cv2.waitKey(0) & 0xFF == ord('q'): break
class RenderCar():
	def __init__(self, batch_size, img_h, img_w, ctx):
		self.h = img_h
		self.w = img_w	
		self.bs = batch_size
		self.ctx = ctx
		self.BIL = PILIMG.BILINEAR

		self.all_car = []
		raw_car_path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/syn_images/02958343'
		for file in os.listdir(raw_car_path):
			for img in os.listdir(os.path.join(raw_car_path, file)):
				img_path = os.path.join(raw_car_path, file, img)
				self.all_car.append(img_path)

		self.pascal_train = []
		self.pascal_valid = []
		self.pascal3d_anno = '/media/nolan/HDD1/PASCAL3D+_release1.1/Annotations/car_imagenet'
		pascal3d_train_path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/pascal_train'
		pascal3d_valid_path = '/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/pascal_valid'
		for img in os.listdir(pascal3d_train_path):
			img_path = os.path.join(pascal3d_train_path, img)
			self.pascal_train.append(img_path)

		for img in os.listdir(pascal3d_valid_path):
			img_path = os.path.join(pascal3d_valid_path, img)
			self.pascal_valid.append(img_path)


		self.augs = image.CreateAugmenter(data_shape=(3, img_h, img_w),	
			inter_method=10, brightness=0.5, contrast=0.5, saturation=0.5, hue=1.0, pca_noise=0.1)
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
		label_batch = nd.ones((self.bs,1,6), ctx=ctx) * (-1) 
		img_batch = nd.zeros((self.bs,3,self.h,self.w), ctx=ctx)
		mask = nd.zeros((self.bs,3,self.h,self.w), ctx=ctx)
		selected = np.random.randint(len(self.all_car), size=self.bs)

		for i in range(self.bs):
			if np.random.rand()>0.8: continue

			img_path = self.all_car[selected[i]]
			pil_img = PILIMG.open(img_path)
			#################### resize ####################
			resize = np.random.rand()*0.5+0.2 # 0.4~1.2
			resize_w = int((np.random.rand()*0.2+0.9) * resize * pil_img.size[0]) # 0.9~1.1
			resize_h = int((np.random.rand()*0.2+0.9) * resize * pil_img.size[1]) # 0.9~1.1
			pil_img = pil_img.resize((resize_w, resize_h), self.BIL)

			box_l, box_t, box_r, box_b = pil_img.getbbox()
			box_w = box_r - box_l
			box_h = box_b - box_t

			pil_img, r = img_enhance(pil_img, M=.0, N=.0,R=30.0, G=0)
			##################### move #####################
			r_box_l, r_box_t, r_box_r, r_box_b = pil_img.getbbox()
			r_box_w = r_box_r - r_box_l
			r_box_h = r_box_b - r_box_t

			paste_x = np.random.randint(low=int(-0.2*r_box_w-r_box_l), high=int(self.w-0.8*r_box_w-r_box_l))
			paste_y = np.random.randint(low=int(-0.2*r_box_h-r_box_t), high=int(self.h-0.8*r_box_h-r_box_t))
			box_x = (r_box_r + r_box_l)/2 + paste_x #+ (box_w*math.cos(r) + abs(box_w*math.sin(r)) - box_w)/2
			box_y = (r_box_b + r_box_t)/2 + paste_y #+ (abs(box_w*math.sin(r)) + box_w*math.cos(r) - box_h)/2

			tmp = PILIMG.new('RGBA', (self.w, self.h))
			tmp.paste(pil_img, (paste_x, paste_y))

			m = nd.array(tmp.split()[-1], ctx=ctx).reshape(1, self.h, self.w)
			mask[i] = nd.tile(m, (3,1,1))

			fg = PILIMG.merge("RGB", (tmp.split()[:3]))
			fg = nd.array(fg)
			for aug in self.augs:
				fg = aug(fg)
			img_batch[i] = fg.as_in_context(ctx).transpose((2,0,1))

			img_cls = int(img_path.split('_no')[0].split('abel')[1])
			#label_batch[i] = nd.array([[img_cls, box_l/self.w, box_t/self.h, box_r/self.w, box_b/self.h]])
			label_batch[i] = nd.array([[img_cls, 
				float(box_y)/self.h, 
				float(box_x)/self.w, 
				float(box_h)/self.h, 
				float(box_w)/self.w,
				r]])

		img_batch = (bg * (1-mask) + img_batch * mask)/255.
		#img_batch = nd.where(mask<200, bg, img_batch)/255.
		img_batch = nd.clip(img_batch, 0, 1) # 0~1 (batch_size, channels, h, w)
		return img_batch, label_batch
	def render_pascal(self, bg, mode):
		'''
		input:
			bg: background ndarray, bs*channel*h*w
			mode: string, train or valid
		output:
			img_batch: bg add car
			label_batch: bs*object*[cls, y(0~1), x(0~1), h(0~1), w(0~1), r(+-pi)]
		'''
		if mode == 'train':
			dataset = self.pascal_train
		elif mode == 'valid':
			dataset = self.pascal_valid
		else: 
			print('arg2 should be train or valid')
			return 0
		ctx = self.ctx
		label_batch = nd.ones((self.bs,1,6), ctx=ctx) * (-1)
		img_batch = nd.zeros((self.bs,3,self.h,self.w), ctx=ctx)
		mask = nd.zeros((self.bs,3,self.h,self.w), ctx=ctx)
		selected = np.random.randint(len(dataset), size=self.bs)

		for i in range(self.bs):
			if np.random.rand()>0.8 and mode == 'train': continue

			img_path = dataset[selected[i]]
			pil_img = PILIMG.open(img_path).convert('RGBA')
			#################### resize ####################
			
			r1 = max([float(pil_img.size[0])/self.w, float(pil_img.size[1])/self.h])

			r2 = 0.2 + np.random.rand()*0.7
			r3 = r2*(0.9 + np.random.rand()*0.2)

			new_w = pil_img.size[0]*r2/r1
			new_h = pil_img.size[1]*r3/r1
			pil_img = pil_img.resize((int(new_w), int(new_h)), self.BIL)

			pil_img, r = img_enhance(pil_img, M=.0, N=.0,R=30.0, G=0)

			img_cls, box_l, box_t, box_r, box_b, skip = self.get_pascal3d_label(img_path)
			if skip: continue

			box_h = (box_b - box_t)*r3/r1
			box_w = (box_r - box_l)*r2/r1

			##################### move #####################
			paste_x = np.random.randint(low=int(-0.1*new_w), high=int(self.w-0.9*new_w))
			paste_y = np.random.randint(low=int(-0.1*new_h), high=int(self.h-0.9*new_h))

			box_x = (box_r + box_l)*r2/r1/2 + paste_x + (new_w*math.cos(r) + abs(new_h*math.sin(r)) - new_w)/2
			box_y = (box_b + box_t)*r3/r1/2 + paste_y + (abs(new_w*math.sin(r)) + new_h*math.cos(r) - new_h)/2

			tmp = PILIMG.new('RGBA', (self.w, self.h))
			tmp.paste(pil_img, (paste_x, paste_y))

			m = nd.array(tmp.split()[-1], ctx=ctx).reshape(1, self.h, self.w)
			mask[i] = nd.tile(m, (3,1,1))

			fg = PILIMG.merge("RGB", (tmp.split()[:3]))
			fg = nd.array(fg)
			for aug in self.augs:
				fg = aug(fg)
			img_batch[i] = fg.as_in_context(ctx).transpose((2,0,1))

			#label_batch[i] = nd.array([[img_cls, box_l/self.w, box_t/self.h, box_r/self.w, box_b/self.h]])
			label_batch[i] = nd.array([[img_cls, 
				float(box_y)/self.h,
				float(box_x)/self.w,
				float(box_h)/self.h,
				float(box_w)/self.w,
				r]])

		img_batch = (bg * (1-mask) + img_batch * mask)/255.
		#img_batch = nd.where(mask<200, bg, img_batch)/255.
		img_batch = nd.clip(img_batch, 0, 1) # 0~1 (batch_size, channels, h, w)
		return img_batch, label_batch
	def get_pascal3d_label(self, img_path):
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
									img_cls = int((q[0][0]+15/2)/15)
									if img_cls>23:img_cls=23
									#print('\t{}'.format(img_cls))
									label[1].append(img_cls)
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
		t=time.time()
		background_iter = image.ImageIter(self.bs, (3, self.h, self.w),
			path_imgrec='/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/sun2012_val.rec',
			path_imgidx='/media/nolan/9fc64877-3935-46df-9ad0-c601733f5888/HP_31/sun2012_val.idx',
			shuffle=True, pca_noise=0, 
			brightness=0.5,	saturation=0.5, contrast=0.5, hue=1.0,
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
	#remove_car_from_sun2012()
	#A = AddLP(160, 384, 0)
	#A.test_render(6)
	B = RenderCar(1, 320, 512, gpu(0))
	B.test()
