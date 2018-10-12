from mxnet import gluon
from mxnet.gluon import nn
from gluoncv.model_zoo.yolo.darknet import DarknetBasicBlockV3, _conv2d
from gluoncv.model_zoo.yolo.yolo3 import YOLODetectionBlockV3, _upsample

class YOLOOutput(gluon.HybridBlock):
	def __init__(self, channel, num_anchors, **kwargs):
		super(YOLOOutput, self).__init__(**kwargs)
		# channel = num_lass + 5
		self.channel = channel
		self.num_anchors = num_anchors

		self.yolooutput = nn.Conv2D(channel * num_anchors, kernel_size=1)
		
	def hybrid_forward(self, F, x, *args):
		x = self.yolooutput(x)
		x = x.transpose((0, 2, 3, 1))
		x = x.reshape((0, -1, self.num_anchors, self.channel))

		return x
