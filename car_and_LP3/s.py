import rospy
from std_msgs.msg import Float32MultiArray
from PIL import Image

import numpy as np


def callback(topic):
	global counter, data
	counter += 1
	data = np.array(topic.data).reshape((480,640,4))[::-1]
	data[:,:,:3] = np.power(data[:,:,:3], 1/2.2) * 255.  # Gamma_correction
	data[:,:,3] = data[:,:,3] * 255.
	data = np.clip(np.around(data),0,255).astype('uint8')
	im = Image.fromarray(data, mode='RGBA')
	#im.save("tt/%d.png"%counter)

counter = 0
rospy.init_node('blender_sub')
rospy.Subscriber("/blender/image", Float32MultiArray, callback)
rospy.spin()

