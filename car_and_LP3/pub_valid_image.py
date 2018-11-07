import sys
import yaml

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError

#sys.path.append('../')
from render_car import *
from modules.utils_gluon import *

ctx = [gpu(0)]
rospy.init_node("valid_image_publisher", anonymous=True)
bridge = CvBridge()
img_pub = rospy.Publisher('/YOLO/valid/img', Image, queue_size=1)
box_pub = rospy.Publisher('/YOLO/valid/box', Float32MultiArray, queue_size=1)
print('\033[7;33mCompare\033[0m')

with open(os.path.join('v1', 'spec.yaml')) as f:
        spec = yaml.load(f)

h, w = spec['size']
classes = spec['classes']

BG_iter = load_background('val', 1, h, w)
car_renderer = RenderCar(h, w, classes, ctx[0])

for bg in BG_iter:
    bg = bg.data[0].as_in_context(ctx[0])  # b*RGB*w*h

    if np.random.rand() > 0.5:
        imgs, labels = car_renderer.render(
            bg, 'valid', pascal=False, render_rate=0.8)
    else:
        imgs, labels = car_renderer.render(
            bg, 'valid', pascal=True, render_rate=0.8)

    imgs = nd.clip(imgs, 0, 1)
    img = batch_ndimg_2_cv2img(imgs)[0]
    img = (img * 255).astype(np.uint8)

    img_pub.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))

    #box_pub.publish(mat)

    raw_input('next')
