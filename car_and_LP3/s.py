import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int16MultiArray
from PIL import Image
import copy
import Queue
import time

import numpy as np

q = Queue.Queue(maxsize=10)
all_im = []


def callback(topic):
    global counter, q
    counter += 1
    data = copy.deepcopy(topic.data)
    q.put(data)
    print((time.time()-t)/counter)

counter = 0
t = time.time()
rospy.init_node('blender_sub')
rospy.Subscriber("/blender/image", Int16MultiArray, callback)#, queue_size=10)
#rospy.spin()


while not rospy.is_shutdown():

    if q.qsize() == 0:
        continue

    data = q.get()
    data = np.array(data).reshape((480, 640, 4))[::-1]
    data[:, :, :3] = np.power(data[:, :, :3], 1/2.2) * 255.  # Gamma_correction
    data[:, :, 3] = data[:, :, 3] * 255.
    data = np.clip(np.around(data), 0, 255).astype('uint8')
    im = Image.fromarray(data, mode='RGBA')
    all_im.append(im)
    #im.save("tt/%d.png" % np.random.randint(10000000))
print(q.qsize(), time.time()-t)

'''
for i, im in enuermate(queue):
    im.save("tt/%d.png" % i)
'''
