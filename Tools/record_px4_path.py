import time
from numpy import inf

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

import matplotlib.pyplot as plt

recorder_x = 0
recorder_y = 0
LP_state = 0
ocr_result = ''

def pose_cb(pose_stamp):
    global recorder_x, recorder_y, LP_state
    
    x = pose_stamp.pose.position.x
    y = pose_stamp.pose.position.y
    z = pose_stamp.pose.position.z
    pose_time = time.time()
    if abs(ocr_time - pose_time) > 0.1:
        LP_state = 0
    else:
        if ocr_result != 'AYM0231':
            LP_state = 1
        else:
            LP_state = 2
    print(LP_state)
    recorder_x = x
    recorder_y = y

def ocr_cb(data):
    global ocr_time, ocr_result
    ocr_time = time.time()
    print(ocr_time)
    ocr_result = data.data

rospy.init_node("record_px4_path", anonymous=True)
rospy.Subscriber('/mavros/local_position/pose', PoseStamped, pose_cb)
rospy.Subscriber('/YOLO/OCR', String, ocr_cb)
ocr_time = inf

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.grid()
while not rospy.is_shutdown():
    if LP_state == 0:
        ax.plot(recorder_x, recorder_y, 'r.', markersize=3)
    elif LP_state == 1:
        ax.plot(recorder_x, recorder_y, 'y.', markersize=3)
    elif LP_state == 2:
        ax.plot(recorder_x, recorder_y, 'b.', markersize=3)
    ax.set_xlim(-2, 10)
    ax.set_ylim(-10, 2)
    plt.pause(0.5)
