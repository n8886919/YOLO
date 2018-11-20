from __future__ import division

import cv2
import numpy as np
from sympy import *
from sympy import Matrix

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def get_corner_points(w, h, pose_6d):
    X = Symbol('X')
    Y = Symbol('Y')
    Z = Symbol('Z')
    r1 = Symbol('r1')
    r2 = Symbol('r2')
    r3 = Symbol('r3')

    fx = Symbol('fx')  # 932.294922
    fy = Symbol('fy')  # 923.134591
    cx = Symbol('cx')  # 268.146099
    cy = Symbol('cy')  # 256.123571

    sub = {
        X: pose_6d[0], Y: pose_6d[1], Z: pose_6d[2],
        r1: pose_6d[3], r2: pose_6d[4], r3: pose_6d[5],
        fx: 890.037231, fy: 889.150513, cx: 314.129602, cy: 220.037739
    }

    P_3d = Matrix([[w, -w, -w, w],
                   [h, h, -h, -h],
                   [0, 0, 0, 0]])
    T_matrix = Matrix([[X]*4, [Y]*4, [Z]*4])

    R1 = Matrix([[1, 0, 0],
                 [0, cos(r1), -sin(r1)],
                 [0, sin(r1), cos(r1)]])
    R2 = Matrix([[cos(r2), 0, sin(r2)],
                 [0, 1, 0],
                 [-sin(r2), 0, cos(r2)]])
    R3 = Matrix([[cos(r3), -sin(r3), 0],
                 [sin(r3), cos(r3), 0],
                 [0, 0, 1]])
    R_matrix = R3 * R2 * R1 * P_3d

    extrinsic_matrix = R_matrix + T_matrix
    intrinsic_matrix = Matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    projection_matrix = intrinsic_matrix * extrinsic_matrix

    points = np.zeros((4, 1, 2))
    ans = projection_matrix.evalf(subs=sub)
    for i in range(4):
        points[i, 0, 0] = ans[0, i] / ans[2, i]
        points[i, 0, 1] = ans[1, i] / ans[2, i]
        '''
        print(projection_matrix[0, i]/projection_matrix[2, i])
        print(projection_matrix[1, i]/projection_matrix[2, i])
        print('')
        '''
    # solve([x + 5*y - 2, -3*x + 6*y - 15], [x, y])

    return points.astype(np.int32)


def image_callback(img):
    global g_img
    img = bridge.imgmsg_to_cv2(img, "bgr8")
    g_img = img

if __name__ == '__main__':
    rospy.init_node("test", anonymous=True)
    rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
    bridge = CvBridge()

    points = get_points([200, 200, 2000, 0, 0, 0])
    rospy.sleep(1)

    while not rospy.is_shutdown():
        cv2.polylines(g_img, points, 1, (0, 0, 255), thickness=10)
        cv2.imshow('img', g_img)
        cv2.waitKey(1)
