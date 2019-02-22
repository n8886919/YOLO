import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def calibration():
    rospy.init_node('image_converter', anonymous=True)
    image_pub = rospy.Publisher("/img", Image)
    bridge = CvBridge()
    #cap = open_cam_onboard(480, 270)
    cap = cv2.VideoCapture(0)
    while not rospy.is_shutdown():
        _, img = cap.read()
        image_pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))
