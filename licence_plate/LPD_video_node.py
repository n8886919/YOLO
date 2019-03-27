import cv2
import os
import sys
import threading
import time

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from yolo_modules.licence_plate_render import ProjectRectangle6D

from yolo_modules.yolo_cv import cv2_flip_and_clip_frame
from yolo_modules import yolo_gluon
from LP_detection import LicencePlateDetectioin, Parser

os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

video_verbose = False
video_threshold = 0.5
#video_record_origin = False
#video_record_processed = False


def main():
    global args, LPD
    args = Parser()
    args.mode = 'video'
    LPD = LicencePlateDetectioin(args)
    video(args)


def video(args):
    global net_img_g, net_out_g, net_thread_start
    h, w = LPD.size

    if args.trt:  # tensorRT Inference can't in thread
        import numpy as np
        from yolo_modules.tensorrt_module import do_inference_wrapper
        from yolo_modules.tensorrt_module import get_engine_wrapper
        engine_wrapper = get_engine_wrapper(
            args.version + '/export/onnx/out.onnx',
            args.version + '/export/onnx/out.trt')

    else:
        import mxnet
        # mx_resize = mxnet.image.ForceResizeAug((w, h), interp=2)  # not always available
        net = yolo_gluon.init_executor(LPD.export_file, (h, w), LPD.ctx[0])
        yolo_gluon.test_inference_rate(net, (1, 3, h, w), cycles=100, ctx=LPD.ctx[0])

    rospy.init_node("LP_Detection_Video_Node", anonymous=True)
    threading.Thread(target=_video_thread).start()
    net_thread_start = True
    while not rospy.is_shutdown():
        if 'img_g' not in globals() or img_g is None:
            print('Wait For Image')
            time.sleep(1.0)
            continue

        if not net_thread_start:
            time.sleep(0.01)
            continue

        net_start_time = time.time()  # tic
        net_img = img_g.copy()  # (480, 640, 3), np.float32
        net_img = cv2.resize(net_img, (w, h))  # (320, 512, 3)

        if args.trt:
            # if cuMemcpyHtoDAsync failed: invalid argument, check input image size
            trt_outputs = do_inference_wrapper(engine_wrapper, net_img)
            net_out = np.array(trt_outputs).reshape((1, 10, 10, 16))  # list to np.array

        else:
            nd_img = yolo_gluon.cv_img_2_ndarray(net_img, LPD.ctx[0])#, mxnet_resize=mx_resize)
            net_out = net.forward(is_train=False, data=nd_img)[0]
            net_out[-1].wait_to_read()

        yolo_gluon.switch_print(time.time()-net_start_time, video_verbose)
        net_img_g = net_img
        net_out_g = net_out
        net_thread_start = False


def _video_thread(record=False):
    global bridge, net_thread_start  # for _image_callback()

    pjct_6d = ProjectRectangle6D(int(380*1.05), int(160*1.05))
    ps_pub = rospy.Publisher(LPD.pub_LP, Float32MultiArray, queue_size=0)
    LP_pub = rospy.Publisher(LPD.pub_clipped_LP, Image, queue_size=0)

    pose_msg = Float32MultiArray()
    bridge = CvBridge()
    rate = rospy.Rate(30)

    if args.dev == 'ros':
        rospy.Subscriber(args.topic, Image, _image_callback)
        print('Image Topic: %s' % args.topic)

    else:
        threading.Thread(target=_get_frame).start()

    # -------------------- video record -------------------- #
    '''
    if record:
        start_time = datetime.datetime.now().strftime("%m-%dx%H-%M")
        out_file = os.path.join('video', 'LPD_ % s.avi' % start_time)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        v_size = (640, 480)
        video_out = cv2.VideoWriter(out_file, fourcc, 30, v_size)
    '''
    while not rospy.is_shutdown():
        if 'net_out_g' not in globals() or 'net_img_g' not in globals():
            time.sleep(1)
            print('Wait For Net')
            continue

        img = net_img_g.copy()
        net_out = net_out_g.copy()
        net_thread_start = True

        img = cv2_flip_and_clip_frame(img, (args.clip_h, args.clip_w), args.flip)
        pred = LPD.predict_LP(net_out)
        ps_pub.publish(pose_msg)

        if pred[0] > video_threshold:
            img, clipped_LP = pjct_6d.add_edges(img, pred[1:])
            clipped_LP = bridge.cv2_to_imgmsg(clipped_LP, 'bgr8')
            LP_pub.publish(clipped_LP)

        if args.show:
            cv2.imshow('img', img)
            cv2.waitKey(1)
        #video_out.write(ori_img)
        #rate.sleep()


def _image_callback(img):
    global img_g
    img_g = bridge.imgmsg_to_cv2(img, "bgr8")


def _get_frame():
    global img_g
    from yolo_modules import global_variable

    print(global_variable.green)
    print('Start OPENCV Video Capture Thread')
    dev = args.dev

    if dev == 'jetson':
        print('Image Source: Jetson OnBoard Camera')
        cap = jetson_onboard_camera(640, 360, dev)

    elif dev.split('.')[-1] in ['mp4', 'avi', 'm2ts']:
        print('Image Source: ' + dev)
        cap = cv2.VideoCapture(dev)
        rate = rospy.Rate(30)

    elif dev.isdigit() and os.path.exists('/dev/video' + dev):
        print('Image Source: /dev/video' + dev)
        cap = cv2.VideoCapture(int(dev))

    else:
        print(global_variable.red)
        print('dev should be jetson / video_path(mp4, avi, m2ts) / device_index')
        rospy.signal_shutdown('')
        sys.exit(0)

    print(global_variable.reset_color)
    while not rospy.is_shutdown():
        ret, img = cap.read()
        if img is None:
            continue
        img_g = img

        if 'rate' in locals():
            rate.sleep()

    cap.release()


if __name__ == '__main__':
    main()
