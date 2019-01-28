from __future__ import print_function
import threading
import time
import Tkinter as tk
import math
import numpy as np
import yaml

import rospy
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import Bool
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

from yolo_modules import global_variable

AXIS = 'xyzw'
PID = 'p'  # 'pid'

with open('ibvs_parameter.yaml') as f:
    IBVS_PARAMETER = yaml.load(f)


class PID_GUI(object):
    def __init__(self):
        self.ibvs_controller = IBVS_Controller()

        self.win = tk.Tk()
        self.win.title('PID Controller')
        self.win.geometry('310x550')

        button_place = IBVS_PARAMETER['button_place']

        self.entry_dict = {}
        for k in self.ibvs_controller.gain_keys:
            x, y = button_place[k]
            lable = tk.Label(self.win, text=k, font=('Arial', 12), width=15, height=1)
            lable.place(x=x, y=y, width=100)
            e = tk.Entry(self.win)
            e.insert('end', str(self.ibvs_controller.gain_default[k]))

            self.entry_dict[k] = e
            self.entry_dict[k].place(x=x, y=y+30, width=100)

        tk.Scale(
            self.win, label='Desire Azimuth', command=self._set_azimuth,
            from_=0, to=180, orient=tk.HORIZONTAL,
            length=300, tickinterval=20, resolution=1, sliderlength=20
            ).place(x=0, y=250)

        tk.Scale(
            self.win, label='Desire Distance', command=self._set_distance,
            from_=1, to=2, orient=tk.HORIZONTAL,
            length=300, tickinterval=0.25, resolution=0.01, sliderlength=20
            ).place(x=0, y=350)

        apply_gain_botton = tk.Button(self.win, text="apply", width=15, height=2, command=self._apply)

        self.fix_pose = tk.BooleanVar()
        self.fix_pose.set(True)

        start_IBVS_botton = tk.Radiobutton(
            self.win, text='IBVS',
            variable=self.fix_pose, value=False, command=self._fix_pose)

        fix_pose_botton = tk.Radiobutton(
            self.win, text='Fix Pose',
            variable=self.fix_pose, value=True, command=self._fix_pose)

        self.land = tk.BooleanVar()
        self.land.set(False)

        landing_botton = tk.Checkbutton(
            self.win, text="land", variable=self.land,
            onvalue=True, offvalue=False, command=self._land)

        start_IBVS_botton.place(x=200, y=430, width=100, height=50)
        fix_pose_botton.place(x=100, y=430, width=100, height=50)
        landing_botton.place(x=0, y=430, width=100, height=50)
        apply_gain_botton.place(x=100, y=500, width=100)

        self._apply()

    def _fix_pose(self):
        b = Bool()
        b.data = self.fix_pose.get()
        self.ibvs_controller.ibvs_fix_pose_pub.publish(b)

    def _land(self):
        b = Bool()
        b.data = self.land.get()
        self.ibvs_controller.ibvs_land_pub.publish(b)

    def _set_azimuth(self, v):
        self.ibvs_controller.desire_azimuth = float(v)

        azi_msgs = Float32()
        azi_msgs.data = self.ibvs_controller.desire_azimuth
        self.ibvs_controller.ibvs_desire_azi_pub.publish(azi_msgs)

    def _set_distance(self, v):
        self.ibvs_controller.desire_distance = float(v)

        dis_msgs = Float32()
        dis_msgs.data = self.ibvs_controller.desire_distance
        self.ibvs_controller.ibvs_desire_dis_pub.publish(dis_msgs)

    def _apply(self):
        print(global_variable.green)
        print('NWE PID GAIN APPLYED !!!')
        print(global_variable.cyan + '========================================')

        self.ibvs_controller.err_log_reset()
        for i, k in enumerate(self.ibvs_controller.gain_keys):
            self.ibvs_controller.gain[k] = float(self.entry_dict[k].get())

            print('%s: %.2f' % (k, self.ibvs_controller.gain[k]), end='\t')
            if i % 3 == 2:
                print()


class IBVS_Controller():
    def __init__(self):
        self.LOSS_TARGET_MAX = 30
        self.car_threshold = 0.2
        self.gain_default = IBVS_PARAMETER['gain_default']

        self.gain_keys = []
        self.gain = {}
        self.err_log = {}
        self.err_pid = {}

        for ax in AXIS:
            self.err_log[ax] = []
            for pid in PID:
                self.err_pid[ax+pid] = 0
                self.gain_keys.append(ax+pid)
                self.gain[ax+pid] = self.gain_default[ax+pid]

        self.desire_azimuth = 0
        self.desire_distance = 0
        self.loss_target_counter = 0

        self._init_ros()

    def _init_ros(self):
        rospy.init_node("IBVS_controller_node", anonymous=True)

        self.t0 = rospy.get_rostime()
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self._pose_callback)
        self.heading = 0
        while not hasattr(self, 'heading'):
            time.sleep(1)
        #rospy.sleep(1)  # wait for local pose
        rospy.Subscriber('/YOLO/box', Float32MultiArray, self._vel_callback)

        self.bridge = CvBridge()

        self.ibvs_desire_azi_pub = rospy.Publisher(
            IBVS_PARAMETER['DESIRE_AZI_TOPIC'],
            Float32, queue_size=1)

        self.ibvs_desire_dis_pub = rospy.Publisher(
            IBVS_PARAMETER['DESIRE_DIS_TOPIC'],
            Float32, queue_size=1)

        self.ibvs_vel_pub = rospy.Publisher(
            IBVS_PARAMETER['CMD_VEL_TOPIC'],
            TwistStamped, queue_size=1)

        self.ibvs_pid_pub = rospy.Publisher(
            IBVS_PARAMETER['PID_TOPIC'],
            Float32MultiArray, queue_size=1)

        self.ibvs_land_pub = rospy.Publisher(
            IBVS_PARAMETER['LAND_TOPIC'],
            Bool, queue_size=1)

        self.ibvs_fix_pose_pub = rospy.Publisher(
            IBVS_PARAMETER['FIX_POSE_TOPIC'],
            Bool, queue_size=1)

        self.pid_output = Float32MultiArray()
        self.pid_output.layout.dim.append(MultiArrayDimension())
        self.pid_output.layout.dim.append(MultiArrayDimension())
        self.pid_output.layout.dim[0].label = "axis"
        self.pid_output.layout.dim[1].label = "pid"
        self.pid_output.data = [0] * len(AXIS) * len(PID)

    def _vel_callback(self, box):
        if not hasattr(self, 'heading'):
            print('Wait For Heading msgs')
            return

        self.t1 = self.t0
        self.t0 = rospy.get_rostime()

        dt = (self.t0 - self.t1).to_sec()
        self._update_error(box.data, dt)

        vel_msgs = TwistStamped()
        vel_msgs.header.stamp = rospy.get_rostime()

        if self.loss_target_counter > self.LOSS_TARGET_MAX:
            vel_msgs.twist.linear.x = 0
            vel_msgs.twist.linear.y = 0
            vel_msgs.twist.linear.z = 0
            vel_msgs.twist.angular.z = 0.1
            print(global_variable.red)
            print('loss target over %d frames! Hover' % self.LOSS_TARGET_MAX)

        else:
            local_x, local_y, vz, vw = self._get_pid_output()
            local_x = self._vel_bound(local_x, 10, 0.03)
            local_y = self._vel_bound(local_y, 10, 0.03)

            global_x = (local_x*math.cos(self.heading) -
                        local_y*math.sin(self.heading))

            global_y = (local_y*math.cos(self.heading) +
                        local_x*math.sin(self.heading))

            vel_msgs.twist.linear.x = global_x
            vel_msgs.twist.linear.y = global_y
            vel_msgs.twist.linear.z = vz
            vel_msgs.twist.angular.z = vw

            #print(self.loss_target_counter)
            print(global_variable.yellow)
            print('local: %.4f\t%.4f\t%.4f\t%.4f' % (
                local_x,
                local_y,
                vel_msgs.twist.linear.z,
                vel_msgs.twist.angular.z))

            print('global: %.4f\t%.4f\t%.4f\t%.4f' % (
                global_x,
                global_y,
                vel_msgs.twist.linear.z,
                vel_msgs.twist.angular.z))

        self.ibvs_vel_pub.publish(vel_msgs)

    def _pose_callback(self, pose):
        z = pose.pose.orientation.z
        w = pose.pose.orientation.w
        heading = math.atan2(z, w) * 2

        if heading > math.pi:
            self.heading = heading - 2 * math.pi
        elif heading < - math.pi:
            self.heading = heading + 2 * math.pi
        else:
            self.heading = heading

    def _update_error(self, box, dt):
        # print(dt)
        if box[0] > self.car_threshold:
            self.loss_target_counter = 0

            print(box[5])
            if box[5] > 0:
                errx = box[5] - self.desire_distance
            else:
                print('No Depth Infomation')
                errx = 0.18 - box[3] * box[4]

            erry = get_erry(box[-24:], self.desire_azimuth)

            err_now = {
                #'y': (box[6] - math.pi) if box[6] > 0 else (box[6] + math.pi),
                'x': errx,  # desire distance
                'y': erry,
                'z': 0.7 - box[1],  # middle of image
                'w': 0.5 - box[2]  # middle of image
            }

            for ax in AXIS:
                self.err_log[ax].append(err_now[ax])

                self.err_pid[ax+'p'] = err_now[ax]
                self.err_pid[ax+'i'] = sum(self.err_log[ax])

                if len(self.err_log[ax]) > 1:
                    self.err_pid[ax+'d'] = (err_now[ax] - self.err_log[ax][-2])/dt
                else:
                    self.err_pid[ax+'d'] = 0
            #print(self.err_log['x'][-2])
            #print(err_now[ax])

        else:
            #print('Loss')
            self.loss_target_counter += 1
            self.err_log_reset()

    def _vel_bound(self, x, high_bound, low_bound):
        x = np.clip(x, -high_bound, high_bound)
        if x < low_bound and x > -low_bound:
            x = 0
        return x

    def _get_pid_output(self):
        xyzw_out = []
        i = 0
        for ax in AXIS:
            out_sum = 0
            for pid in PID:
                out = self.err_pid[ax+pid] * self.gain[ax+pid]
                self.pid_output.data[i] = out
                out_sum += out
                i += 1

            xyzw_out.append(out_sum)

        self.ibvs_pid_pub.publish(self.pid_output)
        return xyzw_out

    def err_log_reset(self):
        for ax in AXIS:
            self.err_log[ax] = []

    def _load_camera_parameter(self):
        with open(camera_parameter_file) as f:
            camera_parameter = yaml.load(f)
            self.fx = camera_parameter['camera_matrix']['data'][0]
            self.fy = camera_parameter['camera_matrix']['data'][4]
            self.cx = camera_parameter['camera_matrix']['data'][2]
            self.cy = camera_parameter['camera_matrix']['data'][5]


_step = 360 / 24
cos_offset = np.array([math.cos(x*math.pi/180) for x in range(0, 360, _step)])
sin_offset = np.array([math.sin(x*math.pi/180) for x in range(0, 360, _step)])


def get_erry(x, desire_azimuth):
    prob = np.exp(x)/np.sum(np.exp(x), axis=0)
    c = sum(cos_offset*prob)
    s = sum(sin_offset*prob)
    vec_ang = math.atan2(s, c)
    vec_rad = (s**2+c**2)**0.5
    #print(vec_ang)

    erry = vec_ang - desire_azimuth * math.pi / 180

    if erry < -math.pi:
        erry += 2 * math.pi
    elif erry > math.pi:
        erry -= 2 * math.pi

    return erry


if __name__ == '__main__':
    pid_gui = PID_GUI()
    pid_gui.win.mainloop()
