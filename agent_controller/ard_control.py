#!/usr/bin/env python
import tf
import rospy
import Tkinter as tk
import threading

from std_msgs.msg import Empty
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped

import sys, select, termios, tty

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
#plt.use('TkAgg')


class PID():
    def __init__(self, P, I=0, D=0):
        self.P = P
        self.I = I
        self.D = D


class GUI():
    def __init__(self, m, n):
        self.w = tk.Tk()
        self.w.title('AR drone Controller')
        self.w.geometry('610x600')

        self.arm_var = tk.BooleanVar()
        c1 = tk.Checkbutton(self.w, text='arming', variable=self.arm_var,
                            width=7, height=2, font=("Arial", 20))

        self.mode_var = tk.StringVar()
        self.mode_var.set('waiting...')

        r1 = self._add_Radiobutton(6, 2, self.mode_var, 'TakeOff')
        r2 = self._add_Radiobutton(6, 2, self.mode_var, 'Land')
        r3 = self._add_Radiobutton(6, 2, self.mode_var, 'KB')
        r4 = self._add_Radiobutton(6, 2, self.mode_var, 'IBVS')
        r0 = self._add_Radiobutton(6, 2, self.mode_var, 'Reset')

        self.key_var = tk.StringVar()
        self.key_var.set('KB waiting...')

        r5 = self._add_Radiobutton(4, 1, self.key_var, 'x+')
        r6 = self._add_Radiobutton(4, 1, self.key_var, 'x-')
        r7 = self._add_Radiobutton(4, 2, self.key_var, 'y+')
        r8 = self._add_Radiobutton(4, 2, self.key_var, 'y-')
        r9 = self._add_Radiobutton(4, 1, self.key_var, 'up')
        r10 = self._add_Radiobutton(4, 1, self.key_var, 'down')
        r11 = self._add_Radiobutton(4, 2, self.key_var, 'stop')

        r12 = self._add_Radiobutton(4, 1, self.key_var, 'left')
        r13 = self._add_Radiobutton(4, 1, self.key_var, 'right')

        b1 = tk.Button(self.w, width=10, height=1, font=("Arial", 20),
                       text='ResetPlot', command=self.clean_plot)
        b2 = tk.Button(self.w, width=10, height=1, font=("Arial", 20),
                       text='Quit', command=quit)

        c1.place(x=0, y=0)

        r1.place(x=0, y=50)
        r2.place(x=100, y=50)
        r3.place(x=200, y=50)
        r4.place(x=300, y=50)
        r0.place(x=400, y=50)

        r5.place(x=100, y=120)
        r6.place(x=100, y=230)
        r7.place(x=0, y=160)
        r8.place(x=200, y=160)
        r9.place(x=320, y=160)
        r10.place(x=320, y=198)
        r11.place(x=100, y=160)

        r12.place(x=0, y=120)
        r13.place(x=200, y=120)

        b1.place(x=200, y=0)
        b2.place(x=400, y=0)

        self.m = m
        self.n = n
        self.x = []
        self.ax = []
        fig = plt.figure()

        for i in range(self.m):
            self.x.append([])
            self.ax.append(fig.add_subplot(111+m*100+i))
            for j in range(self.n):
                self.x[i].append([])
        self.canvas = FigureCanvasTkAgg(fig, master=self.w)
        self.canvas.get_tk_widget().place(x=-20, y=270)

    def _add_Radiobutton(self, w, h, var, text):
        r = tk.Radiobutton(self.w, text=text, variable=var, value=text,
                           width=w, height=h, font=("Arial", 20))
        return r

    def clean_plot(self):
        lock.acquire()
        self.x = []
        for i in range(self.m):
            self.x.append([])
            self.ax[i].clear()
            for j in range(self.n):
                self.x[i].append([])
        self.plot()
        lock.release()

    def plot(self):
        for i in range(self.m):
            self.ax[i].clear()
            for j in range(self.n):
                self.ax[i].plot(self.x[i][j])
        self.canvas.draw()

    def quit():
        global stop_main
        stop_main = True
        self.w.quit()
        self.w.destroy()


def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def imu_cb(imu):
    global way
    ori = imu.orientation
    quaternion = (ori.x, ori.y, ori.z, ori.w)
    roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)


'''
def ibvs_cb(box):
    global vx, vy, vz, vw, box_time, last_err_x, last_err_y, last_err_z, last_err_w

    err_x = desire_x - (box.data[4] - box.data[2])
    err_y = desire_y - (box.data[2] + box.data[4]) / 2
    err_z = desire_z - (box.data[3] + box.data[5]) / 2
    err_w = desire_yaw - yaw

    vx_p = err_x * x_gain.P
    vy_p = err_y * y_gain.P
    vz_p = err_z * z_gain.P
    vw_p = err_w * w_gain.P
    #print('%.4f\t%.4f\t%.4f\t%.4f'%(vx_p, vy_p, vz_p, vw_p))
    gui.x[0][0].append(vx_p)
    gui.x[1][0].append(vy_p)
    #gui.x[2][0].append(vz_p)
    #gui.x[3][0].append(vw_p)


    vx_d = (err_x-last_err_x) * x_gain.D
    vy_d = (err_y-last_err_y) * y_gain.D
    vz_d = (err_z-last_err_z) * z_gain.D
    vw_d = (err_w-last_err_w) * w_gain.D
    #print('%.4f\t%.4f\t%.4f\t%.4f'%(vx_d, vy_d, vz_d, vw_d))
    gui.x[0][1].append(vx_d)
    gui.x[1][1].append(vy_d)
    #gui.x[2][1].append(vz_d)
    #gui.x[3][1].append(vw_d)

    vx = vx_p + vx_d
    vy = vy_p + vy_d
    vz = vz_p + vz_d
    vw = vw_p + vw_d

    last_err_x = err_x
    last_err_y = err_y
    last_err_z = err_z
    last_err_w = err_w
    box_time = rospy.get_rostime()
'''


def ibvs_cb(topic):
    global vx, vy, vz, vw
    vx = topic.twist.linear.x
    vy = topic.twist.linear.y
    vz = topic.twist.linear.z
    vw = topic.twist.angular.z


def init_ros():
    global take_off_pub, land_pub, reset_pub, vel_pub, box_time
    rospy.init_node('ar_drone_control')
    take_off_pub = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=1)
    land_pub = rospy.Publisher('/ardrone/land', Empty, queue_size=1)
    reset_pub = rospy.Publisher('/ardrone/reset', Empty, queue_size=1)
    vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    #rospy.Subscriber('/YOLO/box', Float32MultiArray, ibvs_cb)
    rospy.Subscriber('/drone1/mavros/setpoint_velocity/cmd_vel', TwistStamped, ibvs_cb)
    rospy.Subscriber('/ardrone/imu', Imu, imu_cb)
    box_time = rospy.get_rostime()


def main_thread():
    r = rospy.Rate(20)
    while not rospy.is_shutdown():
        mode = gui.mode_var.get()
        arm = gui.arm_var.get()
        print(mode, arm)
        if stop_main:
            land_pub.publish(Empty())
            print('END Thread')
            break
        if mode == 'TakeOff' and arm:
            take_off_pub.publish(Empty())

        elif mode == 'Land':
            land_pub.publish(Empty())
        elif mode == 'Reset':
            reset_pub.publish(Empty())

        elif mode == 'KB':
            key = gui.key_var.get()
            tw = Twist()
            if key == 'x+':
                tw.linear.x = 0.1
            elif key == 'x-':
                tw.linear.x = -0.1
            elif key == 'y+':
                tw.linear.y = 0.1
            elif key == 'y-':
                tw.linear.y = -0.1
            elif key == 'up':
                tw.linear.z = 0.1
            elif key == 'down':
                tw.linear.z = -0.1
            elif key == 'left':
                tw.angular.z = 0.1
            elif key == 'right':
                tw.angular.z = -0.1
            print('x: %.3f\ty: %.3f\tz: %.3f\tw: %.3f' % (
                tw.linear.x, tw.linear.y, tw.linear.z, tw.angular.z))
            gui.plot()

            if arm:
                print(3)
                vel_pub.publish(tw)

        elif mode == 'IBVS':
            tw = Twist()
            '''
            if (rospy.get_rostime() - box_time) > rospy.Duration(0.5):
                tw.linear.x = 0.0; tw.linear.y = 0.0;
                tw.linear.z = 0.0; tw.angular.z = 0.0
                print('loss target ', (rospy.get_rostime() - box_time))
            else:
            '''
            if True:
                tw.linear.x = vx
                tw.linear.y = vy
                tw.linear.z = vz
                tw.angular.z = vw
                print('%.4f\t%.4f\t%.4f\t%.4f' % (vx, vy, vz, vw))
                gui.plot()
            if arm:
                print('ibvs pub')
                vel_pub.publish(tw)

        else:
            pass
        r.sleep()


desire_yaw = 0
desire_x = 0.4
desire_y = 0.5
desire_z = 0.5

x_gain = PID(0.5, D=0.1)
y_gain = PID(0.5, D=0.1)
z_gain = PID(0.5, D=0.1)
w_gain = PID(0.0)

lock = threading.Lock()

vx, vy, vz, vw, yaw = 0, 0, 0, 0, 0
last_err_x, last_err_y, last_err_z, last_err_w = 0, 0, 0, 0

stop_main = False


gui = GUI(2, 2)

init_ros()
threading.Thread(target=main_thread).start()

gui.w.mainloop()
stop_main = True
land_pub.publish(Empty())
