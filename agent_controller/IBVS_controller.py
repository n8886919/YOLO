import threading
import Tkinter as tk
import math
import numpy as np 
import yaml 

import rospy
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import Bool

camera_parameter_file = 'camera_parameter/C270_1.yaml'

AXIS = ['x', 'y', 'z', 'w']
PID = ['p', 'i', 'd']

class PID_GUI():

	def __init__(self):
		


		self._init_tk()
	
	def _init_tk(self):	
		self.win = tk.Tk()
		self.win.title('PID Controller')
		self.win.geometry('300x450')

		place = {'xp': [0,0], 'xi': [100,0], 'xd': [200,0],
				 'yp': [0,60], 'yi': [100,60], 'yd': [200,60],
				 'zp': [0,120], 'zi': [100,120], 'zd': [200,120],
				 'wp': [0,180], 'wi': [100,180], 'wd': [200,180]
				}

		self.entry_dict = {}

		for k in ibvs_controller.gain_keys:
			l = tk.Label(self.win, text=k, font=('Arial', 12), width=15, height=1)
			l.place(x=place[k][0], y=place[k][1], width=100)
			e = tk.Entry(self.win)
			e.insert('end', str(ibvs_controller.gain_default[k]))

			self.entry_dict[k] = e
			self.entry_dict[k].place(x=place[k][0], y=place[k][1]+30, width=100)

		tk.Scale(self.win, label='Desire Azimuth', from_=0, to=180, orient=tk.HORIZONTAL,
			length=300, tickinterval=20, resolution=1, sliderlength=20, command=self._set_azimuth).place(x=0, y=250)

		self.set_pose = tk.BooleanVar()
		self.set_pose.set(True)

		tk.Radiobutton(self.win, text='IBVS', variable=self.set_pose, value=False, 
			command=self._set_pose).place(x=200, y=330, width=100, height=50)
		tk.Radiobutton(self.win, text='Fix Pose', variable=self.set_pose, value=True, 
			command=self._set_pose).place(x=100, y=330, width=100, height=50)

		self.land = tk.BooleanVar()
		self.land.set(False)

		tk.Checkbutton(self.win, text="land", variable=self.land, 
			onvalue=True, offvalue=False, command=self._land).place(
			x=0, y=330, width=100, height=50)

		tk.Button(self.win, text="apply", width=15, height=2, command=self._apply).place(
			x=100, y=400, width=100)

		self._apply()

	def _set_pose(self):
		b = Bool()
		b.data = self.set_pose.get()
		ibvs_controller.ibvs_set_pose_pub.publish(b)

	def _land(self):
		b = Bool()
		b.data = self.land.get()
		ibvs_controller.ibvs_land_pub.publish(b)

	def _set_azimuth(self, v):
		ibvs_controller.desire_azimuth = float(v)

	def _apply(self):
		ibvs_controller.err_log_reset()
		for k in ibvs_controller.gain_keys:
			try:
				ibvs_controller.gain[k] = float(self.entry_dict[k].get())

			except:
				print('apply %s fail'%k)

			print('%s: %.2f'%(k, ibvs_controller.gain[k]))


class IBVS_CONTROLLER():
	
	def __init__(self):
		
		self.gain = {}
		self.gain_keys = []
		for ax in AXIS:
			for pid in PID:
				self.gain_keys.append(ax+pid)
				self.gain[ax+pid] = 0


		#['xp', 'xi', 'xd', 'yp', 'yi', 'yd', 
		# 'zp', 'zi', 'zd', 'wp', 'wi', 'wd']

		
		self.gain_default = {'xp': 10.0, 'xi': 0.0, 'xd': 0.1,
						'yp':  2.0, 'yi': 0.0, 'yd': 1.0, 
						'zp':  2.0, 'zi': 0.0, 'zd': 0.0, 
						'wp':  2.0, 'wi': 0.0, 'wd': 0.0
		}
		'''
		self.gain_default = {'xp': 1.0, 'xi': 0.0, 'xd': 0.0,
						'yp': 0.2, 'yi': 0.0, 'yd': 0.0, 
						'zp': 0.4, 'zi': 0.0, 'zd': 0.0, 
						'wp': 0.8, 'wi': 0.0, 'wd': 0.0
						}
		'''
		self.desire_azimuth = 90
		self.loss_target_counter = 0
		self.err_log = {}
		self.err_pid = {}
		
		for ax in AXIS:
			self.err_log[ax] = []
			for pid in PID:
				self.err_pid[ax+pid] = 0

		self._init_ros()
		#self._load_camera_parameter()

	def _load_camera_parameter(self):
		with open(camera_parameter_file) as f:
			camera_parameter = yaml.load(f)
			self.fx = camera_parameter['camera_matrix']['data'][0]
			self.fy = camera_parameter['camera_matrix']['data'][4]
			self.cx = camera_parameter['camera_matrix']['data'][2]
			self.cy = camera_parameter['camera_matrix']['data'][5]

	def _init_ros(self):
		rospy.init_node("IBVS_controller_node", anonymous = True)

		self.t0 = rospy.get_rostime()
		self.ang = 0
		rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self._pose_callback)
		rospy.sleep(1)
		rospy.Subscriber('/YOLO/box', Float32MultiArray, self._vel_callback)
		
		self.ibvs_vel_pub = rospy.Publisher('/ibvs_gui/cmd_vel', TwistStamped, queue_size=1)
		self.ibvs_pid_pub = rospy.Publisher('/ibvs_gui/pid', Float32MultiArray, queue_size=1)
		self.ibvs_land_pub = rospy.Publisher('/ibvs_gui/land', Bool, queue_size=1)
		self.ibvs_set_pose_pub = rospy.Publisher('/ibvs_gui/set_pose', Bool, queue_size=1)

		self.mat = Float32MultiArray()
		self.mat.layout.dim.append(MultiArrayDimension())
		self.mat.layout.dim.append(MultiArrayDimension())
		self.mat.layout.dim[0].label = "axis"
		self.mat.layout.dim[1].label = "pid"
		self.mat.data = [0]*len(AXIS) * len(PID)
		
	def _vel_callback(self, box):
		self.t1 = self.t0
		self.t0 = rospy.get_rostime()
		self._update_error(box.data)

		t = TwistStamped()
		t.header.stamp = rospy.get_rostime()

		if self.loss_target_counter > 30:
			t.twist.linear.x  = 0
			t.twist.linear.y  = 0
			t.twist.linear.z  = 0
			t.twist.angular.z = 0.1
			print('loss target over 100 frames!')

		else:	
			local_x = self._vel_bound(self._sum_error('x'), -10, 10, -0.03, 0.03)
			local_y = self._vel_bound(self._sum_error('y'), -10, 10, -0.03, 0.03)

			global_x = local_x * math.cos(self.ang) - local_y * math.sin(self.ang)
			global_y = local_y * math.cos(self.ang) + local_x * math.sin(self.ang)

			t.twist.linear.x = global_x
			t.twist.linear.y = global_y

			t.twist.linear.z  = self._sum_error('z')
			t.twist.angular.z = self._sum_error('w')
			#print(self.loss_target_counter)
			print('local: %.4f\t%.4f\t%.4f\t%.4f'%(local_x, local_y, t.twist.linear.z, t.twist.angular.z))
			print('global: %.4f\t%.4f\t%.4f\t%.4f'%(global_x, global_y, t.twist.linear.z, t.twist.angular.z))
		self.ibvs_vel_pub.publish(t)
		

		i = 0
		for ax in AXIS:
			for pid in PID:
				self.mat.data[i] = self.err_pid[ax+pid] * self.gain[ax+pid]
				i += 1
		#print(self.mat.data)
		self.ibvs_pid_pub.publish(self.mat)
		
	def _pose_callback(self, ps):
		ang = math.atan2(ps.pose.orientation.z, ps.pose.orientation.w) * 2
		if ang > math.pi:
			self.ang = ang - 2 * math.pi
		elif ang < - math.pi:
			self.ang = ang + 2 * math.pi
		else:
			self.ang = ang 

	def _update_error(self, box):
		dt = (self.t0 - self.t1).to_sec()
		#print(dt)
		if box[0] > 0.8:
			self.loss_target_counter = 0

			erry = box[6] - self.desire_azimuth*math.pi/180
			if erry < -math.pi: erry += 2 * math.pi
			elif erry > math.pi: erry -= 2 * math.pi
			
			err_now = {
				#'y': (box[6] - math.pi) if box[6] > 0 else (box[6] + math.pi),
				'y': erry,
				'x': 0.18 - box[3] * box[4],
				'z': 0.5 - box[1],
				'w': 0.5 - box[2]
			}
			
			for ax in AXIS: #['x', 'y', 'z', 'w']
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

	def _vel_bound(self, x, L1, H1, L2, H2):
		x = np.clip(x, L1, H1)
		if x > L2 and x < H2:
			x = 0
		return x	
	
	def _sum_error(self, ax):
		out = 0
		for pid in ['p']:#, 'i', 'd']:
			out += self.err_pid[ax+pid] * self.gain[ax+pid]
		return out
	
	def err_log_reset(self):
		for ax in AXIS:
			self.err_log[ax] = []


ibvs_controller = IBVS_CONTROLLER()

pid_gui = PID_GUI()
pid_gui.win.mainloop()
