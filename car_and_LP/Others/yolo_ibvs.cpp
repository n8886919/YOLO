#include <math.h>
#include <tf/tf.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float32MultiArray.h>
#define pi 3.1415926

float KPx = 0.0005;
float KPy = 0.00035;	
float KPz = 0.0008;
float KProll = 0.5;

float focal_x = 364.891632, focal_y = 366.672943; //mm
float center_x = 239.656725, center_y = 134.927721;	//principle point
float image_w = 480, image_h = 270;
float model_width = 440, model_height = 130; //sim:4600*1600
float desired_distance = 3000;
float camera_offset = 0;							
float desired_heading = 0;

float err_ux, err_uy, err_uz, err_uroll;

ros::Publisher vel_pub;
double roll, pitch, yaw;

using namespace std;

void box_cb(const std_msgs::Float32MultiArray::ConstPtr& box) {
	float xt, yt;
	float erru, errv, errz, err_roll;
	float local_ux, local_uy, uz, uroll, global_ux, global_uy;
	float ux_trans, uy_trans;
	float box_x_center, box_y_center;
	float model_true_area, model_image_area;
	float depth_Z;
	float fov_x;
	float box_w, box_h;

	box_w = abs(box->data[4]*image_w - box->data[2]*image_w);
	box_h = abs(box->data[5]*image_h - box->data[3]*image_h);

	model_true_area = model_width * model_height;
	model_image_area = box_w * box_h;
	box_x_center = (box->data[2] * image_w + box->data[4]*image_w)/2;
	box_y_center = (box->data[3] * image_h + box->data[5]*image_h)/2;
	fov_x = 2 * atan(image_w/(2*focal_x));

	xt = (box_x_center - center_x)/focal_x;
	yt = (box_y_center - center_y)/focal_y;
	depth_Z = sqrt((model_true_area/model_image_area)*focal_x*focal_y);
	//ROS_INFO("x_center:%.2f y_center: %.2f depth_Z:%.2f",box_x_center,box_y_center,depth_Z);

	erru = xt;//erru = xt - xt* , since x*=0(center of image) --> erru = xt
	errv = yt;
	///////////////////// ERR /////////////////////////
	err_ux = (depth_Z-desired_distance);                   						
	err_uy = (yaw-desired_heading)*desired_distance*image_w/(fov_x*focal_x);	
	//control vy via heading of UAV
	err_uz = -depth_Z * errv;
	err_uroll = -(1/(xt*xt+1)) * erru;
	///////////////////// ERR /////////////////////////
	local_ux = KPx * err_ux;
	local_uy = KPy * err_uy;
	uz = KPz * err_uz;
	uroll = KProll * err_uroll;
	
	global_ux = local_ux * cos(yaw) - local_uy * sin(yaw);
	global_uy = local_ux * sin(yaw) + local_uy * cos(yaw);

	geometry_msgs::Twist vs;
	vs.linear.x = global_ux;
	vs.linear.y = global_uy;
	vs.linear.z = uz;
    vs.angular.x = 0;
    vs.angular.y = 0;
	vs.angular.z = uroll;

	vel_pub.publish(vs);
	ROS_INFO("Local Velocity: %.3f\t%.3f\t%.3f\t%.3f\t, Yaw: %.3f", 
		local_ux, local_uy, uz, uroll, yaw);
	ROS_INFO("Global Velocity: %.3f\t%.3f\t%.3f\t%.3f", global_ux, global_ux, uz, uroll);
}

void imu_cb(const sensor_msgs::Imu::ConstPtr& msg) {
	tf::Quaternion q(msg->orientation.x, msg->orientation.y, 
		msg->orientation.z, msg->orientation.w);
    tf::Matrix3x3 m(q);
    m.getRPY(roll, pitch, yaw);	
	yaw -= pi/2;
	if (yaw < -pi) yaw += 2*pi;
}

int main(int argc, char **argv) {
	ros::init(argc, argv, "yolo_ibvs");
	ros::NodeHandle nh;
	ros::Subscriber YOLO_box_sub = nh.subscribe<std_msgs::Float32MultiArray>("/YOLO/box", 1, box_cb);
	ros::Subscriber DJI_imu_sub = nh.subscribe<sensor_msgs::Imu>("/dji_sdk/imu", 1, imu_cb);

	vel_pub = nh.advertise<geometry_msgs::Twist>("/IBVS/local_vel_cmd", 1);
	ros::spin();
    return 0;
}

