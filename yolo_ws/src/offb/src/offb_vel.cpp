/**
 * @file offb_main.cpp
 * @author Julian Oes <julian@oes.ch>
 * @license BSD 3-clause
 *
 * @brief ROS node to do offboard control of PX4 through MAVROS.
 *
 * Initial code taken from http://dev.px4.io/ros-mavros-offboard.html
 */
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int8.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>

#include <cstdio>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

using namespace std;

mavros_msgs::State current_state;
void state_cb(const mavros_msgs::State::ConstPtr& msg) {
    current_state = *msg;
}

geometry_msgs::TwistStamped twist;
void vel_cb(const geometry_msgs::TwistStamped::ConstPtr& msg) {
    twist = *msg;
}

int fly_mode = 1;

void fly_mode_cb(const std_msgs::Int8::ConstPtr& msg) {
    fly_mode = msg->data;
}

bool land = false;
void land_cb(const std_msgs::Bool::ConstPtr& msg) {
    land = msg->data;
}
/*
 * Taken from
 * http://stackoverflow.com/questions/421860/capture-characters-from-standard-input-without-waiting-for-enter-to-be-pressed
 *
 * @return the character pressed.
 */
char getch(){
    int flags = fcntl(0, F_GETFL, 0);
    fcntl(0, F_SETFL, flags | O_NONBLOCK);

    char buf = 0;
    struct termios old = {0};
    if (tcgetattr(0, &old) < 0) {
        perror("tcsetattr()");
    }
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;
    if (tcsetattr(0, TCSANOW, &old) < 0) {
        perror("tcsetattr ICANON");
    }
    if (read(0, &buf, 1) < 0) {
        //perror ("read()");
    }
    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    if (tcsetattr(0, TCSADRAIN, &old) < 0) {
        perror ("tcsetattr ~ICANON");
    }
    return (buf);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "offb_main");
    ros::NodeHandle nh;

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
                                ("/mavros/state", 10, state_cb);
    ros::Subscriber cmd_vel_sub = nh.subscribe<geometry_msgs::TwistStamped>
                                ("/ibvs_gui/cmd_vel", 1, vel_cb);
    ros::Subscriber set_pose_sub = nh.subscribe<std_msgs::Int8>
                                ("/ibvs_gui/fly_mode", 1, fly_mode_cb);
    ros::Subscriber land_sub = nh.subscribe<std_msgs::Bool>
                                ("/ibvs_gui/land", 1, land_cb);

    ros::Publisher local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>
                                   ("/mavros/setpoint_position/local", 10);
    ros::Publisher local_vel_pub = nh.advertise<geometry_msgs::TwistStamped>
                                   ("/mavros/setpoint_velocity/cmd_vel", 10);
    
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
                                       ("/mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
                                         ("/mavros/set_mode");

    // The setpoint publishing rate MUST be faster than 2Hz.
    ros::Rate rate(30.0);

    // Wait for FCU connection.
    while (ros::ok() && current_state.connected) {
        ros::spinOnce();
        rate.sleep();
    }

    geometry_msgs::PoseStamped pose;
    pose.pose.position.x = 0;
    pose.pose.position.y = 0;
    pose.pose.position.z = 0.8;
    pose.pose.orientation.z = 0.7071;
    pose.pose.orientation.w = -0.7071;
    
    geometry_msgs::TwistStamped no_cmd_twist;
    no_cmd_twist.twist.linear.x = 0;
    no_cmd_twist.twist.linear.y = 0;
    no_cmd_twist.twist.linear.z = -0;
    no_cmd_twist.twist.angular.z = 0.0;

    geometry_msgs::TwistStamped down_twist;
    down_twist.twist.linear.x = 0;
    down_twist.twist.linear.y = 0;
    down_twist.twist.linear.z = sin(45);
    down_twist.twist.angular.z = 0.0;

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request(0);

    while (ros::ok()) {
        if (land){
            offb_set_mode.request.custom_mode = "MANUAL";
            set_mode_client.call(offb_set_mode);
            arm_cmd.request.value = false;
            arming_client.call(arm_cmd);

            pose.pose.position.x = 0;
            pose.pose.position.y = 0;
            pose.pose.position.z = 0.8;
            pose.pose.orientation.z = 0.0;
            pose.pose.orientation.w = 1.0;
        }

	    else{
            if (current_state.mode != "OFFBOARD" &&
                    (ros::Time::now() - last_request > ros::Duration(5.0))) {
                if( set_mode_client.call(offb_set_mode) &&
                        offb_set_mode.response.mode_sent) {
                    ROS_INFO("Offboard enabled");
                }
                last_request = ros::Time::now();
            }
            else {
                if (!current_state.armed &&
                        (ros::Time::now() - last_request > ros::Duration(5.0))) {
                    if( arming_client.call(arm_cmd) &&
                            arm_cmd.response.success) {
                        ROS_INFO("Vehicle armed");
                    }
                    last_request = ros::Time::now();
                }
            }
            
            int c = getch();
            //printf("%d",c);
            if (c != EOF) {
                switch (c) {
    	       
                case 119:    // key w
                    pose.pose.position.x += 0.05;
                    break;
                case 115:    // key s
                    pose.pose.position.x -= 0.05;
                    break;
                case 97:    // key a
                    pose.pose.position.y += 0.05;
                    break;
                case 100:    // key d
                    pose.pose.position.y -= 0.05;
                    break;
                
                case 65:    // key up
                    pose.pose.position.z += 0.1;
                    break;
                case 66:    // key down
                    pose.pose.position.z -= 0.1;
                    break;

                case 63:
                    return 0;
                    break;
                }
            }
            
            switch (fly_mode) {

                case 0:
                    ROS_INFO("Down");
                    local_vel_pub.publish(down_twist);
                    break;

                case 1:
                    local_pos_pub.publish(pose);
                    ROS_INFO(
                        "setpoint: %.2f, %.2f, %.2f", pose.pose.position.x, 
                        pose.pose.position.y, pose.pose.position.z);
                    break;

                case 2:
                    ros::Time currtime = ros::Time::now();
                    ros::Duration diff = currtime - twist.header.stamp;            

                    if(diff < ros::Duration(1.0)) {
                        local_vel_pub.publish(twist);
                    }
                    else {
                        ROS_INFO("Loss Command, Hovering");
                        local_vel_pub.publish(no_cmd_twist);
                    }
                    break;
            }
        }
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}
