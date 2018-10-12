/** @file demo_flight_control.cpp
 *  @version 3.3
 *  @date September, 2017
 *
 *  @brief
 *  demo sample of how to use Local position control
 *
 *  @copyright 2017 DJI. All rights reserved.
 *
 */

#include "dji_sdk_demo/demo_local_position_control.h"
#include "dji_sdk/dji_sdk.h"
#include <termios.h>
#include <fcntl.h>
#include <geometry_msgs/Twist.h>

ros::ServiceClient set_local_pos_reference;
ros::ServiceClient sdk_ctrl_authority_service;
ros::ServiceClient drone_task_service;
ros::ServiceClient query_version_service;

// global variables for subscribed topics
ros::Publisher ctrlPosYawPub;

geometry_msgs::Twist desire_vel;
sensor_msgs::NavSatFix current_gps_position;

uint8_t flight_status = 255;
uint8_t display_mode  = 255;
uint8_t current_gps_health = 0;

bool takeoff_result = false;
bool enable_IBVS = false;

void KeyBoard_control(int c);
void IBVS_vel_callback(const geometry_msgs::Twist::ConstPtr& msg);
char getch();

int main(int argc, char** argv) {
  ros::init(argc, argv, "demo_global_velocity_control_node");
  ros::NodeHandle nh;
  // Subscribe to messages from dji_sdk_node
  ros::Subscriber flightStatusSub = nh.subscribe("dji_sdk/flight_status", 10, &flight_status_callback);
  ros::Subscriber displayModeSub  = nh.subscribe("dji_sdk/display_mode", 10, &display_mode_callback);
  ros::Subscriber gpsSub          = nh.subscribe("dji_sdk/gps_position", 10, &gps_position_callback);
  ros::Subscriber gpsHealth       = nh.subscribe("dji_sdk/gps_health", 10, &gps_health_callback);
  //IBVS sub
  ros::Subscriber IBVS_Sub        = nh.subscribe("/IBVS/local_vel_cmd", 10, &IBVS_vel_callback);

  // Publish the control signal
  ctrlPosYawPub = nh.advertise<sensor_msgs::Joy>("dji_sdk/flight_control_setpoint_ENUvelocity_yawrate", 10);

  // Basic services
  sdk_ctrl_authority_service = nh.serviceClient<dji_sdk::SDKControlAuthority> ("dji_sdk/sdk_control_authority");
  drone_task_service         = nh.serviceClient<dji_sdk::DroneTaskControl>("dji_sdk/drone_task_control");
  query_version_service      = nh.serviceClient<dji_sdk::QueryDroneVersion>("dji_sdk/query_drone_version");
  set_local_pos_reference    = nh.serviceClient<dji_sdk::SetLocalPosRef> ("dji_sdk/set_local_pos_ref");
  
  bool obtain_control_result = obtain_control();
  
  
  if (!set_local_position()) {
    ROS_ERROR("GPS health insufficient - No local frame reference for height. Exiting.");
    return 1;
  }

  if(is_M100()) {
    ROS_INFO("M100 taking off!");
    takeoff_result = M100monitoredTakeoff();
  }
  else {
    ROS_INFO("Not M100!");
    return 0;
  }
  ros::Rate rate(20.0);
  while (ros::ok()) { 
    
    int c = getch();
    if (c != EOF) {
      sensor_msgs::Joy controlPosYaw;
      switch (c) {
        case 49:
          enable_IBVS = false;
          ROS_INFO("%d", c);
          ROS_INFO("Unable IBVS");
          break;
        case 50:
          enable_IBVS = true;
          ROS_INFO("Enable IBVS");
          break;
        default:
          break;
      }
      if (enable_IBVS == false && current_gps_health > 3) KeyBoard_control(c);
    }
    ros::spinOnce();
    rate.sleep();
  } // while
  return 0;
}
void KeyBoard_control(int c) {
  sensor_msgs::Joy controlPosYaw;
  //ROS_INFO("%d", c);
  switch (c) {  
    case 65:    // key up
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(1);
      controlPosYaw.axes.push_back(0);
      ctrlPosYawPub.publish(controlPosYaw);
      break;
    case 66:    // key down
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(-1);
      controlPosYaw.axes.push_back(0);
      ctrlPosYawPub.publish(controlPosYaw);
      break;
    case 119:    // key w
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(1);
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(0);
      ctrlPosYawPub.publish(controlPosYaw);
      break;
    case 115:    // key s
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(-1);
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(0);
      ctrlPosYawPub.publish(controlPosYaw);
      break;
    case 97:    // key a
      controlPosYaw.axes.push_back(-1);
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(0);
      ctrlPosYawPub.publish(controlPosYaw);
      break;
    case 100:    // key d
      controlPosYaw.axes.push_back(1);
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(0);
      ctrlPosYawPub.publish(controlPosYaw);
      break;
    case 113:    // key q
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(1);
      ctrlPosYawPub.publish(controlPosYaw);
      break;
    case 101:    // key e
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(0);
      controlPosYaw.axes.push_back(-1);
      ctrlPosYawPub.publish(controlPosYaw);
      break;
    default: 
      break;
  } // switch (c)
}

void IBVS_vel_callback(const geometry_msgs::Twist::ConstPtr& msg) {
  if (current_gps_health > 3) {
    if (enable_IBVS == true) {      
      desire_vel = *msg;
      sensor_msgs::Joy controlPosYaw;
      controlPosYaw.axes.push_back(desire_vel.linear.y);
      controlPosYaw.axes.push_back(desire_vel.linear.x);
      controlPosYaw.axes.push_back(desire_vel.linear.z);
      controlPosYaw.axes.push_back(desire_vel.angular.z);
      ctrlPosYawPub.publish(controlPosYaw);
      ROS_INFO("%.3f\t%.3f\t%.3f\t%.3f", 
        desire_vel.linear.x, desire_vel.linear.y,
        desire_vel.linear.z, desire_vel.angular.z);
    }
  }
  else {
    ROS_INFO("Not enough GPS Satellites");
    enable_IBVS = false;
  }   
}

char getch()
{
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

bool takeoff_land(int task)
{
  dji_sdk::DroneTaskControl droneTaskControl;

  droneTaskControl.request.task = task;

  drone_task_service.call(droneTaskControl);

  if(!droneTaskControl.response.result)
  {
    ROS_ERROR("takeoff_land fail");
    return false;
  }

  return true;
}

bool obtain_control()
{
  dji_sdk::SDKControlAuthority authority;
  authority.request.control_enable=1;
  sdk_ctrl_authority_service.call(authority);

  if(!authority.response.result)
  {
    ROS_ERROR("obtain control failed!");
    return false;
  }

  return true;
}

bool is_M100()
{
  dji_sdk::QueryDroneVersion query;
  query_version_service.call(query);

  if(query.response.version == DJISDK::DroneFirmwareVersion::M100_31)
  {
    return true;
  }

  return false;
}


void gps_position_callback(const sensor_msgs::NavSatFix::ConstPtr& msg) {
  current_gps_position = *msg;
}

void gps_health_callback(const std_msgs::UInt8::ConstPtr& msg) {
  current_gps_health = msg->data;
}

void flight_status_callback(const std_msgs::UInt8::ConstPtr& msg)
{
  flight_status = msg->data;
}

void display_mode_callback(const std_msgs::UInt8::ConstPtr& msg)
{
  display_mode = msg->data;
}

/*!
 * This function demos how to use M100 flight_status
 * to monitor the take off process with some error
 * handling. Note M100 flight status is different
 * from A3/N3 flight status.
 */
bool
M100monitoredTakeoff()
{
  ros::Time start_time = ros::Time::now();

  float home_altitude = current_gps_position.altitude;
  if(!takeoff_land(dji_sdk::DroneTaskControl::Request::TASK_TAKEOFF))
  {
    return false;
  }

  ros::Duration(0.01).sleep();
  ros::spinOnce();

  // Step 1: If M100 is not in the air after 10 seconds, fail.
  while (ros::Time::now() - start_time < ros::Duration(10))
  {
    ros::Duration(0.01).sleep();
    ros::spinOnce();
  }

  if(flight_status != DJISDK::M100FlightStatus::M100_STATUS_IN_AIR)
  {
    ROS_ERROR("Takeoff failed1.");
    return false;
  }
  if(
     current_gps_position.altitude - home_altitude < 1.0)
  {
    ROS_ERROR("Takeoff failed2.");
    return false;
  }
  else
  {
    start_time = ros::Time::now();
    ROS_INFO("Successful takeoff!");
    ros::spinOnce();
  }
  return true;
}

bool set_local_position()
{
  dji_sdk::SetLocalPosRef localPosReferenceSetter;
  set_local_pos_reference.call(localPosReferenceSetter);

  return (bool)localPosReferenceSetter.response.result;
}


