#include <termios.h>
#include <fcntl.h>

#include <tf/tf.h>
#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>

#define RED   "\x1B[1;131m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[1;35m"
#define CYN   "\x1B[1;36m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m\n"

char getch();
int ctr_mode = 0;

double desire_yaw = 0;
double roll, pitch, yaw;

geometry_msgs::Twist ibvs_global_vel;
geometry_msgs::Point local_position;

ros::Publisher local_pos_pub;
ros::Publisher global_vel_pub;
ros::Time ibvs_request_time;
mavros_msgs::State current_state;

void take_off();
void IBVS_control();
void KeyBoard_control(int c);
void IBVS_vel_callback(const geometry_msgs::Twist::ConstPtr& msg);

void state_cb(const mavros_msgs::State::ConstPtr& msg){
	current_state = *msg;
}

void IBVS_vel_callback(const geometry_msgs::Twist::ConstPtr& msg) {
	ibvs_request_time = ros::Time::now();
	ibvs_global_vel = *msg;
}

void local_pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
	tf::Quaternion q(msg->pose.orientation.x, msg->pose.orientation.y, 
		msg->pose.orientation.z, msg->pose.orientation.w);
	tf::Matrix3x3 m(q);
	m.getRPY(roll, pitch, yaw);
	//printf(WHT "Desire_Yaw:%.3f\tYaw:%.3f" RESET, desire_yaw, yaw);
	
	local_position = msg->pose.position;
}

int main(int argc, char **argv) {
	ros::init(argc, argv, "offb_node");
	ros::NodeHandle nh;

	ros::Subscriber state_sub      = nh.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);
	ros::Subscriber IBVS_Sub       = nh.subscribe("/IBVS/local_vel_cmd", 1, &IBVS_vel_callback);
	ros::Subscriber local_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("/mavros/local_position/pose", 1, local_pose_cb);
	
	local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>("mavros/setpoint_position/local", 1);
	global_vel_pub = nh.advertise<geometry_msgs::Twist>("/mavros/setpoint_velocity/cmd_vel_unstamped", 1);

	ros::ServiceClient arming_client   = nh.serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
	ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("mavros/set_mode");
  
	//the setpoint publishing rate MUST be faster than 2Hz
	ros::Rate rate(30);
	ibvs_request_time = ros::Time::now();
	// wait for FCU connection
	while(ros::ok() && !current_state.connected){
		ros::spinOnce();
		rate.sleep();
	}

	//send a few setpoints before starting
	for(int i = 30; ros::ok() && i > 0; --i){
		take_off();
		ros::spinOnce();
		rate.sleep();
	}

	mavros_msgs::SetMode offb_set_mode;
	offb_set_mode.request.custom_mode = "OFFBOARD";

	mavros_msgs::CommandBool arm_cmd;
	arm_cmd.request.value = true;

	ros::Time last_request = ros::Time::now();
    bool land = false;
	while(ros::ok()){
        if (land){
            ROS_INFO("LAND!");
            offb_set_mode.request.custom_mode = "MANUAL";
            set_mode_client.call(offb_set_mode);
            arm_cmd.request.value = false;
            arming_client.call(arm_cmd);
        }

        else {         
		    if( current_state.mode != "OFFBOARD" &&
			    (ros::Time::now() - last_request > ros::Duration(5.0))){
			    if( set_mode_client.call(offb_set_mode) &&
				    offb_set_mode.response.mode_sent){
				    printf(WHT "Offboard enabled" RESET);
			    }
			    last_request = ros::Time::now();
		    }
		    else {
			    if( !current_state.armed &&
				    (ros::Time::now() - last_request > ros::Duration(5.0))){
				    if( arming_client.call(arm_cmd) &&
					    arm_cmd.response.success){
					    printf(WHT "Vehicle armed" RESET);
				    }
				    last_request = ros::Time::now();
			    }
		    }

		    int c = getch();
		    //ROS_INFO("getch: %d", c);
		    if (c != EOF) {
			    switch (c) {				
				    case 49:
					    ctr_mode = 0;
					    printf(YEL "Set Position" RESET);
					    break;
				    case 50:
					    ctr_mode = 1;
					    printf(YEL "KeyBoard Control" RESET);
					    break;
					case 51:
					    ctr_mode = 2;
					    printf(YEL "Land" RESET);
					    break;					    
				    /*
				    case 51:
					    ctr_mode = 2;
					    printf(YEL "Enable IBVS" RESET);
					    break;
				    */
				    default:
					    break;
			    }
		    }//if (c != EOF)
		    switch (ctr_mode) {
			    case 0:
			    	//KeyBoard_control(c);
				    // Setting position may fail
				    take_off();
				    break;
			    case 1:
				    KeyBoard_control(c);
				    break;
                case 2:
				    land = true;
				    break;
			    default:
				    break;
		    }//switch (ctr_mode)
		    ros::spinOnce();
		    rate.sleep();
		}
	}//while
	return 0;
}

void take_off() {
	geometry_msgs::PoseStamped p;
	/*
	p.pose.position = local_position;
	p.pose.position.z = local_position.z + 0.2;
	*/
	p.pose.position.x = 0.0;
	p.pose.position.y = 0.0;
	p.pose.position.z = 1.0;

	local_pos_pub.publish(p);
	printf(CYN "SetP\t\tx:%.2f\t\ty:%.2f\t\tz:%.2f" RESET, p.pose.position.x, p.pose.position.y, p.pose.position.z);
}

void KeyBoard_control(int c) {
  static geometry_msgs::Twist ts;
  switch (c) {  
    case 65:    // key up
      ts.linear.x = 0;
      ts.linear.y = 0;
      ts.linear.z += 0.2;
      break;
    case 66:    // key down
      ts.linear.x = 0;
      ts.linear.y = 0;
      ts.linear.z -= 0.2;
      //ts.angular.z = 0;
      break;
    case 119:    // key w
      ts.linear.x += 0.2;
      ts.linear.y = 0;
      ts.linear.z = 0;
      break;
    case 115:    // key s
      ts.linear.x -= 0.2;
      ts.linear.y = 0;
      ts.linear.z = 0;
      break;
    case 97:    // key a
      ts.linear.x = 0;
      ts.linear.y += 0.2;
      ts.linear.z = 0;
      break;
    case 100:    // key d
      ts.linear.x = 0;
      ts.linear.y -= 0.2;
      ts.linear.z = 0;
      break;
    case 113:    // key q
      ts.angular.z += 0.1;
      break;
    case 101:    // key e
      ts.angular.z -= 0.1;
      break;
    case 32:    // key space
      ts.linear.x = 0;
      ts.linear.y = 0;
      ts.linear.z = 0;
      ts.angular.z = 0; 
    default: 
      break;
    } // switch (c)
    global_vel_pub.publish(ts);
    printf(MAG "KeyB\t\tx:%.2f\t\ty:%.2f\t\tz:%.2f\t\tyaw:%.2f" RESET, ts.linear.x, ts.linear.y, ts.linear.z, ts.angular.z);
}

char getch() {
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