# offb

Offboard control using [ROS](http://www.ros.org) and [MAVROS](https://github.com/mavlink/mavros) for [PX4](https://github.com/PX4/Firmware).

The initial implementation is taken from the [MAVROS offboard control example](http://dev.px4.io/ros-mavros-offboard.html).

## Usage

### Dependencies

- [ROS](http://www.ros.org)
- [MAVROS](https://github.com/mavlink/mavros)
- [Catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)

### Building

```
cd ~/wherever/
git clone https://github.com/julianoes/offb.git
cd ~/catkin_ws
ln -s ~/wherever/offb ~/catkin_ws/src/offb
catkin_make
```

### Running

Start PX4 with e.g.:
```
make posix gazebo
```

Then start MAVROS:

```
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
```

And finally offb:

```
roslaunch offb offb.launch
```
