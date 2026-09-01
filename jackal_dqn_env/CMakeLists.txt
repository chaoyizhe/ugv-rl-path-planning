cmake_minimum_required(VERSION 3.0.2)
project(jackal_dqn_env)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  gazebo_msgs
  gazebo_ros
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs nav_msgs gazebo_msgs gazebo_ros
)

# 代码直接在本目录用 python3 运行 (train.py/test.py)，无需安装到 nodes