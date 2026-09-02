#!/usr/bin/env python3
"""V2.0 强化终点吸引 - 距离系数 60, 吸引 5.0"""
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates

MAP_SIZE = 10
ROBOT_RADIUS = 0.25

MAP = np.array([
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,1],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,1,0,1,0,1,0,1,0],
    [1,0,1,0,0,0,0,0,1,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,1,0,0,0,0,1,1,0,1],
    [0,1,0,1,0,0,0,0,0,0],
    [0,0,0,1,0,1,0,1,1,0],
    [1,0,0,0,0,0,0,0,0,0]
])

START = (0.5, 0.5)
GOAL = (9.5, 9.5)

class JackalEnv:
    def __init__(self):
        self.GOAL = GOAL
        self.scan = None
        self.x = 0
        self.y = 0
        self.previous_distance = None
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_state_callback)
        rospy.Subscriber("/front/scan", LaserScan, self.scan_callback)
        rospy.Subscriber("/jackal_velocity_controller/odom", Odometry, self.odom_callback)
        rospy.sleep(2)

    def reset(self):
        rospy.wait_for_service("/gazebo/set_model_state")
        state_msg = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        ms = ModelState()
        ms.model_name = "jackal"
        ms.pose.position.x, ms.pose.position.y = START
        ms.pose.position.z = 0.1
        state_msg(ms)
        rospy.sleep(0.5)
        self.x, self.y = START
        self.previous_distance = np.sqrt((START[0]-GOAL[0])**2 + (START[1]-GOAL[1])**2)
        return self.get_state()

    def scan_callback(self, msg):
        self.scan = np.array(msg.ranges)

    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y

    def model_state_callback(self, msg):
        if "jackal" in msg.name:
            i = msg.name.index("jackal")
            self.x = msg.pose[i].position.x
            self.y = msg.pose[i].position.y

    def check_collision(self):
        x, y = self.x, self.y
        if x < 0 or x > MAP_SIZE or y < 0 or y > MAP_SIZE:
            return True
        for dx in np.arange(-ROBOT_RADIUS, ROBOT_RADIUS+0.01, 0.10):
            for dy in np.arange(-ROBOT_RADIUS, ROBOT_RADIUS+0.01, 0.10):
                cx, cy = x+dx, y+dy
                if cx < 0 or cx > MAP_SIZE or cy < 0 or cy > MAP_SIZE:
                    return True
                r, c = int(cy), int(cx)
                if 0 <= r < MAP_SIZE and 0 <= c < MAP_SIZE and MAP[r][c] == 1:
                    return True
        if self.scan is not None:
            vs = self.scan[np.isfinite(self.scan)]
            if len(vs) > 0 and np.min(vs) < 0.20:
                return True
        return False

    def check_done(self):
        d = np.sqrt((self.x-GOAL[0])**2 + (self.y-GOAL[1])**2)
        if d < 0.5:
            return True
        if self.check_collision():
            return True
        if self.x < 0 or self.x > 10 or self.y < 0 or self.y > 10:
            return True
        return False

    def get_state(self):
        if self.scan is None:
            return None
        scan = np.clip(self.scan[::10], 0, 5)
        return np.append(scan, [self.x, self.y])

    def step(self, action):
        cmd = Twist()
        if action == 0:
            cmd.linear.x = 0.5
        elif action == 1:
            cmd.linear.x = 0.1
            cmd.angular.z = 0.8
        elif action == 2:
            cmd.linear.x = 0.1
            cmd.angular.z = -0.8
        self.cmd_pub.publish(cmd)
        rospy.sleep(0.1)
        return self.get_state(), self.get_reward(), self.check_done()

    def get_reward(self):
        d = np.sqrt((self.x - GOAL[0])**2 + (self.y - GOAL[1])**2)
        if d < 0.5:
            self.previous_distance = d
            return 500.0
        if self.check_collision():
            self.previous_distance = d
            return -200.0
        reward = -0.1
        if self.previous_distance is not None:
            reward += (self.previous_distance - d) * 60.0
        self.previous_distance = d
        reward += ((10.0 - d) / 10.0) * 5.0
        if self.scan is not None:
            vs = self.scan[np.isfinite(self.scan)]
            if len(vs) > 0:
                md = np.min(vs)
                if md < 0.50:
                    reward -= 3
                elif md < 0.80:
                    reward -= 1
        return reward
