#!/usr/bin/env python3

import rospy
import numpy as np

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
# =====================
# 地图参数
# =====================

MAP_SIZE = 10
ROBOT_RADIUS = 0.25
CELL = 1.0


# 0: 空地
# 1: 障碍物

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


# 起点

START = (
    0.5,
    0.5
)


# 终点

GOAL = (
    9.5,
    9.5
)

class JackalEnv:

    def __init__(self):

        self.GOAL=GOAL
                
        rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.model_state_callback
        )
        
        

        
        
        # 雷达数据
        self.scan = None

        # 位姿
        self.x = 0
        self.y = 0
        self.previous_distance = None

        # 速度控制
        self.cmd_pub = rospy.Publisher(
            "/cmd_vel",
            Twist,
            queue_size=10
        )


        rospy.Subscriber(
            "/front/scan",
            LaserScan,
            self.scan_callback
        )


        rospy.Subscriber(
            "/jackal_velocity_controller/odom",
            Odometry,
            self.odom_callback
        )


        rospy.sleep(2)

    # =====================
    # 环境重置
    # =====================

    def reset(self):

        rospy.loginfo("Reset Jackal")

        rospy.wait_for_service("/gazebo/set_model_state")
        # 设置机器人位置

        state_msg = rospy.ServiceProxy(
            "/gazebo/set_model_state",
            SetModelState
        )


        model_state = ModelState()

        model_state.model_name = "jackal"


        model_state.pose.position.x = START[0]

        model_state.pose.position.y = START[1]

        model_state.pose.position.z = 0.1


        model_state.twist.linear.x = 0
        model_state.twist.linear.y = 0
        model_state.twist.linear.z = 0

        model_state.twist.angular.x = 0
        model_state.twist.angular.y = 0
        model_state.twist.angular.z = 0



        state_msg(model_state)

        rospy.sleep(0.5)


        self.x = START[0]
        self.y = START[1]


        self.previous_distance = np.sqrt(
            (START[0]-GOAL[0])**2+
            (START[1]-GOAL[1])**2
        )

        return self.get_state()

    def scan_callback(self,msg):

        self.scan = np.array(
            msg.ranges
        )


    def odom_callback(self,msg):

        self.odom_x = msg.pose.pose.position.x

        self.odom_y = msg.pose.pose.position.y
        
    def model_state_callback(self,msg):
        if "jackal" in msg.name:

            index = msg.name.index("jackal")

            self.x = msg.pose[index].position.x

            self.y = msg.pose[index].position.y
        
 
    def check_done(self):


        # =====================
        # 到目标
        # =====================

        distance = np.sqrt(
            (self.x-GOAL[0])**2 +
            (self.y-GOAL[1])**2
        )


        if distance < 0.5:

            rospy.loginfo("Reach Goal")

            return True



        # =====================
        # 碰撞
        # =====================

        if self.check_collision():

            rospy.loginfo("Collision")

            return True



        # =====================
        # 越界
        # =====================

        if (
            self.x < 0 or
            self.x > 10 or
            self.y < 0 or
            self.y > 10
        ):

            rospy.loginfo("Out of Map")

            return True



        return False
    # =====================
    # 碰撞检测
    # =====================

    def check_collision(self):


        x = self.x
        y = self.y


        # =====================
        # 1. 边界检测
        # =====================

        if x < 0 or x > MAP_SIZE:
            return True


        if y < 0 or y > MAP_SIZE:
            return True



        # =====================
        # 2. 车体范围检测
        # =====================

        for dx in np.arange(
            -ROBOT_RADIUS,
            ROBOT_RADIUS+0.01,
            0.10
        ):


            for dy in np.arange(
                -ROBOT_RADIUS,
                ROBOT_RADIUS+0.01,
                0.10
            ):


                check_x = x + dx

                check_y = y + dy



                # 超出地图

                if check_x < 0 or check_x > MAP_SIZE:
                    return True


                if check_y < 0 or check_y > MAP_SIZE:
                    return True



                # 转地图索引

                col = int(check_x)

                row = int(check_y)



                # 障碍物

                if (
                    row>=0 and
                    row<MAP_SIZE and
                    col>=0 and
                    col<MAP_SIZE
                 ):

                    if MAP[row][col]==1:
                        return True



        # =====================
        # 3. 雷达检测
        # =====================

        if self.scan is not None:

            valid_scan = self.scan[
                np.isfinite(self.scan)
            ]

            if len(valid_scan)>0:

                if np.min(valid_scan)<0.20:
                    return True



        return False

    # ----------------------
    # 获取状态
    # ----------------------

    def get_state(self):

        if self.scan is None:
            return None


        # 雷达降采样
        scan = self.scan[::10]


        # 最大距离限制
        scan = np.clip(
            scan,
            0,
            5
        )


        state = np.append(
            scan,
            [
                self.x,
                self.y
            ]
        )


        return state



    # ----------------------
    # 执行动作
    # ----------------------

    def step(self,action):


        cmd = Twist()


        if action == 0:
            # 前进

            cmd.linear.x = 0.5
            cmd.angular.z = 0


        elif action == 1:
            # 左转

            cmd.linear.x = 0.1
            cmd.angular.z = 0.8


        elif action == 2:
            # 右转

            cmd.linear.x = 0.1
            cmd.angular.z = -0.8



        self.cmd_pub.publish(cmd)



        rospy.sleep(0.1)



        state = self.get_state()


        reward = self.get_reward()


        done = self.check_done()


        return state,reward,done



    # =====================
    # 奖励函数
    # =====================

    def get_reward(self):

        # ==========================================
        # 当前距离目标
        # ==========================================

        distance = np.sqrt(
            (self.x - GOAL[0]) ** 2 +
            (self.y - GOAL[1]) ** 2
        )


        # ==========================================
        # 1. 到达目标
        # ==========================================

        if distance < 0.5:

            self.previous_distance = distance

            return 500.0


        # ==========================================
        # 2. 碰撞
        # ==========================================

        if self.check_collision():

            self.previous_distance = distance

            return -200.0


        # ==========================================
        # 3. 基础步数惩罚
        # ==========================================

        reward = -0.1


        # ==========================================
        # 4. 距离目标变化奖励
        # ==========================================

        if self.previous_distance is not None:

            distance_change = (
                self.previous_distance - distance
            )

            reward += distance_change * 40.0


        # 更新距离
        self.previous_distance = distance


        # ==========================================
        # 5. 额外的目标距离奖励
        # ==========================================

        # 越接近目标，每一步额外获得一些奖励
        distance_reward = (
            (10.0 - distance) / 10.0
        )

        reward += distance_reward * 0.5


        # ==========================================
        # 6. 激光雷达危险区域
        # ==========================================

        if self.scan is not None:

            valid_scan = self.scan[
                np.isfinite(self.scan)
            ]

            if len(valid_scan) > 0:

                min_distance = np.min(
                    valid_scan
                )


                


                # 接近障碍物
                if min_distance < 0.50:

                    reward -= 3


                # 安全距离
                elif min_distance < 0.80:

                    reward -= 1


        return reward



if __name__=="__main__":

    env = JackalEnv()


    while not rospy.is_shutdown():

        state = env.get_state()

        if state is not None:

            print(
                "state size:",
                state.shape
            )


        rospy.sleep(1)