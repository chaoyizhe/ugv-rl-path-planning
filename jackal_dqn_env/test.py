#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import rospy
import numpy as np

from dqn_agent import DQNAgent
from jackal_env import JackalEnv


MODEL_PATH = os.path.expanduser(
    "~/catkin_ws/src/jackal_dqn_env/models/best_model.pth"
)

MAX_STEPS = 500

# 测试轨迹存档目录(带时间戳,避免覆盖历史)
TEST_TRAJ_DIR = os.path.expanduser(
    "~/catkin_ws/src/jackal_dqn_env/logs/test"
)

os.makedirs(TEST_TRAJ_DIR, exist_ok=True)

TEST_CSV = os.path.join(
    TEST_TRAJ_DIR,
    "test_trajectory_{}.csv".format(
        time.strftime("%Y%m%d_%H%M%S")
    )
)


def main():

    rospy.init_node(
        "dqn_test"
    )


    print("")
    print("===================================")
    print("       Jackal DQN Test")
    print("===================================")
    print("")


    # --------------------------------------------------------
    # 环境
    # --------------------------------------------------------

    env = JackalEnv()


    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    agent = DQNAgent()


    # --------------------------------------------------------
    # 加载最佳模型
    # --------------------------------------------------------

    print(
        "Loading model:"
    )

    print(
        MODEL_PATH
    )


    agent.load(
        MODEL_PATH
    )


    # --------------------------------------------------------
    # 测试时关闭探索
    # --------------------------------------------------------

    agent.epsilon = 0.0


    print("")
    print(
        "Model loaded successfully."
    )

    print(
        "Epsilon:",
        agent.epsilon
    )


    # --------------------------------------------------------
    # reset
    # --------------------------------------------------------

    state = env.reset()


    if state is None:

        print(
            "ERROR: state is None"
        )

        return


    state = np.asarray(
        state,
        dtype=np.float32
    )


    total_reward = 0

    trajectory = []


    # ========================================================
    # 自动执行
    # ========================================================

    for step in range(
        MAX_STEPS
    ):


        # ----------------------------------------------------
        # 根据训练好的网络选择动作
        # ----------------------------------------------------

        action = agent.choose_action(
            state
        )


        # ----------------------------------------------------
        # 打印动作
        # ----------------------------------------------------

        print(
            "step:",
            step,
            "action:",
            action,
            "position:",
            round(env.x, 2),
            round(env.y, 2)
        )


        # ----------------------------------------------------
        # 执行动作
        # ----------------------------------------------------

        next_state, reward, done = env.step(
            action
        )


        # 记录本步轨迹(step, action, x, y, reward)
        trajectory.append([
            step,
            action,
            round(float(env.x), 3),
            round(float(env.y), 3),
            round(float(reward), 3)
        ])


        if next_state is None:

            print(
                "ERROR: next_state is None"
            )

            break


        state = np.asarray(
            next_state,
            dtype=np.float32
        )


        total_reward += reward


        # ----------------------------------------------------
        # 到达目标
        # ----------------------------------------------------

        distance = np.sqrt(
            (env.x - 9.5) ** 2
            +
            (env.y - 9.5) ** 2
        )


        if distance < 0.5:

            print("")
            print(
                "================================"
            )

            print(
                "        SUCCESS!"
            )

            print(
                "        Reach Goal"
            )

            print(
                "================================"
            )

            break


        # ----------------------------------------------------
        # 碰撞/越界
        # ----------------------------------------------------

        if done:

            print("")
            print(
                "================================"
            )

            print(
                "        FAILED"
            )

            print(
                "        Collision or Out"
            )

            print(
                "================================"
            )

            break


    # ========================================================
    # 停车
    # ========================================================

    env.cmd_pub.publish(
        __import__(
            "geometry_msgs.msg",
            fromlist=["Twist"]
        ).Twist()
    )


    print("")
    print(
        "Test finished."
    )

    print(
        "Total reward:",
        total_reward
    )


    # ========================================================
    # 保存本次测试轨迹(带时间戳,不覆盖历史)
    # ========================================================

    with open(
        TEST_CSV,
        "w",
        newline=""
    ) as f:

        w = csv.writer(f)

        w.writerow(["step", "action", "x", "y", "reward"])

        w.writerows(trajectory)


    print(
        "Trajectory saved:",
        TEST_CSV
    )


if __name__ == "__main__":

    main()