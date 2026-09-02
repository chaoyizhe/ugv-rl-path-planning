#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import rospy
import numpy as np
import torch

from dqn_agent import DQNAgent
from jackal_env import JackalEnv


# ============================================================
# 参数
# ============================================================

EPISODES = 500

MAX_STEPS = 500  # V3: 300 -> 500，给更多步数走完地图

TARGET_UPDATE = 20

MODEL_DIR = os.path.expanduser(
    "~/catkin_ws/src/jackal_dqn_env/models"
)

BEST_MODEL = os.path.join(
    MODEL_DIR,
    "best_model.pth"
)

LAST_MODEL = os.path.join(
    MODEL_DIR,
    "last_model.pth"
)

LOGS_DIR = os.path.expanduser(
    "~/catkin_ws/src/jackal_dqn_env/logs"
)

# 每次训练单独一个时间戳目录,避免覆盖历史轨迹
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_LOGS_DIR = os.path.join(LOGS_DIR, RUN_ID)

# 奖励最高那次轨迹的固定保存路径(可回放)
BEST_RUN_CSV = os.path.join(
    MODEL_DIR,
    "best_run_trajectory.csv"
)

# 轨迹CSV列名
TRAJ_HEADER = ["step", "action", "x", "y", "reward"]


# ============================================================
# 创建目录
# ============================================================

for d in (MODEL_DIR, RUN_LOGS_DIR):

    if not os.path.exists(d):

        os.makedirs(d)


# ============================================================
# 主程序
# ============================================================

def main():

    rospy.init_node(
        "dqn_training"
    )


    print("")
    print("===================================")
    print("      Jackal DQN Training")
    print("===================================")
    print("")

    print(
        "本轮轨迹将保存到:",
        RUN_LOGS_DIR
    )


    env = JackalEnv()


    agent = DQNAgent()


    best_reward = -float("inf")

    best_success = False


    # 记录所有轨迹(供最终对比)
    run_trajectories = {}


    for episode in range(
        1,
        EPISODES + 1
    ):


        print("")
        print("-----------------------------------")
        print(
            "Episode:",
            episode,
            "/",
            EPISODES
        )
        print("-----------------------------------")


        state = env.reset()


        if state is None:

            rospy.logwarn(
                "state is None"
            )

            continue


        state = np.asarray(
            state,
            dtype=np.float32
        )


        total_reward = 0

        losses = []

        success = False

        trajectory = []


        for step in range(
            MAX_STEPS
        ):


            action = agent.choose_action(
                state
            )


            next_state, reward, done = env.step(
                action
            )


            if next_state is None:

                rospy.logwarn(
                    "next_state is None"
                )

                break


            next_state = np.asarray(
                next_state,
                dtype=np.float32
            )


            # ------------------------------------------------
            # 记录本步轨迹(step, action, x, y, reward)
            # ------------------------------------------------

            trajectory.append([
                step,
                action,
                round(float(env.x), 3),
                round(float(env.y), 3),
                round(float(reward), 3)
            ])


            agent.store_transition(
                state,
                action,
                reward,
                next_state,
                done
            )


            loss = agent.learn()


            if loss is not None:

                losses.append(
                    loss
                )


            state = next_state

            total_reward += reward


            distance = np.sqrt(
                (env.x - env.GOAL[0]) ** 2
                +
                (env.y - env.GOAL[1]) ** 2
            ) if hasattr(env, "GOAL") else np.sqrt(
                (env.x - 9.5) ** 2
                +
                (env.y - 9.5) ** 2
            )


            if distance < 0.5:

                success = True

                break


            if done:

                break


        agent.update_epsilon()


        if episode % TARGET_UPDATE == 0:

            agent.update_target_network()


        if len(losses) > 0:

            avg_loss = np.mean(
                losses
            )

        else:

            avg_loss = 0


        print("")
        print(
            "Episode:",
            episode
        )

        print(
            "Steps:",
            step + 1
        )

        print(
            "Reward:",
            round(total_reward, 3)
        )

        print(
            "Average Loss:",
            round(avg_loss, 6)
        )

        print(
            "Epsilon:",
            round(agent.epsilon, 4)
        )

        print(
            "Success:",
            success
        )


        # ====================================================
        # 保存本episode轨迹日志
        # ====================================================

        ep_csv = os.path.join(
            RUN_LOGS_DIR,
            "episode_{:04d}_reward_{:.2f}.csv".format(
                episode,
                total_reward
            )
        )

        with open(
            ep_csv,
            "w",
            newline=""
        ) as f:

            w = csv.writer(f)

            w.writerow(TRAJ_HEADER)

            w.writerows(trajectory)


        run_trajectories[episode] = {
            "reward": total_reward,
            "success": success,
            "csv": ep_csv
        }


        # ====================================================
        # 保存最佳模型 + 最佳轨迹
        # ====================================================


        saved_best = False


        if success:

            if (not best_success) or (total_reward > best_reward):


                best_reward = total_reward

                best_success = True


                agent.save(
                    BEST_MODEL
                )

                saved_best = True


                print("")
                print(
                    "***** 保存成功路径模型 *****"
                )

                print(
                    "Reward:",
                    best_reward
                )

                print(
                    "Path:",
                    BEST_MODEL
                )

        elif not best_success:


            if total_reward > best_reward:


                best_reward = total_reward


                agent.save(
                    BEST_MODEL
                )

                saved_best = True


                print("")
                print(
                    "***** 保存临时最佳模型 *****"
                )

                print(
                    "Reward:",
                    best_reward
                )


        # 只要本次刷新了最佳,就把这一轮的轨迹单独保留为可回放文件
        if saved_best:

            with open(
                BEST_RUN_CSV,
                "w",
                newline=""
            ) as f:

                w = csv.writer(f)

                w.writerow(TRAJ_HEADER)

                w.writerows(trajectory)


            print("")
            print(
                "***** 已保存最佳轨迹 *****"
            )

            print(
                "Path:",
                BEST_RUN_CSV
            )


        agent.save(
            LAST_MODEL
        )


    # ========================================================
    # 训练结束:汇总
    # ========================================================

    print("")
    print("===================================")
    print("        Training Finished")
    print("===================================")

    print(
        "Best reward:",
        best_reward
    )

    print(
        "Best model:",
        BEST_MODEL
    )

    print(
        "Best run trajectory:",
        BEST_RUN_CSV
    )

    best_ep = max(
        run_trajectories,
        key=lambda e: run_trajectories[e]["reward"]
    )

    print(
        "Best episode index:",
        best_ep,
        "reward:",
        run_trajectories[best_ep]["reward"]
    )

    with open(
        os.path.join(RUN_LOGS_DIR, "run_summary.txt"),
        "w"
    ) as f:

        f.write("RUN_ID: %s\n" % RUN_ID)

        f.write("Best episode: %d\n" % best_ep)

        f.write("Best reward: %.3f\n" % run_trajectories[best_ep]["reward"])

        f.write("Best success: %s\n" % run_trajectories[best_ep]["success"])

        f.write("Trajectories:\n")

        for e in sorted(run_trajectories):

            f.write(
                "%d reward=%.3f success=%s %s\n" % (
                    e,
                    run_trajectories[e]["reward"],
                    run_trajectories[e]["success"],
                    run_trajectories[e]["csv"]
                )
            )


if __name__ == "__main__":

    main()
