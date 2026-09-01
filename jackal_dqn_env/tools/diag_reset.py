#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# reset 诊断脚本
# 检查 reset 后机器人的实际位置 / 雷达读数是否正常
# 用法: 仿真运行中，新终端执行 python3 diag_reset.py
# ============================================================

import sys
import rospy
import numpy as np

sys.stdout.reconfigure(line_buffering=True)


def main():

    rospy.init_node(
        "diag_reset",
        anonymous=True
    )

    from jackal_env import JackalEnv

    print(">>> 初始化环境...")
    env = JackalEnv()

    for i in range(4):

        print("\n========== reset #%d ==========" % i)

        s = env.reset()

        # 等待模型状态 / 雷达稳定
        rospy.sleep(1.0)

        print("  env.x     = %.3f" % env.x)
        print("  env.y     = %.3f" % env.y)

        if env.scan is not None:
            v = env.scan[np.isfinite(env.scan)]
            print("  scan shape   = %s" % str(env.scan.shape))
            print("  finite count = %d" % len(v))
            print("  scan min     = %.3f" % (v.min() if len(v) else float('nan')))
        else:
            print("  scan = None")

        if s is not None:
            print("  state shape  = %s" % str(s.shape))
            print("  state[0:3]   = %s" % np.asarray(s)[0:3])
            print("  state[-2:]   = %s" % np.asarray(s)[-2:])
        else:
            print("  state = None")

        # 再等一秒，看模型是否稳定在起点
        rospy.sleep(0.5)
        print("  1.5s后 x = %.3f, y = %.3f" % (env.x, env.y))

    print("\n>>> 诊断结束")


if __name__ == "__main__":
    main()