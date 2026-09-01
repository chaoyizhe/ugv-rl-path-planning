# UGV RL Path Planning (仿真部分)

无人车路径规划强化学习实验——ROS Noetic + Gazebo 仿真中训练 Jackal 小车的 DQN 自主避障与寻路。

## 内容

`fangzhen/` 目录为本仓库核心，为 ROS/Gazebo 仿真训练脚本：

| 文件 | 作用 |
|------|------|
| `fengzhen/jackal_env.txt` | Gazebo 仿真环境封装 (JackalEnv)：地图、雷达、碰撞检测、奖励函数 |
| `fengzhen/test.txt` | 加载 `best_model.pth` 进行测试/复现 |
| `fengzhen/train.txt` | DQN 训练主程序 (500 episode) |

## 环境

- Ubuntu 20.04 + ROS Noetic + Gazebo 11 + jackal 仿真包 + PyTorch
- ROS 工作区：`~/catkin_ws/src/jackal_dqn_env`
- 文件实际运行时要重命名为对应 `.py` 名 (`jackal_env.py` / `train.py` / `test.py`)，另需 `dqn_agent.py`。

## 地图与任务

- 10×10 栅格地图，黑色障碍分布见 `jackal_env.txt` 中 `MAP`。
- 起点 `(0.5, 0.5)`，终点 `(9.5, 9.5)`，距离 `< 0.5` 视为到达。
- 状态：24 维雷达(降采样)+坐标；动作：0=前进 1=左转 2=右转；雷达话题 `/front/scan`。

## 快速开始

```bash
source /opt/ros/noetic/setup.bash && source ~/catkin_ws/devel/setup.bash
roslaunch jackal_dqn_env jackal_dqn.launch   # 启动仿真 (终端1)

cd ~/catkin_ws/src/jackal_dqn_env
python3 train.py   # 训练 (终端2)
python3 test.py    # 测试最佳模型
```

## 结果与说明

初版训练 500 轮内小车可稳定避障、深入地图，但尚未成功抵达终点，需微调奖励函数(强化终点吸引)以为获得真正的最优路径。
