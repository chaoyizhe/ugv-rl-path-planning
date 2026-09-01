# jackal_dqn_env 目录说明

核心脚本必须与 `worlds/`、`models/`、`launch/` 保持绝对路径引用，故保留在包根目录。

| 位置 | 内容 |
|------|------|
| 包根 (jackal_dqn_env/) | CMakeLists.txt, package.xml, 训练/环境/智能体脚本 |
| train.py | DQN 训练主程序 |
| test.py | 加载 best_model.pth 进行测试 |
| jackal_env.py | 无人车仿真环境 (JackalEnv) |
| dqn_agent.py | DQN 智能体 |
| nodes/ scripts/ | (可选) 计划放可独立运行的启动脚本 |
| launch/ | roslaunch 启动文件 (jackal_dqn.launch) |
| worlds/ | Gazebo 地图世界 (map.world) |
| models/ | 训练权重 (best_model.pth, last_model.pth, best_run_trajectory.csv) |
| logs/ | 训练/测试轨迹日志 |
| tools/ | 独立诊断/辅助脚本 (diag_reset.py 等) |

## 常用命令
- 启动仿真: `roslaunch jackal_dqn_env jackal_dqn.launch`
- 训练:     `cd ~/catkin_ws/src/jackal_dqn_env && python3 train.py`
- 测试:     `cd ~/catkin_ws/src/jackal_dqn_env && python3 test.py`

---

# 轨迹记录功能 (轨迹回放与最佳那次保留)

## 这个功能是做什么的

无论是训练 (train.py) 还是测试 (test.py)，都会把小车每一步的位置和动作自动记录下来，存成 CSV。
目的是让你**以后能回看任何一次仿真的实际过程**，而不是只看到"哪个模型奖励最高、但没有当时的轨迹"。

额外满足一个更具体的目标：**每次都自动留住"所有轮次里奖励最高的那一次"的完整轨迹**。
因为训练中 `best_model.pth` 权重会被不断覆盖，只有轨迹单独存档，才能还原出"最好那次走了什么路线"。

## 关键输出文件

| 文件 | 来源 | 含义 |
|------|------|------|
| `models/best_run_trajectory.csv` | train.py | **奖励最高那次**的轨迹，持续被更高奖励覆盖更新 |
| `logs/YYYYMMDD_HHMMSS/episode_XXXX_reward_XX.csv` | train.py | 每一轮一份轨迹，永远不覆盖，可追溯历史所有轮次 |
| `logs/YYYYMMDD_HHMMSS/run_summary.txt` | train.py | 本次训练汇总：哪一轮最好、奖励多少、每轮文件位置 |
| `logs/test/test_trajectory_YYYYMMDD_HHMMSS.csv` | test.py | 每次测试的轨迹，带时间戳，不覆盖历史 |

CSV 表头统一为: `step, action, x, y, reward`
- `step` 步数
- `action` 动作 (0=前进, 1=左转, 2=右转)
- `x` / `y` 小车在 Gazebo 世界中的坐标
- `reward` 该步奖励

地图为 10×10，起点 `(0.5, 0.5)`，终点 `(9.5, 9.5)`，障碍物见 `jackal_env.py` 中的 `MAP`。

## 怎么查看/复现

1. 用 CSV 画轨迹图(推荐,在用 Noetic 环境时)：
   ```bash
   cd ~/catkin_ws/src/jackal_dqn_env
   python3 tools/draw_trajectory.py models/best_run_trajectory.csv   # 若已有绘图脚本
   ```
   (如无绘图脚本，可用任意支持 CSV 的工具/脚本读取 `x,y` 列在地图上描点。)

2. 跑一次测试、把当前模型轨迹存一份：
   ```bash
   cd ~/catkin_ws/src/jackal_dqn_env && python3 test.py
   ```
   会自动生成 `logs/test/test_trajectory_*.csv`。

## 以后每个版本怎么保持这个功能(重要)

此功能是写进 `train.py` / `test.py` 代码本身的。只要沿用这一套 `jackal_dqn_env`
代码去改项目，记录逻辑会一直生效。**但如果你新复制了一份代码(新包/新目录)，
以下三处逻辑必须同步，否则会丢失轨迹记录：**

1. **`train.py` 中**
   - 每个 episode 里, `for step in range(MAX_STEPS)` 循环内要维护一个 `trajectory` 列表，
     在每次 `env.step(action)` 后追加 `[step, action, env.x, env.y, reward]`。
   - 每轮结束后把 `trajectory` 写入 `logs/<RUN_ID>/episode_XXXX_*.csv`。
   - **关键:** 在"保存 best_model.pth"的分支里(成功刷新 best 或临时 best 刷新时)，
     同时把该轮 `trajectory` 写入 `models/best_run_trajectory.csv`，即"最好那次"的轨迹。
   - 顶部需定义并创建 `MODEL_DIR`、`LOGS_DIR`、`RUN_LOGS_DIR`、`BEST_RUN_CSV` 等目录。

2. **`test.py` 中**
   - 循环内同样维护 `trajectory` 并在 `env.step` 后追加。
   - 结束时写入带时间戳的 `logs/test/test_trajectory_YYYYMMDD_HHMMSS.csv`。

3. **不要改动模型保存路径**
   `train.py`/`test.py` 用绝对路径 `~/catkin_ws/src/jackal_dqn_env/models/` 读写权重。
   若改动，需同步改脚本内的 `MODEL_DIR` 和 `MODEL_PATH`。

> 简易校验: 重训或测试后，检查 `models/best_run_trajectory.csv` 与 `logs/test/` 下是否生成了 CSV；
> 若存在说明轨迹记录功能正常工作。