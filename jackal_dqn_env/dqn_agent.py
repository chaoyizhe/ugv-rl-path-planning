#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 参数
# ============================================================

# 当前 jackal_env.py：
# 72 个激光雷达数据
# + x
# + y
# = 74 维
STATE_DIM = 74

# 动作：
# 0 = 前进
# 1 = 左转
# 2 = 右转
ACTION_DIM = 3


# ============================================================
# 学习参数
# ============================================================

LR = 0.0005

GAMMA = 0.99


# ============================================================
# Epsilon 参数
# ============================================================

EPSILON = 1.0

EPSILON_MIN = 0.05

# 原来是 0.995
# 现在减慢探索衰减
EPSILON_DECAY = 0.98


# ============================================================
# Replay Buffer
# ============================================================

BATCH_SIZE = 64

MEMORY_SIZE = 50000

# Replay Buffer 至少积累这么多经验以后才开始学习
LEARN_START = 1000


# ============================================================
# Target Network
# ============================================================

# 每多少次 learn 更新一次 target network
TARGET_UPDATE_STEPS = 500


# ============================================================
# DQN网络
# ============================================================

class DQNNet(nn.Module):

    def __init__(self):

        super(DQNNet, self).__init__()


        # ----------------------------------------------------
        # 第一层
        # ----------------------------------------------------

        self.fc1 = nn.Linear(
            STATE_DIM,
            256
        )


        # ----------------------------------------------------
        # 第二层
        # ----------------------------------------------------

        self.fc2 = nn.Linear(
            256,
            256
        )


        # ----------------------------------------------------
        # 第三层
        # ----------------------------------------------------

        self.fc3 = nn.Linear(
            256,
            128
        )


        # ----------------------------------------------------
        # 输出层
        # ----------------------------------------------------

        self.out = nn.Linear(
            128,
            ACTION_DIM
        )


    def forward(self, x):

        x = F.relu(
            self.fc1(x)
        )

        x = F.relu(
            self.fc2(x)
        )

        x = F.relu(
            self.fc3(x)
        )

        return self.out(x)


# ============================================================
# DQN Agent
# ============================================================

class DQNAgent:

    def __init__(self):

        # ----------------------------------------------------
        # 使用 CPU
        # ----------------------------------------------------

        self.device = torch.device(
            "cpu"
        )


        # ----------------------------------------------------
        # Eval Network
        # ----------------------------------------------------

        self.eval_net = DQNNet().to(
            self.device
        )


        # ----------------------------------------------------
        # Target Network
        # ----------------------------------------------------

        self.target_net = DQNNet().to(
            self.device
        )


        # 初始时两个网络完全一样

        self.target_net.load_state_dict(
            self.eval_net.state_dict()
        )


        # Target Network 不参与梯度更新

        self.target_net.eval()


        # ----------------------------------------------------
        # 优化器
        # ----------------------------------------------------

        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(),
            lr=LR
        )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        # 原来是 MSELoss
        # 改成 Huber Loss
        #
        # 对碰撞 -100
        # 到达 +200
        # 这种较大的 reward 更稳定

        self.loss_fn = nn.SmoothL1Loss()


        # ----------------------------------------------------
        # Epsilon
        # ----------------------------------------------------

        self.epsilon = EPSILON


        # ----------------------------------------------------
        # Replay Buffer
        # ----------------------------------------------------

        self.memory = []

        self.memory_size = MEMORY_SIZE


        # ----------------------------------------------------
        # 学习次数
        # ----------------------------------------------------

        self.learn_step = 0


    # ========================================================
    # 选择动作
    # ========================================================

    def choose_action(
        self,
        state
    ):

        # ----------------------------------------------------
        # Epsilon-Greedy
        # ----------------------------------------------------

        # 随机探索

        if random.random() < self.epsilon:

            return random.randint(
                0,
                ACTION_DIM - 1
            )


        # ----------------------------------------------------
        # 神经网络选择动作
        # ----------------------------------------------------

        state = np.asarray(
            state,
            dtype=np.float32
        )


        state = torch.FloatTensor(
            state
        ).unsqueeze(
            0
        ).to(
            self.device
        )


        with torch.no_grad():

            q_values = self.eval_net(
                state
            )


        action = torch.argmax(
            q_values,
            dim=1
        ).item()


        return action


    # ========================================================
    # 保存经验
    # ========================================================

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        transition = (

            np.asarray(
                state,
                dtype=np.float32
            ),

            int(action),

            float(reward),

            np.asarray(
                next_state,
                dtype=np.float32
            ),

            bool(done)

        )


        # ----------------------------------------------------
        # Replay Buffer 满了以后
        # 删除最旧的数据
        # ----------------------------------------------------

        if len(self.memory) >= self.memory_size:

            self.memory.pop(0)


        self.memory.append(
            transition
        )


    # ========================================================
    # 学习
    # ========================================================

    def learn(self):

        # ----------------------------------------------------
        # Replay Buffer 预热
        # ----------------------------------------------------

        if len(self.memory) < LEARN_START:

            return None


        # ----------------------------------------------------
        # 随机采样
        # ----------------------------------------------------

        batch = random.sample(
            self.memory,
            BATCH_SIZE
        )


        # ----------------------------------------------------
        # 拆分 batch
        # ----------------------------------------------------

        states = np.array(
            [x[0] for x in batch],
            dtype=np.float32
        )


        actions = np.array(
            [x[1] for x in batch],
            dtype=np.int64
        )


        rewards = np.array(
            [x[2] for x in batch],
            dtype=np.float32
        )


        next_states = np.array(
            [x[3] for x in batch],
            dtype=np.float32
        )


        dones = np.array(
            [x[4] for x in batch],
            dtype=np.float32
        )


        # ----------------------------------------------------
        # 转 Tensor
        # ----------------------------------------------------

        states = torch.FloatTensor(
            states
        ).to(
            self.device
        )


        actions = torch.LongTensor(
            actions
        ).unsqueeze(
            1
        ).to(
            self.device
        )


        rewards = torch.FloatTensor(
            rewards
        ).unsqueeze(
            1
        ).to(
            self.device
        )


        next_states = torch.FloatTensor(
            next_states
        ).to(
            self.device
        )


        dones = torch.FloatTensor(
            dones
        ).unsqueeze(
            1
        ).to(
            self.device
        )


        # ====================================================
        # 当前 Q 值
        # ====================================================

        current_q = self.eval_net(
            states
        ).gather(
            1,
            actions
        )


        # ====================================================
        # Double DQN
        # ====================================================
        #
        # 普通 DQN：
        #
        # target_net 直接选最大 Q
        #
        # Double DQN：
        #
        # eval_net 负责选择动作
        # target_net 负责评价动作
        #
        # 可以降低 Q 值高估问题
        # ====================================================

        with torch.no_grad():

            # ------------------------------------------------
            # 1. eval_net 选择下一步最优动作
            # ------------------------------------------------

            next_actions = self.eval_net(
                next_states
            ).argmax(
                dim=1,
                keepdim=True
            )


            # ------------------------------------------------
            # 2. target_net 评价这个动作
            # ------------------------------------------------

            next_q = self.target_net(
                next_states
            ).gather(
                1,
                next_actions
            )


        # ====================================================
        # TD Target
        # ====================================================

        target_q = (

            rewards

            +

            GAMMA
            *
            next_q
            *
            (1 - dones)

        )


        # ====================================================
        # Loss
        # ====================================================

        loss = self.loss_fn(
            current_q,
            target_q
        )


        # ====================================================
        # 反向传播
        # ====================================================

        self.optimizer.zero_grad()


        loss.backward()


        # ----------------------------------------------------
        # 梯度裁剪
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            self.eval_net.parameters(),
            5.0
        )


        self.optimizer.step()


        # ====================================================
        # 学习次数 +1
        # ====================================================

        self.learn_step += 1


        # ====================================================
        # 更新 Target Network
        # ====================================================

        if (
            self.learn_step
            %
            TARGET_UPDATE_STEPS
            == 0
        ):

            self.update_target_network()


        return loss.item()


    # ========================================================
    # 更新 Target Network
    # ========================================================

    def update_target_network(self):

        self.target_net.load_state_dict(
            self.eval_net.state_dict()
        )


    # ========================================================
    # Epsilon 衰减
    # ========================================================

    def update_epsilon(self):

        if self.epsilon > EPSILON_MIN:

            self.epsilon *= EPSILON_DECAY


            if self.epsilon < EPSILON_MIN:

                self.epsilon = EPSILON_MIN


    # ========================================================
    # 保存模型
    # ========================================================

    def save(
        self,
        path
    ):

        torch.save(
            {

                "eval_net":
                self.eval_net.state_dict(),

                "target_net":
                self.target_net.state_dict(),

                "optimizer":
                self.optimizer.state_dict(),

                "epsilon":
                self.epsilon,

                "learn_step":
                self.learn_step

            },
            path
        )


    # ========================================================
    # 加载模型
    # ========================================================

    def load(
        self,
        path
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )


        # ----------------------------------------------------
        # Eval Network
        # ----------------------------------------------------

        self.eval_net.load_state_dict(
            checkpoint["eval_net"]
        )


        # ----------------------------------------------------
        # Target Network
        # ----------------------------------------------------

        self.target_net.load_state_dict(
            checkpoint["target_net"]
        )


        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        if "optimizer" in checkpoint:

            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )


        # ----------------------------------------------------
        # Epsilon
        # ----------------------------------------------------

        if "epsilon" in checkpoint:

            self.epsilon = checkpoint[
                "epsilon"
            ]


        # ----------------------------------------------------
        # Learn Step
        # ----------------------------------------------------

        if "learn_step" in checkpoint:

            self.learn_step = checkpoint[
                "learn_step"
            ]


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":

    agent = DQNAgent()


    # --------------------------------------------------------
    # 随机生成一个 74 维状态
    # --------------------------------------------------------

    state = np.random.randn(
        STATE_DIM
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # 选择动作
    # --------------------------------------------------------

    action = agent.choose_action(
        state
    )


    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    print(
        "state shape:",
        state.shape
    )


    print(
        "action:",
        action
    )


    print(
        "epsilon:",
        agent.epsilon
    )


    print(
        "device:",
        agent.device
    )