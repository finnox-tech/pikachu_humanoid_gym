# Pikachu V025 Transfer — 站起任务奖励函数设计文档

## 任务目标

让机器人从初始趴卧/倒地状态，自主运动到**稳定直立站立**姿态。
本阶段**不需要**速度跟踪或步态控制，后续行走任务再逐步开启对应奖励。

---

## 激活的奖励函数

### 核心站立目标

| 奖励函数 | Scale | 说明 |
|----------|------:|------|
| `upright_progress` | +3.0 | **全程连续梯度**，解决稀疏奖励问题，见下方详细说明 |
| `stand_up` | +2.0 | 综合站立奖励（指数核，接近目标时精确奖励） |
| `orientation` | +2.0 | 底盘水平度（euler + projected_gravity 双重检测） |
| `base_height` | +0.5 | 底盘高度接近目标值（`base_height_target = 0.155 m`） |

### 手脚支撑引导（解决甩腿问题）

| 奖励函数 | Scale | 说明 |
|----------|------:|------|
| `support_contact` | +1.5 | 未站立时奖励手脚同时撑地；已站立时奖励手部离地 |
| `base_upward_vel` | +2.0 | 未站立时奖励底盘向上线速度，激励主动发力抬身 |

### 稳定性

| 奖励函数 | Scale | 说明 |
|----------|------:|------|
| `feet_distance` | +0.2 | 双脚横向间距保持合理范围 |
| `knee_distance` | +0.2 | 双膝横向间距保持合理范围 |
| `base_acc` | +0.2 | 惩罚底盘加速度剧烈晃动 |
| `feet_contact_forces` | -0.01 | 脚部接触力超限惩罚 |
| `collision` | -1.0 | 碰撞惩罚 |

### 能量效率与动作平滑

| 奖励函数 | Scale | 说明 |
|----------|------:|------|
| `action_smoothness` | -0.002 | 惩罚相邻帧动作突变 |
| `torques` | -1e-5 | 惩罚关节力矩过大 |
| `dof_vel` | **-5e-3** | 惩罚关节速度（**原 -5e-4 提升 10x，抑制甩腿**） |
| `dof_acc` | -1e-7 | 惩罚关节加速度过大 |

---

## `_reward_upright_progress` — 稀疏奖励问题的解决方案

### 根本原因

`stand_up` 和 `orientation` 均使用指数核（`exp(-x * 20)`）：
- 机器人趴着时 `‖projected_gravity_xy‖ ≈ 1.0`，导致 `exp(-20) ≈ 0`
- **无论策略做什么，奖励都接近 0，梯度消失，策略无法学习**

### 解决方案

```
_reward_upright_progress = -projected_gravity[:, 2]
```

| 姿态 | `projected_gravity[z]` | 返回值 |
|------|------------------------|--------|
| 完全直立 | ≈ -1.0 | **+1.0** |
| 水平趴卧 | ≈  0.0 | **~0.0** |
| 完全倒置 | ≈ +1.0 | **-1.0** |

全程提供连续梯度，引导策略从任意姿态旋转到直立。与 `stand_up` 互补：前者提供全程方向，后者在接近目标时精确奖励。

---

## `_reward_stand_up` 详细说明

```
返回值 = instant_reward × (1 + time_bonus) - prone_penalty
```

### 站立条件（三者同时满足）

| 条件 | 判断方式 |
|------|----------|
| 高度达标 | `root_states[:, 2] > base_height_target × 0.75` |
| 姿态竖直 | `‖projected_gravity[:, :2]‖ < 0.3`（约 17° 倾斜范围内） |
| 手部离地 | `contact_forces[:, hand_indices, 2] ≤ hand_contact_force` |

### 即时站立奖励 `instant_reward ∈ [0, 1]`

```
orientation_reward = exp(-‖projected_gravity_xy‖ × 20)
height_reward      = exp(-|base_height - 0.155| × 100)
instant_reward     = orientation_reward × height_reward
```

### 长时间站立加成 `time_bonus ∈ [0, 1)`

- 维护 `stand_up_timer`（秒）：连续满足站立条件时累加，否则清零
- `time_bonus = tanh(stand_up_timer / 2.0)`
- 约 **2 秒**后饱和到接近 1，即满分时奖励翻倍

### 趴下惩罚 `prone_penalty ∈ [0, 1]`

```
prone_penalty = hand_contact.float() × (1 - orientation_reward)
```
- 手部接触地面 **且** 姿态倾斜时才触发
- 纯手接触但姿态良好（如俯卧撑起身瞬间）惩罚较小

---

## 暂时关闭的奖励函数（步态相关）

| 奖励函数 | 关闭原因 |
|----------|----------|
| `joint_pos` | 参考运动跟踪，针对行走步态相位设计，与站起动作冲突 |
| `feet_clearance` | 脚部抬离高度，站立不需要抬脚 |
| `feet_contact_number` | 脚部接触时序与步态相位对齐，站立无步态 |
| `hand_contact_number` | 手部接触时序与步态相位对齐，站起过程手需自由使用 |
| `feet_air_time` | 腾空时间奖励，仅行走有意义 |
| `foot_slip` | 脚底滑动惩罚，站立时脚不移动无需约束 |
| `hand_slip` | 手部滑动惩罚，站起推地过程手会主动移动，开启会产生冲突 |
| `contact_no_vel` | 接触时无速度惩罚，行走步态稳定性约束，站立无需 |
| `tracking_lin_vel` | 速度指令跟踪，本阶段无速度目标 |
| `tracking_ang_vel` | 角速度指令跟踪，本阶段无目标 |
| `stand_still` | 静止惩罚，与站起过程冲突 |

---

## 后续行走阶段建议开启顺序

1. 先确认站立稳定后，开启 `foot_slip`、`feet_distance`（已开启）
2. 引入速度指令：开启 `tracking_lin_vel`、`tracking_ang_vel`
3. 添加步态：开启 `feet_contact_number`、`feet_air_time`、`feet_clearance`
4. 精调：开启 `joint_pos`、`hand_contact_number`、`contact_no_vel`
