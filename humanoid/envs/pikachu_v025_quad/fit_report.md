# 双足到四足机器人适配修改报告

## 项目概述
本项目将原有的双足机器人模型适配为四足机器人模型，支持双手双脚（18自由度）行走。主要修改了配置文件和环境文件，鼓励机器人从双足形态转换为四足形态进行移动。

## 修改内容

### 1. 配置文件修改 (`pikachu_v025_quad_config.py`)

#### 1.1 环境参数
- `num_actions`: 从10增加到18（10个腿部关节 + 8个手臂关节）
- `num_single_obs`: 从47增加到59（计算公式：5 + 3×18 + 6 = 59）
- `single_num_privileged_obs`: 从81增加到97（计算公式：25 + 4×18 = 97）
- `num_privileged_obs`: 相应更新以匹配新的观察维度

#### 1.2 默认关节角度
- 腿部关节保持双足初始姿态，便于模型学习转换
- 手臂关节设置为中性位置（接近0），鼓励模型学习如何降低身体并使用手臂支撑

#### 1.3 控制参数
- 更新了手臂关节的刚度和阻尼参数，使其更适合四足行走
- 降低了手臂关节的刚度，便于姿态调整

#### 1.4 奖励函数权重
- 添加了专门针对四足模式的奖励权重：
  - `quadrupedal_posture`: 2.0 - 鼓励四足身体姿态
  - `arms_contact`: 1.5 - 鼓励手臂与地面接触
  - `body_orientation_for_quadruped`: 1.0 - 鼓励身体保持水平

### 2. 环境文件修改 (`pikachu_v025_quad_env.py`)

#### 2.1 关节索引缓存
- 更新了手臂关节索引，包括pitch、roll、yaw和elbow四个关节
- 修正了关节名称（从`left_elbow_ankle_joint`改为`left_elbow_joint`）

#### 2.2 参考状态生成
- 更新了参考状态生成函数，支持18自由度的对角步态
- 实现了四足步态：左前腿+右后臂同步，右前腿+左后臂同步

#### 2.3 奖励函数
新增了以下奖励函数：
- `_reward_quadrupedal_posture`: 鼓励四足姿态，惩罚双足直立姿态
- `_reward_arms_contact`: 鼓励手臂与地面接触，支持四足行走
- `_reward_body_orientation_for_quadruped`: 鼓励身体保持水平姿态

## 设计理念

### 1. 渐进式学习
- 保持初始姿态与双足一致，让模型从熟悉的姿态开始学习
- 通过奖励函数引导模型逐步探索四足姿态

### 2. 四足步态实现
- 实现了对角步态模式，模拟真实四足动物的行走方式
- 左侧腿与右侧臂同步运动，右侧腿与左侧臂同步运动

### 3. 奖励机制
- 通过降低身体高度的奖励鼓励四足姿态
- 通过手臂使用奖励鼓励四肢协调运动
- 通过接触奖励鼓励手臂参与行走

## 验证结果

所有修改均通过Python语法检查，维度匹配正确，奖励函数逻辑合理。

## 使用建议

1. 训练初期模型可能会保持双足姿态，这是正常现象
2. 随着训练进行，模型应逐渐学会降低身体并使用手臂支撑
3. 可根据训练效果调整奖励权重比例
4. Z轴朝前的四足行走模式将在训练中逐渐显现

## 注意事项

1. 由于增加了自由度，训练可能需要更多时间收敛
2. 建议使用更大的神经网络容量以处理18自由度的复杂控制
3. 需要平衡四足奖励与运动性能奖励，避免过度降低身体影响移动效率

---

## 双足/四足模式切换功能（新增）

### 修改概述

在原有纯四足训练框架的基础上，新增了运行时可切换的**双足/四足模式命令**（`mode` command），使策略能够在单次训练中同时学习两种形态，并在 play 阶段通过键盘实时切换。

---

### 配置文件修改 (`pikachu_v025_quad_config.py`)

#### 观测维度更新

| 字段 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `num_single_obs` | 65 | **66** | command_input 新增 mode 维度 |
| `single_num_privileged_obs` | 97 | **98** | 同上 |
| `num_observations` | 975 | **990** | frame_stack × 66 |
| `num_privileged_obs` | 291 | **294** | c_frame_stack × 98 |

#### 指令扩展

```python
num_commands = 5  # 新增第5个: mode（0=双足, 1=四足）

class ranges:
    ...
    mode = [0, 1]
```

#### 奖励参数扩展

```python
base_height_target = 0.155   # 四足模式目标高度（原有）
base_height_biped  = 0.35    # 双足模式目标高度（新增）
```

---

### 环境文件修改 (`pikachu_v025_quad_env.py`)

#### `__init__`

新增 `self._keyboard_mode = 0`，play 时保存键盘模式状态（初始为双足）。

#### `_resample_commands`（新增覆写）

训练时对每个 episode 随机分配模式（50% 双足 / 50% 四足），使策略同时学习两种形态：

```python
mode_rand = torch.rand(len(env_ids), device=self.device)
self.commands[env_ids, 4] = (mode_rand > 0.5).float()
```

#### `_validate_observation_dims`

期望维度更新为 `6 + 3n + 6`（actor）和 `26 + 4n`（critic）。

#### `_get_noise_scale_vec`

`action_offset` 从 5 改为 **6**，`noise_vec[0:6] = 0`（mode 不加噪声）。

#### `compute_observations`

1. **T 键切换**（仅在键盘控制模式下生效）：

```python
for event in pygame.event.get():
    if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
        self._keyboard_mode = 1 - self._keyboard_mode
        print(f"[Mode切换] → {'四足(QUAD)' if ...} command[4]={...}")
```

2. 每帧将 `_keyboard_mode` 写入 `commands[:, 4]`，优先级高于 `_resample_commands`。

3. `command_input` 扩展为 6 维：

```python
self.command_input = torch.cat(
    (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale,
     self.commands[:, 4:5]), dim=1)  # 6 dims
```

#### `compute_ref_state`（重构）

分别计算双足参考和四足参考，按 mode 线性混合：

```python
# 双足：仅腿部交替摆动，手臂保持默认（ref=0）
# 四足：对角步态，左臂↔右腿同相，右臂↔左腿同相
mode = self.commands[:, 4].unsqueeze(1)
self.ref_dof_pos = biped_ref * (1.0 - mode) + quad_ref * mode
```

#### 奖励函数 mode 门控

| 奖励函数 | 门控条件 | 说明 |
|---------|---------|------|
| `_reward_base_height` | 按 mode 选不同目标高度 | 双足 0.35m，四足 0.155m |
| `_reward_default_arm_joint_pos` | 仅双足激活（`biped_mask`） | 四足时手臂由 `joint_pos` 引导 |
| `_reward_quadrupedal_posture` | 仅四足激活（`quad_mask`） | 降低躯干+使用手臂 |
| `_reward_arms_contact` | 仅四足激活 | 鼓励手肘接触地面 |
| `_reward_body_orientation_for_quadruped` | 仅四足激活 | 身体水平（与 orientation 不重叠）|

---

### play.py 修改

`FIX_COMMAND=True` 时，仅对 `Pikachu_V025_Quad` 任务写入 mode 指令：

```python
if args.task == 'Pikachu_V025_Quad':
    env.commands[:, 4] = 1.0  # 1=四足; 改为 0.0 可切换到双足
```

---

### 使用方式

| 场景 | 操作 |
|------|------|
| **训练** | 正常运行 `train.py`，随机 50/50 分配模式 |
| **Play 键盘模式**（`FIX_COMMAND=False`）| 按 **T** 键实时切换，终端打印当前模式 |
| **Play 固定命令模式**（`FIX_COMMAND=True`）| 修改 `play.py` 中 `commands[:, 4]` 的值（0=双足，1=四足）|