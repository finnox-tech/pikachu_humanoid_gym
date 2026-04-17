# `Response.py` 使用说明

## 功能
`Response.py` 用于做任务关节的自动化阶跃响应测试。

脚本会根据 `--task` 自动关联到已注册任务，并自动读取：
- 任务对应的 `asset.file`
- URDF 中的关节限位 / effort / velocity
- 任务配置中的 PD 参数
- 任务配置和 URDF 共同作用后的有效扭矩限制与位置限制

然后它会对每个关节自动执行阶跃测试，输出：
- 每个关节的目标位置 / 实际位置响应曲线
- 动态响应指标
  - 调节时间 `settling time`
  - 上升时间 `rise time`
  - 峰值时间 `peak time`
  - 最大超调量 `max overshoot`
  - 稳态误差 `steady-state error`
  - 峰值扭矩 `peak torque`
- 汇总 CSV
- 明细 CSV
- JSON 报告

每张图像最多放 5 个关节；如果测试项更多，会自动拆成多张图。

---

## 脚本位置
- [Response.py](/home/finnox/Pikachu/PikachuRobot/pikachu_humanoid_gym/humanoid/scripts/Response.py)

---

## 运行前准备
建议在你平时训练/播放同一套环境中运行，例如：

```bash
cd /home/finnox/Pikachu/PikachuRobot/pikachu_humanoid_gym
python humanoid/scripts/Response.py --help
```

环境中需要可用：
- `isaacgym`
- `torch`
- `numpy`
- `matplotlib`

---

## 常用命令

1. 对 `Pikachu_V025_Quad` 做默认自动阶跃响应测试
```bash
python humanoid/scripts/Response.py --task Pikachu_V025_Quad
```

2. 只测试名字里包含 `ankle` 的关节
```bash
python humanoid/scripts/Response.py --task Pikachu_V025_Quad --joint_regex ankle
```

3. 正负两个方向都测
```bash
python humanoid/scripts/Response.py --task Pikachu_V025_Quad --step_direction both
```

4. 改大阶跃幅度
```bash
python humanoid/scripts/Response.py --task Pikachu_V025_Quad --step_fraction 0.5
```

5. 开 viewer 观察仿真过程
```bash
python humanoid/scripts/Response.py --task Pikachu_V025_Quad --show_viewer
```

6. 使用 free-base 测试
```bash
python humanoid/scripts/Response.py --task Pikachu_V025_Quad --free_base
```

7. 指定输出目录
```bash
python humanoid/scripts/Response.py \
  --task Pikachu_V025_Quad \
  --output_dir /tmp/pikachu_quad_response
```

---

## 默认行为说明

脚本默认会做这些隔离设置，但**只在脚本内部生效**，不会改你的原始任务文件：

1. `num_envs = 1`
2. 关闭噪声、推搡、质量随机化、摩擦随机化
3. 地形切到 `plane`
4. 关闭键盘指令和 debug plot
5. 默认固定基座

这样做的目的是尽量把结果聚焦到关节 PD + 扭矩限制 + 任务动作链路本身。

注意：
- 默认是“固定基座”，不是“关闭重力”。
- 只要任务里 `asset.disable_gravity = False`，重力仍然存在。
- 因此机器人虽然看起来悬在空中，但连杆重量、关节耦合、PD 刚度不足这些效应仍然会体现在响应结果里。

---

## 输出文件

默认输出目录：

```text
logs/response/<task>/<timestamp>/
```

会生成：

- `response_summary.csv`
  每个关节/方向一行，记录核心评估指标
- `response_detail.csv`
  完整时序数据，便于你二次分析
- `response_report.json`
  包含任务配置摘要和所有测试结果
- `response_group_01.png`, `response_group_02.png`, ...
  响应曲线图

图中除了文本指标，还会直接画出：
- 阶跃开始时刻
- 调节带
- 峰值点 / 峰值时间
- 调节时间
- `t90`

---

## 参数说明

- `--task`
  任务名，必须是 `humanoid/envs/__init__.py` 里已经注册的任务。
- `--show_viewer`
  打开 Isaac Gym viewer。默认不打开，避免图形环境影响自动测试。
- `--output_dir`
  指定输出目录。
- `--step_fraction`
  阶跃目标占可用关节余量的比例，默认 `0.35`。
- `--position_margin`
  距离有效关节限位保留的安全边界，默认 `0.03 rad`。
- `--step_direction`
  `auto / positive / negative / both`。
  `auto` 会优先选默认姿态附近余量更大的方向。
- `--pre_steps`
  阶跃前保持默认目标的步数。
- `--response_steps`
  阶跃后记录的步数。
- `--settle_tol_ratio`
  调节时间判据的相对带宽，默认 `2%`。
- `--settle_tol_abs`
  调节时间判据的最小绝对带宽，默认 `0.01 rad`。
- `--joint_regex`
  只测试匹配该正则的关节。
- `--free_base`
  不固定基座。
- `--no_detail_csv`
  不保存完整时序 CSV。默认会保存。

---

## 关于“根据 URDF 限制和 config 扭矩限制”

脚本当前的处理逻辑是：

1. 从 URDF 读取原始 `lower / upper / effort / velocity`
2. 从环境实例读取 Isaac Gym 实际使用的：
   - `dof_pos_limits`
   - `dof_vel_limits`
   - `torque_limits`
3. 响应测试使用的是**环境实际生效**的限制

也就是说：
- 图和指标反映的是任务里真正跑起来的控制响应
- 同时 CSV / JSON 里会保留 URDF 原始限制，方便你核对

另外，阶跃幅值会尽量优先选成比较整齐的数，例如：
- `0.1`
- `0.2`
- `0.3`
- `0.5`

如果关节可用余量比较小，才会退到更小的整齐值，例如 `0.05`。

---

## 如果需要新增参数，应该加在哪里

这个脚本没有改你全局的 `get_args()`，新增参数都只放在：

- [Response.py](/home/finnox/Pikachu/PikachuRobot/pikachu_humanoid_gym/humanoid/scripts/Response.py)

里面的 `parse_args()` 函数。

如果你后面想加例如：
- `--max_step_rad`
- `--torque_excitation`
- `--joint_list`
- `--save_npz`

建议都继续加在这个脚本自己的 `parse_args()` 里，不要去改：

- [helpers.py](/home/finnox/Pikachu/PikachuRobot/pikachu_humanoid_gym/humanoid/utils/helpers.py)

这样不会影响原本的 `train.py` / `play.py` / `sim2sim`。

---

## 结果解读建议

1. 先看 `response_summary.csv`
   找出超调大、调节时间长、稳态误差大的关节。
2. 再看对应的 `response_group_*.png`
   对比目标和实际曲线。
3. 如果某个关节：
   - `peak torque` 常贴近 `torque_limit`
   - `steady_state_error` 偏大
   - `settling time` 很长

   通常意味着该关节更可能受限于：
   - 扭矩上限太低
   - `kp/kd` 不合适
   - 负载/惯量耦合太强
   - 任务动作链路里还带有延迟

---

## 注意事项

1. 某些任务环境如果在 `step()` 里额外实现了动作延迟或滤波，那么测试结果会包含这部分效应。
2. 默认固定基座更适合看“电机/关节闭环”本身；如果你想看整机耦合响应，再打开 `--free_base`。
3. 如果某个关节默认姿态已经非常靠近限位，脚本可能会跳过它，避免做出几乎没有意义的极小阶跃。
