你这个对比很有价值。结合你项目里的 `env + config + URDF`，结论是：

**不是单一超参数问题，而是“同一套奖励/时序假设”在两种机器人结构上可行性不同。**  
XBot 的结构天然更适配这套步态假设，Pikachu_V025 更容易收敛到“弹跳/双脚同相”局部最优。

**关键差异（按影响排序）**

1. **关节运动学约束差异很大，Pikachu 的步态自由度更受限**
- Pikachu 髋俯仰、膝关节基本是单侧区间（如 `left_hip_pitch_joint: [0, 2.44]`, `left_knee_joint: [0, 1.57]`），更像“只能屈、难以自然伸展摆腿”。
- 参考位置在 `Pikachu` 的 URDF：`left_hip_pitch_joint`、`left_knee_joint`、`left_ankle_joint` 限位在 [Pikachu_V025_flat.urdf:112](/home/finnox/Pikachu/pikachu-humanoid-gym/resources/robots/Pikachu_V025/urdf/Pikachu_V025_flat.urdf:112)、[Pikachu_V025_flat.urdf:298](/home/finnox/Pikachu/pikachu-humanoid-gym/resources/robots/Pikachu_V025/urdf/Pikachu_V025_flat.urdf:298)、[Pikachu_V025_flat.urdf:360](/home/finnox/Pikachu/pikachu-humanoid-gym/resources/robots/Pikachu_V025/urdf/Pikachu_V025_flat.urdf:360)。
- XBot 腿关节限位更宽且更对称（`left_leg_pitch_joint`, `left_knee_joint`, `left_ankle_pitch_joint`），在 [XBot-L.urdf:1537](/home/finnox/Pikachu/pikachu-humanoid-gym/resources/robots/XBot/urdf/XBot-L.urdf:1537)、[XBot-L.urdf:1598](/home/finnox/Pikachu/pikachu-humanoid-gym/resources/robots/XBot/urdf/XBot-L.urdf:1598)、[XBot-L.urdf:1659](/home/finnox/Pikachu/pikachu-humanoid-gym/resources/robots/XBot/urdf/XBot-L.urdf:1659)。
- 这会直接影响“交替步态”的可达性。

2. **动力学尺度完全不同，但奖励阈值很多是同量级模板**
- 我按 URDF 求和得到总质量：XBot 约 `53.04 kg`，Pikachu 约 `3.24 kg`。
- 但 `max_contact_force` 仅从 700 改到 200：[humanoid_config.py:186](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/custom/humanoid_config.py:186) vs [pikachu_v025_config.py:194](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/pikachu_v025/pikachu_v025_config.py:194)。
- 对 XBot 来说 700N 大约是体重数量级；对 Pikachu 来说 200N 是体重的很多倍，允许高冲击“砸地弹跳”。

3. **执行器能力和阻尼对比：Pikachu 更像“低力矩+低阻尼”，更容易弹簧化**
- Pikachu 关节 effort 多在 `9~12.5`，速度上限也低（部分只有 1~3 rad/s）；见 [Pikachu_V025_flat.urdf:112](/home/finnox/Pikachu/pikachu-humanoid-gym/resources/robots/Pikachu_V025/urdf/Pikachu_V025_flat.urdf:112)、[Pikachu_V025_flat.urdf:174](/home/finnox/Pikachu/pikachu-humanoid-gym/resources/robots/Pikachu_V025/urdf/Pikachu_V025_flat.urdf:174)。
- XBot 腿部关节 effort 到 `100~250`，速度到 `12`；见 [XBot-L.urdf:1548](/home/finnox/Pikachu/pikachu-humanoid-gym/resources/robots/XBot/urdf/XBot-L.urdf:1548)、[XBot-L.urdf:1609](/home/finnox/Pikachu/pikachu-humanoid-gym/resources/robots/XBot/urdf/XBot-L.urdf:1609)。
- 配置里的控制阻尼也差很多：XBot 基本 10 ([humanoid_config.py:122](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/custom/humanoid_config.py:122))，Pikachu 很低，踝仅 `0.01` ([pikachu_v025_config.py:129](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/pikachu_v025/pikachu_v025_config.py:129))。

4. **足端定义不同，接触语义不同**
- XBot 用 `foot_name="ankle_roll"`([humanoid_config.py:61](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/custom/humanoid_config.py:61))，URDF里还有 foot ee link。
- Pikachu 用 `foot_name="ankle"`([pikachu_v025_config.py:61](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/pikachu_v025/pikachu_v025_config.py:61))，实质是踝链接接触地面。
- 同样的接触奖励/惩罚，物理意义不完全一致，容易出现“踝部拍地推进”。

5. **初始姿态和几何尺度使 Pikachu 更像“蹲姿弹跳器”**
- XBot 初始高度 `0.95` 且关节默认 0：[humanoid_config.py:101](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/custom/humanoid_config.py:101)、[humanoid_config.py:103](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/custom/humanoid_config.py:103)。
- Pikachu 初始高度 `0.15`，默认是明显屈膝屈踝姿态：[pikachu_v025_config.py:101](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/pikachu_v025/pikachu_v025_config.py:101)、[pikachu_v025_config.py:103](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/pikachu_v025/pikachu_v025_config.py:103)。
- 同样 `target_feet_height=0.06` 与 `cycle_time=0.64`（两者都一样）对小机器人是“更激进”的摆动目标。

6. **奖励实现里有一处会放大“非真实交替步态”**
- `_reward_feet_air_time` 把 `stance_mask` 混进 `contact_filt`：[pikachu_v025_env.py:401](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/pikachu_v025/pikachu_v025_env.py:401)。
- 这使得“相位满足但未真实着地”也可能触发 air-time 计分，促进投机 gait。
- 再叠加 `only_positive_rewards=True`（[pikachu_v025_config.py:191](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/pikachu_v025/pikachu_v025_config.py:191)；裁剪在 [legged_robot.py:229](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/base/legged_robot.py:229)），负项约束被弱化。

7. **随机化对 Pikachu 的“相对扰动”更大**
- 两者都 `action_delay=0.5`、`action_noise=0.02`、`friction_range=[0.1,2.0]`，但 Pikachu 更轻、更低惯量，等效扰动更重。
- `added_mass_range` 也是相对体重更大扰动：Pikachu `[-0.2,0.8]` ([pikachu_v025_config.py:160](/home/finnox/Pikachu/pikachu-humanoid-gym/humanoid/envs/pikachu_v025/pikachu_v025_config.py:160)) 对 3.24kg 影响显著。

---

所以你看到的结果很合理：  
**XBot 在 100 步（你说的训练早期）就出现正常步行，而 Pikachu_V025 同配置不行，核心是结构尺度/约束不同导致奖励最优解不同。**

如果你愿意，下一步我可以给你一份“只针对结构差异做归一化”的改动清单（不是拍脑袋调参），按“力学尺度归一化、接触阈值归一化、步态时序归一化”三组来改。