1.PD Controller:before you start your RL training,you must adjust your PD controller at Mujoco or Gazebo with each joint,because the AC-PPO NN output are targeted positions of your each joint,if the basic PD controller is not available,the target output will not approch the right place that the NN desire. 

2.As a humanoid robot,there will be a comfortable base height when the robot are walking or do other works.So you would better to test the base height at your early experiments.


3.初始姿态，默认姿态不对，落地就倒，很难突破，陷入怪圈。之后感觉可以随机初始姿态，让机器人学会站立和倒地自救

4.初始离地高度，应该是根据正运动学算出来离地高度，让机器人平稳接触地面，防止有下落过程导致踮脚尖

5.
```
You must change that to fit your robot!
def _reward_hip_pos(self):
    """髋关节位置奖励：惩罚髋关节偏离初始位置"""
    # 提取髋关节索引（[3,4,9,10]，计算位置平方和作为惩罚
    return torch.sum(torch.square(self.dof_pos[:, [3,4,9,10]]), dim=1)
```


6.尝试keep_yaw,惩罚角速度变化，惩罚yaw轴偏差

7.尝试仅用腿部5关节，fixd other joints



8.惩罚项的指数核函数
r=exp(-|q-q_target|^2/sigma)
这种方式能够在误差较大时惩罚较平缓，误差小时奖励更丰富

9.exp_cfg 上下对比，自动报告，reward,result plot

10.Soft limits ,频繁超出URDF限制，添加一个指数极的限位惩罚项，当q>q_max时，惩罚项随超限距离呈平方级增长


11.平脚,根据陀螺仪


<!-- stand -->
python play.py --task=Pikachu_V01 --num_envs=10 --load_run=Jan28_04-41-45 --checkpoint=20000

python play.py --task=Pikachu_V01 --num_envs=10 --checkpoint=6600 --load_run=Jan28_09-37-56

# [512 256 128]
python play.py --task=Pikachu_V01 --num_envs=10 --checkpoint=50000 --load_run=Jan28_13-28-11


walk well
 python play.py --task=Pikachu_V01 --num_envs=10 --checkpoint=62000 --load_run=Jan29_08-06-47_

 随机姿态开始