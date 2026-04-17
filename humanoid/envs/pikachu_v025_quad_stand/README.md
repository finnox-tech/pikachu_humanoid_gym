# Pikachu V025 Quad Stand

## Goal

`Pikachu_V025_Quad_Stand` is a standing-recovery task for the 14-DoF quadruped configuration.

The policy is trained to:

- recover from randomized initial joint poses
- recover from randomized base roll/pitch/yaw
- damp initial linear and angular velocity disturbances
- return to the default standing pose and stay stable under pushes

## Task Registration

The task is registered in:

- `humanoid/envs/__init__.py`

Task name:

- `Pikachu_V025_Quad_Stand`

Example training command:

```bash
python humanoid/scripts/train.py --task=Pikachu_V025_Quad_Stand
```

## Main Files

- `humanoid/envs/pikachu_v025_quad_stand/pikachu_v025_quad_config.py`
- `humanoid/envs/pikachu_v025_quad_stand/pikachu_v025_quad_env.py`

## Random Initial Pose

Random reset is controlled by `PikachuQuadStandCfg.random_init`.

Current randomization includes:

- base xy position offset
- base z offset
- base roll/pitch/yaw
- base linear velocity
- base angular velocity
- per-joint position offsets around the default standing pose
- per-joint initial velocity

The environment samples these values during reset by overriding:

- `_reset_dofs(...)`
- `_reset_root_states(...)`

Joint positions are clipped by the robot DOF limits after sampling, so resets stay inside a safe controllable range.

## Recovery Metrics

The task logs recovery statistics through `infos["episode"]`, so they appear in the standard training logger automatically.

Current metrics:

- `recovery_rate`: fraction of finished episodes that successfully returned to a stable standing posture before reset
- `recovery_time`: mean recovery time in seconds, computed only over episodes that successfully recovered

Recovery is considered complete only when the robot satisfies the standing condition for several consecutive policy steps.

The thresholds are configured in `PikachuQuadStandCfg.recovery`:

- `orientation_threshold`
- `ang_vel_threshold`
- `lin_vel_threshold`
- `joint_error_threshold`
- `stable_steps`

In practice, this means the robot must simultaneously satisfy:

- small roll/pitch error
- small base angular velocity
- small horizontal base velocity
- small joint error relative to the default standing pose

for `stable_steps` consecutive control steps before the episode is marked as recovered.

## Observation Design

This stand task keeps the observation size compatible with the original quad setup:

- `53` single-step observations
- `15` frame stack

But gait phase is no longer used as a locomotion signal:

- `sin` is fixed to `0`
- `cos` is fixed to `1`
- commands are fixed to zero by config

This means the policy mainly relies on:

- joint error relative to default pose
- joint velocity
- previous action
- base angular velocity
- base Euler attitude

to recover back to the nominal standing posture.

## Reward Design

The stand task removes gait-specific objectives and keeps standing-recovery objectives.

Kept or emphasized:

- `orientation`
- `base_height`
- `default_joint_pos`
- `tracking_lin_vel`
- `tracking_ang_vel`
- `vel_mismatch_exp`
- `stand_still`
- `base_acc`
- `action_smoothness`
- `torques`
- `dof_vel`
- `dof_acc`
- `collision`
- `feet_contact_forces`

Removed from the effective reward scales:

- gait reference tracking
- feet clearance
- hand clearance
- feet contact phase reward
- hand contact phase reward
- feet air time
- foot slip gait reward
- hand slip gait reward

## Important Tuning Knobs

If the robot falls too often at the beginning:

- reduce `random_init.root_rot_range`
- reduce `random_init.joint_pos_range`
- reduce `random_init.root_ang_vel_range`
- reduce `random_init.joint_vel`

If recovery becomes too conservative:

- increase `random_init` ranges gradually
- slightly increase `action_scale`
- reduce `default_joint_pos` reward if the policy becomes too stiff

If the policy oscillates near the target posture:

- increase `action_smoothness` penalty magnitude
- increase `dof_vel` penalty magnitude
- decrease `random_init.joint_vel`

## Suggested Curriculum

Recommended order:

1. Start with smaller roll/pitch and joint offsets.
2. Train until the robot can consistently return to stand.
3. Increase root rotation and joint offset ranges.
4. Increase push strength only after recovery is stable.

## Notes

- This task is designed for recovery to the default standing pose, not walking.
- Commands are fixed at zero, so any non-zero base velocity is treated as disturbance that should be rejected.
- The same standing default pose is used both as control target and recovery target.
