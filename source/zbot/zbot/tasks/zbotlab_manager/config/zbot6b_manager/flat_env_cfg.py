# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import Zbot6BRoughEnvCfg


@configclass
class Zbot6BFlatEnvCfg(Zbot6BRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # rewards

        # self.rewards.track_lin_vel_xy_exp.weight = 1.0
        # self.rewards.track_ang_vel_z_exp.weight = 0.5
        # self.rewards.termination_penalty.weight = -200.0
        # self.rewards.dof_torques_l2.weight = -1.0e-5
        # self.rewards.dof_acc_l2.weight = -2.5e-7
        # self.rewards.action_rate_l2 = -0.01
        # self.rewards.foot_step_length.weight = 2.0
        # self.rewards.foot_downward.weight = -1.0
        # self.rewards.gait.weight = 0.5
        # self.rewards.feet_slide.weight = -0.2
        # self.rewards.feet_clearance.weight = 1.0
        # self.rewards.feet_air_time.weight = 2.5
        # self.rewards.air_time_variance.weight = -1.0
        # self.rewards.base_vel_forward.weight = 1.0
        # self.rewards.feet_force_pattern.weight = 1.0
        # self.rewards.undesired_contacts.weight = -1.0

        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        # self.rewards.termination_penalty = None
        self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2 = None
        self.rewards.action_rate_l2 = None
        self.rewards.foot_step_length = None
        # self.rewards.foot_downward = None
        self.rewards.gait = None
        self.rewards.feet_slide = None
        self.rewards.feet_clearance = None
        self.rewards.feet_air_time = None
        self.rewards.air_time_variance = None
        # self.rewards.base_vel_forward = None
        # self.rewards.feet_force_pattern = None
        # self.rewards.undesired_contacts = None
        
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.7, 1.0)


class Zbot6BFlatEnvCfg_PLAY(Zbot6BFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
