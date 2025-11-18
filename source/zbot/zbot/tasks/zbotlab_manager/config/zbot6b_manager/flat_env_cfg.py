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
        # self.rewards.dof_torques_l2.weight = -5.0e-6
        # self.rewards.dof_acc_l2.weight *= 1.5
        # self.rewards.action_rate_l2 *= 1.5
        self.rewards.foot_step_length.weight = 5.0
        # self.rewards.gait.weight = 0.5
        # self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_clearance.weight = 0.5
        # self.rewards.feet_air_time.weight = 2.5
        # self.rewards.air_time_variance.weight = -1.0
        # self.rewards.undesired_contacts = None
        # self.rewards.flat_orientation_l2 = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


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
