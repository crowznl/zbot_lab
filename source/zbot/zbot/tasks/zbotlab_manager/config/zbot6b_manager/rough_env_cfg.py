# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import zbot.tasks.zbotlab_manager.mdp as mdp
from zbot.tasks.zbotlab_manager.zbotlab_env_cfg import ZbotLabRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from zbot.assets import ZBOT_6S_2_CFG, ZBOT_6S_V1_CFG


@configclass
class Zbot6BRewardsCfg(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot.*"),
            "command_name": "base_velocity",
            "threshold": 0.3,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot.*"),
        },
    )


@configclass
class Zbot6BRoughEnvCfg(ZbotLabRoughEnvCfg):
    """Zbot 6B rough environment configuration."""

    rewards: Zbot6BRewardsCfg = Zbot6BRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # scene
        self.scene.robot = ZBOT_6S_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # commands reduce
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (-0.314, 0.314)

        # actions
        self.actions.joint_pos.scale = 0.5

        # events
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["base|a.*|b.*"]
        self.terminations.base_contact = None
        self.terminations.base_height.params["minimum_height"] = 0.18

        # rewards
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -5.0e-6
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.action_rate_l2.weight *= 1.5
        self.rewards.dof_acc_l2.weight *= 1.5


@configclass
class Zbot6BRoughEnvCfg_PLAY(Zbot6BRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.7, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
