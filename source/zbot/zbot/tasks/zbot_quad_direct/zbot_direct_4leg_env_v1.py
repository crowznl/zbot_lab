# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# obs修改：相对角度指令commands[1]改为显示的heading_err，否则网络难以学习！！！

# 训练过程中动态重采样命令（Command Resampling）,不仅仅在reset时发生，通过EventManager实现随机时间间隔改变指令

# [physics stepping(apply action)] > _get_dones() > _get_rewards() > _reset_idx()(S + event"reset") > event"interval" > _get_observations()
# 之前想过在_compute_intermediate_values()里通过resample标志位（env_ids）来更新target_heading_yaw，但_compute是在_get_dones()开头被调用的，而_get_dones()又在apply action后被调用，
# 那显然不能用已经动过的current_yaw再去计算target，要么在event"interval"之后_get_observations()开头再次调用_compute，但是很多计算不必要。所以确实有必要维护一个全局变量，
# 那么与其在调用resample_commands时记录current_yaw，不如直接在其中算出target_heading_yaw

# 又去看了下官方文档，发现完善了很多内容，一些需要自己摸索的现在都有说明了，真的是常看常新！！！

# [域随机化]
# [EN] https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html#domain-randomization
# [ZH] https://docs.robotsfan.com/isaaclab/source/tutorials/03_envs/create_direct_rl_env.html#domain-randomization
# you can find it in local document also. '~/IsaacLab/docs/source/tutorials/03_envs/create_direct_rl_env.rst'
# In the direct workflow, domain randomization configuration uses the configclass module to specify a configuration class consisting of EventTermCfg variables.
# 我第一次是在'~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env_cfg.py'看到这个用法的，发现direct workflow居然也设计提供了event manager。

# [动作和观测噪声]
# [EN] https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html#action-and-observation-noise
# [ZH] https://docs.robotsfan.com/isaaclab/source/tutorials/03_envs/create_direct_rl_env.html#action-and-observation-noise
# you can find it in local document also. '~/IsaacLab/docs/source/tutorials/03_envs/create_direct_rl_env.rst'

# maker visualization
# 参考manager based workflow的command相关配置，set_debug_vis，_set_debug_vis_impl，_debug_vis_callback，主要还是direct workflow也提供了待实现的相关接口。
# [EN] https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/training_jetbot_gt.html 
# [ZH] https://docs.robotsfan.com/isaaclab/source/setup/walkthrough/training_jetbot_gt.html
# [EN] https://isaac-sim.github.io/IsaacLab/main/source/how-to/draw_markers.html 
# [ZH] https://docs.robotsfan.com/isaaclab/source/how-to/draw_markers.html
# 这里也讲解了更详细的VisualizationMarkers用法。

# 为了速度指令既有正负值，绝对值又不会太小（>0.3），即避免在均匀采样如[-0.5, 0.5]时采样到大量接近0的指令，
# 修改resample_commands函数的逻辑：将 velocity_range 解释为速度的绝对值范围（大小），然后随机赋予正负号。

# 在Direct Workflow(DirectRLEnv)中实现课程学习Curriculum Learning，
# 随着tracking reward的提升，逐步扩大reset_command_resample.params["velocity_range"]的范围。
# 由于没有command_manager和curriculum_manager，还是借助event_manager实现。
# 但是这么说的话，DirectRLEnv又提供了self.common_step_counter  # for curriculum generation

from __future__ import annotations

import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg 
from isaaclab.utils import configclass

from zbot.assets import ZBOT_4L_CFG

def reset_root_state_uniform(
    env: Zbot4LEnvV1,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """

    # get default root state
    root_states = env._robot.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    # default_root_state的yaw通常是0，如果不是，需要加上default_yaw = math_utils.euler_xyz_from_quat(root_states[:, 3:7])[2]
    env.current_yaw[env_ids] = rand_samples[:, 5]

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # orientations = math_utils.random_yaw_orientation(len(env_ids), device=env.device)  # another methods, if only random yaw

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    env._robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    env._robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)

# --- Command Resampling (供 EventManager 调用) ---
def resample_commands(env: Zbot4LEnvV1, env_ids: torch.Tensor, velocity_range: tuple[float, float], yaw_range: tuple[float, float]):
    """Resample velocity and yaw commands."""

    # 1. Linear Velocity X (+- m/s)
    low, high = velocity_range
    # env.commands[env_ids, 0] = torch.rand(len(env_ids), device=env.device) * (high - low) + low
    # 随机生成符号 {-1, 1}
    vel_sign = torch.randint(0, 2, (len(env_ids),), device=env.device) * 2.0 - 1.0
    env.commands[env_ids, 0] = (torch.rand(len(env_ids), device=env.device) * (high - low) + low) * vel_sign
    
    # 2. Relative Yaw Command (-Pi ~ Pi)
    # 这是相对于当前朝向的目标偏角
    low, high = yaw_range
    env.commands[env_ids, 1] = torch.rand(len(env_ids), device=env.device) * (high - low) + low
    
    env.target_heading_yaw[env_ids] = math_utils.wrap_to_pi(env.current_yaw[env_ids] + env.commands[env_ids, 1])

def vel_range_curriculum(
    env: Zbot4LEnvV1,
    env_ids: torch.Tensor,
    reward_term_name: str = "track_lin_vel_x",
    limit_ranges: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:

    reward = torch.mean(env._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > env.reward_scales[reward_term_name] * 0.8:

            reset_term = env.cfg.events.reset_command_resample
            interval_term = env.cfg.events.interval_command_resample
            current_range = reset_term.params["velocity_range"]

            delta_range = torch.tensor([-0.1, 0.05], device=env.device)
            new_range = torch.clamp(
                torch.tensor(current_range, device=env.device) + delta_range,
                limit_ranges[0],
                limit_ranges[1],
            ).tolist()

            reset_term.params["velocity_range"] = tuple(new_range)
            interval_term.params["velocity_range"] = tuple(new_range)
            # 如果把velocity_range定义为list，应直接params["velocity_range"]=torch.clamp(...).tolist()，
            # 而避免中间变量，导致引用断裂。不过这里根据建议，还是保持使用tuple

@configclass
class EventCfg:
    """Configuration for randomization."""

    # startup
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         # "static_friction_range": (0.8, 0.8),
    #         # "dynamic_friction_range": (0.6, 0.6),
    #         "static_friction_range": (0.6, 1.0),
    #         "dynamic_friction_range": (0.6, 1.0),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-0.5, 1.0),
    #         "operation": "add",
    #     },
    # )

    # reset
    vel_range = EventTerm(
        func=vel_range_curriculum,
        mode="reset",
    )
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    reset_base = EventTerm(
        # func=mdp.reset_root_state_uniform,  # 由于需要更新current_yaw，自己实现了类似功能
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {  # default + range
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # 1. Reset: 环境重置时生成新指令
    reset_command_resample = EventTerm(
        func=resample_commands,
        mode="reset",
        params={
            # "velocity_range": (0.4, 0.6),
            # "velocity_range": (0.3, 0.5),
            # "velocity_range": (0.1, 0.5),
            # "velocity_range": (-0.3, 0.5),
            # "velocity_range": (-0.5, 0.5),
            # "velocity_range": (-0.8, 0.8),
            # "velocity_range": (0.3, 0.5),
            "velocity_range": (0.2, 0.5),
            "yaw_range": (-0.2, 0.2), # Reset初期给一个小一点的角度，容易起步
        },
    )

    # interval
    # 2. Interval: 每隔一段时间改变指令
    interval_command_resample = EventTerm(
        func=resample_commands,
        mode="interval",
        interval_range_s=(3.0, 6.0),  # 每 3~6 秒变一次指令
        params={
            # "velocity_range": (0.4, 0.6),  # -ln(1.595/2)*4= 0.057 m/s
            # "velocity_range": (0.3, 0.5),  # -ln(1.74/2)*4= 0.035 m/s
            # "velocity_range": (0.1, 0.5),  # not move
            # "velocity_range": (-0.3, 0.5),  # 还可行
            # "velocity_range": (-0.5, 0.5),  # not move
            # "velocity_range": (-0.8, 0.8),  # not move
            # "velocity_range": (0.3, 0.5),  # not so good in 2000 iter # resample_commands先采样大小，再随机正负
            "velocity_range": (0.2, 0.5),  # 还行 in 5000 or 8000 iter
            # "yaw_range": (-0.5, 0.5),  # OK: 0.974, -ln(0.97)*4*180/pi= 0.38 deg
            # "yaw_range": (-1.0, 1.0),  # OK: 0.921, -ln(0.92)*4*180/pi= 1.19 deg
            "yaw_range": (-0.8, 0.8),  # OK: 0.939, -ln(0.94)*4*180/pi= 0.90 deg
        },
    )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class Zbot4LEnvV1Cfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4  # 2
    action_space = 12  #48 for sin ;  12 for pd
    observation_space = 42
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200.0,  # 1 / 60.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # commands makers
    debug_vis=False  # train
    # debug_vis=True  # play

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/heading_velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""
    current_vel_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/heading_velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to RED_ARROW_X_MARKER_CFG."""
    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    arrow_scale = (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = arrow_scale
    current_vel_visualizer_cfg.markers["arrow"].scale = arrow_scale

    # events
    events: EventCfg = EventCfg()

    # [动作和观测噪声]
    # [EN] https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html#action-and-observation-noise
    # [ZH] https://docs.robotsfan.com/isaaclab/source/tutorials/03_envs/create_direct_rl_env.html#action-and-observation-noise
    # you can find it in local document also. '~/IsaacLab/docs/source/tutorials/03_envs/create_direct_rl_env.rst'

    # from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
    # # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    # action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    # noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
    # bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    # )
    # # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    # observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    # noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    # bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    # )
    # # If only per-step noise is desired, GaussianNoiseCfg can be used.

    # robot
    robot: ArticulationCfg = ZBOT_4L_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=True,
    )

    termination_height = 0.18
    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # #   working
    # reward_cfg = {
    #     "reward_scales": {
    #         # --- Tracking ---
    #         "track_lin_vel_x": 2.0,
    #         "track_heading_yaw": 1.0,

    #         # --- Penalties ---
    #         # "lin_vel_y": -1.0,         # 侧向漂移惩罚

    #         "action_rate": -0.1,
    #         "torques": -2e-4,
    #         "joint_vel": -0.001,
    #         "joint_acc": -2.5e-7,
    #         # "flat_orientation_l2": -2.5,
    #         "feet_downward": -1.0,

    #         # "feet_air_time": 1.0,
    #         # "airtime_variance": -1.0,
    #         # "feet_slide": -1.0,
    #     },
    # }
    # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    #   working
    reward_cfg = {
        "reward_scales": {
            # --- Tracking ---
            "track_lin_vel_x": 2.0,
            "track_heading_yaw": 1.0,

            # --- Penalties ---
            "lin_vel_y": -1.0,         # 侧向漂移惩罚

            "action_rate": -0.1,
            "torques": -2e-4,
            "joint_vel": -0.001,
            "joint_acc": -2.5e-7,
            "flat_orientation_l2": -2.5,
            "feet_downward": -1.0,

            "feet_air_time": 1.0,  # threshold = 0.2
            "airtime_variance": -1.0,
            "feet_slide": -1.0,
        },
    }


class Zbot4LEnvV1(DirectRLEnv):
    cfg: Zbot4LEnvV1Cfg

    def __init__(self, cfg: Zbot4LEnvV1Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        # self._debug_vis_handle = None  # already defined in super().__init__()
        # set initial state of debug visualization
        self.base_lin_vel_forward_w = torch.zeros(self.num_envs, device=self.device) # 不定义的话会有warning
        self.set_debug_vis(self.cfg.debug_vis)

        # Commands: [target_vel_x, target_yaw_relative]
        self.commands = torch.zeros(self.num_envs, 2, device=self.device)
        # 记录resample_commands时的目标朝向 (World Frame)
        self.current_yaw = torch.zeros(self.num_envs, device=self.device)
        self.target_heading_yaw = torch.zeros(self.num_envs, device=self.device)
        
        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self.p_delta = torch.zeros_like(self._robot.data.default_joint_pos)
        # self.joint_speed_limit = 0.2 + 1.8 * torch.rand(self.num_envs, 1, device=self.device)
        self.joint_speed_limit = 1.0 * torch.ones((self.num_envs, 1), device=self.device)  # 固定为1.0，不再作为obs

        # Get specific body indices
        self._feet_ids, _ = self._contact_sensor.find_bodies("foot.*")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies("base|a.*|b.*")
        self.base_body_idx = self._robot.find_bodies("base")[0]
        self.feet_body_idx = self._robot.find_bodies("foot.*")[0]

        self.z_w = torch.tensor([0, 0, 1], device=self.sim.device, dtype=torch.float32).repeat((self.num_envs, 4, 1))  # torch.Size([4096, 4, 3])
        self.axis_x_feet = torch.tensor(
            [[-1, 0, 0], [1, 0, 0], [1, 0, 0], [-1, 0, 0]], device=self.sim.device, dtype=torch.float32
        ).repeat((self.num_envs, 1, 1))
        self.axis_z_feet = torch.tensor(
            [[0, 1, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0]], device=self.sim.device, dtype=torch.float32
        ).repeat((self.num_envs, 1, 1))

        self.feet_contact_forces_last = 15.0 * torch.ones((self.num_envs, 4), device=self.device)
        self.feet_down_pos_last = torch.zeros((self.num_envs, 4, 3), device=self.device)
        self.feet_step_length = torch.zeros((self.num_envs, 4), device=self.device)
        self.feet_force_sum = torch.zeros(self.num_envs, device=self.device)
        self.heading_err_sum = torch.zeros(self.num_envs, device=self.device)
        
        self.reward_scales = cfg.reward_cfg["reward_scales"]
        # print('**** self.reward_scales = ', cfg.reward_cfg, cfg.reward_cfg["reward_scales"])
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self._episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            # self.reward_scales[name] *= self.step_dt  # 改到_get_rewards()中计算, 更直观
            self.reward_functions[name] = getattr(self, "_reward_" + name)

            # Logging
            self._episode_sums[name] = torch.zeros(
                (self.num_envs,), device=self.device, dtype=torch.float
            )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # #mode 1
        self._actions = torch.tanh(actions.clone())
        self.p_delta[:] += (
            torch.pi
            * self._actions
            * self.joint_speed_limit
            * self.step_dt
        )
        self.p_delta = torch.clip(self.p_delta, -1.0 * torch.pi, 1.0 * torch.pi)
        self._processed_actions = self.p_delta + self._robot.data.default_joint_pos

        # # mode 2
        # self.num_dof = 6
        # self._actions = torch.tanh(actions.clone())
        # ctl_d = self._actions.view(self.num_envs, self.num_dof, 4)
        # vmax = 2 * np.pi * self.joint_speed_limit
        # off = (ctl_d[..., 0] + 0) * vmax
        # amp = (1 - torch.abs(ctl_d[..., 0])) * (ctl_d[..., 1] + 0) * vmax
        # phi = (ctl_d[..., 2] + 0) * 2 * np.pi
        # omg = (ctl_d[..., 3] + 1.0) * np.pi
        # # omg = 2*np.pi
        # t = (self.episode_length_buf * self.step_dt) * torch.ones(
        #     self.num_envs, self.num_dof, device=self.device
        # )
        # v_d = off + amp * torch.sin(omg * t + phi)
        # self.p_delta += v_d * self.step_dt
        # self.p_delta = torch.clip(self.p_delta, -1.0 * torch.pi, 1.0 * torch.pi)
        # self._processed_actions = self.p_delta + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _compute_intermediate_values(self):
        self.base_pos_w = self._robot.data.body_link_pos_w[:, self.base_body_idx].squeeze(1)
        self.base_quat_w = self._robot.data.body_link_quat_w[:, self.base_body_idx].squeeze(1)
        self.feet_quat_w = self._robot.data.body_link_quat_w[:, self.feet_body_idx]
        self.feet_pos_w = self._robot.data.body_link_pos_w[:, self.feet_body_idx]
        # print(self.base_pos_w[:4, 2])  # tensor([0.2545, 0.2545, 0.2545, 0.2545], device='cuda:0')
        # print(self.base_quat_w[:2])  # [ 0.6003, -0.6003, -0.3735, -0.3739]
        # print(self.feet_pos_w[:2, :, 2])  # tensor([[0.0000e+00, 5.3035e-02],[1.8626e-09, 5.3035e-02]], device='cuda:0')

        axis_y = torch.tensor([0, 1, 0], device=self.sim.device, dtype=torch.float32).repeat((self.num_envs, 1))
        # base body axis y point to world Y
        self.base_shoulder_w = math_utils.quat_apply(self.base_quat_w, axis_y)  # torch.Size([4096, 3])
        self.base_dir_forward_w = torch.cross(self._robot.data.GRAVITY_VEC_W, self.base_shoulder_w, dim=-1)  # torch.Size([4096, 3])

        # # 如果USD中，articulation root就在base_link上，可以用下面这种更简单的方法计算
        # axis_x = torch.tensor([1, 0, 0], device=self.sim.device, dtype=torch.float32).repeat((self.num_envs, 1))
        # self.base_dir_forward_w = math_utils.quat_apply(self._robot.data.root_link_quat_w, axis_x)  # torch.Size([4096, 3])

        # self.current_yaw = math_utils.euler_xyz_from_quat(self.base_quat_w)[2]  # 目前没有提供四元数单独获取yaw的函数，反而计算量更大
        self.current_yaw = torch.atan2(self.base_dir_forward_w[:, 1], self.base_dir_forward_w[:, 0])  # torch.Size([4096])
        # print(self.current_yaw[0], self.target_heading_yaw[0])
        # tensor(-3.1367, device='cuda:0') tensor(-2.9934, device='cuda:0')
        # tensor(3.1384, device='cuda:0') tensor(-2.9934, device='cuda:0') # 可能跳变！
        # tensor(0.0782, device='cuda:0') tensor(0.0782, device='cuda:0')

        # self.heading_err = self.target_heading_yaw - self.current_yaw  # torch.Size([4096])
        # Wrap heading error to [-pi, pi] to avoid sudden ±π jumps when crossing the -π/π boundary.
        diff = self.target_heading_yaw - self.current_yaw
        self.heading_err = torch.atan2(torch.sin(diff), torch.cos(diff))
        # self.heading_err = torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi  # 法二

        self.base_lin_vel_w = self._robot.data.body_link_lin_vel_w[:, self.base_body_idx, :].squeeze()  # torch.Size([4096, 3])
        self.base_lin_vel_forward_w = torch.sum(self.base_lin_vel_w * self.base_dir_forward_w, dim=-1)  # 法一 torch.Size([4096])
        # self.base_lin_vel_forward_w = torch.einsum('ij,ij->i', self.base_lin_vel_w, self.base_dir_forward_w)  # 法二
        # self.base_lin_vel_forward_w = (self.base_lin_vel_w @ self.base_dir_forward_w.unsqueeze(-1)).squeeze(-1)  # 法三

        self.feet_contact_forces = torch.mean(
            self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids, 2],
            dim=1,
        ).squeeze()
        # self.feet_contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids, 2].squeeze()

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        # reset\event后，可能需要更新下缓存，
        self.base_quat_w = self._robot.data.body_link_quat_w[:, self.base_body_idx].squeeze(1)  # 要不就直接root_quat获取
        diff = self.target_heading_yaw - self.current_yaw
        self.heading_err = torch.atan2(torch.sin(diff), torch.cos(diff))
        # print(self.heading_err.shape)  # torch.Size([4096])
        # 神经网络算不过来：网络需要自己学会 Target_Yaw = Quat_to_Yaw(base_quat) + command_yaw，
        # 然后还要计算 Error = Target_Yaw - Current_Yaw。这对一个简单的 MLP 网络来说太难了，尤其是四元数是非线性的。
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.base_quat_w,  # 注意s2r时，IMU配置初始值修改
                    # self._robot.data.projected_gravity_b,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                    # self.joint_speed_limit,  # 固定为1.0，即1.0 * torch.pi
                    # self.commands, # [vel_x, yaw_relative]
                    self.commands[:, 0:1],  # vel_x
                    self.heading_err.unsqueeze(-1),
                )
                if tensor is not None
            ],
            dim=-1,
        )
        # print(obs.shape)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # compute reward
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name] * self.step_dt
            reward += rew
            self._episode_sums[name] += rew
        
        terminated_ids = self.reset_terminated.nonzero(as_tuple=False).squeeze(-1)
        reward[terminated_ids] -= 20.0  # vip to Penalize rew when reset

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        died = torch.any(
            torch.max(
                torch.norm(self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_body_ids], dim=-1), 
                dim=1, 
            )[0]
            > 1.0,
            dim=1,
        )
        # died_1 = (self.base_pos_w[:, 2] < self.cfg.termination_height)
        # died |= died_1
        # died_2 = (self.heading_err.abs() > 0.5 * torch.pi)
        # died |= died_2

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.p_delta[env_ids] = 0.0

        # Sample new yaw commands # 不用显示地更新了，super()._reset_idx(env_ids) calls EventManager (which calls resample_commands)
        # self.yaw_commands[env_ids] = torch.zeros_like(self.yaw_commands[env_ids]).uniform_(-1.0 * torch.pi, 1.0 * torch.pi)

        # Reset robot state
        # default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        
        # Randomize Root State & Refresh Current Yaw
        # self._reset_root_state_uniform(env_ids)  # 也挪到event里了，并在resample_commands之前

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.feet_contact_forces_last[env_ids] = 15.0 * torch.ones((len(env_ids), 4), device=self.device)
        self.feet_down_pos_last[env_ids] = (self._robot.data.body_link_pos_w[:, self.feet_body_idx])[env_ids]
        self.feet_step_length[env_ids] = torch.zeros((len(env_ids), 4), device=self.device)
        self.feet_force_sum[env_ids] = 0.0
        self.heading_err_sum[env_ids] = 0.0

        # self._compute_intermediate_values()  # 不太必要，大部分obs用不到

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        # 由于前面的spread机制，在第一次（或全量）重置时，
        # torch.randint_like将所有环境的episode_length_buf在[0, max_episode_length)之间均匀地采样整数。
        # 在这种异步重置机制下，单帧的 time_out 计数确实失去了表征“存活率”的意义，
        # 稳定在大约 num_envs / max_episode_length 个环境（.e.g 4096/1000 = 4.1）。
        # 因此，关注它，不如关注OnpolicyRunner log的平均回合长度mean_episode_length！它越接近max_episode_length，说明训练得好。
        if self.cfg.events.reset_command_resample is not None:
            extras["Curriculum/vel_lower_bound"] = self.cfg.events.reset_command_resample.params["velocity_range"][0]
        
        self.extras["log"].update(extras)

    def _reward_track_lin_vel_x(self):
        # Tracking Linear Velocity X
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel_forward_w)
        return torch.exp(-lin_vel_error / 0.25)

    def _reward_track_heading_yaw(self):
        # Tracking Heading (Relative)
        # heading_err is computed in _compute_intermediate_values
        return torch.exp(-torch.square(self.heading_err) / 0.25)

    def _reward_lin_vel_y(self):
        # Penalize lateral velocity (Drift)
        # Compute velocity in base frame Y direction
        vel_y = torch.sum(self.base_lin_vel_w * self.base_shoulder_w, dim=-1)
        return torch.square(vel_y)
    
    # def _reward_feet_forward(self):
    #     feet_x_w = math_utils.quat_apply(self.feet_quat_w, self.axis_x_feet)  # torch.Size([4096, 4, 3])
    #     # print(feet_x_w[0])
    #     feet_forward = torch.sum(
    #         torch.norm(
    #             feet_x_w - self.base_dir_forward_w.unsqueeze(1),
    #             dim=-1,
    #         ),
    #         dim=-1,
    #     )
    #     return feet_forward
    
    def _reward_feet_downward(self):
        feet_z_w = math_utils.quat_apply(self.feet_quat_w, self.axis_z_feet)  # torch.Size([4096, 4, 3])
        # print(feet_z_w[0])
        feet_downward = torch.sum(
            torch.norm(
                (feet_z_w - self.z_w),
                dim=-1,
            ),
            dim=-1,
        )
        return feet_downward

    # def _reward_heading_err(self):
    #     return torch.abs(self.heading_err)

    # def _reward_heading_err_sum(self):
    #     self.heading_err_sum += 0.01 * self.heading_err
    #     self.heading_err_sum = torch.clamp(self.heading_err_sum, -0.5*torch.pi, 0.5*torch.pi)
    #     return torch.abs(self.heading_err_sum)

    # def _reward_base_vel_forward(self):
    #     base_vel_forward = torch.tanh(10.0 * self.base_lin_vel_forward_w / self.joint_speed_limit.squeeze())
    #     return base_vel_forward

    def _reward_step_length(self):
        # Reward z axis base linear velocity

        force_c = 10.0
        # feet_just_down
        feet_down_idx = torch.logical_and(
            (self.feet_contact_forces > force_c),
            (self.feet_contact_forces_last < force_c),
        )  # "掩码索引"（boolean indexing）

        feet_step_vec_w = self.feet_pos_w - self.feet_down_pos_last
        feet_step_length_w = torch.sum(feet_step_vec_w * self.base_dir_forward_w.unsqueeze(1), dim=-1)  # 法一
        # feet_step_length_w = (feet_step_vec_w @ self.base_dir_forward_w.unsqueeze(-1)).squeeze(-1)  # 法二
        # feet_step_length_w = torch.einsum(
        #     'bij,bj->bi', 
        #     feet_step_vec_w, 
        #     self.base_dir_forward_w
        # )  # 法三
        self.feet_step_length[feet_down_idx] = feet_step_length_w[feet_down_idx]

        rew_feet_step_length = torch.min(self.feet_step_length, dim=-1)[0]
        
        self.feet_down_pos_last[feet_down_idx, :] = self.feet_pos_w[feet_down_idx, :]
        self.feet_contact_forces_last[:] = self.feet_contact_forces[:]  # refresh last
        return torch.tanh(15.0*rew_feet_step_length)

    def _reward_airtime_variance(self):
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        last_contact_time = self._contact_sensor.data.last_contact_time[:, self._feet_ids]
        return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
            torch.clip(last_contact_time, max=0.5), dim=1
        )

    def _reward_airtime_sum(self):
        feet_air_times = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        # airtime_sum = torch.tanh(torch.sum(feet_air_times, dim=-1))
        airtime_sum = torch.clamp(torch.sum(feet_air_times, dim=-1), max=2.0)
        return airtime_sum
    
    def _reward_feet_air_time(self):
        """Reward long steps taken by the feet using L2-kernel.

        This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
        that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
        the time for which the feet are in the air.

        If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
        """
        threshold = 0.2
        # compute the reward
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
        # no reward for zero command
        # reward *= torch.norm(self._command[:, :2], dim=1) > 0.1
        return reward

    def _reward_feet_slide(self):
        """Penalize feet sliding.

        This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
        norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
        agent is penalized only when the feet are in contact with the ground.
        """
        # Penalize feet sliding
        contacts = self.feet_contact_forces > 1.0
        feet_vel = self._robot.data.body_com_lin_vel_w[:, self.feet_body_idx, :2]
        feet_slide = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)
        return feet_slide
    
    def _reward_action_rate(self):
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        return action_rate
    
    def _reward_torques(self):
        # Penalize joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        return joint_torques
    
    def _reward_joint_vel(self):
        # joint velocity
        rew_joint_vel = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
        return rew_joint_vel

    def _reward_joint_acc(self):
        # joint acceleration
        rew_joint_acc = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        return rew_joint_acc

    def _reward_feet_height(self):
        feet_heights = self.feet_pos_w[:, :, 2]
        feet_heights[:,1] -= 0.053
        feet_height_reward = torch.sum(feet_heights, dim=1)
        return feet_height_reward
    
    def _reward_feet_gait(
        self,
        period: float = 1.0,
        offset: list[float] = [0.0, 0.5, 0.0, 0.5],  # 脚部ID顺序是[FL, RL, RR, FR]（左前，左后Rear，右后，右前）
        threshold: float = 0.55,
    ):
        is_contact = self._contact_sensor.data.current_contact_time[:, self._feet_ids] > 0

        global_phase = ((self.episode_length_buf * self.step_dt) % period / period).unsqueeze(1)
        phases = []
        for offset_ in offset:
            phase = (global_phase + offset_) % 1.0
            phases.append(phase)
        leg_phase = torch.cat(phases, dim=-1)

        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(len(self._feet_ids)):
            is_stance = leg_phase[:, i] < threshold
            reward += ~(is_stance ^ is_contact[:, i])
        
        # logging
        # if self.common_step_counter % 800 == 0:
        #     print(f"is_stanced: {leg_phase[:2] < threshold}")
        #     print(f"is_contact: {is_contact[:2]}")
        
        return reward
    
    def _reward_base_height(self):
        base_height = self.base_pos_w[:, 2] - self._terrain.env_origins[:, 2] - 0.25
        return base_height
    
    def _reward_flat_orientation_l2(self):
        """Penalize non-flat base orientation using L2 squared kernel.

        This is computed by penalizing the xy-components of the projected gravity vector.
        """
        return torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

    ################################################################################
    # Debug Visualization
    ################################################################################

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale = torch.tensor(self.cfg.arrow_scale, device=self.device).repeat(self.num_envs, 1)
        vel_des_arrow_scale[:, 0] *= self.commands[:, 0] * 5.0
        vel_arrow_scale = torch.tensor([self.cfg.arrow_scale[0], self.cfg.arrow_scale[1]*2, self.cfg.arrow_scale[2]*2], 
                                       device=self.device).repeat(self.num_envs, 1)
        vel_arrow_scale[:, 0] *= self.base_lin_vel_forward_w * 5.0
        zeros = torch.zeros_like(self.target_heading_yaw)
        vel_des_arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, self.target_heading_yaw)
        vel_arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, self.current_yaw)
        # vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        # vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self._robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    # def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Converts the XY base velocity command to arrow direction rotation."""
    #     # obtain default scale of the marker
    #     default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
    #     # arrow-scale
    #     arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
    #     arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
    #     # arrow-direction
    #     heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
    #     zeros = torch.zeros_like(heading_angle)
    #     arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
    #     # convert everything back from base to world frame
    #     base_quat_w = self._robot.data.root_quat_w
    #     arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

    #     return arrow_scale, arrow_quat

