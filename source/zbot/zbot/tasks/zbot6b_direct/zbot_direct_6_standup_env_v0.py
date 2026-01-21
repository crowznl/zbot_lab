# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# try to stand up

from __future__ import annotations

import gymnasium as gym
import torch
from collections import deque
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

from zbot.assets import ZBOT_6S_CFG_2

def reset_root_state_uniform(
    env: Zbot6SUpEnv,
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
    
    在四元数运算中, 如果q1代表当前的姿态(从体坐标系到世界坐标系的变换), 右乘q2即(q1 * q2)通常表示**相对于当前体坐标系(Intrinsic / Local Frame)**进行的旋转。
    
    quat_from_euler_xyz(roll, pitch, yaw) 返回的四元数对应的是 Extrinsic X-Y-Z 顺序（即先绕世界 X, 再绕世界 Y, 再绕世界 Z), 
    数学上等于 qdelta = qz ⊗ qy ⊗ qx. 当 pitch=0 时,它生成的是qz ⊗ qx.
    
    我现在需要在root_states的基础上, 先绕体坐标系x轴转动, 再绕体坐标系z轴转动; 或者先绕世界坐标系x轴转动, 再绕世界坐标系z轴转动。

    1. 方案一: 绕体坐标系 (Body Frame) X 轴转动，再绕 体坐标系 Z 轴转动
    问题: root_states 右乘 orientations_delta 确实是在体坐标系下进行旋转，但 quat_from_euler_xyz 生成的 delta = qz ⊗ qy ⊗ qx(隐含顺序是先 Z 后 X 的内旋，或先 X 后 Z 的外旋)。
    如果想要先 Body-X 后 Body-Z, 需要手动构造qx ⊗ qz
    q_x = math_utils.quat_from_euler_xyz(roll, torch.zeros_like(roll), torch.zeros_like(roll))
    q_z = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    orientations_delta = math_utils.quat_mul(q_x, q_z)

    2. 方案二：绕世界坐标系 (World Frame) X 轴转动，再绕 世界坐标系 Z 轴转动
    问题: 绕世界坐标系旋转需要使用左乘。你需要的目标旋转是 qz ⊗ qx ⊗ qroot(注意: 连续外旋时，后发生的旋转在最左边）。
    解决方法: 改为左乘, 且幸运的是 quat_from_euler_xyz(roll, 0, yaw) 生成的正是 qz ⊗ qx. 
    orientations = math_utils.quat_mul(orientations_delta, root_states[:, 3:7])
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
    orientations = math_utils.quat_mul(orientations_delta, root_states[:, 3:7])
    # orientations = math_utils.random_yaw_orientation(len(env_ids), device=env.device)  # another methods, if only random yaw

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    env._robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    env._robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def my_curriculum(env: Zbot6SUpEnv, env_ids: torch.Tensor):
    # if env.common_step_counter >= (env.max_episode_length * 16) and env.curriculum_stage == 0:  # in the 200 episodes 6*50*a/24
    # if env.common_step_counter >= (env.max_episode_length * 120) and env.curriculum_stage == 0:  # in the 1500 episodes
    if env.common_step_counter >= (env.max_episode_length * 80) and env.curriculum_stage == 0:  # in the 1000 episodes
        env.reward_scales["feet_downward_4"] = 2.0
        env.reward_scales["shape_symmetry"] = -2.0

        env.curriculum_stage += 1

    # elif env.common_step_counter >= (env.max_episode_length * 24) and env.curriculum_stage == 1:  # in the 1000 episodes
    #     env.reward_scales["airtime_variance"] = -40.0
    #     env.reward_scales["feet_downward"] = -5.0

    #     reset_term = env.event_manager.get_term_cfg("reset_command_resample")
    #     interval_term = env.event_manager.get_term_cfg("interval_command_resample")
    #     reset_term.params["prob_pos"] = 0.8
    #     interval_term.params["prob_pos"] = 0.8

    #     env.curriculum_stage += 1

@configclass
class EventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # "static_friction_range": (0.8, 0.8),
            # "dynamic_friction_range": (0.6, 0.6),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

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
            # "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "roll": (-0.7854, 0.7854), "yaw": (-3.14, 3.14)},  # 相对于世界坐标系XYZ顺序
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
   
    my_curric = EventTerm(
        func=my_curriculum,
        mode="reset",
    )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class Zbot6SUpEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 6.0
    decimation = 4  # 2
    action_space = 6  #24 for sin ;  6 for pd
    observation_space = 22
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
    robot: ArticulationCfg = ZBOT_6S_CFG_2.replace(prim_path="/World/envs/env_.*/Robot")
    # contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/.*",
    #     history_length=3,
    #     update_period=0.0,
    #     track_air_time=True,
    # )

    termination_height = 0.20
    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # # 脚能翻下去
    # reward_cfg = {
    #     "reward_scales": {
    #         "upward": 5.0,
    #         "shape_symmetry": -1.0,
    #         # --- Penalties ---
    #         "action_rate": -0.1,
    #         "torques": -2e-4,
    #         "joint_vel": -0.001,
    #         "joint_acc": -2.5e-7,

    #         "feet_downward": -1.0,
    #         # "feet_forward": 0.0,
    #     },
    # }

    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # # 减小fall down惩罚 能站起来 但是脚没翻下去
    # reward_cfg = {
    #     "reward_scales": {
    #         "upward": 5.0,
    #         "shape_symmetry": -1.0,
    #         "feet_downward": -1.0,
    #         # "feet_forward": 0.0,
    #     },
    # }

    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # # feetd-2 not stand up; feetd-1.5 有的脚没翻就直接尝试站起来
    # reward_cfg = {
    #     "reward_scales": {
    #         "upward": 5.0,
    #         "shape_symmetry": -1.0,
    #         "feet_downward": -1.5,
    #         # "feet_forward": 0.0,
    #     },
    # }

    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # # 尝试了curriculum not work，感觉应该设计reward把脚翻下去才有奖励
    # # <0 up 5 not stand up; 50 bad; 10 脚没彻底翻就开始站起来
    # # <0.5 up 10 
    # # <0.8 up 10 not stand up
    # # <0.7 up 10 1000it没有完全站起来；2000it stand up，有必要吗？？和0.5对比下【TODO】
    # reward_cfg = {
    #     "reward_scales": {
    #         "upward": 10.0,
    #         "shape_symmetry": -1.0,
    #         # # --- Penalties ---
    #         # "action_rate": -0.1,
    #         # "torques": -2e-4,
    #         # "joint_vel": -0.001,
    #         # "joint_acc": -2.5e-7,

    #         "feet_downward": -1.0,
    #         # "feet_forward": 0.0,
    #     },
    # }

    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # # 尝试下反对称
    # # <0.7 up 10 not stand up
    # # <0.5 up 10 stand up
    # reward_cfg = {
    #     "reward_scales": {
    #         "upward": 10.0,
    #         "shape_symmetry_2": -1.0,
    #         "feet_downward": -1.0,
    #     },
    # }

    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # # 还是脚没彻底翻就开始站起来，感觉得惩罚提前向上；
    # # <0.7 0.7 up 10 not stand up
    # # <0.5 0.7 up 10 stand up，但没什么用
    # # <0.5 0.8(v+h) up 10 not stand up
    # # <0.0 0.8(v+h) up 10 not stand up
    # # 合并了 <0.7 feetd -5 up 10 stand up，但没什么用
    # # 合并了 <0.8 feetd -10 up 10 not stand up
    # # 合并了 <0.8 feetd_2 -5 up 10 not stand up
    # # 合并了_0.1 <0.8 feetd -5 up 10 not stand up
    # # 合并了 <0.7 feetd -10 up 10 not stand up
    # # 合并了_2.0 <0.7 feetd -5 up 10 not stand up
    # # >0.1 <0.8 feetd -5 up 10 not stand up
    # # >0.1 <0.8 feetd -1 up 10 not stand up
    # # >0.1 <0.7 feetd -1 up 10 还不错，有点那意思了 2026-01-15_23-49-06 
    # # >0.08 <0.7 feetd -1 up 10 感觉没之前好
    # # >0.1 <0.75 feetd -1 up 10 better 2026-01-16_17-06-39 
    # # >0.08 <0.75 feetd -1 up 10 faster 1000it 2026-01-16_18-34-40
    # # >0.08 <0.75 feetd -1 up 5 not stand up
    # # _5.0 >0.08 <0.75 feetd -1 up 10 1000it 2026-01-16_19-45-02
    # # _5.0 >0.08 <0.8 feetd -1 up 10 not stand up
    # # _5.0 >0.07 <0.75 feetd -1 up 10 感觉差不多 2026-01-16_20-52-17
    # reward_cfg = {
    #     "reward_scales": {
    #         "upward": 10.0,
    #         "shape_symmetry": -1.0,
    #         "feet_downward": -1.0,
    #         # "upearly": -2.0,
    #     },
    # }

    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # # >0.1 <0.7 feetd -1 up 10 not stand up
    # # _5.0 >0.07 <0.75 feetd -1 up 10 not stand up, 
    # # no actionrate still not stand up
    # # no joint_acc still not stand up
    # # no joint_vel still not stand up
    # # no torques stand up 都没加了
    # reward_cfg = {
    #     "reward_scales": {
    #         "upward": 10.0,
    #         "shape_symmetry": -1.0,
    #         "feet_downward": -1.0,
            
    #         # --- Penalties ---
    #         # "action_rate": -0.1,
    #         # "torques": -2e-4,
    #         # "joint_vel": -0.001,
    #         # "joint_acc": -2.5e-7,
    #     },
    # }

    
    # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # _5.0 >0.07 <0.8 feetd -1 up 10 stand up 2026-01-17_17-26-49
    # rebase roll, yaw w.r.t world
    # _5.0 >0.07 <0.8 feetd -1 up 10 not stand up
    # _2.0 >0.1 <0.75 feetd -1 up 10 not stand up
    # _5.0 >0.1 <0.5 feetd -1 up 10 stand up 2026-01-17_19-16-32  <<==========
    # _5.0 >0.1 <0.7 feetd -1 up 10 not stand up
    # _5.0 >0.1 <0.6 feetd -1 up 10 not stand up
    # _5.0 >0.07 <0.5 feetd -1 up 10 not stand up
    # _5.0 >0.08 <0.5 feetd -1 up 10 not stand up
    # _5.0 >0.1 <0.5 feetd -1 up 10 feetd_3 10 not stand up
    # _5.0 >0.1 <0.5 feetd -1 up 10 feetd_3 5 not stand up
    # _5.0 >0.1 <0.5 feetd -1 up 10 feetd_3 2 not stand up
    # _5.0 >0.1 <0.5 feetd -1 up 10 feetd_3 1 not stand up
    # _5.0 >0.1 <0.5 feetd -1 up 10 feetd_3 0.5 not stand up
    # _5.0 >0.1 <0.5 feetd -1 up 10 feetd_3 0.1 not stand up
    # 说白了脚彻底放平几乎就是死区，彻底放平了再站起来几乎不可能，除非脚垫较高打破死区
    # 试试课程，>1500it 增加对称性和 <1.5 feetd_4 奖励【TODO】
    # _5.0 >0.1 <0.5 feetd -1 up 10 stand up >1500it sy -1.5 feetd_4 <1.5 1 not better
    # _5.0 >0.1 <0.5 feetd -1 up 10 stand up >1000it sy -2.0 feetd_4 <1.5 1 太高了，感觉拱形好些，修改upward 高度>0.22后直接固定奖励
    # _5.0 >0.1 <0.5 >0.22 feetd -1 up_2 10 stand up >1000it sy -2.0 feetd_4 <1.5 1 good good 稍微有点高 2026-01-20_15-20-25 <<==========
    # _5.0 >0.1 <0.5 >0.20 feetd -1 up_2 10 stand up >1000it sy -2.0 feetd_4 <1.5 1 not better
    # _5.0 >0.1 <0.5 >0.21 feetd -1 up_2 10 stand up >1000it sy -2.0 feetd_4 <1.5 1 not better
    # _5.0 >0.1 <0.5 >0.212 feetd -1 up_2 10 stand up >1000it sy -2.0 feetd_4 <1.5 1 太高了
    # _5.0 >0.1 <0.5 >0.212 feetd -1 up_2 10 stand up >1000it sy -1.5 feetd_4 <1.5 1 bad
    # _5.0 >0.1 <0.5 >0.212 feetd -1 up_2 10 stand up >1000it sy -2.0 feetd_4 <1.5 2 not better
    # _5.0 >0.1 <0.5 >0.22 feetd -1 up_2 10 stand up >1000it sy -2.0 feetd_4 <1.5 2 fine 2026-01-21_19-58-48 <<==========
    reward_cfg = {
        "reward_scales": {
            # "upward": 10.0,
            "upward_2": 10.0,
            "shape_symmetry": -1.0,
            "feet_downward": -1.0,
            # "feet_downward_3": 0.1,
            "feet_downward_4": 0.0,
        },
    }

    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # # 尝试下反对称 rebase roll, yaw w.r.t world
    # # _5.0 >0.1 <0.5 feetd -1 up 10 not stand up
    # # _5.0 >0.1 <0.05 feetd -1 up 10 stand up 但是脚没翻下去
    # # _0.0 >0.1 <0.5 feetd -1 up 10 bad
    # # _5.0 >0.1 <0.35 feetd -1 up 10 stand up
    # # _5.0 >0.1 <0.45 feetd -1 up 10 bad
    # # _5.0 >0.1 <0.4 feetd -1 up 10 OK 2026-01-19_14-12-05 <<==========
    # # _5.0 >0.1 <0.4 feetd -1 up 10 sy -2.0 bad
    # # _5.0 >0.1 <0.4 feetd -1 up 10 sy -1.5 bad
    # # _5.0 >0.1 <0.4 feetd -2 up 10 sy -2.0 bad 脚进入死区了
    # # _5.0 >0.1 <0.4 feetd -1 up 10 sy -1.2 stand up
    # # add penalty not stand up
    # reward_cfg = {
    #     "reward_scales": {
    #         "upward": 10.0,
    #         "shape_symmetry_2": -1.2,
    #         "feet_downward": -1.0,
    #     },
    # }

    # # play script
    # debug_vis=True
    # events.my_curric = None


class Zbot6SUpEnv(DirectRLEnv):
    cfg: Zbot6SUpEnvCfg

    def __init__(self, cfg: Zbot6SUpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        # self._debug_vis_handle = None  # already defined in super().__init__()
        # set initial state of debug visualization
        # self.base_lin_vel_forward_w = torch.zeros(self.num_envs, device=self.device) # 不定义的话会有warning
        # self.set_debug_vis(self.cfg.debug_vis)

        # Commands: [target_vel_x, target_yaw_relative]
        self.commands = torch.zeros(self.num_envs, 2, device=self.device)
        # 记录resample_commands时的目标朝向 (World Frame)
        self.current_yaw = torch.zeros(self.num_envs, device=self.device)
        self.target_heading_yaw = torch.zeros(self.num_envs, device=self.device)

        self.curriculum_stage = 0
        # 用于课程学习的滑动平均缓冲区，是 按每个reset环境的奖励值（用extend），还是 按每次step的平均奖励值（用append）来储存，在_reset_idx中设计实现
        self.curriculum_vel_reward_buffer = deque(maxlen=1*24)
        self.curriculum_yaw_reward_buffer = deque(maxlen=1*24)
        
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
        # self._feet_ids, _ = self._contact_sensor.find_bodies("foot.*")
        # self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies("base|a.*|b.*")
        self.base_body_idx = self._robot.find_bodies("base")[0]
        self.feet_body_idx = self._robot.find_bodies("foot.*")[0]

        self.z_w = torch.tensor([0, 0, 1], device=self.sim.device, dtype=torch.float32).repeat((self.num_envs, 2, 1))  # torch.Size([4096, 2, 3])
        self.axis_x_feet = torch.tensor(
            [1, 0, 0], device=self.sim.device, dtype=torch.float32
        ).repeat((self.num_envs, 2, 1))
        self.axis_z_feet = torch.tensor(
            [[0, 0, 1], [0, 0, -1]], device=self.sim.device, dtype=torch.float32
        ).repeat((self.num_envs, 1, 1))

        self.feet_down_pos_last = torch.zeros((self.num_envs, 2, 3), device=self.device)
        self.feet_step_length = torch.zeros((self.num_envs, 2), device=self.device)
        self.feet_force_sum = torch.zeros(self.num_envs, device=self.device)
        self.heading_err_sum = torch.zeros(self.num_envs, device=self.device)
        self.center_z_last = 0.05*torch.ones(self.num_envs, device=self.device)
        
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

        # self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # self.scene.sensors["contact_sensor"] = self._contact_sensor

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

        axis_z = torch.tensor([0, 0, 1], device=self.sim.device, dtype=torch.float32).repeat((self.num_envs, 1))
        # base body axis z point to world Y  # -------------------------------------------------------------------------------------------------------attention
        self.base_shoulder_w = math_utils.quat_apply(self.base_quat_w, axis_z)  # torch.Size([4096, 3])
        self.base_dir_forward_w = torch.cross(self._robot.data.GRAVITY_VEC_W, self.base_shoulder_w, dim=-1)  # torch.Size([4096, 3])

        self.current_yaw = torch.atan2(self.base_dir_forward_w[:, 1], self.base_dir_forward_w[:, 0])  # torch.Size([4096])
        # Wrap heading error to [-pi, pi] to avoid sudden ±π jumps when crossing the -π/π boundary.
        diff = self.target_heading_yaw - self.current_yaw
        self.heading_err = torch.atan2(torch.sin(diff), torch.cos(diff))

        self.base_lin_vel_w = self._robot.data.body_link_lin_vel_w[:, self.base_body_idx, :].squeeze()  # torch.Size([4096, 3])
        self.base_lin_vel_forward_w = torch.sum(self.base_lin_vel_w * self.base_dir_forward_w, dim=-1)  # 法一 torch.Size([4096])

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        # reset\event后，可能需要更新下缓存，
        self.base_quat_w = self._robot.data.body_link_quat_w[:, self.base_body_idx].squeeze(1)  # 要不就直接root_quat获取
        diff = self.target_heading_yaw - self.current_yaw
        self.heading_err = torch.atan2(torch.sin(diff), torch.cos(diff))
        # print(self.heading_err.shape)  # torch.Size([4096])

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.base_quat_w,  # 注意s2r时，IMU配置初始值修改
                    # self._robot.data.projected_gravity_b,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
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
        # reward[terminated_ids] -= 20.0  # vip to Penalize rew when reset
        reward[terminated_ids] -= 2.0  # vip to Penalize rew when reset

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        died = (self.center_z_last - self.base_pos_w[:, 2]) > 0.05

        self.center_z_last = torch.where((self.episode_length_buf % 50 == 49), self.base_pos_w[:, 2], self.center_z_last) # or put in _reset_idx()

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        actual_episode_duration = self.episode_length_buf[env_ids].float() * self.step_dt
        actual_episode_duration = torch.clamp(actual_episode_duration, min=self.step_dt)
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_per_s = self._episode_sums[key][env_ids] / actual_episode_duration
            extras["Episode_Reward/" + key] = torch.mean(episodic_sum_per_s)
            self._episode_sums[key][env_ids] = 0.0
        
        # if "track_lin_vel_x" in self._episode_sums:
        #     self.curriculum_vel_reward_buffer.append(extras["Episode_Reward/track_lin_vel_x"].item())
        # if "track_heading_yaw" in self._episode_sums:
        #     self.curriculum_yaw_reward_buffer.append(extras["Episode_Reward/track_heading_yaw"].item())

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        # 由于spread机制，在第一次（或全量）重置时，
        # torch.randint_like将所有环境的episode_length_buf在[0, max_episode_length)之间均匀地采样整数。
        # 在这种异步重置机制下，单帧的 time_out 计数确实失去了表征“存活率”的意义，
        # 稳定在大约 num_envs / max_episode_length 个环境（.e.g 4096/1000 = 4.1）。
        # 因此，关注它，不如关注OnpolicyRunner log的平均回合长度mean_episode_length！它越接近max_episode_length，说明训练得好。
        
        # extras["Curriculum/curriculum_stage"] = self.curriculum_stage
        self.extras["log"].update(extras)

        # ============================================================================================================
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

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.feet_down_pos_last[env_ids] = (self._robot.data.body_link_pos_w[:, self.feet_body_idx])[env_ids]
        self.feet_step_length[env_ids] = torch.zeros((len(env_ids), 2), device=self.device)
        self.feet_force_sum[env_ids] = 0.0
        self.heading_err_sum[env_ids] = 0.0
        self.center_z_last[env_ids] = 0.05

        # self._compute_intermediate_values()  # 不太必要，大部分obs用不到

    def _reward_track_lin_vel_x(self):
        # Tracking Linear Velocity X
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel_forward_w)
        return torch.exp(-lin_vel_error / 0.25)

    def _reward_track_heading_yaw(self):
        # Tracking Heading (Relative)
        # heading_err is computed in _compute_intermediate_values
        return torch.exp(-torch.square(self.heading_err) / 0.25)

    def _reward_lin_vel_x(self):
        return torch.square(self.base_lin_vel_forward_w)
    
    def _reward_lin_vel_y(self):
        # Penalize lateral velocity (Drift)
        # Compute velocity in base frame Y direction
        vel_y = torch.sum(self.base_lin_vel_w * self.base_shoulder_w, dim=-1)
        return torch.square(vel_y)
    
    def _reward_feet_forward(self):
        feet_x_w = math_utils.quat_apply(self.feet_quat_w, self.axis_x_feet)  # torch.Size([4096, 2, 3])
        # print(feet_x_w[0])
        feet_forward = torch.sum(
            torch.norm(
                feet_x_w - self.base_dir_forward_w.unsqueeze(1),
                dim=-1,
            ),
            dim=-1,
        )
        return feet_forward
    
    def _reward_feet_downward(self):
        feet_z_w = math_utils.quat_apply(self.feet_quat_w, self.axis_z_feet)  # torch.Size([4096, 2, 3])
        # print(feet_z_w[0])
        feet_downward = torch.sum(
            torch.norm(
                (feet_z_w - self.z_w),
                dim=-1,
            ),
            dim=-1,
        )
        return feet_downward

    
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
    
    def _reward_base_height(self):
        base_height = self.base_pos_w[:, 2] - self._terrain.env_origins[:, 2] - 0.25
        return base_height
    
    def _reward_upward(self):
        body_states = self._robot.data.body_link_state_w  # (num_envs, num_bodies, 13)
        rew_height = body_states[:, 6, 2] + 0.5*body_states[:, 4, 2] + 0.5*body_states[:, 8, 2] - 0.1*torch.ones_like(body_states[:, 6, 2])
        rew_upward = torch.where(body_states[:, 6, 2] < 0.22,
                                 rew_height + 0.5*body_states[:, 6, 9] + 0.5*body_states[:, 5, 9],
                                 rew_height + 1.0)
        
        feet_z_w = math_utils.quat_apply(self.feet_quat_w, self.axis_z_feet)
        rew_upward = torch.where(((feet_z_w[:, 0, 2] < 0.5) | (feet_z_w[:, 1, 2] < 0.5)) & (body_states[:, 6, 2] > 0.1), -5.0 * rew_upward, rew_upward)

        return rew_upward
    
    def _reward_upearly(self):
        body_states = self._robot.data.body_link_state_w  # (num_envs, num_bodies, 13)
        rew_height = body_states[:, 6, 2] + 0.5*body_states[:, 4, 2] + 0.5*body_states[:, 8, 2] - 0.1*torch.ones_like(body_states[:, 6, 2])
        feet_z_w = math_utils.quat_apply(self.feet_quat_w, self.axis_z_feet)
        rew = (rew_height + self._robot.data.body_link_state_w[:, 6, 9]) * ((feet_z_w[:, 0, 2] < 0.8) | (feet_z_w[:, 1, 2] < 0.8))
        return rew
    
    def _reward_shape_symmetry(self):
        jp = self.p_delta
        symmetry_err = (
            torch.abs(jp[:, 0] + jp[:, 5])
            + torch.abs(jp[:, 1] + jp[:, 4])
            + torch.abs(jp[:, 2] + jp[:, 3])
        )
        return symmetry_err
    
    def _reward_shape_symmetry_2(self):
        jp = self.p_delta
        symmetry_err = (
            torch.abs(jp[:, 0] - jp[:, 5])
            + torch.abs(jp[:, 1] - jp[:, 4])
            + torch.abs(jp[:, 2] - jp[:, 3])
        )
        return symmetry_err
    
    def _reward_feet_downward_2(self):
        feet_z_w = math_utils.quat_apply(self.feet_quat_w, self.axis_z_feet)  # torch.Size([4096, 2, 3])
        # print(feet_z_w[0])
        feet_downward_proj = torch.sum(
            torch.sum(
                (feet_z_w * self.z_w),
                dim=-1,
            ),
            dim=-1,
        )
        return 2-feet_downward_proj
    
    def _reward_feet_downward_3(self):
        feet_z_w = math_utils.quat_apply(self.feet_quat_w, self.axis_z_feet)  # torch.Size([4096, 2, 3])
        # print(feet_z_w[0])
        feet_downward_proj = torch.sum(
            torch.sum(
                (feet_z_w * self.z_w),
                dim=-1,
            ),
            dim=-1,
        )
        return feet_downward_proj
    
    def _reward_feet_downward_4(self):
        feet_z_w = math_utils.quat_apply(self.feet_quat_w, self.axis_z_feet)  # torch.Size([4096, 2, 3])
        # print(feet_z_w[0])
        feet_downward_proj = torch.sum(
            torch.sum(
                (feet_z_w * self.z_w),
                dim=-1,
            ),
            dim=-1,
        )
        feet_downward_proj = torch.where((self.base_pos_w[:, 2] < 0.15), feet_downward_proj, 1.6)
        return feet_downward_proj
    
    def _reward_upward_2(self):
        body_states = self._robot.data.body_link_state_w  # (num_envs, num_bodies, 13)
        rew_height = body_states[:, 6, 2] + 0.5*body_states[:, 4, 2] + 0.5*body_states[:, 8, 2] - 0.1*torch.ones_like(body_states[:, 6, 2])
        rew_upward = torch.where(body_states[:, 6, 2] < 0.22,
                                 rew_height + 0.5*body_states[:, 6, 9] + 0.5*body_states[:, 5, 9],
                                 0.35 + 1.0)
        
        feet_z_w = math_utils.quat_apply(self.feet_quat_w, self.axis_z_feet)
        rew_upward = torch.where(((feet_z_w[:, 0, 2] < 0.5) | (feet_z_w[:, 1, 2] < 0.5)) & (body_states[:, 6, 2] > 0.1), -5.0 * rew_upward, rew_upward)

        return rew_upward