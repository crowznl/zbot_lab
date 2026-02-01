# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# obs修改：相对角度指令commands[1]改为显示的heading_err，否则网络难以学习！！！
# 参考zbot_quad_direct/zbot_direct_4leg_env_v1.py

# 修改_rew_step_length，适应正反行走

# 控制台/TensorBoard的信息是在on_policy_runner.py的log方法中实现的，一定要意识到它记录的是统计平均值！！！
# 每次iteration，计算num_steps_per_env（i.e. 24）个steps的平均值。
# 收集阶段：OnPolicyRunner 会运行 num_steps_per_env 步（比如 24 步）。
# 累积：在这 24 步的过程中，extras["log"]都会被收集到 ep_infos 列表中。
# 平均：log 函数执行 value = torch.mean(infotensor)。
# 因此，在课程学习（Curriculum Learning）中，最好也使用滑动平均(Rolling Average)缓冲区来计算奖励的平均值，而不是某一帧的数据，避免过早提升难度。

# 将日志记录逻辑移到 _reset_idx 的最前面，并使用episode_length_buf计算实际时长，进而计算单位时间内奖励的平均。

# DirectRLEnv.step()遵循较新的Gymnasium接口，返回5个值 (obs, rewards, terminated, truncated, extras)，
# 而OnPolicyRunner期望的env.step()返回4个值 (obs, rewards, dones, extras)，这是通过包装器(Wrapper)实现的。
# env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # wrap around environment for rsl-rl

# 一个非常微妙但致命的问题。原因在于Python 对象引用和 @configclass 装饰器的行为机制。
# 简单来说：修改 env.cfg.events.reset_command_resample 这个配置对象没有用，
# EventManager(事件管理器)在初始化时会读取配置，并将其转化为内部的列表 _mode_term_cfgs。
# 通过 env.cfg.events... 访问到的配置对象，与 EventManager 内部持有的对象不是同一个实例。

# 在 mode="reset" 的回调函数中，永远不要使用 == 来判断全局步数，因为你是无法保证重置事件恰好落在那个特定步数上的。
# EventTerm(mode="reset")是注入到 self._reset_idx() super()中的，而 _reset_idx并不是每个step都会被调用。
# 我一直认为step不管怎样都会调用self._reset_idx。刚刚看了源码，原来它有len(reset_env_ids) > 0的判断。
# 这就是为什么用 == 几乎肯定会漏掉触发时机（除非那帧恰好有环境重置）。

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

from zbot.assets import ZBOT_6S_CFG

def reset_root_state_uniform(
    env: Zbot6SEnvV4,
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
def resample_commands(
        env: Zbot6SEnvV4, 
        env_ids: torch.Tensor, 
        velocity_range: tuple[float, float], 
        yaw_range: tuple[float, float],
        dual_sign: bool = True, 
        offset: float = 0.0,
        prob_pos: float = 0.5,
    ):
    """Resample velocity and yaw commands."""

    # 1. Linear Velocity X (+- m/s)
    low, high = velocity_range
    if dual_sign:
        # 随机生成符号 {-1, 1}
        # vel_sign = torch.randint(0, 2, (len(env_ids),), device=env.device) * 2.0 - 1.0
        vel_sign = torch.bernoulli(torch.full((len(env_ids),), prob_pos, device=env.device)) * 2.0 - 1.0
        
        high = high + offset * (vel_sign - 1.0)  # 降低反方向的速度指令
        env.commands[env_ids, 0] = (torch.rand(len(env_ids), device=env.device) * (high - low) + low) * vel_sign
    else:
        env.commands[env_ids, 0] = torch.rand(len(env_ids), device=env.device) * (high - low) + low
    
    # 2. Relative Yaw Command (-Pi ~ Pi)
    # 这是相对于当前朝向的目标偏角
    low, high = yaw_range
    env.commands[env_ids, 1] = torch.rand(len(env_ids), device=env.device) * (high - low) + low
    
    env.target_heading_yaw[env_ids] = math_utils.wrap_to_pi(env.current_yaw[env_ids] + env.commands[env_ids, 1])

def my_curriculum(env: Zbot6SEnvV4, env_ids: torch.Tensor):
    # if env.common_step_counter == (env.max_episode_length * 24):  # in the 1000 episodes
    # if env.common_step_counter >= (env.max_episode_length * 24) and env.curriculum_stage == 0:  # in the 1000 episodes
    if env.common_step_counter >= (env.max_episode_length * 12) and env.curriculum_stage == 0:  # in the 500 episodes
        env.reward_scales["airtime_variance"] = -10.0
        env.reward_scales["feet_forward"] = -1.0
        env.reward_scales["feet_slide"] = -2.0

        env.curriculum_stage += 1

    elif env.common_step_counter >= (env.max_episode_length * 24) and env.curriculum_stage == 1:  # in the 1000 episodes
        env.reward_scales["airtime_variance"] = -40.0
        env.reward_scales["feet_downward"] = -5.0

        reset_term = env.event_manager.get_term_cfg("reset_command_resample")
        interval_term = env.event_manager.get_term_cfg("interval_command_resample")
        reset_term.params["prob_pos"] = 0.8
        interval_term.params["prob_pos"] = 0.8

        env.curriculum_stage += 1

    # elif env.common_step_counter == (env.max_episode_length * 48):  # in the 2000 episodes
    #     reset_term = env.event_manager.get_term_cfg("reset_command_resample")
    #     interval_term = env.event_manager.get_term_cfg("interval_command_resample")
    #     reset_term.params["prob_pos"] = 0.7
    #     interval_term.params["prob_pos"] = 0.7
    #     # reset_term.params["prob_pos"] = 1.0
    #     # interval_term.params["prob_pos"] = 1.0
    
    # elif env.common_step_counter == (env.max_episode_length * 72):  # in the 3000 episodes
    #     reset_term = env.event_manager.get_term_cfg("reset_command_resample")
    #     interval_term = env.event_manager.get_term_cfg("interval_command_resample")
    #     reset_term.params["prob_pos"] = 0.6
    #     interval_term.params["prob_pos"] = 0.6
    #     # reset_term.params["prob_pos"] = 0.8
    #     # interval_term.params["prob_pos"] = 0.8

    # elif env.common_step_counter == (env.max_episode_length * 96):  # in the 4000 episodes
    #     reset_term = env.event_manager.get_term_cfg("reset_command_resample")
    #     interval_term = env.event_manager.get_term_cfg("interval_command_resample")
    #     reset_term.params["prob_pos"] = 0.5
    #     interval_term.params["prob_pos"] = 0.5

    # elif env.common_step_counter >= (env.max_episode_length * 96) and env.curriculum_stage == 2:  # in the 4000 episodes
    #     env.reward_scales["step_length"] = 8.0
    #     env.curriculum_stage += 1

    elif env.common_step_counter >= (env.max_episode_length * 144) and env.curriculum_stage == 2:  # in the 6000 episodes
    # elif env.common_step_counter >= (env.max_episode_length * 144) and env.curriculum_stage == 3:  # in the 6000 episodes
        env.reward_scales["feet_harmony"] = 1.0
        env.reward_scales["feet_downward"] = -10.0  # -8.0

        env.reward_scales["step_length"] = 7.0  # 6.0

        env.reward_scales["track_heading_yaw"] = 2.0

        reset_term = env.event_manager.get_term_cfg("reset_command_resample")
        interval_term = env.event_manager.get_term_cfg("interval_command_resample")
        reset_term.params["prob_pos"] = 0.6  # 0.7
        interval_term.params["prob_pos"] = 0.6  # 0.7

        env.reward_scales["feet_close"] = -120.0
        env.curriculum_stage += 1

def range_curriculum(
    env: Zbot6SEnvV4,
    env_ids: torch.Tensor,
    # reward_term_name: str = "track_lin_vel_x",
    limit_ranges: tuple[float, float] = (0.0, 0.3),
    limit_yaw_ranges: tuple[float, float] = (-0.3, 0.3),
):

    if len(env.curriculum_vel_reward_buffer) < 20:
        return
    
    # if env.common_step_counter == (env.max_episode_length * 12):
    #     reset_term = env.event_manager.get_term_cfg("reset_command_resample")
    #     interval_term = env.event_manager.get_term_cfg("interval_command_resample")
    #     reset_term.params["dual_sign"] = True
    #     interval_term.params["dual_sign"] = True

    # if (env.common_step_counter > 24 * 1000) & (env.common_step_counter % env.max_episode_length == 0):
    # if env.common_step_counter % (env.max_episode_length * 6) == 0:  # per 250 episodes
    # if env.common_step_counter % (env.max_episode_length * 12) == 0:  # per 500 episodes
    # if env.common_step_counter % (env.max_episode_length * 24) == 0:  # per 1000 episodes
    if (env.common_step_counter >= (env.max_episode_length * 48)) & (env.common_step_counter % (env.max_episode_length * 12) == 0):  # >= 2000 episode per 500 episodes
    # if (env.common_step_counter >= (env.max_episode_length * 72)) & (env.common_step_counter % (env.max_episode_length * 12) == 0):  # >= 3000 episode per 500 episodes
    

        # reset_term = env.cfg.events.reset_command_resample
        # interval_term = env.cfg.events.interval_command_resample
        # print(env.commands)
        # 错误的方法，你会发现log的参数变了但实际resample command时的range还是初始范围
        # [Fix] Obtain the actual configuration object used by the EventManager.
        # Accessing env.cfg.events directly might return a different instance or copy.
        reset_term = env.event_manager.get_term_cfg("reset_command_resample")
        interval_term = env.event_manager.get_term_cfg("interval_command_resample")

        # reward = torch.mean(env._episode_sums["track_lin_vel_x"][env_ids]) / env.max_episode_length_s
        reward = sum(env.curriculum_vel_reward_buffer) / len(env.curriculum_vel_reward_buffer)
        if reward > env.reward_scales["track_lin_vel_x"] * 0.85:
            current_range = reset_term.params["velocity_range"]
            delta_range = torch.tensor([-0.05, 0.05], device=env.device)
            new_range = torch.clamp(
                torch.tensor(current_range, device=env.device) + delta_range,
                limit_ranges[0],
                limit_ranges[1],
            ).tolist()
            reset_term.params["velocity_range"] = tuple(new_range)
            interval_term.params["velocity_range"] = tuple(new_range)

            # if not reset_term.params["dual_sign"]:
            #     # 如果已经能够正向运动，则增加双向范围
            #     reset_term.params["dual_sign"] = True
            #     interval_term.params["dual_sign"] = True

        # if env.common_step_counter > 24 * 1000:  # num_steps_per_env * 1000 i.e. 1000 episodes
        # reward = torch.mean(env._episode_sums["track_heading_yaw"][env_ids]) / env.max_episode_length_s
        reward = sum(env.curriculum_yaw_reward_buffer) / len(env.curriculum_yaw_reward_buffer)
        if reward > env.reward_scales["track_heading_yaw"] * 0.85:
            current_range = reset_term.params["yaw_range"]
            delta_range = torch.tensor([-0.05, 0.05], device=env.device)
            new_range = torch.clamp(
                torch.tensor(current_range, device=env.device) + delta_range,
                limit_yaw_ranges[0],
                limit_yaw_ranges[1],
            ).tolist()
            reset_term.params["yaw_range"] = tuple(new_range)
            interval_term.params["yaw_range"] = tuple(new_range)

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
    
    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # vel_range = None

    # # 1. Reset: 环境重置时生成新指令
    # reset_command_resample = EventTerm(
    #     func=resample_commands,
    #     mode="reset",
    #     params={
    #         "velocity_range": (0.3, 0.3),
    #         # "velocity_range": (0.2, 0.4),
    #         "yaw_range": (-0.0, 0.0),
    #         # "yaw_range": (-0.1, 0.1),
    #         # "yaw_range": (-0.2, 0.2),  # not good
    #         "dual_sign": False,
    #     },
    # )

    # # interval
    # # 2. Interval: 每隔一段时间改变指令
    # interval_command_resample = EventTerm(
    #     func=resample_commands,
    #     mode="interval",
    #     interval_range_s=(3.0, 6.0),  # 每 3~6 秒变一次指令
    #     params={
    #         "velocity_range": (0.3, 0.3),
    #         # "velocity_range": (0.2, 0.4),
    #         "yaw_range": (-0.0, 0.0),
    #         # "yaw_range": (-0.1, 0.1),
    #         # "yaw_range": (-0.2, 0.2),
    #         "dual_sign": False,
    #     },
    # )

    # ××××××××××××××××××××××××正反运动××××××××××××××××××××××××××××××××××
    my_curric = EventTerm(
        func=my_curriculum,
        mode="reset",
    )

    vel_range = EventTerm(
        func=range_curriculum,
        mode="reset",
        params={
            "limit_ranges": (0.0, 0.3),
            "limit_yaw_ranges": (-0.3, 0.3),
        },
    )

    # 1. Reset: 环境重置时生成新指令
    reset_command_resample = EventTerm(
        func=resample_commands,
        mode="reset",
        params={
            "velocity_range": (0.3, 0.3),
            "yaw_range": (-0.1, 0.1),
            "dual_sign": True,
            "offset": 0.0,
            "prob_pos": 1.0,
        },
    )

    # interval
    # 2. Interval: 每隔一段时间改变指令
    interval_command_resample = EventTerm(
        func=resample_commands,
        mode="interval",
        interval_range_s=(3.0, 6.0),  # 每 3~6 秒变一次指令
        params={
            "velocity_range": (0.3, 0.3),
            "yaw_range": (-0.1, 0.1),
            "dual_sign": True,
            "offset": 0.0,
            "prob_pos": 1.0,
        },
    )

    # # ×××××××××××××尝试过不用dual_sign，使得正反limit不一样。但后面设计了offset，又可以继续使用dual_sign了。而且测试发现机器人其实更容易学会倒着走。
    # vel_range = EventTerm(
    #     func=range_curriculum,
    #     mode="reset",
    #     params={
    #         "limit_ranges": (-0.2, 0.3),
    #         "limit_yaw_ranges": (-0.3, 0.3),
    #     },
    # )

    # # 1. Reset: 环境重置时生成新指令
    # reset_command_resample = EventTerm(
    #     func=resample_commands,
    #     mode="reset",
    #     params={
    #         "velocity_range": (0.0, 0.1),
    #         "yaw_range": (-0.1, 0.1),
    #         "dual_sign": False,
    #     },
    # )

    # # interval
    # # 2. Interval: 每隔一段时间改变指令
    # interval_command_resample = EventTerm(
    #     func=resample_commands,
    #     mode="interval",
    #     interval_range_s=(3.0, 6.0),  # 每 3~6 秒变一次指令
    #     params={
    #         "velocity_range": (0.0, 0.1),
    #         "yaw_range": (-0.1, 0.1),
    #         "dual_sign": False,
    #     },
    # )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class Zbot6SEnvV4Cfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4  # 2
    action_space = 6  #24 for sin ;  6 for pd
    observation_space = 24
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
    robot: ArticulationCfg = ZBOT_6S_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=True,
    )

    termination_height = 0.20
    # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    #   working
    reward_cfg = {
        "reward_scales": {
            # --- Tracking ---
            "track_lin_vel_x": 1.0,
            "track_heading_yaw": 1.0,
            # "lin_vel_x": 1.0,

            # --- Penalties ---
            "lin_vel_y": -1.0,         # 侧向漂移惩罚

            "action_rate": -0.1,
            "torques": -2e-4,
            "joint_vel": -0.001,
            "joint_acc": -2.5e-7,

            "feet_downward": -1.0,
            "feet_forward": -0.5,
            "step_length": 5.0,
            # "step_length": 2.0,

            "feet_air_time_biped": 1.0,
            "airtime_variance": -5.0,  # -1.0,
            "feet_slide": -1.0,
            
            "feet_harmony": 0.0,
            "feet_close": -10.0,  # -10.0,
        },
    }

    # 依然只学会了倒着走
    # events.reset_command_resample.params["velocity_range"] = (0.0, 0.1)
    # events.interval_command_resample.params["velocity_range"] = (0.0, 0.1)

    # per 500, 1000 episodes adjust velocity range
    # events.reset_command_resample.params["dual_sign"] = False
    # events.interval_command_resample.params["dual_sign"] = False
    # events.reset_command_resample.params["velocity_range"] = (0.3, 0.3)
    # events.interval_command_resample.params["velocity_range"] = (0.3, 0.3)

    # per 500 episodes adjust velocity range, 0.4 not good
    # events.reset_command_resample.params["dual_sign"] = False
    # events.interval_command_resample.params["dual_sign"] = False
    # events.vel_range.params["limit_ranges"] = (0.0, 0.4)
    # events.reset_command_resample.params["velocity_range"] = (0.4, 0.4)
    # events.interval_command_resample.params["velocity_range"] = (0.4, 0.4)

    # per 1000 episodes adjust velocity range, set dual_sign=True in the 500 episodes 2026-01-09_18-06-50
    # events.reset_command_resample.params["dual_sign"] = False
    # events.interval_command_resample.params["dual_sign"] = False
    # events.reset_command_resample.params["velocity_range"] = (0.3, 0.3)
    # events.interval_command_resample.params["velocity_range"] = (0.3, 0.3)

    # 
    # events.vel_range = None 
    # events.reset_command_resample.params["dual_sign"] = False
    # events.interval_command_resample.params["dual_sign"] = False
    # events.reset_command_resample.params["velocity_range"] = (0.3, 0.3)
    # events.interval_command_resample.params["velocity_range"] = (0.3, 0.3)  # ok
    # events.reset_command_resample.params["velocity_range"] = (-0.3, -0.3)
    # events.interval_command_resample.params["velocity_range"] = (-0.3, -0.3)  #stepl5 not move, stepl3 not move, stepl2 not good, stepl5+feetf0.5 good
    # events.reset_command_resample.params["velocity_range"] = (-0.2, -0.2)
    # events.interval_command_resample.params["velocity_range"] = (-0.2, -0.2)  #stepl5 not move, stepl2 not move
    # events.reset_command_resample.params["velocity_range"] = (-0.4, -0.4)
    # events.interval_command_resample.params["velocity_range"] = (-0.4, -0.4)  #stepl5 not move

    # new curriculum, adjust resample command func  # fine, but prob_pos how many? 0.8正》反，0.7反》正
    # events.vel_range = None 
    # airv-1(default) 1000 1，2500 0.8，later vel_range 正走光抬腿 2026-01-10_23-19-41

    # airtime_variance -10, 2000 1，3000 0.8，later vel_range 2026-01-11_01-14-36 非常诡异的步态
    # airtime_variance -10 not good for 正走

    # maybe 1 0.8 1 0.8? 2026-01-11_03-12-27
    # events.vel_range = None 

    # # test airv-10, feetf 0.0 1000, add feetf-1.0 2000 not good，应该是因为airv-10本身就不太好，后面checkpoint1000看了下
    # # test airv-10, feetf-0.5 no
    # # test airv-5, feetf-1.0 no
    # # test airv-5, feetf-0.5 1000it OK 右脚略大于左脚；2000it 右脚抬得更猛了 2026-01-12_00-05-49
    # events.vel_range = None
    # events.my_curric = None

    # # my_curric不能正常触发!!!!![fix bug]
    # # 1 1000, 0.5 2000 2026-01-12_16-09-19
    # # 1 1000, 0.8 2000 2026-01-12_16-58-09
    # # 增大airv
    # # 1 airv-5 1000, 0.8 airv-10 2000 2026-01-12_18-18-06 better
    # events.vel_range = None 

    # 先只关注正方向，prob_pos=1 data/0112
    # # test airv-5, feetf-0.5 1000it，airv-10, feetf-0.5 2000it，没什么变化，还是右脚抬得更猛
    # # test airv-5, feetf-0.5 1000it，airv-10, feetf-1.0 2000it，没什么变化，还是右脚抬得更猛
    # # test airv-5, feetf-0.5 1000it，airv-20, feetf-1.0 2000it，没什么变化，还是右脚抬得更猛
    # # test airv-5, feetf-0.5 1000it，airv-10, feetf-1.0 feets-2.0 2000it 还是右脚抬得更猛
    # # test airv-5, feetf-0.5 1000it，airv-10, feetf-1.0 feets-5.0 2000it 还是右脚抬得更猛
    # # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it Ok，2000it Ok 2026-01-12_23-24-31
    # # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, 0.8 2000it Ok 2026-01-13_00-59-49 > 2026-01-12_18-18-06 
    # events.vel_range = None 

    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, 0.8 2000it，later vel_range 还是右脚抬得更猛 2026-01-13_02-00-04

    # # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-20, feetd-2.0 2000it 2026-01-13_14-23-17 > 2026-01-12_23-24-31
    # # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-20, feetd-2.0 0.8 2000it 2026-01-13_15-02-11
    # # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-20, feetd-2.0 0.8 2000it，later vel_range 6000it OK 还是右脚抬得更猛 2026-01-13_15-37-46
    # # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-2.0 2000it 也还行 2026-01-13_17-20-58
    # events.vel_range = None 
    # # events.vel_range.params["limit_yaw_ranges"] = (-0.5, 0.5)

    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-2.0 0.8 2000it，later vel_range 6000it 要好些 2026-01-13_18-12-13 
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-3.0 0.8 2000it，later vel_range 8000it 2026-01-14_01-14-28
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 8000it 2026-01-14_03-03-33 > 2026-01-14_01-14-28 <<==========
    # > 6000it 增加抬脚高度和协调性
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 8000it feetha_2 1.0 6000it 还不错
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 10000it feetha_4 1.0 6000it 左右脚还不错，脚不是太平
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 10000it feetha_3 1.0 feetd-8.0 6000it 脚比较平了，但是反走靠得有点近 2026-01-22_17-18-06
    # 增加died3 <0.115bad <12bad <0.13bad <0.14alldeid 不行
    # 增加feet_close 0.11
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 10000it/20000it feetha_3 1.0 feetd-8.0 6000it 2026-01-22_22-00-14 / 2026-01-23_15-20-39 good,就是反走步长较小，朝向能力一般
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 10000it stepl 8.0 4000it feetha_3 1.0 feetd-8.0 6000it 2026-01-23_20-24-50 步子是大了，但感觉容易摔
    # feet_close 0.13 bad >7800it move; 0.12 worse not move; 0.115 ok
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 10000it/20000it feetha_3 1.0 feetd-8.0 6000it OK 2026-01-25_16-24-24 / 2026-01-27_15-00-47 <<==========
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 10000it stepl 7.0 feetha_3 1.0 feetd-8.0 6000it 步子大了些，但感觉容易摔
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 10000it stepl 6.0 feetha_3 1.0 feetd-8.0 6000it 感觉差不多，没必要，tracking yaw还下降了
    # 尝试 增加tracking yaw 和 prob_pos
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 10000it feetha_3 1.0 feetd-8.0 0.7 yaw 2.0 6000it 感觉步子小，反走脚不平 2026-01-27_21-35-05
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 10000it stepl 6.0 feetha_3 1.0 feetd-10.0 0.7 yaw 2.0 6000it 反走脚还是不平
    # 调小自碰撞阈值
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 10000it stepl 6.0 feetha_3 1.0 feetd-10.0 0.6 yaw 2.0 6000it 感觉脚更平了 2026-01-28_20-26-12
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 20000it stepl 7.0 feetha_3 1.0 feetd-10.0 0.6 yaw 2.0 6000it 其它都不错，但是反走脚靠得很近 2026-01-29_15-23-02
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 20000it stepl 7.0 feetha_3 1.0 feetd-10.0 0.6 yaw 2.0 feetc-100 6000it 2026-01-29_20-54-23 OK
    # 如果feet_close一开始就是-100.0，bad 正走无法启动，>10000it只学会了反走；一开始就是-50.0 bad；一开始就是-20.0 bad
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 20000it stepl 7.0 feetha_3 1.0 feetd-10.0 0.6 yaw 2.0 feetc-200 6000it tracking yaw下降，feetfoward也下降
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 20000it stepl 7.0 feetha_3 1.0 feetd-10.0 0.6 yaw 2.0 feetc-150 6000it tracking yaw下降，feetfoward也下降
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 20000it stepl 7.0 feetha_3 1.0 feetd-10.0 0.6 yaw 2.0 feetc-120 6000it 2026-02-01_01-51-36 OK <<==========
    # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-5.0 0.8 2000it，later vel_range 20000it stepl 7.0 feetha_3 1.0 feetd-10.0 0.6 yaw 2.0 feetc-110 6000it tracking yaw下降，feetfoward也下降
    events.vel_range.params["limit_yaw_ranges"] = (-0.5, 0.5)

    # # 修改了下airtime_variance的计算方式，以及steplength no reward for zero command 
    # # bad，对feetdownward的影响较大，改回去了，主要取消clip后airv的参数可能不能这么大
    # # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-2.0 0.8 2000it 改了airv好像确实有点用
    # # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-20, feetd-2.0 0.8 2000it 没什么变化，对feetdownward的影响一样较大
    # events.vel_range = None
    # # test airv-5, feetf-0.5 500it，airv-10, feetf-1.0 feets-2.0 1000it, airv-40, feetd-2.0 0.8 2000it，later vel_range 8000it bad
    # events.vel_range.params["limit_yaw_ranges"] = (-0.5, 0.5)

    # # play script
    # debug_vis=True
    # events.vel_range = None  # disable automatic curriculum
    # events.my_curric = None
    # events.reset_command_resample.params["dual_sign"] = False
    # events.reset_command_resample.params["velocity_range"] = (-0.3, 0.3)
    # events.reset_command_resample.params["yaw_range"] = (-0.3, 0.3) # (-0.1, 0.1)
    # events.interval_command_resample.params["dual_sign"] = False
    # events.interval_command_resample.params["velocity_range"] = (-0.3, 0.3)
    # events.interval_command_resample.params["yaw_range"] = (-0.3, 0.3)  # (-0.3, 0.3)


class Zbot6SEnvV4(DirectRLEnv):
    cfg: Zbot6SEnvV4Cfg

    def __init__(self, cfg: Zbot6SEnvV4Cfg, render_mode: str | None = None, **kwargs):
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
        self._feet_ids, _ = self._contact_sensor.find_bodies("foot.*")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies("base|a.*|b.*")
        self.base_body_idx = self._robot.find_bodies("base")[0]
        self.feet_body_idx = self._robot.find_bodies("foot.*")[0]

        self.z_w = torch.tensor([0, 0, 1], device=self.sim.device, dtype=torch.float32).repeat((self.num_envs, 2, 1))  # torch.Size([4096, 2, 3])
        self.axis_x_feet = torch.tensor(
            [1, 0, 0], device=self.sim.device, dtype=torch.float32
        ).repeat((self.num_envs, 2, 1))
        self.axis_z_feet = torch.tensor(
            [[0, 0, 1], [0, 0, -1]], device=self.sim.device, dtype=torch.float32
        ).repeat((self.num_envs, 1, 1))

        self.feet_contact_forces_last = 15.0 * torch.ones((self.num_envs, 2), device=self.device)
        self.feet_down_pos_last = torch.zeros((self.num_envs, 2, 3), device=self.device)
        self.feet_step_length = torch.zeros((self.num_envs, 2), device=self.device)
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

        axis_z = torch.tensor([0, 0, 1], device=self.sim.device, dtype=torch.float32).repeat((self.num_envs, 1))
        # base body axis z point to world Y  # -------------------------------------------------------------------------------------------------------attention
        self.base_shoulder_w = math_utils.quat_apply(self.base_quat_w, axis_z)  # torch.Size([4096, 3])
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
            > 0.5,
            # > 1.0,
            dim=1,
        )
        died_1 = (self.base_pos_w[:, 2] < self.cfg.termination_height)
        died |= died_1
        # died_2 = (self.heading_err.abs() > 0.5 * torch.pi)
        # died |= died_2
        # died_3 = torch.norm(self.feet_pos_w[:, 0] - self.feet_pos_w[:, 1], dim=-1) < 0.115
        # died_3 = torch.norm(self.feet_pos_w[:, 0, :2] - self.feet_pos_w[:, 1, :2], dim=-1) < 0.115 # 只看水平距离
        # died |= died_3

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
        
        if "track_lin_vel_x" in self._episode_sums:
            self.curriculum_vel_reward_buffer.append(extras["Episode_Reward/track_lin_vel_x"].item())
        if "track_heading_yaw" in self._episode_sums:
            self.curriculum_yaw_reward_buffer.append(extras["Episode_Reward/track_heading_yaw"].item())

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
        
        extras["Curriculum/curriculum_stage"] = self.curriculum_stage
        if self.cfg.events.reset_command_resample is not None:
            # extras["Curriculum/vel_lower_bound"] = self.cfg.events.reset_command_resample.params["velocity_range"][0]
            # extras["Curriculum/vel_upper_bound"] = self.cfg.events.reset_command_resample.params["velocity_range"][1]
            # extras["Curriculum/yaw_bound"] = self.cfg.events.reset_command_resample.params["yaw_range"][0]
            # 因为range_curriculum修复后逻辑变了，所以这里也从EventManager里取
            # [Fix] Log the ACTUAL active parameters from EventManager, not the static self.cfg
            reset_term_cfg = self.event_manager.get_term_cfg("reset_command_resample")
            extras["Curriculum/vel_lower_bound"] = reset_term_cfg.params["velocity_range"][0]
            extras["Curriculum/vel_upper_bound"] = reset_term_cfg.params["velocity_range"][1]
            extras["Curriculum/yaw_bound"] = reset_term_cfg.params["yaw_range"][0]
        
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

        self.feet_contact_forces_last[env_ids] = 15.0 * torch.ones((len(env_ids), 2), device=self.device)
        self.feet_down_pos_last[env_ids] = (self._robot.data.body_link_pos_w[:, self.feet_body_idx])[env_ids]
        self.feet_step_length[env_ids] = torch.zeros((len(env_ids), 2), device=self.device)
        self.feet_force_sum[env_ids] = 0.0
        self.heading_err_sum[env_ids] = 0.0

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
        
        # self.feet_step_length[feet_down_idx] = feet_step_length_w[feet_down_idx]
        # self.feet_step_length[feet_down_idx] = feet_step_length_w[feet_down_idx].abs()  # 这样不行，不是正反迈步都奖励；而是速度指令为正时，迈正步奖励，负时迈反步奖励。
        # self.feet_step_length[feet_down_idx] = feet_step_length_w[feet_down_idx] * torch.sign(self.commands[:, 0:1]).repeat(1, 2)[feet_down_idx]
        self.feet_step_length[feet_down_idx] = feet_step_length_w[feet_down_idx] * torch.sign(self.commands[:, 0:1]).expand(-1, 2)[feet_down_idx]
        # 推荐使用expand，零内存复制，修改视图(View)步长(Stride)，效率高。
        # 创建视图只需要计算新的形状和步长，耗时是O(1)，它不会分配新的内存来存储重复的数据。
        # 常见的视图操作包括：view(/reshape)、squeeze、unsqueeze、expand、切片操作如tensor[:, 0:1]、transpose、permute等。

        rew_feet_step_length = torch.min(self.feet_step_length, dim=-1)[0]

        self.feet_step_length *= 0.99  # decay old step lengths
        
        self.feet_down_pos_last[feet_down_idx, :] = self.feet_pos_w[feet_down_idx, :]
        self.feet_contact_forces_last[:] = self.feet_contact_forces[:]  # refresh last
        
        rew = torch.tanh(15.0*rew_feet_step_length)
        # arctan和tanh是两种完全不同的数学函数，前者输出(-π/2, π/2)，后者输出(-1, 1)。tanh(1)≈0.7616，tanh(2)≈0.96，tanh(5)≈0.9999
        
        # no reward for zero command
        # rew *= self.commands[:, 0].abs() >= 0.05
        return rew

    def _reward_airtime_variance(self):
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        last_contact_time = self._contact_sensor.data.last_contact_time[:, self._feet_ids]
        return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
            torch.clip(last_contact_time, max=0.5), dim=1
        )
        # return torch.var(last_air_time, dim=1) + torch.var(last_contact_time, dim=1)

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
    
    def _reward_feet_air_time_biped(self):
        """Reward long steps taken by the feet for bipeds.

        This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
        a time in the air.
        """
        # compute the reward
        air_time = self._contact_sensor.data.current_air_time[:, self._feet_ids]
        contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_ids]
        in_contact = contact_time > 0.0
        in_mode_time = torch.where(in_contact, contact_time, air_time)
        single_stance = torch.sum(in_contact.int(), dim=1) == 1
        reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
        reward = torch.clamp(reward, max=2.0)
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
    
    def _reward_feet_harmony(self):
        feet_air_times = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        airtime_sum = torch.sum(feet_air_times, dim=-1)
        airtime_balance = torch.abs(feet_air_times[:, 0] - feet_air_times[:, 1])
        # return airtime_sum - 2.0 * airtime_balance
        # return airtime_sum - 4.0 * airtime_balance
        return airtime_sum - 3.0 * airtime_balance
    
    def _reward_feet_close(self):
        feet_pos_xy = self.feet_pos_w[:, :, :2]
        feet_dist = torch.norm(feet_pos_xy[:, 0, :] - feet_pos_xy[:, 1, :], dim=-1)
        # rew = torch.clamp(0.11 - feet_dist, min=0.0)
        rew = torch.clamp(0.115 - feet_dist, min=0.0)
        return rew

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
    
    # def _reward_feet_gait(
    #     self,
    #     period: float = 1.0,
    #     offset: list[float] = [0.0, 0.5, 0.0, 0.5],  # 脚部ID顺序是[FL, RL, RR, FR]（左前，左后Rear，右后，右前）
    #     threshold: float = 0.55,
    # ):
    #     is_contact = self._contact_sensor.data.current_contact_time[:, self._feet_ids] > 0

    #     global_phase = ((self.episode_length_buf * self.step_dt) % period / period).unsqueeze(1)
    #     phases = []
    #     for offset_ in offset:
    #         phase = (global_phase + offset_) % 1.0
    #         phases.append(phase)
    #     leg_phase = torch.cat(phases, dim=-1)

    #     reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    #     for i in range(len(self._feet_ids)):
    #         is_stance = leg_phase[:, i] < threshold
    #         reward += ~(is_stance ^ is_contact[:, i])
        
    #     # logging
    #     # if self.common_step_counter % 800 == 0:
    #     #     print(f"is_stanced: {leg_phase[:2] < threshold}")
    #     #     print(f"is_contact: {is_contact[:2]}")
        
    #     return reward
    
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
        # base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w = self._robot.data.body_link_pos_w[:, self.base_body_idx].squeeze(1)
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

