# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Feet rewards.
"""

def init_my_data(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None):
    # Initialize foot-related data in the env
    env.feet_force_sum = torch.zeros(env.num_envs, device=env.device)
    env.feet_step_length = torch.zeros((env.num_envs, 2), device=env.device)
    env.feet_contact_forces_last = torch.zeros((env.num_envs, 2), device=env.device)
    env.feet_down_pos_last = torch.zeros((env.num_envs, 2, 3), device=env.device)

def reset_my_data(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    # Reset foot-related data for given env ids
    asset = env.scene[asset_cfg.name]
    env.feet_force_sum[env_ids] = 0.0
    env.feet_step_length[env_ids] = 0.0
    env.feet_contact_forces_last[env_ids] = 0.0
    env.feet_down_pos_last[env_ids] = (asset.data.body_link_pos_w[:, asset_cfg.body_ids])[env_ids]

def foot_step_length(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    sensor_cfg: SceneEntityCfg, 
    command_name: str = None
) -> torch.Tensor:
    # Reward foot step length
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # feet_just_down
    feet_contact_forces = torch.mean(
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2],
            dim=1,
        ).squeeze()
    feet_down_idx = torch.logical_and(
        (feet_contact_forces > 10.0),
        (env.feet_contact_forces_last < 10.0),
    )  # "掩码索引"（boolean indexing）

    y_w = torch.tensor([0, 1, 0], device=env.device, dtype=torch.float32).repeat((env.num_envs, 1))
    base_shoulder_w = quat_apply(asset.data.root_quat_w, y_w)  # torch.Size([4096, 3])
    base_dir_forward_w = torch.cross(asset.data.GRAVITY_VEC_W, base_shoulder_w, dim=-1)  # torch.Size([4096, 3])
    
    # 如果提供了command_name，则使用命令方向作为前进方向
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        # 获取XY平面的命令方向
        # command_dir_w = torch.stack([
        #     command[:, 0],
        #     command[:, 1],
        #     torch.zeros_like(command[:, 0])
        # ], dim=1)
        # # 只有有效命令时才使用命令方向
        # has_valid_command = torch.norm(command[:, :2], dim=1) > 1e-3  # threshold
        # base_dir_forward_w = torch.where(
        #     has_valid_command.unsqueeze(1),
        #     command_dir_w,
        #     base_dir_forward_w
        # )
        base_dir_forward_w = command
        base_dir_forward_w[:, 2] = 0.0
    
    # Normalize the forward direction vector
    base_dir_forward_w = base_dir_forward_w / (torch.norm(base_dir_forward_w, dim=1, keepdim=True) + 1e-6)

    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    feet_step_vec_w = feet_pos_w - env.feet_down_pos_last
    feet_step_length_w = torch.abs(torch.sum(feet_step_vec_w * base_dir_forward_w.unsqueeze(1), dim=-1))  # 法一
    # feet_step_length_w = (feet_step_vec_w @ base_dir_forward_w.unsqueeze(-1)).squeeze(-1)  # 法二
    # feet_step_length_w = torch.einsum(
    #     'bij,bj->bi',
    #     feet_step_vec_w,
    #     base_dir_forward_w
    # )  # 法三
    env.feet_step_length[feet_down_idx] = feet_step_length_w[feet_down_idx]

    rew_feet_step_length = torch.min(env.feet_step_length, dim=-1)[0]

    env.feet_down_pos_last[feet_down_idx, :] = feet_pos_w[feet_down_idx, :]
    env.feet_contact_forces_last[:] = feet_contact_forces[:]  # refresh last
    return torch.tanh(15.0*rew_feet_step_length)

def foot_downward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    z_w = torch.tensor([0, 0, 1], device=env.device, dtype=torch.float32).repeat((env.num_envs, 2, 1))  # torch.Size([4096, 2, 3])
    axis_z_feet = torch.tensor(
            [[0, 1, 0], [0, -1, 0]], device=env.device, dtype=torch.float32
        ).repeat((env.num_envs, 1, 1))
    feet_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    feet_z_w = quat_apply(feet_quat_w, axis_z_feet)  # torch.Size([4096, 2, 3])
    feet_downward = torch.sum(
            torch.norm(
                (feet_z_w - z_w),
                dim=-1,
            ),
            dim=-1,
        )
    return feet_downward

def foot_forward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    y_w = torch.tensor([0, 1, 0], device=env.device, dtype=torch.float32).repeat((env.num_envs, 1))
    base_shoulder_w = quat_apply(asset.data.root_quat_w, y_w)  # torch.Size([4096, 3])
    base_dir_forward_w = torch.cross(asset.data.GRAVITY_VEC_W, base_shoulder_w, dim=-1)  # torch.Size([4096, 3])
    
    axis_x_feet = torch.tensor(
            [1, 0, 0], device=env.sim.device, dtype=torch.float32
        ).repeat((env.num_envs, 2, 1))
    feet_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    feet_x_w = quat_apply(feet_quat_w, axis_x_feet)  # torch.Size([4096, 2, 3])
    feet_forward = torch.sum(
        torch.norm(
            feet_x_w - base_dir_forward_w.unsqueeze(1),
            dim=-1,
        ),
        dim=-1,
    )
    return feet_forward

def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    #摆动腿的error越小越好，而支撑腿的error大，但速度为0
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        # reward *= cmd_norm > 0.1
        reward *= cmd_norm > 0.05

    return reward

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )

def air_time_balance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    airtime_balance = torch.abs(last_air_time[:, 0] - last_air_time[:, 1])
    return airtime_balance

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def base_vel_forward(env: ManagerBasedRLEnv, which_forward: int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    
    y_w = torch.tensor([0, 1 * which_forward, 0], device=env.device, dtype=torch.float32).repeat((env.num_envs, 1))
    base_shoulder_w = quat_apply(asset.data.root_quat_w, y_w)  # torch.Size([4096, 3])
    base_dir_forward_w = torch.cross(asset.data.GRAVITY_VEC_W, base_shoulder_w, dim=-1)  # torch.Size([4096, 3])

    base_lin_vel_forward_w = torch.sum(asset.data.root_link_lin_vel_w * base_dir_forward_w, dim=-1)  # 法一 torch.Size([4096])
    # base_lin_vel_forward_w = torch.tanh(base_lin_vel_forward_w)
    return base_lin_vel_forward_w

def feet_force_pattern(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_contact_forces = torch.mean(
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2],
            dim=1,
        ).squeeze()
    feet_force_diff = (feet_contact_forces[:, 1] - feet_contact_forces[:, 0]) * torch.sign(env.feet_force_sum)
    env.feet_force_sum += 0.001 * (
        feet_contact_forces[:, 0] - feet_contact_forces[:, 1]
    )
    return 0.5 * feet_force_diff - 0.1 * torch.abs(env.feet_force_sum)

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_com_lin_vel_w[:, :3])
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_link_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_w[:, 2])
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_link_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)
