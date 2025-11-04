# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg 
from isaaclab.utils import configclass

from zbot.assets import ZBOT_6S_CFG
import numpy as np
import math

@configclass
class ZbotDirectEnvCfgV2(DirectRLEnvCfg):
    # robot
    robot: ArticulationCfg = ZBOT_6S_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=5,
        update_period=0.0,
        track_air_time=True,
        track_pose=True,
    )

    # env
    episode_length_s = 20.0
    decimation = 4  # 2
    action_space = 6  #24 for sin ;  6 for pd
    observation_space = 23
    state_space = 0
    termination_height = 0.22

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
    
    # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    #   train reward for just stepping walk base 2000 step0
    reward_cfg = {
        "reward_scales": {
            "base_vel_forward": 1.0,
            "feet_downward": -1.0,
            "feet_forward": -1.0,  # -0.5,
            "base_heading_x": -1.0,
            # "base_heading_x_sum": -1.0,
            "feet_force_diff": 0.5,
            "feet_force_sum": -0.1,
            "base_pos_y_err": -1.0,
            # "feet_slide": -10.0,
            # "airtime_sum": 10.0,
        },
    }

    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # #   train reward 2000 step1
    # reward_cfg = {
    #     "reward_scales": {
    #         "base_vel_forward": 1.0,
    #         "feet_downward": -1.0,
    #         "feet_forward": -1.0,
    #         "base_heading_x": -1.0,
    #         "base_heading_x_sum": -3.0,
    #         "step_length": 5.0,
    #         "airtime_balance": -15.0,
    #         "action_rate": -0.1,
    #         "torques": -0.002,
    #         "feet_slide": -10.0,
    #         "base_pos_y_err": -1.0,
    #     },
    # }

    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # #   train reward 2000 step2
    # reward_cfg = {
    #     "reward_scales": {
    #         "base_vel_forward": 1.0,
    #         "feet_downward": -1.0,
    #         "feet_forward": -1.0,
    #         "base_heading_x": -1.0,
    #         "base_heading_x_sum": -5.0,
    #         "step_length": 5.0,
    #         "airtime_balance": -15.0,
    #         "action_rate": -0.1,
    #         "torques": -0.002,
    #         "feet_slide": -10.0,
    #         "base_pos_y_err": -1.5,
    #     },
    # }

    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # #   train reward 2000 step3
    # reward_cfg = {
    #     "reward_scales": {
    #         "base_vel_forward": 1.0,
    #         "feet_downward": -1.0,
    #         "feet_forward": -1.0,
    #         "base_heading_x": -1.0,
    #         "base_heading_x_sum": -5.0,
    #         "step_length": 5.0,
    #         "airtime_balance": -15.0,
    #         "action_rate": -0.1,
    #         "torques": -0.002,
    #         "feet_slide": -10.0,
    #         "base_pos_y_err": -3.0,
    #     },
    # }

    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # #   train reward 2000 step4
    # reward_cfg = {
    #     "reward_scales": {
    #         "base_vel_forward": 1.0,
    #         "feet_downward": -1.5,
    #         "feet_forward": -1.0,
    #         "base_heading_x": -1.0,
    #         "base_heading_x_sum": -5.0,
    #         "step_length": 5.0,
    #         "airtime_balance": -15.0,
    #         "action_rate": -0.1,
    #         "torques": -0.002,
    #         "feet_slide": -10.0,
    #         "base_pos_y_err": -1.5,
    #         "base_pos_y_err_sum": -1.5,
    #     },
    # }

class ZbotDirectEnvV2(DirectRLEnv):
    cfg: ZbotDirectEnvCfgV2

    def __init__(self, cfg: ZbotDirectEnvCfgV2, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

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

        # Get specific body indices
        self._feet_ids, _ = self._contact_sensor.find_bodies("foot.*")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies("base|a.*|b.*")
        self.base_body_idx = self._robot.find_bodies("base")[0]
        self.feet_body_idx = self._robot.find_bodies("foot.*")[0]
        self.feet_contact_forces_last = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float
        )

        self.feet_down_pos_last = torch.zeros((self.num_envs, 2, 3), device=self.device)
        self.feet_step_length = torch.zeros((self.num_envs, 2), device=self.device)
        self.feet_air_times = torch.zeros((self.num_envs, 2), device=self.device)
        self.feet_force_sum = torch.zeros(self.num_envs, device=self.device)
        self.base_heading_x_sum = torch.zeros(self.num_envs, device=self.device)
        self.base_pos_y_err_sum = torch.zeros(self.num_envs, device=self.device)

        self.joint_speed_limit = 0.2 + 1.8 * torch.rand(self.num_envs, 1, device=self.device)

        self.p_delta = torch.zeros_like(self._robot.data.default_joint_pos)
        self.reward_scales = cfg.reward_cfg["reward_scales"]
        # print('**** self.reward_scales = ', cfg.reward_cfg, cfg.reward_cfg["reward_scales"])
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self._episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.step_dt
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

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        self.base_pos_w = self._robot.data.body_link_pos_w[:, self.base_body_idx].squeeze()
        self.base_quat_w = self._robot.data.body_link_quat_w[:, self.base_body_idx].squeeze()
        self.feet_quat_w = self._robot.data.body_link_quat_w[:, self.feet_body_idx]
        self.feet_pos_w = self._robot.data.body_link_pos_w[:, self.feet_body_idx]

        axis_z = torch.tensor([0, 0, 1], device=self.sim.device, dtype=torch.float32).repeat((self.num_envs, 1))
        # base body axis z point to world Y
        self.base_shoulder_w = math_utils.quat_apply(self.base_quat_w, axis_z)  # torch.Size([4096, 3])
        self.base_dir_forward_w = torch.cross(self._robot.data.GRAVITY_VEC_W, self.base_shoulder_w)  # torch.Size([4096, 3])
        self.base_heading_x_err = -self.base_dir_forward_w[..., 1]  # torch.Size([4096])
        
        self.base_lin_vel_w = self._robot.data.body_com_lin_vel_w[:, self.base_body_idx, :].squeeze()  # torch.Size([4096, 3])
        self.base_lin_vel_forward_w = torch.sum(self.base_lin_vel_w * self.base_dir_forward_w, dim=-1)  # 法一 torch.Size([4096])
        # self.base_lin_vel_forward_w = torch.einsum('ij,ij->i', self.base_lin_vel_w, self.base_dir_forward_w)  # 法二
        # self.base_lin_vel_forward_w = (self.base_lin_vel_w @ self.base_dir_forward_w.unsqueeze(-1)).squeeze(-1)  # 法三

        self.z_w = torch.tensor([0, 0, 1], device=self.sim.device, dtype=torch.float32).repeat((self.num_envs, 2, 1))  # torch.Size([4096, 2, 3])
        # axis_x_feet = torch.tensor(
        #     [[1, 0, 0], [-0.7071, 0.0, -0.7071]], device=self.sim.device, dtype=torch.float32
        # ).repeat((self.num_envs, 1, 1))
        # axis_z_feet = torch.tensor(
        #     [[0, 0, 1], [0.7071, 0.0, -0.7071]], device=self.sim.device, dtype=torch.float32
        # ).repeat((self.num_envs, 1, 1))
        axis_x_feet = torch.tensor(
            [1, 0, 0], device=self.sim.device, dtype=torch.float32
        ).repeat((self.num_envs, 2, 1))
        axis_z_feet = torch.tensor(
            [[0, 0, 1], [0, 0, -1]], device=self.sim.device, dtype=torch.float32
        ).repeat((self.num_envs, 1, 1))
        self.feet_z_w = math_utils.quat_apply(self.feet_quat_w, axis_z_feet)  # torch.Size([4096, 2, 3])
        self.feet_x_w = math_utils.quat_apply(self.feet_quat_w, axis_x_feet)  # torch.Size([4096, 2, 3])
        # print(self.feet_z_w[0])
        # print(self.feet_x_w[0])
        # ----------------------------------------------------------------------关键是目前usd的feet1坐标系不在底面
        # ----------------------------------------------------------------------还有注意b(feet1)的坐标系是Xform并没有旋转
        
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    # self.feet_quat_w[:, [0, ], 3], # 2 * torch.atan2(z, w)
                    self.base_quat_w,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                    self.joint_speed_limit,
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
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self._episode_sums[name] += rew
        
        terminated_ids = self.reset_terminated.nonzero(as_tuple=False).squeeze(-1)
        reward[terminated_ids] -= 20.0  # vip to Penalize rew when reset

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        self.feet_contact_forces = torch.mean(
            self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids, 2],
            dim=1,
        ).squeeze()
        self.feet_air_times = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        self.feet_contact_times = self._contact_sensor.data.current_contact_time[
            :, self._feet_ids
        ]
        # self.feet_contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids, 2].squeeze()
        died = torch.any(
            torch.max(
                torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1
            )[0]
            > 1.0,
            dim=1,
        )
        
        died_1 = (self.base_pos_w[:, 2] < self.cfg.termination_height)
        self.base_pos_y_err = self.base_pos_w[:,1] - self._terrain.env_origins[:,1]
        died_6 = (self.base_pos_y_err.abs() > 0.5)
        died |= died_1
        died |= died_6

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

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.p_delta[env_ids] = 0.0
        self.feet_down_pos_last[env_ids] = (self._robot.data.body_link_pos_w[:, self.feet_body_idx])[env_ids]
        self.feet_force_sum[env_ids] = 0.0
        self.base_heading_x_sum[env_ids] = 0.0
        self.base_pos_y_err_sum[env_ids] = 0.0

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
        extras["Episode_Termination/body_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"].update(extras)

    def _reward_feet_forward(self):
        feet_forward = torch.sum(
            torch.norm(
                self.feet_x_w - self.base_dir_forward_w.unsqueeze(1),
                dim=-1,
            ),
            dim=-1,
        )
        return feet_forward
    
    def _reward_feet_downward(self):
        feet_downward = torch.sum(
            torch.norm(
                (self.feet_z_w - self.z_w),
                dim=-1,
            ),
            dim=-1,
        )
        return feet_downward

    def _reward_base_heading_x(self):
        return torch.abs(self.base_heading_x_err)
    
    def _reward_base_heading_x_sum(self):
        self.base_heading_x_sum += 0.01 * (self.base_heading_x_err)
        self.base_heading_x_sum = torch.clip(self.base_heading_x_sum, -1, 1)
        return torch.abs(self.base_heading_x_sum)

    def _reward_base_vel_forward(self):
        base_vel_forward = torch.tanh(10.0 * self.base_lin_vel_forward_w / self.joint_speed_limit.squeeze())
        return base_vel_forward

    def _reward_base_pos_y_err(self):
        y_err = torch.abs(self.feet_pos_w[:,0, 1] + self.feet_pos_w[:,1, 1] - 2.0*self._terrain.env_origins[:,1]) + torch.abs(self.base_pos_w[:,1] - self._terrain.env_origins[:,1])
        return y_err

    def _reward_base_pos_y_err_sum(self):
        self.base_pos_y_err_sum += 0.01 * (self.base_pos_y_err)
        self.base_pos_y_err_sum = torch.clip(self.base_pos_y_err_sum, -1, 1)
        return torch.abs(self.base_pos_y_err_sum)

    def _reward_action_rate(self):
        # action rate
        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )
        return action_rate

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

    def _reward_airtime_balance(self):
        airtime_balance = torch.abs(
            self.feet_air_times[:, 0] - self.feet_air_times[:, 1]
        )
        return airtime_balance

    def _reward_airtime_sum(self):
        airtime_sum = torch.tanh(torch.sum(self.feet_air_times, dim=-1))
        return airtime_sum
    
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
    
    def _reward_torques(self):
        # Penalize joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        return joint_torques
    
    def _reward_feet_force_diff(self):
        feet_force_diff = (self.feet_contact_forces[:, 1] - self.feet_contact_forces[:, 0]) * torch.sign(self.feet_force_sum)
        return feet_force_diff
    
    def _reward_feet_force_sum(self):
        self.feet_force_sum += 0.001 * (
            self.feet_contact_forces[:, 0] - self.feet_contact_forces[:, 1]
        )
        return torch.abs(self.feet_force_sum)
    
