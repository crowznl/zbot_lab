# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# add reset_root_state_uniform function
# 调整了更新状态的代码位置，因为↓
# _get_dones() > _get_rewards() > _reset_idx() > _get_observations()

from __future__ import annotations

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

import csv
import os

@configclass
class ZbotDirectEnvCfgV2V1(DirectRLEnvCfg):
    # robot
    robot: ArticulationCfg = ZBOT_6S_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # env
    episode_length_s = 20.0
    decimation = 4  # 2
    action_space = 6  #24 for sin ;  6 for pd
    observation_space = 23
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
    
    termination_height = 0.20
    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # #   train reward for just stepping walk base 2000 step0
    # reward_cfg = {
    #     "reward_scales": {
    #         "base_vel_forward": 1.0,
    #         "feet_downward": -1.0,
    #         "feet_forward": -1.0,
    #         "heading_err": -1.0,  # -0.5,
    #         "heading_err_sum": -1.0,
    #         "feet_force_diff": 0.5,
    #         "feet_force_sum": -0.1,
    #         # "feet_slide": -10.0,
    #         # "airtime_sum": 10.0,
    #     },
    # }
    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # #   train reward 2000 step1
    # reward_cfg = {
    #     "reward_scales": {
    #         "base_vel_forward": 1.0,
    #         "feet_downward": -1.0,
    #         "feet_forward": -1.0,
    #         "heading_err": -1.0,
    #         "heading_err_sum": -3.0,
    #         "step_length": 5.0,
    #         "airtime_balance": -15.0,
    #         "action_rate": -0.1,
    #         "torques": -0.002,
    #         "feet_slide": -10.0,
    #     },
    # }
    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # #   train reward 2000 step2
    # reward_cfg = {
    #     "reward_scales": {
    #         "base_vel_forward": 1.0,
    #         "feet_downward": -2.0,
    #         "feet_forward": -1.0,
    #         "heading_err": -1.0,
    #         "heading_err_sum": -3.0,
    #         "step_length": 5.0,
    #         "airtime_sum": 2.0,
    #         "airtime_balance": -15.0,
    #         "action_rate": -0.1,
    #         "torques": -0.002,
    #         "feet_slide": -10.0,
    #     },
    # }
    # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    #   train reward 2000 step3
    reward_cfg = {
        "reward_scales": {
            "base_vel_forward": 1.0,
            "feet_downward": -2.0,
            "feet_forward": -1.0,
            "heading_err": -1.0,
            "heading_err_sum": -5.0,
            "step_length": 5.0,
            "airtime_sum": 3.0,
            "airtime_balance": -15.0,
            "action_rate": -0.1,
            "torques": -0.002,
            "feet_slide": -10.0,
        },
    }

    # # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # #   train reward for just stepping
    # reward_cfg = {
    #     "reward_scales": {
    #         # "base_vel_forward": 1.0,
    #         "feet_downward": -1.0,
    #         "feet_forward": -1.0,  # -0.5,
    #         # "base_heading_x": -1.0,
    #         # "base_heading_x_sum": -1.0,
    #         "feet_height": 10.0,
    #         "feet_gait": 2.0,
    #         # "feet_force_diff": 1.0,
    #         # "feet_force_sum": -0.1,
    #         # "feet_slide": -10.0,
    #         # "feet_air_time_biped": 10.0,
    #         "airtime_balance": -2.0,
    #         "airtime_sum": 2.0,
    #         # "action_rate": -0.1,
    #         # "torques": -0.002,
    #     },
    # }


class ZbotDirectEnvV2V1(DirectRLEnv):
    cfg: ZbotDirectEnvCfgV2V1

    def __init__(self, cfg: ZbotDirectEnvCfgV2V1, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.heading_yaw = torch.tensor([0.0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs,))
        self.yaw_commands = self.heading_yaw.clone()  # torch.Size([4096])
        
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
        self.joint_speed_limit = 0.2 + 1.8 * torch.rand(self.num_envs, 1, device=self.device)
        # self.joint_speed_limit = 1.0 * torch.ones((self.num_envs, 1), device=self.device)  # play

        # Get specific body indices
        self._feet_ids, _ = self._contact_sensor.find_bodies("foot.*")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies("base|a.*|b.*")
        self.base_body_idx = self._robot.find_bodies("base")[0]
        self.feet_body_idx = self._robot.find_bodies("foot.*")[0]
        
        self.z_w = torch.tensor([0, 0, 1], device=self.sim.device, dtype=torch.float32).repeat((self.num_envs, 2, 1))  # torch.Size([4096, 2, 3])
        # axis_x_feet = torch.tensor(
        #     [[1, 0, 0], [-0.7071, 0.0, -0.7071]], device=self.sim.device, dtype=torch.float32
        # ).repeat((self.num_envs, 1, 1))
        # axis_z_feet = torch.tensor(
        #     [[0, 0, 1], [0.7071, 0.0, -0.7071]], device=self.sim.device, dtype=torch.float32
        # ).repeat((self.num_envs, 1, 1))
        self.axis_x_feet = torch.tensor(
            [1, 0, 0], device=self.sim.device, dtype=torch.float32
        ).repeat((self.num_envs, 2, 1))
        self.axis_z_feet = torch.tensor(
            [[0, 0, 1], [0, 0, -1]], device=self.sim.device, dtype=torch.float32
        ).repeat((self.num_envs, 1, 1))
        # ----------------------------------------------------------------------关键是目前usd的feet1坐标系不在底面
        # ----------------------------------------------------------------------还有注意b(feet1)的坐标系是Xform并没有旋转

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

    def _compute_intermediate_values(self):
        self.base_pos_w = self._robot.data.body_link_pos_w[:, self.base_body_idx].squeeze()
        self.base_quat_w = self._robot.data.body_link_quat_w[:, self.base_body_idx].squeeze()
        self.feet_quat_w = self._robot.data.body_link_quat_w[:, self.feet_body_idx]
        self.feet_pos_w = self._robot.data.body_link_pos_w[:, self.feet_body_idx]
        # print(self.base_pos_w[:4, 2])  # tensor([0.2545, 0.2545, 0.2545, 0.2545], device='cuda:0')
        # print(self.base_quat_w[:2])  # [ 0.6003, -0.6003, -0.3735, -0.3739]
        # print(self.feet_pos_w[:2, :, 2])  # tensor([[0.0000e+00, 5.3035e-02],[1.8626e-09, 5.3035e-02]], device='cuda:0')

        axis_z = torch.tensor([0, 0, 1], device=self.sim.device, dtype=torch.float32).repeat((self.num_envs, 1))
        # base body axis z point to world Y
        self.base_shoulder_w = math_utils.quat_apply(self.base_quat_w, axis_z)  # torch.Size([4096, 3])
        self.base_dir_forward_w = torch.cross(self._robot.data.GRAVITY_VEC_W, self.base_shoulder_w, dim=-1)  # torch.Size([4096, 3])
        self.current_yaw = torch.atan2(self.base_dir_forward_w[:, 1], self.base_dir_forward_w[:, 0])  # torch.Size([4096])
        self.heading_err = self.current_yaw - self.heading_yaw  # torch.Size([4096])

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
        
        self.base_quat_w = self._robot.data.body_link_quat_w[:, self.base_body_idx].squeeze()  # reset后，可能需要更新下缓存，要不就直接获取
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.base_quat_w,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                    self.joint_speed_limit,
                    # self.yaw_commands.unsqueeze(-1),
                )
                if tensor is not None
            ],
            dim=-1,
        )
        # print(obs.shape)
        # self.save_tensor_to_csv(obs, csv_file_path="logs/csv/v2_1_obs_env0.csv")
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
        died_1 = (self.base_pos_w[:, 2] < self.cfg.termination_height)
        died_2 = (self.heading_err.abs() > 0.5 * torch.pi)
        died |= died_1
        died |= died_2

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

        # Sample new yaw commands
        # self.yaw_commands[env_ids] = torch.zeros_like(self.yaw_commands[env_ids]).uniform_(-1.0 * torch.pi, 1.0 * torch.pi)

        # Reset robot state
        # default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._reset_root_state_uniform(env_ids, self.yaw_commands)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.feet_contact_forces_last[env_ids] = 15.0 * torch.ones((len(env_ids), 2), device=self.device)
        self.feet_down_pos_last[env_ids] = (self._robot.data.body_link_pos_w[:, self.feet_body_idx])[env_ids]
        self.feet_step_length[env_ids] = torch.zeros((len(env_ids), 2), device=self.device)
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
        self.extras["log"].update(extras)

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

    def _reward_heading_err(self):
        return torch.abs(self.heading_err)
    
    def _reward_heading_err_sum(self):
        self.heading_err_sum += 0.01 * self.heading_err
        self.heading_err_sum = torch.clamp(self.heading_err_sum, -0.5*torch.pi, 0.5*torch.pi)
        return torch.abs(self.heading_err_sum)

    def _reward_base_vel_forward(self):
        base_vel_forward = torch.tanh(10.0 * self.base_lin_vel_forward_w / self.joint_speed_limit.squeeze())
        return base_vel_forward

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
        feet_air_times = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        airtime_balance = torch.abs(
            feet_air_times[:, 0] - feet_air_times[:, 1]
        )
        return airtime_balance

    def _reward_airtime_sum(self):
        feet_air_times = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        # airtime_sum = torch.tanh(torch.sum(feet_air_times, dim=-1))
        airtime_sum = torch.clamp(torch.sum(feet_air_times, dim=-1), max=2.0)
        return airtime_sum
    
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
        joint_vel = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
        return joint_vel

    def _reward_joint_acc(self):
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        return joint_accel

    def _reward_feet_force_diff(self):
        feet_force_diff = (self.feet_contact_forces[:, 1] - self.feet_contact_forces[:, 0]) * torch.sign(self.feet_force_sum)
        return feet_force_diff
    
    def _reward_feet_force_sum(self):
        self.feet_force_sum += 0.001 * (
            self.feet_contact_forces[:, 0] - self.feet_contact_forces[:, 1]
        )
        return torch.abs(self.feet_force_sum)
    
    def _reward_feet_height(self):
        feet_heights = self.feet_pos_w[:, :, 2]
        feet_heights[:,1] -= 0.053
        feet_height_reward = torch.sum(feet_heights, dim=1)
        return feet_height_reward
    
    def _reward_feet_gait(
        self,
        period: float = 2.0,
        offset: list[float] = [0.0, 0.5],
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

        return reward

    def _reward_shape_symmetry(self):
        jp = self.p_delta
        symmetry_err = (
            torch.abs(jp[:, 0] + jp[:, 5])
            + torch.abs(jp[:, 1] + jp[:, 4])
            + torch.abs(jp[:, 2] + jp[:, 3])
        )
        return symmetry_err

    def _reset_root_state_uniform(
        self,
        env_ids: torch.Tensor,
        yaw_commands: torch.Tensor,
        pose_range: dict[str, tuple[float, float]] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        velocity_range: dict[str, tuple[float, float]] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
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
        root_states = self._robot.data.default_root_state[env_ids].clone()

        # poses
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)

        self.heading_yaw[env_ids] = rand_samples[:, 5] + yaw_commands[env_ids]

        positions = root_states[:, 0:3] + self._terrain.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # orientations = math_utils.random_yaw_orientation(len(env_ids), device=self.device)  # another methods, if only random yaw

        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)

        velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
        self._robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    def save_tensor_to_csv(self, var: torch.Tensor, csv_file_path: str = "xxx_env0.csv"):
        """
        将第一个环境的 self._processed_actions 保存到 CSV 文件中。
        
        CSV 格式：
        - 第一列：当前时间 (self.episode_length_buf * self.step_dt)
        - 第二至七列：各关节的 action 值 (self._processed_actions[0, :6])
        """
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        
        # 检查文件是否存在以决定是否需要写入表头
        file_exists = os.path.isfile(csv_file_path)
        
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # 如果是新文件，则写入表头
            if not file_exists:
                header = ['time'] + [f'tensor_{i}' for i in range((var.shape[1]))]
                writer.writerow(header)
            
            # 获取第一个环境的时间和动作数据
            # print(self.episode_length_buf[0])  # tensor(791, device='cuda:0')
            # print(self.step_dt)  # 0.02
            current_time = round((self.episode_length_buf[0] * self.step_dt).item(), 2)  # 浮点数运算的精度问题
            # tensor_values = getattr(self, tensor_name)[0, :6].cpu().numpy()
            # row = [current_time] + list(tensor_values)
            # 写入数据行

            row = [current_time] + var[0].cpu().tolist()
            writer.writerow(row)