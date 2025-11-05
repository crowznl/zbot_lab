# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

from zbot.assets import ZBOT_D_6S_CFG

@configclass
class ZbotDirectEnvCfgV0(DirectRLEnvCfg):
    # robot
    robot: ArticulationCfg = ZBOT_D_6S_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/a1", history_length=3, update_period=0.0, track_air_time=False, 
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/b4", 
                                "/World/envs/env_.*/Robot/a5", "/World/envs/env_.*/Robot/b5", 
                                "/World/envs/env_.*/Robot/a6", "/World/envs/env_.*/Robot/b6"]
    )
    contact_sensor_2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/b6", history_length=3, update_period=0.0, track_air_time=False, 
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/a3", 
                                "/World/envs/env_.*/Robot/b2", "/World/envs/env_.*/Robot/a2", 
                                "/World/envs/env_.*/Robot/b1"]
    )
    contact_sensor_3: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/b1", history_length=3, update_period=0.0, track_air_time=False, 
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/a5", "/World/envs/env_.*/Robot/b5", 
                                "/World/envs/env_.*/Robot/a6"]
    )
    contact_sensor_4: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/a6", history_length=3, update_period=0.0, track_air_time=False, 
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/b2", "/World/envs/env_.*/Robot/a2"]
    )  # TODO: a2, b5

    # env
    episode_length_s = 16.0
    decimation = 4  # 2
    action_space = 6  # 24 for sin ;  6 for pd
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
    
    # ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    #   train reward for just stepping walk base 2000 step0
    reward_cfg = {
        "reward_scales": {
            "base_vel_forward": 5.0,
            "base_up_z": -0.5,
            "base_heading_y": -1.0,
            "base_heading_y_sum": -1.0,
            "base_pos_x_err": -1.0,
            # "base_pos_x_err_sum": -1.0,
            "action_rate": -0.1,
            "torques": -0.002,
        },
    }

class ZbotDirectEnvV0(DirectRLEnv):
    cfg: ZbotDirectEnvCfgV0

    def __init__(self, cfg: ZbotDirectEnvCfgV0, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print(self._robot.find_bodies(".*"))
        print(self._robot.find_joints(".*"))

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

        self.heading_vec = torch.tensor([0, -1, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.up_vec = torch.tensor([-1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        # base body axis -Y point to world Y, axis -X point to world Z
        self.base_heading_y_sum = torch.zeros(self.num_envs, device=self.device)
        self.base_pos_x_err_sum = torch.zeros(self.num_envs, device=self.device)

        self.joint_speed_limit = (torch.rand(self.num_envs, 1, device=self.device) * 1.8 + 0.2) * torch.pi

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

        self._contact_sensor_1 = ContactSensor(self.cfg.contact_sensor_1)
        self.scene.sensors["contact_sensor_1"] = self._contact_sensor_1
        self._contact_sensor_2 = ContactSensor(self.cfg.contact_sensor_2)
        self.scene.sensors["contact_sensor_2"] = self._contact_sensor_2
        self._contact_sensor_3 = ContactSensor(self.cfg.contact_sensor_3)
        self.scene.sensors["contact_sensor_3"] = self._contact_sensor_3
        self._contact_sensor_4 = ContactSensor(self.cfg.contact_sensor_4)
        self.scene.sensors["contact_sensor_4"] = self._contact_sensor_4
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # #mode 1
        self._actions = torch.tanh(actions.clone())
        self.p_delta[:] += (
            self._actions
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

        self.base_pos_w = self._robot.data.body_link_pos_w[:, 6]
        self.base_quat_w = self._robot.data.body_link_quat_w[:, 6]

        self.base_heading_w = math_utils.quat_apply(self.base_quat_w, self.heading_vec)  # torch.Size([4096, 3])
        self.base_up_w = math_utils.quat_apply(self.base_quat_w, self.up_vec)  # torch.Size([4096, 3])
        self.base_heading_y_err = -self.base_heading_w[..., 0]  # torch.Size([4096])

        self.base_lin_vel_w = self._robot.data.body_link_vel_w[:, 6, :3].squeeze()  # torch.Size([4096, 3])
        self.base_lin_vel_forward_w = torch.sum(self.base_lin_vel_w * self.base_heading_w, dim=-1)  # 法一 torch.Size([4096])
        # self.base_lin_vel_forward_w = torch.einsum('ij,ij->i', self.base_lin_vel_w, self.base_heading_w)  # 法二
        # self.base_lin_vel_forward_w = (self.base_lin_vel_w @ self.base_heading_w.unsqueeze(-1)).squeeze(-1)  # 法三
        
        obs = torch.cat(
            [
                tensor
                for tensor in (
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

        filter_contact_forces = torch.cat((self._contact_sensor_1.data.force_matrix_w, 
                                           self._contact_sensor_2.data.force_matrix_w, 
                                           self._contact_sensor_3.data.force_matrix_w, 
                                           self._contact_sensor_4.data.force_matrix_w), dim=2)
        
        # died = torch.any(torch.max(torch.norm(filter_contact_forces, dim=-1), dim=1)[0] > 1.0, dim=1)
        died = torch.any(
            torch.max(
                torch.norm(filter_contact_forces, dim=-1), dim=1
            )[0]
            > 1.0,
            dim=1,
        )

        self.base_pos_x_err = self.base_pos_w[:,0] - self._terrain.env_origins[:,0] + 0.318
        died_1 = (self.base_pos_x_err.abs() > 0.2)
        died |= died_1

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
        self.base_heading_y_sum[env_ids] = 0.0
        self.base_pos_x_err_sum[env_ids] = 0.0

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

    def _reward_base_vel_forward(self):
        base_vel_forward = torch.tanh(10.0 * self.base_lin_vel_forward_w / self.joint_speed_limit.squeeze())
        return base_vel_forward
    
    def _reward_base_up_z(self):
        return torch.abs(self.base_up_w[:, 1])
    
    def _reward_base_heading_y(self):
        return torch.abs(self.base_heading_y_err)
    
    def _reward_base_heading_y_sum(self):
        self.base_heading_y_sum += 0.01 * (self.base_heading_y_err)
        self.base_heading_y_sum = torch.clip(self.base_heading_y_sum, -1, 1)
        return torch.abs(self.base_heading_y_sum)

    def _reward_base_pos_x_err(self):
        x_err = torch.abs(self._robot.data.body_com_pos_w[:, 0, 0] 
                          + self._robot.data.body_com_pos_w[:, 11, 0] 
                          - 2.0*self._terrain.env_origins[:,0]
                          + 0.636) 
        + torch.abs(self.base_pos_x_err)
        return x_err

    def _reward_base_pos_x_err_sum(self):
        self.base_pos_x_err_sum += 0.01 * (self.base_pos_x_err)
        self.base_pos_x_err_sum = torch.clip(self.base_pos_x_err_sum, -1, 1)
        return torch.abs(self.base_pos_x_err_sum)

    def _reward_action_rate(self):
        # action rate
        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )
        return action_rate
    
    def _reward_torques(self):
        # Penalize joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        return joint_torques
    
