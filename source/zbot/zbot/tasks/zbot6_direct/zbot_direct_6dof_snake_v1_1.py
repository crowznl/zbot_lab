"""zbot 6-DoF snake v1

训练目标（不再显式跟踪速度）：
1) 学会整体前/后运动与原地转向；
2) 运动时尽量保持直线，抑制侧向漂移；
3) 通过相位观测 + 小权重节律先验，鼓励近似正弦的有节律动作。

默认配置采用 forward-only 作为第一阶段（可切换为前后+转向混合）。
"""

from __future__ import annotations

import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from zbot.assets import ZBOT_6S_CFG_2


def reset_root_state_uniform(
    env: "Zbot6SnakeEnvV1",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
):
    """Reset root pose/velocity uniformly within ranges."""

    root_states = env._robot.data.default_root_state[env_ids].clone()

    # poses: x, y, z, roll, pitch, yaw
    range_list = [pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    # default_root_state 的 yaw 通常是 0；如不是可进一步补偿
    env.current_yaw[env_ids] = rand_samples[:, 5]

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(orientations_delta, root_states[:, 3:7])

    # velocities
    range_list = [velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    velocities = root_states[:, 7:13] + rand_samples

    env._robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    env._robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def resample_commands(
    env: "Zbot6SnakeEnvV1",
    env_ids: torch.Tensor,
    forward_only: bool = True,
    prob_turn: float = 0.25,
    prob_backward: float = 0.35,
):
    """Resample high-level motion commands.

    commands[:, 0] -> move command in {-1, 0, +1}
        +1: forward, -1: backward, 0: no translation command
    commands[:, 1] -> turn command in {-1, 0, +1}
        +1: turn left, -1: turn right, 0: no yaw command
    """

    if forward_only:
        env.commands[env_ids, 0] = 1.0
        env.commands[env_ids, 1] = 0.0
    else:
        n = len(env_ids)
        turn_mask = torch.rand(n, device=env.device) < prob_turn
        turn_sign = torch.where(torch.rand(n, device=env.device) < 0.5, -torch.ones(n, device=env.device), torch.ones(n, device=env.device))

        back_mask = torch.rand(n, device=env.device) < prob_backward
        move_sign = torch.where(back_mask, -torch.ones(n, device=env.device), torch.ones(n, device=env.device))

        env.commands[env_ids, 0] = torch.where(turn_mask, torch.zeros(n, device=env.device), move_sign)
        env.commands[env_ids, 1] = torch.where(turn_mask, turn_sign, torch.zeros(n, device=env.device))

    # move 模式下，以当前朝向作为“保持直线”的目标朝向
    env.target_heading_yaw[env_ids] = env.current_yaw[env_ids]


@configclass
class EventCfg:
    """Configuration for randomization and command events."""

    # reset_base = EventTerm(
    #     func=reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": (-0.5, 0.5),
    #             "y": (-0.5, 0.5),
    #             "roll": (-0.7854, 0.7854),
    #             # "pitch": (-0.15, 0.15),
    #             "yaw": (-3.14, 3.14),
    #         },
    #         "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
    #     },
    # )

    reset_command_resample = EventTerm(
        func=resample_commands,
        mode="reset",
        params={
            # 先易后难：先只向前，再改 forward_only=False
            "forward_only": True,
            "prob_turn": 0.25,
            "prob_backward": 0.35,
        },
    )

    interval_command_resample = EventTerm(
        func=resample_commands,
        mode="interval",
        interval_range_s=(2.5, 5.0),
        params={
            "forward_only": True,
            "prob_turn": 0.25,
            "prob_backward": 0.35,
        },
    )


@configclass
class Zbot6SnakeEnvV1Cfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 4
    action_space = 6
    observation_space = 27
    state_space = 0

    # rhythm prior settings
    # 参考你历史可用模板: 0.5 * sin(2*t + 1.3*pi*i)
    rhythm_omega = 2.0  # rad/s
    rhythm_joint_phase_coef = 4.08407045  # 1.3 * pi
    rhythm_orthogonal_phase = 1.5708  # pi/2, for orthogonal connected joints (odd joints)
    rhythm_use_orthogonal_phase = True
    rhythm_use_triangle_wave = False
    rhythm_move_gain = 0.50
    rhythm_turn_gain = 0.30

    # limit integrated joint excursion to suppress body over-twisting
    joint_delta_limit = 1.8849556  # 0.6 * pi

    # heading smoothing for robust yaw estimation
    heading_ema_alpha = 0.30

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200.0,
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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    events: EventCfg = EventCfg()

    robot: ArticulationCfg = ZBOT_6S_CFG_2.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/foot_0",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/b4",
            "/World/envs/env_.*/Robot/a5",
            "/World/envs/env_.*/Robot/b5",
            "/World/envs/env_.*/Robot/a6",
            "/World/envs/env_.*/Robot/foot_1",
        ],
    )
    contact_sensor_2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/foot_1",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/a3",
            "/World/envs/env_.*/Robot/b2",
            "/World/envs/env_.*/Robot/a2",
            "/World/envs/env_.*/Robot/b1",
        ],
    )
    contact_sensor_3: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/b1",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/a5",
            "/World/envs/env_.*/Robot/b5",
            "/World/envs/env_.*/Robot/a6",
        ],
    )
    contact_sensor_4: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/a6",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/b2",
            "/World/envs/env_.*/Robot/a2",
        ],
    )

    reward_cfg = {
        "reward_scales": {
            # task rewards (no explicit speed tracking)
            "alive": 0.2,
            "motion_progress": 6.0,
            "turn_rate": 3.0,

            # straightness / anti-drift
            "straight_heading": -1.5,
            "lateral_drift": -2.0,
            "yaw_drift": -0.6,

            # rhythm prior
            "rhythm_tracking": 0.35,
            "shape_smooth": -0.25,

            # regularization
            "action_rate": -0.05,
            "torques": -2e-4,
            "joint_vel": -2e-4,
            "joint_acc": -1e-7,
        }
    }


class Zbot6SnakeEnvV1(DirectRLEnv):
    cfg: Zbot6SnakeEnvV1Cfg

    def __init__(self, cfg: Zbot6SnakeEnvV1Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Commands: [move_cmd {-1,0,+1}, turn_cmd {-1,0,+1}]
        self.commands = torch.zeros(self.num_envs, 2, device=self.device)
        self.current_yaw = torch.zeros(self.num_envs, device=self.device)
        self.target_heading_yaw = torch.zeros(self.num_envs, device=self.device)
        self.current_yaw_ema = torch.zeros(self.num_envs, device=self.device)
        self.heading_err = torch.zeros(self.num_envs, device=self.device)

        # actions
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = torch.zeros_like(self._actions)

        self.p_delta = torch.zeros_like(self._robot.data.default_joint_pos)
        self.joint_speed_limit = 1.0 * torch.ones((self.num_envs, 1), device=self.device)

        # body indices
        base_body_ids = self._robot.find_bodies("base")[0]
        self.base_body_idx = int(base_body_ids[0])

        # reusable axes (按你的坐标关系：local -z 为 heading, local y 为 lateral)
        self.axis_forward_local = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self.axis_lateral_local = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # rhythm buffers
        self.phase_offset = torch.zeros(self.num_envs, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_sin = torch.zeros(self.num_envs, device=self.device)
        self.phase_cos = torch.zeros(self.num_envs, device=self.device)

        self.num_dof = self._robot.data.default_joint_pos.shape[1]
        joint_ids = torch.arange(self.num_dof, device=self.device, dtype=torch.float32)
        # pair index for orthogonal-chain modules: [0,0,1,1,2,2] for 6 dof
        self.joint_is_odd = (joint_ids % 2).unsqueeze(0)
        self.joint_ids = joint_ids.unsqueeze(0)
        self.joint_phase_offsets = self.joint_ids * self.cfg.rhythm_joint_phase_coef
        pair_ids = torch.div(joint_ids, 2, rounding_mode="floor").unsqueeze(0)
        self.joint_pair_sign = torch.where(
            (pair_ids % 2) == 0,
            torch.ones_like(pair_ids),
            -torch.ones_like(pair_ids),
        )
        # turning bias for orthogonal chain: primary axis stronger, orthogonal axis weaker
        self.turn_profile = torch.where(
            self.joint_is_odd == 0,
            torch.ones((1, self.num_dof), device=self.device),
            0.35 * torch.ones((1, self.num_dof), device=self.device),
        )
        self.cpg_ref = torch.zeros((self.num_envs, self.num_dof), device=self.device)

        # reward registry
        self.reward_scales = cfg.reward_cfg["reward_scales"]
        self.reward_functions, self._episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self._episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)

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

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = torch.tanh(actions.clone())
        self.p_delta[:] += torch.pi * self._actions * self.joint_speed_limit * self.step_dt
        self.p_delta = torch.clip(self.p_delta, -self.cfg.joint_delta_limit, self.cfg.joint_delta_limit)
        self._processed_actions = self.p_delta + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _compute_intermediate_values(self):
        self.base_pos_w = self._robot.data.body_link_pos_w[:, self.base_body_idx].squeeze(1)
        self.base_quat_w = self._robot.data.body_link_quat_w[:, self.base_body_idx].squeeze(1)

        # heading/lateral from base local axes directly (不与重力叉乘)
        self.base_dir_forward_w = math_utils.quat_apply(self.base_quat_w, self.axis_forward_local)
        self.base_dir_forward_w = torch.nn.functional.normalize(self.base_dir_forward_w, dim=-1)
        self.base_dir_lateral_w = math_utils.quat_apply(self.base_quat_w, self.axis_lateral_local)
        self.base_dir_lateral_w = torch.nn.functional.normalize(self.base_dir_lateral_w, dim=-1)

        yaw_raw = torch.atan2(self.base_dir_forward_w[:, 1], self.base_dir_forward_w[:, 0])
        yaw_delta = torch.atan2(torch.sin(yaw_raw - self.current_yaw_ema), torch.cos(yaw_raw - self.current_yaw_ema))
        self.current_yaw_ema = math_utils.wrap_to_pi(self.current_yaw_ema + self.cfg.heading_ema_alpha * yaw_delta)
        self.current_yaw = self.current_yaw_ema

        diff = self.target_heading_yaw - self.current_yaw
        self.heading_err = torch.atan2(torch.sin(diff), torch.cos(diff))

        self.base_lin_vel_w = self._robot.data.body_link_lin_vel_w[:, self.base_body_idx, :]
        self.base_lin_vel_forward_w = torch.sum(self.base_lin_vel_w * self.base_dir_forward_w, dim=-1)
        self.base_lin_vel_lateral_w = torch.sum(self.base_lin_vel_w * self.base_dir_lateral_w, dim=-1)
        self.base_ang_vel_z_w = self._robot.data.body_link_ang_vel_w[:, self.base_body_idx, 2]

        # rhythm phase
        self.phase = self.cfg.rhythm_omega * (self.episode_length_buf.float() * self.step_dt) + self.phase_offset
        self.phase_sin = torch.sin(self.phase)
        self.phase_cos = torch.cos(self.phase)

        # rhythm reference (CPG prior): move term follows 0.5*sin(omega*t + phase_coef*i)
        orth_phase = self.joint_is_odd * self.cfg.rhythm_orthogonal_phase if self.cfg.rhythm_use_orthogonal_phase else 0.0
        raw_wave = self.phase.unsqueeze(1) + self.joint_phase_offsets + orth_phase
        if self.cfg.rhythm_use_triangle_wave:
            wave = (2.0 / torch.pi) * torch.asin(torch.sin(raw_wave))
        else:
            wave = torch.sin(raw_wave)
        move_cmd = self.commands[:, 0:1]
        turn_cmd = self.commands[:, 1:2]
        self.cpg_ref = (
            self.cfg.rhythm_move_gain * move_cmd * wave
            + self.cfg.rhythm_turn_gain * turn_cmd * self.turn_profile * self.joint_pair_sign
        )
        self.cpg_ref = torch.clamp(self.cpg_ref, -1.0, 1.0)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        self._compute_intermediate_values()

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.base_quat_w,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                    self.commands,
                    self.heading_err.unsqueeze(-1),
                    self.phase_sin.unsqueeze(-1),
                    self.phase_cos.unsqueeze(-1),
                )
                if tensor is not None
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name] * self.step_dt
            reward += rew
            self._episode_sums[name] += rew

        terminated_ids = self.reset_terminated.nonzero(as_tuple=False).squeeze(-1)
        reward[terminated_ids] -= 2.0
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # safety termination
        # died = ~torch.isfinite(self.base_pos_w).all(dim=-1)

        drift_y = torch.abs(self.base_pos_w[:, 1] - self._terrain.env_origins[:, 1])
        died = drift_y > 1.0

        # self-collision (参考 snake_v0)
        filter_contact_forces = torch.cat(
            (
                self._contact_sensor_1.data.force_matrix_w,
                self._contact_sensor_2.data.force_matrix_w,
                self._contact_sensor_3.data.force_matrix_w,
                self._contact_sensor_4.data.force_matrix_w,
            ),
            dim=2,
        )
        self_collision = torch.any(
            torch.max(torch.norm(filter_contact_forces, dim=-1), dim=1)[0] > 1.0,
            dim=1,
        )
        died |= self_collision

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # logging
        actual_episode_duration = self.episode_length_buf[env_ids].float() * self.step_dt
        actual_episode_duration = torch.clamp(actual_episode_duration, min=self.step_dt)
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_per_s = self._episode_sums[key][env_ids] / actual_episode_duration
            extras["Episode_Reward/" + key] = torch.mean(episodic_sum_per_s)
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        # reset
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.p_delta[env_ids] = 0.0

        # Reset robot state
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # random phase offset to avoid all envs synchronized
        self.phase_offset[env_ids] = torch.rand(len(env_ids), device=self.device) * 2.0 * torch.pi

        # initialize heading caches
        self.current_yaw_ema[env_ids] = self.current_yaw[env_ids]
        self.target_heading_yaw[env_ids] = self.current_yaw[env_ids]

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------

    def _reward_alive(self):
        return torch.ones(self.num_envs, device=self.device)

    def _reward_motion_progress(self):
        # encourage commanded forward/back motion without explicit speed target tracking
        cmd = self.commands[:, 0]
        active = (cmd.abs() > 0.5).float()
        return torch.tanh(4.0 * cmd * self.base_lin_vel_forward_w) * active

    def _reward_turn_rate(self):
        # encourage commanded turning direction
        cmd = self.commands[:, 1]
        active = (cmd.abs() > 0.5).float()
        return torch.tanh(2.0 * cmd * self.base_ang_vel_z_w) * active

    def _reward_straight_heading(self):
        # during translational motion, keep heading near sampled target heading
        move_active = (self.commands[:, 0].abs() > 0.5).float()
        return torch.abs(self.heading_err) * move_active

    def _reward_lateral_drift(self):
        move_active = (self.commands[:, 0].abs() > 0.5).float()
        return torch.square(self.base_lin_vel_lateral_w) * move_active

    def _reward_yaw_drift(self):
        # during straight move, suppress unnecessary yaw spin
        move_active = (self.commands[:, 0].abs() > 0.5).float()
        return torch.square(self.base_ang_vel_z_w) * move_active

    def _reward_rhythm_tracking(self):
        # soft prior: action should be close to traveling-wave reference
        err = torch.mean(torch.square(self._actions - self.cpg_ref), dim=1)
        return torch.exp(-err / 0.5)

    def _reward_shape_smooth(self):
        # discourage adjacent-joint abrupt bending to reduce body contortion
        d = self.p_delta[:, 1:] - self.p_delta[:, :-1]
        return torch.mean(torch.square(d), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self._robot.data.applied_torque), dim=1)

    def _reward_joint_vel(self):
        return torch.sum(torch.square(self._robot.data.joint_vel), dim=1)

    def _reward_joint_acc(self):
        return torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
