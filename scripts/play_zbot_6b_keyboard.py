"""
此脚本演示了使用键盘交互式控制 Zbot 双足机器人。
需先加载 Isaac Sim。

用法:
    ./isaaclab.sh -p scripts/play_zbot_6b_keyboard.py
    python scripts/play_zbot_6b_keyboard.py
"""

# 参考 https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
# https://docs.robotsfan.com/isaaclab/source/overview/showroom.html `./isaaclab.sh -p scripts/demos/h1_locomotion.py`
# <path-to>/IsaacLab/scripts/demos/h1_locomotion.py
# 在 h1_locomotion.py 中使用 RslRlVecEnvWrapper 和 OnPolicyRunner，与我们直接加载 JIT 模型（torch.jit.load）相比，各有优劣。
# OnPolicyRunner 加载的是训练过程中的完整检查点Checkpoint（model_*.pt），而不仅仅是策略网络。这意味着它包含了优化器状态、Critic 网络、归一化层的均值和方差（Running Mean/Std）等。

import argparse
import sys


from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(
    description="Interactive demo for Zbot with keyboard control."
)
# 添加 AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余导入在启动应用后进行"""
import torch
import carb
import omni
# import gymnasium as gym

from zbot.tasks.zbot6b_direct.zbot_direct_6dof_bipedal_env_v4 import Zbot6SEnvV4, Zbot6SEnvV4Cfg

class ZbotKeyboardController:
    """
    Zbot 键盘控制器类。
    """

    def __init__(self):
        # 1. 配置环境
        env_cfg = Zbot6SEnvV4Cfg()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 100000.0  # 延长回合时间
        env_cfg.debug_vis = True  # 开启可视化箭头

        # 2. 禁用自动指令重采样 (由键盘接管)
        # 将事件设为 None，防止环境自动改变指令
        env_cfg.events.vel_range = None
        env_cfg.events.my_curric = None
        env_cfg.events.reset_command_resample = None
        env_cfg.events.interval_command_resample = None
        
        # 3. 创建环境
        self.env = Zbot6SEnvV4(cfg=env_cfg, render_mode="rgb_array")
        self.device = self.env.device

        # 4. 加载策略
        # 路径指向你提供的 exported policy.pt
        # policy_path = "/home/crowznl/Dev/isaac/myExt/zbot_lab/logs/rsl_rl/zbot_6b_flat_direct_v4/2026-01-04_15-55-45/exported/policy.pt"
        # policy_path = "/home/crowznl/Dev/isaac/myExt/zbot_lab/logs/rsl_rl/zbot_6b_flat_direct_v4/2026-01-13_18-12-13/exported/policy.pt"
        # policy_path ="/home/crowznl/Dev/isaac/myExt/zbot_lab/logs/rsl_rl/zbot_6b_flat_direct_v4/2026-01-14_03-03-33/exported/policy.pt"
        policy_path = "/home/crowznl/Dev/isaac/myExt/zbot_lab/logs/rsl_rl/zbot_6b_flat_direct_v4/2026-01-23_15-20-39/exported/policy_walk_keyboard.pt"
        # policy_path = "/home/crowznl/Dev/isaac/myExt/zbot_lab/logs/rsl_rl/zbot_6b_flat_direct_v4/2026-01-23_20-24-50/exported/policy.pt"
        print(f"Loading policy from: {policy_path}")
        try:
            self.policy = torch.jit.load(policy_path, map_location=self.device)
            self.policy.eval()
        except Exception as e:
            print(f"Error loading policy: {e}")
            print("请确保路径正确且文件是 torch.jit 导出的模型。")
            sys.exit(1)

        # 5. 初始化控制变量
        self.cmd_vel_x = 0.0
        self.cmd_yaw_target = 0.0
        
        # 6. 设置键盘监听
        self.create_keyboard_listener()
        
        print("-" * 80)
        print("Zbot Keyboard Control Ready")
        print("Controls:")
        print("  W / S : Increase/Decrease Linear Velocity X (+/- 0.05 m/s)")
        print("  A / D : Increase/Decrease Target Yaw (+/- 0.05 rad)")
        print("  R     : Reset Environment")
        print("  ESC   : Quit")
        print("-" * 80)
        print(">>> 请点击 Isaac Sim 视口窗口以确保捕获键盘输入 <<<")

    def create_keyboard_listener(self):
        """注册键盘事件回调"""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def _on_keyboard_event(self, event):
        """处理键盘按键"""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            
            if key == "W":
                self.cmd_vel_x += 0.05
                print(f"Velocity X: {self.cmd_vel_x:.2f}, Target Yaw: {self.cmd_yaw_target:.2f}")
            elif key == "S":
                self.cmd_vel_x -= 0.05
                print(f"Velocity X: {self.cmd_vel_x:.2f}, Target Yaw: {self.cmd_yaw_target:.2f}")
            elif key == "A":
                self.cmd_yaw_target += 0.05
                print(f"Velocity X: {self.cmd_vel_x:.2f}, Target Yaw: {self.cmd_yaw_target:.2f}")
            elif key == "D":
                self.cmd_yaw_target -= 0.05
                print(f"Velocity X: {self.cmd_vel_x:.2f}, Target Yaw: {self.cmd_yaw_target:.2f}")
            elif key == "R":
                self.env.reset()
                self.cmd_vel_x = 0.0
                self.cmd_yaw_target = self.env.current_yaw[0].item() # 重置时对齐当前朝向
                print("Environment Reset")
            elif key == "ESCAPE":
                simulation_app.close()

    def run(self):
        """主循环"""
        obs, _ = self.env.reset()
        
        # 初始化目标朝向为当前朝向，防止一开始就猛转
        self.cmd_yaw_target = self.env.current_yaw[0].item()

        while simulation_app.is_running():
            # 1. 将键盘控制量应用到环境
            # DirectRLEnv 的 step 会在内部调用 _get_observations，
            # 而 _get_observations 依赖 self.commands 和 self.target_heading_yaw。
            # 因此我们在 step 之前更新这些变量。
            
            self.env.commands[:, 0] = self.cmd_vel_x
            # 注意：env.commands[:, 1] 是相对 yaw，但在 _get_observations 中，
            # 真正使用的是 self.target_heading_yaw 来计算 heading_err。
            # 所以我们直接更新 target_heading_yaw。
            self.env.target_heading_yaw[:] = self.cmd_yaw_target

            # 2. 推理
            with torch.inference_mode():
                # 这里的 obs 是上一步 step 返回的，包含了上一步的指令信息。
                # 虽然有一帧的延迟，但对于键盘控制来说是可以接受的。

                # DirectRLEnv 返回的是字典 {'policy': tensor, ...}
                # 而 JIT 导出的模型 forward(Tensor) 只接受 Tensor
                if isinstance(obs, dict) and "policy" in obs:
                    obs_input = obs["policy"]
                else:
                    obs_input = obs
                action = self.policy(obs_input)
                
                # 3. 仿真步进
                obs, _, _, _, _ = self.env.step(action)

def main():
    controller = ZbotKeyboardController()
    controller.run()

if __name__ == "__main__":
    main()
    simulation_app.close()