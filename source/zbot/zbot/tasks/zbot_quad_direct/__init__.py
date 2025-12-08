import gymnasium as gym

from . import agents
from .zbot_direct_4leg_env_v0 import Zbot4LEnvCfg, Zbot4LEnv  # 12 dof

##
# Register Gym environments.
##

gym.register(
    id="zbot-quad-walking-v0",
    entry_point="zbot.tasks.zbot_quad_direct:Zbot4LEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Zbot4LEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)