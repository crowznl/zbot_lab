import gymnasium as gym

from . import agents
from .zbot_direct_4leg_env_v0 import Zbot4LEnvCfg, Zbot4LEnv  # 12 dof
from .zbot_direct_4leg_env_v1 import Zbot4LEnvV1Cfg, Zbot4LEnvV1  # commands

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

gym.register(
    id="zbot-quad-walking-v1",
    entry_point="zbot.tasks.zbot_quad_direct:Zbot4LEnvV1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Zbot4LEnvV1Cfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerV1Cfg",
    },
)