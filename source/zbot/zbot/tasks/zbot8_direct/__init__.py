import gymnasium as gym

from . import agents
from .zbot8_bipedal_env_v0 import Zbot8SEnvV0Cfg, Zbot8SEnvV0  # add command, curriculum by event. also debug visualization.

##
# Register Gym environments.
##

gym.register(
    id="zbot-8b-walking-v0",
    entry_point="zbot.tasks.zbot8_direct:Zbot8SEnvV0",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Zbot8SEnvV0Cfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Zbot8SEnvV0PPOCfg",
    },
)