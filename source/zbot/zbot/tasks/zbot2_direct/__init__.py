import gymnasium as gym

from . import agents
# from .zbot2_env_v0 import Zbot2Env, Zbot2EnvCfg
from .zbot2_env_v1 import Zbot2Env, Zbot2EnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Zbot-2s-walk-v0",
    entry_point="zbot.tasks.zbot2_direct:Zbot2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Zbot2EnvCfg, 
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Zbot2FlatPPORunnerCfg",
    },
)

