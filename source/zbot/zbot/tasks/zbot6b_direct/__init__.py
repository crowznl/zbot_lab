import gymnasium as gym

from . import agents
# from .zbot6b_env_v0 import ZbotBEnv, ZbotBEnvCfg
# from .zbot6b_env_v1 import ZbotBEnv, ZbotBEnvCfg
from .zbot6b_env_v2 import ZbotBEnv, ZbotBEnvCfg
from .zbot_direct_6dof_bipedal_env import ZbotDirectEnvCfg, ZbotDirectEnv
from .zbot_direct_6dof_bipedal_env_v2 import ZbotDirectEnvCfgV2, ZbotDirectEnvV2  # not project to base
from .zbot_direct_6dof_bipedal_env_v2_1 import ZbotDirectEnvCfgV2V1, ZbotDirectEnvV2V1 # reset base uniform
from .zbot_direct_6dof_bipedal_env_v2_2 import ZbotDirectEnvCfgV2V2, ZbotDirectEnvV2V2  # lay down policy
from .zbot_direct_6dof_bipedal_env_v3 import ZbotDirectEnvCfgV3, ZbotDirectEnvV3  # add node module
from .zbot_direct_12dof_bipedal_env import Zbot12BEnvCfg, Zbot12BEnv  # 12 dof

##
# Register Gym environments.
##

gym.register(
    id="zbot-6b-walking-v0",
    entry_point="zbot.tasks.zbot6b_direct:ZbotBEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ZbotBEnvCfg, 
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZbotSBFlatPPORunnerCfg",
    },
)

gym.register(
    id="zbot-6b-walking-v1",
    entry_point="zbot.tasks.zbot6b_direct:ZbotDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ZbotDirectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgV1",
    },
)

gym.register(
    id="zbot-6b-walking-v2",
    entry_point="zbot.tasks.zbot6b_direct:ZbotDirectEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ZbotDirectEnvCfgV2,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgV2",
    },
)

gym.register(
    id="zbot-6b-walking-v2-1",
    entry_point="zbot.tasks.zbot6b_direct:ZbotDirectEnvV2V1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ZbotDirectEnvCfgV2V1,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgV2V1",
    },
)

gym.register(
    id="zbot-6b-laying-v1",
    entry_point="zbot.tasks.zbot6b_direct:ZbotDirectEnvV2V2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ZbotDirectEnvCfgV2V2,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgV2V2",
    },
)

gym.register(
    id="zbot-6b-walking-v3",
    entry_point="zbot.tasks.zbot6b_direct:ZbotDirectEnvV3",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ZbotDirectEnvCfgV3,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgV3",
    },
)

gym.register(
    id="zbot-12b-walking-v0",
    entry_point="zbot.tasks.zbot6b_direct:Zbot12BEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Zbot12BEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgV4",
    },
)