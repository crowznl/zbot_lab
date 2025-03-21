import gymnasium as gym

from . import agents
# from .zbot_env_v00 import ZbotSEnv, ZbotSEnvCfg
# from .zbot_env_v01 import ZbotSEnv, ZbotSEnvCfg
# from .zbot_env_v02 import ZbotSEnv, ZbotSEnvCfg  # snake movement
# from .zbot_env_v03 import ZbotSEnv, ZbotSEnvCfg
# from .zbot_env_v04 import ZbotSEnv, ZbotSEnvCfg
# from .zbot_env_v05 import ZbotSEnv, ZbotSEnvCfg
# from .zbot_env_v06 import ZbotSEnv, ZbotSEnvCfg
# from .zbot_env_v07 import ZbotSEnv, ZbotSEnvCfg
# from .zbot_env_v08 import ZbotSEnv, ZbotSEnvCfg
# from .zbot_env_v09 import ZbotSEnv, ZbotSEnvCfg
from .zbot_env_v10 import ZbotSEnv, ZbotSEnvCfg # for 6 modules wiht 1 node module

##
# Register Gym environments.
##

gym.register(
    id="zbot-6s-direct-v0",
    entry_point="zbot.tasks.zbot6_direct:ZbotSEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ZbotSEnvCfg, 
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZbotSFlatPPORunnerCfg",
    },
)

'''
Because we create our task in template, the script list_envs.py will not show it.
Once you have done the task, you need to setup your python package by

python -m pip install -e source/[your template name]/.

Go to the scripts, you can find the RL library interface there. Then, you need to 
modify the following part to make the train script to find import your tasks.

# import omni.isaac.lab_tasks  # noqa: F401
import [your template name].tasks

'''

