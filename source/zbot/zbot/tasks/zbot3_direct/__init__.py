# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
zbot 3s environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="zbot-3s-direct-v0",
    entry_point=f"{__name__}.zbot_3s_env:Zbot3SEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.zbot_3s_env_cfg:Zbot3SEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Zbot3SPPORunnerCfg",
    },
)

'''
# https://jih189.github.io/isaaclab
# https://jih189.github.io/isaaclab_train_play
# https://github.com/isaac-sim/IsaacLab/issues/754  # How to register a manager based RL environment #754 

Because we create our task in template, the script list_envs.py will not show it.
Once you have done the task, you need to setup your python package by

python -m pip install -e exts/[your template name]/.

Go to the scripts, you can find the RL library interface there. Then, you need to 
modify the following part to make the train script to find import your tasks.

# import omni.isaac.lab_tasks  # noqa: F401
import [your template name].tasks

'''
