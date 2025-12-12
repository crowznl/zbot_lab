# ZBOT_LAB

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

This repository serves as a **ZBOT training ZOO** based on Isaac Lab. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Comprehensive` This library provide both direct and manager-based workflows.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), [Chinese installation guide](https://docs.robotsfan.com/isaaclab/). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone https://github.com/crowznl/zbot_lab.git

# Option 2: SSH
git clone git@github.com:crowznl/zbot_lab.git
```

- Using a **python interpreter that has Isaac Lab installed**, install the library

This ensures that the gym.register() function is called correctly.

```bash
conda activate your_env
python -m pip install -e source/zbot
# or
./isaaclab.sh -p -m pip install -e source/zbot
```

- **Note 1:** Using scripts that matches the installed rsl_rl library version

As the rsl_rl library is installed with Isaac Lab, you can find the matching version of scripts in Isaac Lab directory (e.g. `${your-path-to-isaac-lab}/scripts/.../rsl_rl`). Copy the `rsl_rl` folder to the zbot_lab's `scripts` folder，replacing the source file.

In other words，if you are using a different version of isaac lab, (usually means a different version of rsl_rl), you can replace the `rsl_rl` folder in `zbot_lab/scripts` with the corresponding version of `./scripts/.../rsl_rl` from the Isaac Lab directory.

- **Note 2:** Eidt the scripts, if you replace the `rsl_rl` folder following the Note 1.

you need to edit the `scripts/rsl_rl/train.py`, `scripts/rsl_rl/play.py` file to import your customize tasks.

```python
# find the following code
# import omni.isaac.lab_tasks  # noqa: F401 
# or 
# import isaaclab_tasks  # noqa: F401
# change to
import zbot.tasks
```

- Verify that this library is correctly installed by running the following command

```bash
python scripts/rsl_rl/train.py --task=zbot-6b-walking-v2
# python scripts/rsl_rl/train.py --task=zbot-6b-walking-v2 --num_envs 4096 --headless --max_iterations 2000 --resume --load_run=2025-11-04_13-15-04
# python scripts/rsl_rl/play.py --task=Zbot-6b-walking-v2 --num_envs=64 --checkpoint=model_500.pt --video --video_length 500  # record video
```

## Tensorboard

To view tensorboard, run:

```bash
tensorboard --logdir=your_logs
# or
python -m tensorboard --logdir=your_logs
```

## Set up IDE (Optional)

To setup the IDE, please follow these [instructions](https://docs.robotsfan.com/isaaclab/source/overview/developer-guide/vs_code.html):

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create `launch.json` and `settings.json` in the `.vscode` directory. This helps in indexing all the python modules for intelligent suggestions while writing code.

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-isaaclab>/source/isaaclab",  // e.g. "~/IsaacLab/source/isaaclab",
        "<path-to-isaaclab>/source/isaaclab_assets",
        "<path-to-isaaclab>/source/isaaclab_mimic",
        "<path-to-isaaclab>/source/isaaclab_rl",
        "<path-to-isaaclab>/source/isaaclab_tasks",

        "${workspaceFolder}/source/zbot"
    ]
}
```

## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure and parts of the implementation.
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl): The core library for reinforcement learning.
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab): Referenced for the reward function and robot actuator configuration.

## Citation

Please cite the following if you use this code or parts of it:

```text
@InProceedings{zhounanlin2025zbot,
  author={Nanlin Zhou, Sikai Zhao, Hang Luo, Kai Han,  Zhiyuan Yang, Jian Qi, Ning Zhao, Jie Zhao, Yanhe Zhu},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title={ZBOT: A Novel Modular Robot Capable of Active Transformation from Snake to Bipedal Configuration through RL},
  year={2025},
  pages={12093-12099},
  doi={https://doi.org/10.1109/IROS60139.2025.11246225}
}
```

```
@software{crowznl2025zbot_lab,
  author = {Nanlin Zhou},
  title = {zbot_lab: zbot training zoo based on IsaacLab.},
  url = {https://github.com/crowznl/zbot_lab},
  year = {2025}
}
```