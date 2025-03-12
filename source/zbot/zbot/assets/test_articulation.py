# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_articulation.py

"""
# but we copy it here, eidt and use this script to test my robot_CFG
# referrence: https://github.com/isaac-sim/IsaacLab/issues/1046 
# [Question] How should I create an Articulation will unactuated joints? #1046 


"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import time
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab.actuators import ImplicitActuatorCfg  # DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Pre-defined configs
##
from zbot.assets import ZBOT_D_6S_CFG, ZBOT_D_6W_CFG, ZBOT_D_6B_CFG, ZBOT_D_6B_1_CFG, ZBOT_D_2S_CFG, ZBOT_D_2S_A_CFG, JOINT_TEST_CFG


G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/crowznl/Dev/isaac/asset/Unitree/G1/g1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_pitch_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            "left_one_joint": 1.0,
            "right_one_joint": -1.0,
            "left_two_joint": 0.52,
            "right_two_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "torso_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_five_joint": 0.001,
                ".*_three_joint": 0.001,
                ".*_six_joint": 0.001,
                ".*_four_joint": 0.001,
                ".*_zero_joint": 0.001,
                ".*_one_joint": 0.001,
                ".*_two_joint": 0.001,
            },
        ),
    },
)

H1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/crowznl/Dev/isaac/asset/Unitree/H1/h1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0,
            ".*_hip_pitch": -0.28,  # -16 degrees
            ".*_knee": 0.79,  # 45 degrees
            ".*_ankle": -0.52,  # -30 degrees
            "torso": 0.0,
            ".*_shoulder_pitch": 0.28,
            ".*_shoulder_roll": 0.0,
            ".*_shoulder_yaw": 0.0,
            ".*_elbow": 0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw": 150.0,
                ".*_hip_roll": 150.0,
                ".*_hip_pitch": 200.0,
                ".*_knee": 200.0,
                "torso": 200.0,
            },
            damping={
                ".*_hip_yaw": 5.0,
                ".*_hip_roll": 5.0,
                ".*_hip_pitch": 5.0,
                ".*_knee": 5.0,
                "torso": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle"],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={".*_ankle": 20.0},
            damping={".*_ankle": 4.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_pitch": 40.0,
                ".*_shoulder_roll": 40.0,
                ".*_shoulder_yaw": 40.0,
                ".*_elbow": 40.0,
            },
            damping={
                ".*_shoulder_pitch": 10.0,
                ".*_shoulder_roll": 10.0,
                ".*_shoulder_yaw": 10.0,
                ".*_elbow": 10.0,
            },
        ),
    },
)

JOINT_1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/crowznl/Dev/isaac/asset/zbot/joint1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.05),
        # rot=(0.707, 0.0, -0.707, 0.0),
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),  # (w, x, y, z)
        joint_pos={
            # "joint1": -3.141593,  # -180 degrees
            "joint1": 0.0,  # -180 degrees
        },
        joint_vel={
            "joint1": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "zbot_joint": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            effort_limit=20,
            velocity_limit=10,
            stiffness=20,
            damping=0.5,
            armature=0.01,
            friction=0.0,
        ),
    },
)

JOINT_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/crowznl/Dev/isaac/asset/zbot/joint2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.05),
        # rot=(0.707, 0.0, -0.707, 0.0),
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),  # (w, x, y, z)
        joint_pos={
            # "joint1": -3.141593,  # -180 degrees
            "joint": 0.0,  # -180 degrees
        },
        joint_vel={
            "joint": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "zbot_joint": ImplicitActuatorCfg(
            joint_names_expr=["join.*"],
            effort_limit=20,
            velocity_limit=10,
            stiffness=20,
            damping=0.5,
            armature=0.01,
            friction=0.0,
        ),
    },
)


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # Articulation
    # zbot_cfg = ZBOT_D_6S_CFG.copy()
    # zbot_cfg = ZBOT_D_6B_1_CFG.copy()
    # zbot_cfg = ZBOT_D_2S_A_CFG.copy()
    # zbot_cfg = ZBOT_D_2S_CFG.copy()
    zbot_cfg = JOINT_TEST_CFG.copy()
    # zbot_cfg = G1_CFG.copy()
    # zbot_cfg = JOINT_1_CFG.copy()
    # zbot_cfg = JOINT_2_CFG.copy()
    # zbot_cfg = H1_CFG.copy()
    '''
    纯纯因为没触发git hook, USD修改没同步过来, 导致始终显示[-3.4028e+38,  3.4028e+38], 
    同步后就正常了！！！！啊
    '''

    zbot_cfg.prim_path = "/World/Origin.*/Robot"
    zbot6s = Articulation(cfg=zbot_cfg)

    # return the scene information
    scene_entities = {"zbot6s": zbot6s}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["zbot6s"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    joint = robot.data.default_joint_pos.clone()

    print(robot.data.default_joint_limits)
    print(robot.data.joint_limits)
    # joints_limits = robot.data.default_joint_limits.clone()
    # joints_limits[:, :, 0] = -3.141593*4
    # joints_limits[:, :, 1] = 3.141593*4
    # print(joints_limits)
    # robot.write_joint_limits_to_sim(joints_limits)
    # print(robot.data.default_joint_limits)
    # print(robot.data.joint_limits)

    cc = -1
    reset_max_steps = 500 # 10 # 500
    # Simulation loop
    # while simulation_app.is_running():
    #     # Reset
    #     if count % reset_max_steps == 0:
    #         # reset counter
    #         count = 0
    #         joint = robot.data.default_joint_pos.clone()
    #         # cc *= -1

    #         # reset the scene entities
    #         # root state
    #         # we offset the root state by the origin since the states are written in simulation world frame
    #         # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
    #         root_state = robot.data.default_root_state.clone()
    #         root_state[:, :3] += origins
    #         # print(root_state)
    #         robot.write_root_state_to_sim(root_state)
    #         # set joint positions with some noise
    #         init_joint_pos, init_joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    #         # print(joint_pos)
    #         # print(joint_vel)
    #         # joint_pos += torch.rand_like(joint_pos) * 0.1
    #         robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel)
    #         # clear internal buffers
    #         robot.reset()
    #         print("[INFO]: Resetting robot state...")
    #     # # Apply random action
    #     # # -- generate random joint efforts
    #     # efforts = torch.randn_like(robot.data.joint_pos) * 5.0
    #     # # -- apply action to the robot
    #     # robot.set_joint_effort_target(efforts)
    #     # # -- write data to sim
    #     # robot.write_data_to_sim()

    #     time.sleep(0.1)
    #     # print("c", robot.data.joint_pos/3.141593*180)
        
    #     joint += 0.1 * cc
    #     print("t", joint/3.141593*180)
    #     robot.set_joint_position_target(joint)
    #     robot.write_data_to_sim()
    #     print("c", robot.data.joint_pos/3.141593*180)

    #     # Perform step
    #     sim.step()
    #     # Increment counter
    #     count += 1
    #     # Update buffers
    #     robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    # # Set gravity to zero
    # sim_cfg.gravity = (0.0, 0.0, 0.0)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    # sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    sim.set_camera_view([0.5, 0.0, 2.0], [0.0, 0.0, 0.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)
    time.sleep(100)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
