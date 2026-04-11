import os
from typing import Any, Dict, List, Optional

import torch

from curobo.geom.types import WorldConfig
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


MODEL: Dict[str, Any] = {}

DEFAULT_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _default_world() -> Dict[str, Any]:
    return {
        "cuboid": {
            "table": {
                "dims": [2.0, 2.0, 0.1],
                "pose": [0.6, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
            }
        }
    }


def load_model():
    global MODEL

    robot_cfg_path = os.getenv("CUROBO_ROBOT_CFG", "/models/ur5e_rg6.yml")
    interpolation_dt = float(os.getenv("CUROBO_INTERPOLATION_DT", "0.02"))
    collision_cache_obb = int(os.getenv("CUROBO_COLLISION_CACHE_OBB", "50"))
    collision_cache_mesh = int(os.getenv("CUROBO_COLLISION_CACHE_MESH", "20"))

    world_dict = _default_world()

    mg_config = MotionGenConfig.load_from_robot_config(
        robot_cfg_path,
        world_dict,
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": collision_cache_obb, "mesh": collision_cache_mesh},
    )
    motion_gen = MotionGen(mg_config)
    motion_gen.warmup()

    MODEL = {
        "motion_gen": motion_gen,
        "robot_cfg_path": robot_cfg_path,
        "interpolation_dt": interpolation_dt,
    }


def _to_joint_state(joint_positions: List[float], joint_names: Optional[List[str]]) -> JointState:
    names = joint_names or DEFAULT_JOINT_NAMES
    q = torch.tensor([joint_positions], device="cuda", dtype=torch.float32)
    return JointState.from_position(q, joint_names=names)


def _to_pose(goal_pose: List[float]) -> Pose:
    return Pose.from_list(goal_pose)


def _traj_to_dict(traj: Any) -> Dict[str, Any]:
    q = traj.position.detach().cpu().tolist() if hasattr(traj, "position") else []
    qd = traj.velocity.detach().cpu().tolist() if hasattr(traj, "velocity") else []
    qdd = traj.acceleration.detach().cpu().tolist() if hasattr(traj, "acceleration") else []

    return {
        "position": q,
        "velocity": qd,
        "acceleration": qdd,
        "joint_names": list(traj.joint_names) if hasattr(traj, "joint_names") else [],
    }


def plan_motion(
    joint_positions: List[float],
    goal_pose: List[float],
    joint_names: Optional[List[str]] = None,
    world: Optional[Dict[str, Any]] = None,
    max_attempts: int = 1,
    time_dilation_factor: float = 1.0,
    interpolation_dt: Optional[float] = None,
) -> Dict[str, Any]:
    if not MODEL:
        raise RuntimeError("Model not loaded")

    motion_gen: MotionGen = MODEL["motion_gen"]

    if world is not None:
        motion_gen.update_world(WorldConfig.from_dict(world))

    start_state = _to_joint_state(joint_positions, joint_names)
    goal = _to_pose(goal_pose)

    plan_cfg = MotionGenPlanConfig(
        max_attempts=max_attempts,
        time_dilation_factor=time_dilation_factor,
    )

    result = motion_gen.plan_single(start_state, goal, plan_cfg)

    success = bool(result.success.item()) if hasattr(result.success, "item") else bool(result.success)

    response: Dict[str, Any] = {
        "success": success,
        "status": str(result.status) if hasattr(result, "status") else None,
    }

    if not success:
        return response

    if interpolation_dt is not None:
        traj = result.get_interpolated_plan(interpolation_dt=interpolation_dt)
        used_dt = interpolation_dt
    else:
        traj = result.get_interpolated_plan()
        used_dt = getattr(result, "interpolation_dt", MODEL["interpolation_dt"])

    response["trajectory"] = _traj_to_dict(traj)
    response["interpolation_dt"] = float(used_dt)
    response["solve_time"] = float(result.solve_time) if hasattr(result, "solve_time") else None

    return response