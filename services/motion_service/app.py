from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

import service_core as sc


app = FastAPI(title="cuRobo Motion Service")


class GoalPoseRequest(BaseModel):
    # [x, y, z, qw, qx, qy, qz]
    pose: List[float] = Field(..., min_length=7, max_length=7)


class MotionPlanRequest(BaseModel):
    joint_positions: List[float]
    goal_pose: GoalPoseRequest
    joint_names: Optional[List[str]] = None
    world: Optional[Dict[str, Any]] = None
    max_attempts: int = 1
    time_dilation_factor: float = 1.0
    interpolation_dt: Optional[float] = None


@app.on_event("startup")
def startup():
    sc.load_model()


@app.get("/health")
def health():
    return {"status": "ok", "service": "motion"}


@app.post("/plan")
def plan(req: MotionPlanRequest):
    try:
        result = sc.plan_motion(
            joint_positions=req.joint_positions,
            goal_pose=req.goal_pose.pose,
            joint_names=req.joint_names,
            world=req.world,
            max_attempts=req.max_attempts,
            time_dilation_factor=req.time_dilation_factor,
            interpolation_dt=req.interpolation_dt,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))