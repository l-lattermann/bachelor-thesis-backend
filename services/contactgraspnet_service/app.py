from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import service_core as cs
from grasp_selection import process_contact_graspnet_result
from projection_utils import save_projected_grasp_overlay


app = FastAPI(title="Contact-GraspNet Service")

CONFIG_PATH = Path("/app/config.yaml")
CFG = None


class InferenceRequest(BaseModel):
    npz_path: str
    object_id: Optional[int] = None


def load_config() -> dict:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def to_shape(x):
    return list(np.asarray(x).shape) if x is not None else None


def to_list(x):
    return np.asarray(x).tolist() if x is not None else None


@app.on_event("startup")
def startup_event():
    global CFG
    CFG = load_config()
    cs.load_model()


@app.get("/health")
def health():
    return {"status": "ok", "service": "contact_graspnet"}


@app.post("/inference")
def inference(req: InferenceRequest):
    try:
        npz_path = Path(req.npz_path)
        if not npz_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {npz_path}")

        object_id = int(req.object_id)
            
        run_result = cs.run_contact_graspnet(
            str(npz_path),
            object_id=object_id,
        )

        proc_result = process_contact_graspnet_result(
            result=run_result,
            sel_cfg=CFG["contact_graspnet"]["selection"],
        )

        projection_cfg = CFG["contact_graspnet"]["gripper_projection"]

        overlay_path = save_projected_grasp_overlay(
            rgb=proc_result.get("rgb"),
            K=proc_result.get("K"),
            pred_grasps_cam=proc_result["pred_grasps_cam"],
            scores=proc_result["scores"],
            gripper_openings=proc_result["gripper_openings"],
            draw_default_opening=projection_cfg["use_default_opening"],
            default_opening=projection_cfg["default_opening"],
            draw_confidence=projection_cfg["draw_confidence"],
            output_path=str(Path(CFG["paths"]["pipeline_file_share"]) / "cgn_output.png"),
        )

        response = {
            "result_path": overlay_path,
            "num_grasps": sum(len(v) for v in proc_result["pred_grasps_cam"].values()),
        }

        if CFG.get("project", {}).get("debug"):
            response["debug"] = {
                "npz_path": str(npz_path),
                "object_id": object_id,
                "rgb_shape": to_shape(proc_result.get("rgb")),
                "K": tuple(float(x) for x in np.asarray(proc_result.get("K")).flatten()) if proc_result.get("K") is not None else None,
                "scores_stats": {
                    str(k): {
                        "min": float(np.min(np.asarray(v))) if len(np.asarray(v)) > 0 else None,
                        "max": float(np.max(np.asarray(v))) if len(np.asarray(v)) > 0 else None,
                        "mean": float(np.mean(np.asarray(v))) if len(np.asarray(v)) > 0 else None,
                    }
                    for k, v in proc_result["scores"].items()
                }
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))