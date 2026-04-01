from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

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


@app.on_event("startup")
def startup_event():
    global CFG
    CFG = load_config()
    cs.load_model()

    print("============================")
    print("===     CGN SERVICE       ==")
    print("============================")


@app.get("/health")
def health():
    return {"status": "ok", "service": "contact_graspnet"}



@app.post("/inference")
def inference(req: InferenceRequest):
    try:
        npz_path = Path(req.npz_path)
        if not npz_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {npz_path}")

        object_id = (
            int(req.object_id)
            if req.object_id is not None
            else int(CFG["contact_graspnet"]["prediction"]["segmap_id"])
        )

        result = cs.run_contact_graspnet(
            str(npz_path),
            object_id=object_id,
        )

        result = process_contact_graspnet_result(
            result=result,
            sel_cfg=CFG["contact_graspnet"]["selection"],
        )

        print("\n=== SELECTED GRASPS DEBUG ===")
        for key in result["pred_grasps_cam"]:
            grasps = np.asarray(result["pred_grasps_cam"][key])
            scores = np.atleast_1d(result["scores"][key])
            contacts = np.atleast_2d(result["contact_pts"][key])
            openings = np.atleast_1d(result["gripper_openings"][key])

            if len(grasps) == 0:
                print(f"[key={key}] no grasps")
                continue

            for i in range(len(grasps)):
                x, y, z = contacts[i]
                score = scores[i]
                opening = openings[i] if len(openings) > 0 else -1.0

                print(
                    f"[key={key}] grasp {i+1}: "
                    f"score={score:.3f}, "
                    f"opening={opening:.3f} m, "
                    f"contact=({x:.3f}, {y:.3f}, {z:.3f})",
                    flush=True,
                )
        print("================================\n")

        projection_cfg = CFG["contact_graspnet"]["gripper_projection"]

        overlay_path = save_projected_grasp_overlay(
            rgb=result.get("rgb"),
            K=result.get("K"),
            pred_grasps_cam=result["pred_grasps_cam"],
            scores=result["scores"],
            gripper_openings=result["gripper_openings"],
            draw_default_opening=projection_cfg["use_default_opening"],
            default_opening=projection_cfg["default_opening"],
            draw_confidence=projection_cfg["draw_confidence"],
            output_path=str(Path(CFG["paths"]["pipeline_file_share"]) / "cgn_output.png"),
        )

        return {
            "result_path": overlay_path,
            "num_grasps": sum(len(v) for v in result["pred_grasps_cam"].values()),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))