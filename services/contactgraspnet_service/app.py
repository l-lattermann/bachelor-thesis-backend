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

CFG = CGN_CFG = SELECTION_CFG = KMEANS_CFG = PROJECTION_CFG = None
OUTPUT_PATH = CHECKPOINT_DIR = None


class InferenceRequest(BaseModel):
    npz_path: str
    object_id: Optional[int] = None


@app.on_event("startup")
def startup():
    global CFG, CGN_CFG, SELECTION_CFG, KMEANS_CFG, PROJECTION_CFG
    global OUTPUT_PATH, CHECKPOINT_DIR

    CFG = yaml.safe_load(CONFIG_PATH.read_text())

    CGN_CFG = CFG["contact_graspnet"]

    SELECTION_CFG = CGN_CFG["selection"]
    KMEANS_CFG = SELECTION_CFG["k_means"]
    PROJECTION_CFG = CGN_CFG["gripper_projection"]

    OUTPUT_PATH = CGN_CFG["output_png"]
    CHECKPOINT_DIR = CGN_CFG["checkpoint_dir"]

    cs.load_model(checkpoint_dir=CHECKPOINT_DIR)

    print("============================")
    print("===      CGN SERVICE     ===")
    print("============================")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inference")
def inference(req: InferenceRequest):
    try:
        npz_path = Path(req.npz_path)
        if not npz_path.exists():
            raise HTTPException(404, f"File not found: {npz_path}")

        object_id = int(req.object_id) if req.object_id is not None else None

        run_result = cs.run_contact_graspnet(
            str(npz_path),
            object_id=object_id,
        )

        proc_result = process_contact_graspnet_result(
            result=run_result,
            num_grasps=SELECTION_CFG["num_grasps"],
            top_score_candidates=SELECTION_CFG["top_score_candidates"],
            use_k_means=SELECTION_CFG["k_means_clustering"],
            kmeans_n_init=KMEANS_CFG["kmeans_n_init"],
            kmeans_random_state=KMEANS_CFG["kmeans_random_state"],
            orientation_weight=KMEANS_CFG["orientation_weight"],
        )

        overlay_path = save_projected_grasp_overlay(
            rgb=proc_result["rgb"],
            K=proc_result["K"],
            pred_grasps_cam=proc_result["pred_grasps_cam"],
            gripper_openings=proc_result["gripper_openings"],
            draw_default_opening=PROJECTION_CFG["use_default_opening"],
            default_opening=PROJECTION_CFG["default_opening"],
            gripper_line_width=PROJECTION_CFG["gripper_line_width"],
            number_line_width=PROJECTION_CFG["number_line_width"],
            output_path=OUTPUT_PATH,
        )

        return {
            "result_path": overlay_path,
            "num_grasps": sum(len(v) for v in proc_result["pred_grasps_cam"].values()),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))