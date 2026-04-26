from pathlib import Path
from typing import Optional
import traceback

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import service_core as cs
from grasp_selection import process_contact_graspnet_result, save_selected_cgn_output
from projection_utils import save_grasp_score_heatmap


app = FastAPI(title="Contact-GraspNet Service")

CONFIG_PATH = Path("/app/config.yaml")

CFG = CGN_CFG = SELECTION_CFG = DBSCAN_CFG = None
OUTPUT_PATH = CHECKPOINT_DIR = HEATMAP_DIR = SEL_GRASPS_NPZ_PATH = None


class InferenceRequest(BaseModel):
    npz_path: str
    object_id: Optional[int] = None


@app.on_event("startup")
def startup():
    global CFG, CGN_CFG, SELECTION_CFG, DBSCAN_CFG
    global OUTPUT_PATH, CHECKPOINT_DIR, HEATMAP_DIR, SEL_GRASPS_NPZ_PATH

    CFG = yaml.safe_load(CONFIG_PATH.read_text())

    CGN_CFG = CFG["contact_graspnet"]

    SELECTION_CFG = CGN_CFG["selection"]
    DBSCAN_CFG = SELECTION_CFG["dbscan"]

    OUTPUT_PATH = CGN_CFG["output_png"]
    HEATMAP_DIR = CGN_CFG["heatmap_dir"]
    SEL_GRASPS_NPZ_PATH = CGN_CFG["sel_grasps"]
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

        if CFG["project"]["debug"]:
            print("\n===== HEATMAP DEBUG =====")

            for k in run_result["pred_grasps_cam"].keys():
                pts = run_result["contact_pts"].get(k, [])
                grasps = run_result["pred_grasps_cam"].get(k, [])
                sc = run_result["scores"].get(k, [])

                print(f"key: {k}")
                print(f"  contact_pts: {len(pts)}")
                print(f"  pred_grasps_cam: {len(grasps)}")
                print(f"  scores: {len(sc)}")

            print("=========================\n")
            heatmap_path = save_grasp_score_heatmap(
                rgb=run_result["rgb"],
                K=run_result["K"],
                contact_pts=run_result["contact_pts"],
                pred_grasps_cam=run_result["pred_grasps_cam"],
                scores=run_result["scores"],
            )

        else:
            heatmap_path = None

        proc_result = process_contact_graspnet_result(
            result=run_result,
            num_grasps=SELECTION_CFG["num_grasps"],
            top_score_candidates=SELECTION_CFG["top_score_candidates"],
            use_dbscan=SELECTION_CFG["dbscan_clustering"],
            dbscan_min_score=DBSCAN_CFG["min_score"],
            dbscan_eps=DBSCAN_CFG["eps"],
            dbscan_min_samples=DBSCAN_CFG["min_samples"],
            orientation_weight=DBSCAN_CFG["orientation_weight"],
        )

        sel_grasps_npz = save_selected_cgn_output(
            SEL_GRASPS_NPZ_PATH,
            pred_grasps_cam=proc_result["pred_grasps_cam"],
            scores=proc_result["scores"],
            gripper_openings=proc_result["gripper_openings"],
            pc_full=run_result["pc_full"],
            segmap=run_result["segmap"],
            rgb=run_result["rgb"],
            pc_colors=run_result["pc_colors"],
        )

        return {
            "sel_grasps_npz": sel_grasps_npz,
            "heatmap_path": heatmap_path,
            "num_grasps": sum(len(v) for v in proc_result["pred_grasps_cam"].values()),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
        )