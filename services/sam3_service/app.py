from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml
import os
import time 

import service_core as sc
import data_io as io_utils

app = FastAPI()

CONFIG_PATH = "/app/config.yaml"

CFG = SAM_CFG = PATHS_CFG = None
OUT_DIR = DEBUG_DIR = None
OUT_PATH_NPZ = OUT_PATH_PNG = None

CHECKPOINT_PATH = None


class SAMRequest(BaseModel):
    npz_path: str
    image_path: str


@app.on_event("startup")
def startup():
    global CFG, SAM_CFG, PATHS_CFG
    global OUT_DIR, DEBUG_DIR, OUT_PATH_NPZ, OUT_PATH_PNG, CHECKPOINT_PATH

    with open(CONFIG_PATH, "r") as f:
        CFG = yaml.safe_load(f)

    SAM_CFG = CFG["sam"]
    PATHS_CFG = CFG["paths"]

    OUT_DIR = PATHS_CFG["pipeline_file_share"]
    DEBUG_DIR = PATHS_CFG["output_debug"]

    OUT_PATH_NPZ = SAM_CFG["output_npz"]
    OUT_PATH_PNG = SAM_CFG["output_png"]
    CHECKPOINT_PATH = SAM_CFG["checkpoint"]

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PATH_NPZ), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PATH_PNG), exist_ok=True)

    sc.load_model(
        checkpoint_path=CHECKPOINT_PATH,
        device=SAM_CFG["device"],
        warmup=True,
    )

    print("============================")
    print("===      SAM SERVICE     ===")
    print("============================")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: SAMRequest):
    try:
        t0 = time.perf_counter()

        if not os.path.exists(req.npz_path):
            raise HTTPException(404, f"NPZ not found: {req.npz_path}")
        if not os.path.exists(req.image_path):
            raise HTTPException(404, f"Image not found: {req.image_path}")

        t1 = time.perf_counter()
        rgb_xyz_aligned, xyz, cam = io_utils.load_rc_cube_npz(req.npz_path)
        left_img = io_utils.load_left_rgb(req.image_path)
        t2 = time.perf_counter()

        result = sc.run_sam_pipeline(
            image=left_img,
            target_hw=xyz.shape[:2],
            min_area=SAM_CFG["min_area"],
            max_area_ratio=SAM_CFG["max_area_ratio"],
        )
        t3 = time.perf_counter()

        io_utils.save_annotated_masks_outline(
            image=left_img,
            masks=result["ordered_masks_fullres"],
            output_path=OUT_PATH_PNG,
        )

        path = OUT_PATH_PNG + "_raw.png"
        io_utils.save_annotated_masks_overlay(
            image=left_img,
            masks=result["raw_masks"],
            output_path=path,
        )
        t4 = time.perf_counter()

        io_utils.save_output_npz(
            output_path=OUT_PATH_NPZ,
            rgb=rgb_xyz_aligned,
            xyz=xyz,
            seg=result["seg"],
            cam=cam,
        )
        t5 = time.perf_counter()

        return {
            "status": "ok",
            "result_path": OUT_PATH_NPZ,
            "rgb_annotated_path": OUT_PATH_PNG,
            "num_masks_raw": len(result["raw_masks"]),
            "num_masks_filtered": len(result["ordered_masks_fullres"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))