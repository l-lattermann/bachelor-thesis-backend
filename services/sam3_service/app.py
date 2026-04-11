from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml

import service_core as sc
import data_io as io_utils

app = FastAPI()

CONFIG_PATH = "/app/config.yaml"

CFG = None
SAM_CFG = None

OUT_PATH_NPZ = None
OUT_PATH_PNG = None


class SAMRequest(BaseModel):
    npz_path: str
    image_path: str
    text_prompt: str


@app.on_event("startup")
def startup():
    global CFG, SAM_CFG
    global OUT_PATH_NPZ, OUT_PATH_PNG

    with open(CONFIG_PATH, "r") as f:
        CFG = yaml.safe_load(f)

    SAM_CFG = CFG["sam3"]
    OUT_PATH_NPZ = SAM_CFG["output_npz"]
    OUT_PATH_PNG = SAM_CFG["output_png"]

    Path(OUT_PATH_NPZ).parent.mkdir(parents=True, exist_ok=True)

    sc.load_model(
        hf_token=os.getenv("HF_TOKEN"),
        device=SAM_CFG["device"],
        confidence_threshold=SAM_CFG["confidence_threshold"],
        warmup=False,
    )

    print("==============================")
    print("===      SAM 3 SERVICE     ===")
    print("==============================")


@app.get("/health")
def health():
    return {"status": "ok", "service": "sam3"}


@app.post("/predict")
def predict(req: SAMRequest):
    try:
        if not os.path.exists(req.npz_path):
            raise HTTPException(404, f"NPZ not found: {req.npz_path}")
        if not os.path.exists(req.image_path):
            raise HTTPException(404, f"Image not found: {req.image_path}")

        rgb, xyz, cam = io_utils.load_rc_cube_npz(req.npz_path)
        image = io_utils.load_left_rgb(req.image_path)

        result = sc.generate_masks(
            image=image,
            text_prompt=req.text_prompt,
            confidence_threshold=SAM_CFG["confidence_threshold"],
        )

        seg = io_utils.masks_to_segmentation(result["masks"], image.shape)

        io_utils.save_output_npz(
            output_path=OUT_PATH_NPZ,
            rgb=rgb,
            xyz=xyz,
            seg=seg,
            cam=cam,
        )

        io_utils.save_debug_plot(
            image=image,
            state=result["state"],
            output_path=OUT_PATH_PNG,
        )

        return {
            "status": "ok",
            "result_path": OUT_PATH_NPZ,
            "rgb_annotated_path": OUT_PATH_PNG,
            "mask_count": result["mask_count"],
            "boxes": result["boxes"].tolist() if result["boxes"] is not None else [],
            "scores": result["scores"].tolist() if result["scores"] is not None else [],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))