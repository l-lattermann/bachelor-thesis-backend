from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import io
import yaml

import service_core as cs


app = FastAPI()

CONFIG_PATH = "/app/config.yaml"
cfg = None


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class InferenceRequest(BaseModel):
    npz_path: str


@app.on_event("startup")
def startup_event():
    global cfg
    cfg = load_config()
    cs.load_model()
    print("============================", flush=True)
    print("===     CGN SERVICE       ==", flush=True)
    print("============================", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "service": "contact_graspnet"}


@app.post("/inference")
def inference(req: InferenceRequest):
    try:
        npz_path = Path(req.npz_path)
        if not npz_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {npz_path}")

        result = cs.run_contact_graspnet(str(npz_path))

        # optional visualization output
        image_paths = cs.save_cgn_output_image(
            pc_full=result["pc_full"],
            pred_grasps_cam=result["pred_grasps_cam"],
            scores=result["scores"],
            segmap=result.get("segmap"),
            rgb=result.get("rgb"),
            pc_colors=result.get("pc_colors"),
            gripper_openings=result.get("gripper_openings"),
        )

        buf = io.BytesIO()
        np.savez(
            buf,
            pred_grasps_cam=result["pred_grasps_cam"],
            scores=result["scores"],
            contact_pts=result["contact_pts"],
            gripper_openings=result["gripper_openings"],
        )
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="application/octet-stream",
            headers={
                "X-CGN-Grasps-Image": image_paths["grasps_img"],
                "X-CGN-Segmap-Image": image_paths["segmap_img"] or "",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))