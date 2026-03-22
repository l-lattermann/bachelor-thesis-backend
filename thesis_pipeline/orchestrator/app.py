import io
from pathlib import Path

import numpy as np
import requests
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

CONFIG_PATH = "/app/config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


CFG = load_config()

UOIS_URL = CFG["network"]["uois_service_url"]
CGN_URL = CFG["network"]["contact_graspnet_service_url"]
RC_CUBE_URL = CFG["network"]["rc_cube_service_url"]


class PipelineRequest(BaseModel):
    npy_path: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator"}


def resolve_input_npy(req: PipelineRequest) -> str:
    """
    Decide where the pipeline input comes from.

    Priority:
    1. explicit request npy_path
    2. RC-Cube if enabled in config
    3. fallback to config-defined input npy
    """
    use_rc_cube = CFG["pipeline"]["use_rc_cube"]

    if req.npy_path:
        return req.npy_path

    if use_rc_cube:
        fetch_mode = CFG["rc_cube"]["fetch_mode"]
        out_npy = CFG["paths"]["rc_cube"]["uois_npy"]
        rc_cube_ip = CFG["rc_cube"]["ip"]

        if fetch_mode != "disparity":
            raise RuntimeError(
                f"Currently only rc_cube fetch_mode='disparity' is supported, got '{fetch_mode}'"
            )

        rc_resp = requests.post(
            f"{RC_CUBE_URL}/fetch_disparity_npy",
            json={
                "rc_cube_ip": rc_cube_ip,
                "out_path": out_npy,
            },
            timeout=300,
        )
        rc_resp.raise_for_status()
        rc_result = rc_resp.json()

        return rc_result["npy_path"]

    fallback_path = CFG["paths"]["rc_cube"]["uois_npy"]
    if not Path(fallback_path).exists():
        raise FileNotFoundError(
            f"No request npy_path provided, RC-Cube disabled, and fallback input does not exist: {fallback_path}"
        )
    return fallback_path


@app.post("/run_pipeline")
def run_pipeline(req: PipelineRequest):
    try:
        # 0) resolve pipeline input
        npy_path = resolve_input_npy(req)

        # 1) run UOIS
        uois_resp = requests.post(
            f"{UOIS_URL}/predict",
            json={"npy_path": npy_path},
            timeout=300,
        )
        uois_resp.raise_for_status()
        uois_result = uois_resp.json()
        mask_path = uois_result["mask_path"]

        # 2) load original input + attach segmentation
        data = np.load(npy_path, allow_pickle=True).item()
        seg = np.load(mask_path)
        data["seg"] = seg

        # 3) send binary npy to CGN
        buf = io.BytesIO()
        np.save(buf, data, allow_pickle=True)
        buf.seek(0)

        cgn_resp = requests.post(
            f"{CGN_URL}/inference",
            data=buf.getvalue(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=600,
        )
        cgn_resp.raise_for_status()

        result_npz = np.load(io.BytesIO(cgn_resp.content), allow_pickle=True)

        # 4) save final output
        out_path = CFG["paths"]["contact_graspnet"]["output_npz"]
        np.savez(
            out_path,
            pred_grasps_cam=result_npz["pred_grasps_cam"],
            scores=result_npz["scores"],
            contact_pts=result_npz["contact_pts"],
        )

        return {
            "status": "ok",
            "input_npy": npy_path,
            "uois_mask": mask_path,
            "cgn_output": out_path,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))