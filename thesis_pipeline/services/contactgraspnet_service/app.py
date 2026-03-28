from io import BytesIO
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import service_core as cs
from grasp_selection import process_contact_graspnet_result
from projection_utils import save_projected_grasp_overlay


app = FastAPI(title="Contact-GraspNet Service")

CONFIG_PATH = Path("/app/config.yaml")
CFG = None


class InferenceRequest(BaseModel):
    npz_path: str


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

        result = cs.run_contact_graspnet(str(npz_path))

        result = process_contact_graspnet_result(
            result=result,
            sel_cfg=CFG["contact_graspnet"]["selection"],
        )

        # --- DEBUG PRINT ---
        print("\n=== SELECTED GRASPS DEBUG ===", flush=True)

        for key in result["pred_grasps_cam"]:
            grasps_k = np.asarray(result["pred_grasps_cam"][key])
            scores_k = np.atleast_1d(result["scores"][key])
            contacts_k = np.atleast_2d(result["contact_pts"][key])
            openings_k = np.atleast_1d(result["gripper_openings"][key])

            if len(grasps_k) == 0:
                print(f"[key={key}] no grasps", flush=True)
                continue

            for i in range(len(grasps_k)):
                x, y, z = contacts_k[i]
                score = scores_k[i]
                opening = openings_k[i] if len(openings_k) > 0 else -1.0

                print(
                    f"[key={key}] grasp {i+1}: "
                    f"score={score:.3f}, "
                    f"opening={opening:.3f} m, "
                    f"contact=({x:.3f}, {y:.3f}, {z:.3f})",
                    flush=True,
                )

        print("================================\n", flush=True)

        use_default_opening = CFG["contact_graspnet"]["gripper_projection"]["use_default_opening"]
        default_opening = CFG["contact_graspnet"]["gripper_projection"]["default_opening"]
        draw_confidence = CFG["contact_graspnet"]["gripper_projection"]["draw_confidence"]

        overlay_path = save_projected_grasp_overlay(
            rgb=result.get("rgb"),
            K=result.get("K"),
            pred_grasps_cam=result["pred_grasps_cam"],
            scores=result["scores"],
            gripper_openings=result["gripper_openings"],
            draw_default_opening=use_default_opening,
            default_opening=default_opening,
            draw_confidence=draw_confidence,
            output_path="/shared/pipeline_io/cgn_output.png",
        )

        buf = BytesIO()
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
                "X-CGN-Overlay": overlay_path or "",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))