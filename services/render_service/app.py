from pathlib import Path
import time
import traceback

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import service_core


APP_TITLE = "Render Service"
CONFIG_PATH = Path("/app/config.yaml")


class RenderRequest(BaseModel):
    npz_path: str


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_runtime_config(cfg: dict) -> dict:
    render_cfg = cfg["rendering"]["render"]
    out_cfg = cfg["rendering"]["output"]
    cam_cfg = cfg["rendering"]["camera_views"]

    return {
        "width": int(render_cfg["width"]),
        "height": int(render_cfg["height"]),
        "max_points": int(render_cfg["max_points"]),
        "gripper_width": float(render_cfg["gripper_width"]),
        "gripper_opening_offset": float(render_cfg["gripper_opening_offset"]),
        "finger_len": float(render_cfg["finger_len"]),
        "point_size": float(render_cfg["point_size"]),
        "line_radius": float(render_cfg["line_radius"]),
        "label_offset": float(render_cfg["label_offset"]),
        "approach_offset": float(render_cfg["approach_offset"]),
        "label_left_shift_px": int(render_cfg["label_left_shift_px"]),
        "label_font_scale": float(render_cfg["label_font_scale"]),
        "label_text_thickness": int(render_cfg["label_text_thickness"]),
        "label_outline_thickness": int(render_cfg["label_outline_thickness"]),
        "background_rgba": [float(x) for x in render_cfg["background_rgba"]],
        "colormap": str(render_cfg["colormap"]),
        "left_view_filename": str(out_cfg["left_view_filename"]),
        "right_view_filename": str(out_cfg["right_view_filename"]),
        "output_dir": str(out_cfg["output_dir"]),
        "top_eye_dir": [float(x) for x in cam_cfg["top"]["eye_dir"]],
        "top_up": [float(x) for x in cam_cfg["top"]["up"]],
        "top_dist": float(cam_cfg["top"]["dist"]),
        "top_fov": float(cam_cfg["top"]["fov"]),
        "front_eye_dir": [float(x) for x in cam_cfg["front"]["eye_dir"]],
        "front_up": [float(x) for x in cam_cfg["front"]["up"]],
        "front_dist": float(cam_cfg["front"]["dist"]),
        "front_fov": float(cam_cfg["front"]["fov"]),
    }


app = FastAPI(title=APP_TITLE)

CFG = RUNTIME_CFG = None


@app.on_event("startup")
def startup_event():
    global CFG, RUNTIME_CFG

    CFG = load_config()
    RUNTIME_CFG = build_runtime_config(CFG)
    service_core.validate_config(RUNTIME_CFG)
    service_core.start_render_worker(RUNTIME_CFG)

    print("===============================")
    print("===     RENDER SERVICE       ==")
    print("===============================")


@app.get("/health")
def health():
    return {"status": "ok", "service": "render_service"}


@app.post("/render")
def render(req: RenderRequest):
    try:
        result = service_core.render_from_npz(
            npz_path=req.npz_path,
            runtime_cfg=RUNTIME_CFG,
        )

        return {
            "status": "ok",
            "npz_path": result["npz_path"],
            "top_render_path": result["top_render_path"],
            "front_render_path": result["front_render_path"],
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
        )