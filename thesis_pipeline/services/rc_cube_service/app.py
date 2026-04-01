from fastapi import FastAPI, HTTPException
import yaml
import os

from service_core import RcCubeGrpcClient
import data_io as io_utils

app = FastAPI()

CONFIG_PATH = "/app/config.yaml"
cfg = None


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@app.on_event("startup")
def startup_event():
    global cfg
    cfg = load_config()

    print("===============================")
    print("===     RC CUBE SERVICE      ==")
    print("===============================")


@app.get("/health")
def health():
    return {"status": "ok", "service": "rc_cube_service"}


@app.post("/fetch_disparity_and_left")
def fetch_disparity_and_left():
    try:
        global cfg

        if not cfg.get("pipeline", {}).get("use_rc_cube", False):
            return {
                "status": "skipped",
                "reason": "pipeline.use_rc_cube = false",
            }

        rc = cfg.get("rc_cube", {})

        paths_cfg = cfg.get("paths", {})
        out_dir = paths_cfg.get("pipeline_file_share", "/shared/pipeline_io")
        debug_base_dir = paths_cfg.get("output_debug", "/shared/debug")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(debug_base_dir, exist_ok=True)


        debug = cfg.get("project", {}).get("debug", False)

        client = RcCubeGrpcClient(CONFIG_PATH)

        left_rgb, disp_arr, cam, disp_params = client.get_disparity_and_left(
            left_enabled=rc.get("left_enabled", True),
            right_enabled=rc.get("right_enabled", False),
            disparity_enabled=rc.get("disparity_enabled", True),
            disparity_error_enabled=rc.get("disparity_error_enabled", False),
            confidence_enabled=rc.get("confidence_enabled", False),
            mesh_enabled=rc.get("mesh_enabled", False),
            color=rc.get("color", True),
            timeout=rc.get("timeout_sec"),
        )

        rc_output = io_utils.process_rc_cube_output(
            left_rgb=left_rgb,
            disp_arr=disp_arr,
            cam=cam,
            disp_params=disp_params,
            out_dir=out_dir,                  
            debug=debug,
            debug_base_dir=debug_base_dir,   
        )
        return {
            "status": "ok",
            "result_path": rc_output["rc_out_npz"],
            "debug": rc_output,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))