from fastapi import FastAPI, HTTPException
import yaml
import os
import numpy as np

from service_core import RcCubeGrpcClient
import data_io as io_utils
app = FastAPI()

CONFIG_PATH = "/app/config.yaml"
cfg = None

@app.on_event("startup")
def startup_event():
    global cfg
    cfg = load_config()
    print("===============================")
    print("===     RC CUBE SERVICE      ==")
    print("===============================")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@app.get("/health")
def health():
    return {"status": "ok", "service": "rc_cube_service"}


@app.post("/fetch_disparity_and_left")
def fetch_disparity_and_left():
    try:
        global cfg

        # --- guard ---
        if not cfg.get("pipeline", {}).get("use_rc_cube", False):
            return {
                "status": "skipped",
                "reason": "pipeline.use_rc_cube = false",
            }

        # --- config ---
        rc = cfg.get("rc_cube", {})
        out_dir = cfg["paths"]["pipeline_file_share"]
       
        # --- fetch ---
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

        # --- processing ---
        uois_dict = io_utils.disparity_to_uois_dict(
            disp_arr=disp_arr,
            left_rgb=left_rgb,
            cam=cam,
            disp_params=disp_params,
            conf=None,
            seg_id=1,
            background_label=0,
        )

        # --- pipeline outputs ---
        os.makedirs(out_dir, exist_ok=True)
        npz_path = os.path.join(out_dir, "rc_cube_output.npz")
        np.savez_compressed(
            npz_path,
            rgb=uois_dict["rgb"],
            xyz=uois_dict["xyz"],
            label=uois_dict["label"],
        )

        cam_yaml = os.path.join(out_dir, "cam.yaml")
        with open(cam_yaml, "w") as f:
            yaml.safe_dump(cam, f, sort_keys=False)

        result = {
            "status": "ok",
            "rc_out_npz": npz_path,
            "cam_yaml": cam_yaml,
        }

        # --- debug outputs ---
        debug = cfg.get("project", {}).get("debug", False)
        debug_out_dir = os.path.join(cfg.get("paths", {}).get("output_debug", "/shared/debug"), "rc_cube")
        os.makedirs(debug_out_dir, exist_ok=True)

        if debug:
            os.makedirs(debug_out_dir, exist_ok=True)

            debug_paths = io_utils.save_rc_cube_output(
                left_rgb=left_rgb,
                disp_arr=disp_arr,
                cam=cam,
                disp_params=disp_params,
                base_dir=debug_out_dir,
            )

            uois_ply_path = os.path.join(debug_out_dir, "rc_cube_debug_uois_format.ply")
            io_utils.save_uois_dict_to_ply(uois_dict, uois_ply_path)

            result.update(debug_paths)
            result["uois_ply_path"] = uois_ply_path

            # --- debug stats ---
            result["debug"] = {
                "left_rgb_shape": list(left_rgb.shape),
                "rgb_min": int(left_rgb.min()),
                "rgb_max": int(left_rgb.max()),
                "rgb_mean": float(left_rgb.mean()),
                "disp_shape": list(disp_arr.shape),
                "cam_hw": [int(cam.get("height", 0)), int(cam.get("width", 0))],
            }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))