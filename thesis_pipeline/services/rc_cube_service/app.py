from fastapi import FastAPI, HTTPException
import yaml

from service_core import RcCubeGrpcClient
import data_io as io_utils
app = FastAPI()

CONFIG_PATH = "/app/config.yaml"


@app.on_event("startup")
def startup_event():
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
        cfg = load_config()

        if not cfg["pipeline"].get("use_rc_cube", False):
            return {
                "status": "skipped",
                "reason": "pipeline.use_rc_cube = false",
            }

        rc = cfg["rc_cube"]
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

        debug_save_output = rc.get("debug_save_output", False)
        base_dir = cfg["paths"]["output_dir"]

        result = {"status": "ok"}

        if debug_save_output:
            paths = io_utils.save_rc_cube_output(
                left_rgb=left_rgb,
                disp_arr=disp_arr,
                cam=cam,
                disp_params=disp_params,
                base_dir=base_dir,
            )
            result.update(paths)

        uois_dict = io_utils.disparity_to_uois_dict(
            disp_arr=disp_arr,
            left_rgb=left_rgb,
            cam=cam,
            disp_params=disp_params,
            conf=None,
            seg_id=1,
            background_label=0,
        )

        result["debug"] = {
            "left_rgb_shape": list(left_rgb.shape),
            "rgb_min": int(left_rgb.min()),
            "rgb_max": int(left_rgb.max()),
            "rgb_mean": float(left_rgb.mean()),
            "disp_shape": list(disp_arr.shape),
            "cam_hw": [int(cam["height"]), int(cam["width"])],
        }


        uois_npy_path = f"{base_dir}/uois_input"
        io_utils.save_uois_npy(uois_dict, uois_npy_path)
        result["uois_npy_path"] = uois_npy_path

        uois_ply_path = f"{base_dir}/uois_debug.ply"
        io_utils.save_uois_dict_to_ply(
            uois_dict,
            uois_ply_path,
        )
        result["uois_ply_path"] = uois_ply_path

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))