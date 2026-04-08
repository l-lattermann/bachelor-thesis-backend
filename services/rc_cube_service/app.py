from fastapi import FastAPI, HTTPException
import yaml
import os

from service_core import RcCubeGrpcClient
import data_io as io_utils

app = FastAPI()

CONFIG_PATH = "/app/config.yaml"

CFG = RC_CFG = PIPELINE_CFG = PATHS_CFG = None
OUT_DIR = DEBUG_DIR = RC_CAM_MOCK_DIR = RC_FULL_MOCK_DIR = None

client = None


@app.on_event("startup")
def startup():
    global CFG, RC_CFG, PIPELINE_CFG, PATHS_CFG
    global client, OUT_DIR, DEBUG_DIR, RC_CAM_MOCK_DIR, RC_FULL_MOCK_DIR
    global OUTPUT_NPZ_PATH, OUTPUT_LEFT_IMG_PATH

    CFG = yaml.safe_load(open(CONFIG_PATH).read())

    RC_CFG = CFG["rc_cube"]
    PIPELINE_CFG = CFG["pipeline"]
    PATHS_CFG = CFG["paths"]

    ip = os.environ["RC_CUBE_IP"]

    OUT_DIR = PATHS_CFG["pipeline_file_share"]

    DEBUG_BASE_DIR = PATHS_CFG["output_debug"]
    DEBUG_DIR = os.path.join(DEBUG_BASE_DIR, "rc_cube")

    RC_CAM_MOCK_DIR = RC_CFG["rc_mock_cam_input"]
    RC_FULL_MOCK_DIR = RC_CFG["rc_mock_full_input"]

    OUTPUT_NPZ_PATH = RC_CFG["output_npz"]
    OUTPUT_LEFT_IMG_PATH = RC_CFG["output_left_img"]

    # ensure dirs exist
    os.makedirs(os.path.dirname(OUTPUT_NPZ_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_LEFT_IMG_PATH), exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(RC_CAM_MOCK_DIR, exist_ok=True)

    client = RcCubeGrpcClient(RC_CFG, ip)

    print("===============================")
    print("===     RC CUBE SERVICE      ==")
    print("===============================")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/fetch_disparity_and_left")
def fetch_disparity_and_left():
    try:
        debug = CFG["project"]["debug"]

        if PIPELINE_CFG["mock_rc_cube_cam"]:
            left_rgb, disp_arr, cam, disp_params = io_utils.load_rc_cube_mock_cam_output(RC_CAM_MOCK_DIR)

        elif PIPELINE_CFG["mock_rc_cube_full"]:
            npz_path, left_path = io_utils.gen_mock_output_from_npz(
                RC_FULL_MOCK_DIR, 
                OUTPUT_NPZ_PATH, 
                OUTPUT_LEFT_IMG_PATH,
            )

            return {
                "status": "ok",
                "rc_out_npz": npz_path,
                "left_png_path": left_path,
                "debug": "USING FULL RC CUBE MOCK OUTPUT",
            }

        else:
            left_rgb, disp_arr, conf_arr, cam, disp_params = client.get_disparity_and_left(
                left_enabled=RC_CFG["left_enabled"],
                right_enabled=RC_CFG["right_enabled"],
                disparity_enabled=RC_CFG["disparity_enabled"],
                disparity_error_enabled=RC_CFG["disparity_error_enabled"],
                confidence_enabled=RC_CFG["confidence_enabled"],
                mesh_enabled=RC_CFG["mesh_enabled"],
                color=RC_CFG["color"],
                timeout=RC_CFG["timeout_sec"],
            )

        rc_output = io_utils.process_rc_cube_output(
            left_rgb=left_rgb,
            disp_arr=disp_arr,
            cam=cam,
            disp_params=disp_params,
            conf=conf_arr,
            conf_thr=RC_CFG["conf_threshhold"],
            output_npz_path=OUTPUT_NPZ_PATH,
            output_left_img_path=OUTPUT_LEFT_IMG_PATH,
            debug=debug,
            debug_base_dir=DEBUG_DIR,
            save_pointcloud_npc_with_timestamp=debug,
        )

        return {
            "status": "ok",
            "rc_out_npz": rc_output["rc_out_npz"],
            "left_png_path": rc_output["left_png_path"],
            "debug": rc_output,
        }

    except Exception as e:
        raise HTTPException(500, str(e))