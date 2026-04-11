from pathlib import Path
import time
import traceback
from typing import Optional

import requests
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Orchestrator")

CONFIG_PATH = Path("/app/config.yaml")

CFG = None
NET = None
PIPELINE_CFG = None


class PipelineRequest(BaseModel):
    object_query: Optional[str] = None


def wait_for_health(name: str, url: str, retries: int = 120, delay: float = 2.0) -> None:
    last = None
    for _ in range(retries):
        try:
            r = requests.get(f"{url}/health", timeout=10)
            if r.status_code == 200:
                return
            last = r.text
        except Exception as e:
            last = str(e)
        time.sleep(delay)

    raise RuntimeError(f"{name} not reachable: {last}")


def post_json(url: str, payload: dict, name: str) -> dict:
    try:
        r = requests.post(url, json=payload, timeout=120)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{name} failed: {e}")

    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"{name} error: {r.text}")

    return r.json()


@app.on_event("startup")
def startup() -> None:
    global CFG, NET, PIPELINE_CFG

    CFG = yaml.safe_load(CONFIG_PATH.read_text())
    NET = CFG["network"]
    PIPELINE_CFG = CFG["pipeline"]

    services = {
        "rc_cube": NET["rc_cube_service_url"],
        "sam": NET["sam2_service_url"],
        "cgn": NET["contact_graspnet_service_url"],
        "llm": NET["llm_service_url"],
    }

    for name, url in services.items():
        wait_for_health(name, url)

    time.sleep(1)
    print("============================")
    print("===     ORCHESTRATOR     ===")
    print("============================")



@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run_pipeline")
def run_pipeline(req: PipelineRequest) -> dict:
    step = "init"
    rc = sam = cgn = llm_obj = llm_grasp = obj_id = None

    try:
        print("[ORCH] Start")

        rc_url = NET["rc_cube_service_url"]
        sam_url = NET["sam2_service_url"]
        cgn_url = NET["contact_graspnet_service_url"]
        llm_url = NET["llm_service_url"]

        step = "rc"
        t0 = time.perf_counter()
        rc = post_json(f"{rc_url}/fetch_disparity_and_left", {}, "rc")
        t_rc = time.perf_counter() - t0

        step = "sam"
        t0 = time.perf_counter()
        sam = post_json(
            f"{sam_url}/predict",
            {
                "npz_path": rc["rc_out_npz"],
                "image_path": rc["left_png_path"],
            },
            "sam",
        )
        t_sam = time.perf_counter() - t0

        t_llm_obj = 0.0
        obj_id = None

        if PIPELINE_CFG["use_obj_selection"] and req.object_query:
            step = "llm_obj"
            t0 = time.perf_counter()

            llm_obj = post_json(
                f"{llm_url}/generate",
                {
                    "prompt_name": "select_obj_id",
                    "full_img_path": sam["rgb_annotated_path"],
                    "prompt_vars": {"object_query": req.object_query},
                },
                "llm_obj",
            )

            t_llm_obj = time.perf_counter() - t0

            resp = llm_obj.get("response", {})
            if isinstance(resp, dict):
                val = resp.get("object_id")
                if isinstance(val, int):
                    obj_id = val

            if obj_id is None:
                return {
                    "status": "no_object_found",
                    "message": f"No object_id found for query '{req.object_query}'",
                }
        else:
            obj_id = CFG["contact_graspnet"]["prediction"]["segmap_id"]

        step = "cgn"
        t0 = time.perf_counter()

        cgn = post_json(
            f"{cgn_url}/inference",
            {
                "npz_path": sam["result_path"],
                "object_id": obj_id,
            },
            "cgn",
        )

        t_cgn = time.perf_counter() - t0

        num_grasps = cgn.get("num_grasps", 0)

        if not isinstance(num_grasps, int) or num_grasps == 0:
            return {
                "status": "no_grasps_found",
                "message": f"No grasps found for object_id={obj_id}",
            }

        step = "llm_grasp"
        t0 = time.perf_counter()
        llm_grasp = post_json(
            f"{llm_url}/generate",
            {
                "prompt_name": "select_grasp",
                "full_img_path": cgn["annotated_full_size"],
                "zoomed_img_path": cgn["annotated_cropped"],
            },
            "llm_grasp",
        )
        t_llm_grasp = time.perf_counter() - t0

        print("[ORCH] Done")

        response = {
            "status": "ok",
            "raw_pointcloud": rc["rc_out_npz"],
            "segmented_pointcloud": sam["result_path"],
            "rgb_annotated_img": sam["rgb_annotated_path"],
            "selected_object_id": obj_id,
            "grasp_annotation_full_img": cgn["annotated_full_size"],
            "grasp_annotation_zoomed_img": cgn["annotated_cropped"],
            "llm_grasp_response": llm_grasp["response"],
            "timings": {
                "rc": round(t_rc, 4),
                "sam": round(t_sam, 4),
                "llm_obj": round(t_llm_obj, 4),
                "cgn": round(t_cgn, 4),
                "llm_grasp": round(t_llm_grasp, 4),
            },
        }

        if CFG["project"]["debug"]:
            response["debug"] = {
                "rc_cube": rc,
                "sam": sam,
                "cgn": cgn,
                "llm_obj": llm_obj,
                "llm_grasp": llm_grasp,
            }

        return response

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "step": step,
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "context": {
                    "rc": rc,
                    "sam": sam,
                    "cgn": cgn,
                    "llm_obj": llm_obj,
                    "llm_grasp": llm_grasp,
                    "object_id": obj_id,
                },
            },
        )