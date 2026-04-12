from pathlib import Path
import time
import traceback

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
    object_query: str


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
        "sam3": NET["sam3_service_url"],
        "cgn": NET["contact_graspnet_service_url"],
        "render_url": NET["render_service_url"],
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
    rc = sam = cgn = render = llm_obj = llm_grasp = None
    obj_id = None

    try:
        print("[ORCH] Start")

        rc_url = NET["rc_cube_service_url"]
        sam_url = NET["sam3_service_url"]
        cgn_url = NET["contact_graspnet_service_url"]
        render_url = NET["render_service_url"]
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
                "text_prompt": req.object_query,
                "npz_path": rc["rc_out_npz"],
                "image_path": rc["left_png_path"],
            },
            "sam",
        )
        t_sam = time.perf_counter() - t0

        t_llm_obj = 0.0
        llm_obj = None

        mask_count = sam.get("mask_count")
        if not isinstance(mask_count, int):
            raise ValueError("SAM response missing valid mask_count.")

        if mask_count < 1:
            return {
                "status": "no_object_found",
                "message": "SAM found no objects.",
                "raw_pointcloud": rc["rc_out_npz"],
                "segmented_pointcloud": sam["result_path"],
                "rgb_annotated_img": sam["rgb_annotated_path"],
            }

        if mask_count == 1:
            val = sam.get("selected_mask_label")
            if isinstance(val, int):
                obj_id = val
            else:
                obj_id = 1

        elif mask_count > 1:
            if not req.object_query:
                return {
                    "status": "ambiguous_scene",
                    "message": "Multiple objects detected but no object_query was provided.",
                    "mask_count": mask_count,
                    "raw_pointcloud": rc["rc_out_npz"],
                    "segmented_pointcloud": sam["result_path"],
                    "rgb_annotated_img": sam["rgb_annotated_path"],
                }

            step = "llm_obj"
            t0 = time.perf_counter()
            llm_obj = post_json(
                f"{llm_url}/generate",
                {
                    "prompt_name": "select_obj_id",
                    "image_paths": [sam["rgb_annotated_path"]],
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
                    "mask_count": mask_count,
                    "raw_pointcloud": rc["rc_out_npz"],
                    "segmented_pointcloud": sam["result_path"],
                    "rgb_annotated_img": sam["rgb_annotated_path"],
                }

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
                "selected_object_id": obj_id,
                "segmented_pointcloud": sam["result_path"],
                "rgb_annotated_img": sam["rgb_annotated_path"],
            }

        step = "render"
        t0 = time.perf_counter()
        render = post_json(
            f"{render_url}/render",
            {
                "npz_path": cgn["sel_grasps_npz"],
            },
            "render",
        )
        t_render = time.perf_counter() - t0

        step = "llm_grasp"
        t0 = time.perf_counter()
        llm_grasp = post_json(
        f"{llm_url}/generate",
            {
                "prompt_name": "select_grasp",
                "prompt_vars": {
                    "object_query": req.object_query,
                },
                "image_paths": [
                    rc["left_png_path"],
                    render["top_render_path"],
                    render["front_render_path"],
                ],
            },
            "llm_grasp",
        )
        t_llm_grasp = time.perf_counter() - t0

        total_time = t_rc + t_sam + t_llm_obj + t_cgn + t_render + t_llm_grasp

        response = {}

        if CFG["project"]["debug"]:
            response["debug"] = {
                "rc_cube": rc,
                "sam": sam,
                "cgn": cgn,
                "render": render,
                "llm_obj": llm_obj,
                "llm_grasp": llm_grasp,
            }

            response["timings"] = {
                "rc": round(t_rc, 2),
                "sam": round(t_sam, 2),
                "llm_obj": round(t_llm_obj, 2),
                "cgn": round(t_cgn, 2),
                "render": round(t_render, 2),
                "llm_grasp": round(t_llm_grasp, 2),
                "total": round(total_time, 2),
            }

        llm_resp = llm_grasp["response"]

        response["llm_grasp_response"] = {
            "inferred_task": llm_resp["inferred_task"],
            "selected_grasp_id": llm_resp["selected_grasp_id"],
            "final_decision_reason": llm_resp["final_decision_reason"],
            "evaluations": llm_resp["evaluations"],
        }
        response["status"] = "ok"

        print("[ORCH] Done")
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
                    "render": render,
                    "llm_obj": llm_obj,
                    "llm_grasp": llm_grasp,
                    "object_id": obj_id,
                },
            },
        )