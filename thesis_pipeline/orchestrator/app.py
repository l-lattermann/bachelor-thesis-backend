from pathlib import Path
import yaml
import requests
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional


app = FastAPI(title="Orchestrator")

CONFIG_PATH = Path("/app/config.yaml")
CFG = None



class PipelineRequest(BaseModel):
    object_query: Optional[str] = None


def load_config() -> dict:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def wait_for_health(name: str, base_url: str, retries: int = 120, delay: float = 2.0) -> None:
    last_error = None

    for _ in range(retries):
        try:
            res = requests.get(f"{base_url}/health", timeout=10)
            if res.status_code == 200:
                return
            last_error = f"status {res.status_code}: {res.text}"
        except requests.RequestException as e:
            last_error = str(e)

        time.sleep(delay)

    raise RuntimeError(f"{name} not reachable after {retries} retries: {last_error}")


@app.on_event("startup")
def startup_event():
    global CFG
    CFG = load_config()

    services = {
        "rc_cube": CFG["network"]["rc_cube_service_url"],
        "uois": CFG["network"]["uois_service_url"],
        "contact_graspnet": CFG["network"]["contact_graspnet_service_url"],
        "llm": CFG["network"]["llm_service_url"],
    }

    try:
        for name, url in services.items():
            wait_for_health(name, url)

        time.sleep(1)
        print("============================")
        print("===     ORCHESTRATOR     ===")
        print("============================\n\n")
        print("[HEALTH OK] all subcontainers are healthy")

    except Exception as e:
        print(f"[STARTUP ERROR] {e}")
        raise


@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator"}


@app.get("/health_all")
def health_all():
    results = {}

    services = {
        "rc_cube": CFG["network"]["rc_cube_service_url"],
        "uois": CFG["network"]["uois_service_url"],
        "contact_graspnet": CFG["network"]["contact_graspnet_service_url"],
        "llm": CFG["network"]["llm_service_url"],
    }

    overall_ok = True

    for name, url in services.items():
        try:
            res = requests.get(f"{url}/health", timeout=5)
            if res.status_code == 200:
                results[name] = "ok"
            else:
                results[name] = f"error ({res.status_code})"
                overall_ok = False
        except Exception as e:
            results[name] = f"unreachable ({e})"
            overall_ok = False

    return {
        "status": "ok" if overall_ok else "degraded",
        "services": results,
    }


def post_json(url: str, payload: dict, step_name: str) -> dict:
    try:
        res = requests.post(url, json=payload, timeout=120)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"{step_name} request failed: {e}")

    if res.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"{step_name} failed: {res.status_code} {res.text}"
        )

    try:
        return res.json()
    except ValueError:
        raise HTTPException(status_code=500, detail=f"{step_name} returned invalid JSON")


@app.post("/run_pipeline")
def run_pipeline(req: PipelineRequest):
    try:
        print("[ORCH] Starting pipeline")

        rc_cube_url = CFG["network"]["rc_cube_service_url"]
        uois_url = CFG["network"]["uois_service_url"]
        cgn_url = CFG["network"]["contact_graspnet_service_url"]
        llm_url = CFG["network"]["llm_service_url"]

        t0 = time.perf_counter()
        print("[ORCH] Calling RC Cube...")
        rc_data = post_json(
            f"{rc_cube_url}/fetch_disparity_and_left",
            {},
            "RC Cube"
        )
        rc_cube_time_sec = time.perf_counter() - t0
        raw_pointcloud = rc_data["result_path"]

        t0 = time.perf_counter()
        print("[ORCH] Calling UOIS...")
        uois_data = post_json(
            f"{uois_url}/predict",
            {"npz_path": raw_pointcloud},
            "UOIS"
        )
        uois_time_sec = time.perf_counter() - t0
        segmented_pointcloud = uois_data["result_path"]
        rgb_annotated_img = uois_data["rgb_annotated_path"]

        selected_object_id = None
        llm_object_time_sec = 0.0
        llm_object_data = None

        if getattr(req, "object_query", None):
            t0 = time.perf_counter()
            print("[ORCH] Calling LLM for object selection...")
            llm_object_data = post_json(
                f"{llm_url}/generate",
                {
                    "prompt_name": "select_object_id",
                    "image_path": rgb_annotated_img,
                    "prompt_vars": {
                        "object_query": req.object_query
                    }
                },
                "LLM object selection"
            )
            llm_object_time_sec = time.perf_counter() - t0

            try:
                selected_object_id = int(llm_object_data["response"])
            except Exception:
                selected_object_id = None

        t0 = time.perf_counter()
        print("[ORCH] Calling Contact-GraspNet...")

        cgn_payload = {"npz_path": segmented_pointcloud}
        if selected_object_id is not None:
            cgn_payload["object_id"] = selected_object_id

        cgn_data = post_json(
            f"{cgn_url}/inference",
            cgn_payload,
            "Contact-GraspNet"
        )

        contact_graspnet_time_sec = time.perf_counter() - t0
        grasp_annotated_img = cgn_data["result_path"]

        combined_inference_time_sec = (
            rc_cube_time_sec + uois_time_sec + contact_graspnet_time_sec
        )

        t0 = time.perf_counter()
        print("[ORCH] Calling LLM for grasp selection...")
        llm_grasp_data = post_json(
            f"{llm_url}/generate",
            {
                "prompt_name": "select_grasp",
                "image_path": grasp_annotated_img,
            },
            "LLM grasp selection"
        )
        llm_grasp_time_sec = time.perf_counter() - t0
        llm_grasp_response = llm_grasp_data["response"]

        print("[ORCH] Pipeline finished")

        return {
            "status": "ok",
            "raw_pointcloud": raw_pointcloud,
            "segmented_pointcloud": segmented_pointcloud,
            "rgb_annotated_img": rgb_annotated_img,
            "selected_object_id": selected_object_id,
            "grasp_annotated_img": grasp_annotated_img,
            "llm_grasp_response": llm_grasp_response,
            "debug": {
                "rc_cube": rc_data,
                "uois": uois_data,
                "llm_object_selection": llm_object_data,
                "contact_graspnet": cgn_data,
                "llm_grasp_selection": llm_grasp_data,
            },
            "timings": {
                "rc_cube_time_sec": round(rc_cube_time_sec, 4),
                "uois_time_sec": round(uois_time_sec, 4),
                "llm_object_selection_time_sec": round(llm_object_time_sec, 4),
                "contact_graspnet_time_sec": round(contact_graspnet_time_sec, 4),
                "combined_inference_time_sec": round(combined_inference_time_sec, 4),
                "llm_grasp_selection_time_sec": round(llm_grasp_time_sec, 4),
            },
        }

    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing expected key: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))