from pathlib import Path
import yaml
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from service_core import run_uois_on_npz, load_model
import data_io as io_utils

app = FastAPI()

CONFIG_PATH = "/app/config.yaml"
cfg = None


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@app.on_event("startup")
def startup_event():
    global cfg
    cfg = load_config()
    load_model()

    print("============================")
    print("===     UOIS SERVICE      ==")
    print("============================")


class UOISRequest(BaseModel):
    npz_path: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "uois"}


@app.post("/predict")
def predict(req: UOISRequest) -> dict:
    global cfg

    try:
        t_total_start = time.perf_counter()

        npz_path = Path(req.npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Input file not found: {req.npz_path}")

        t0 = time.perf_counter()
        run_result = run_uois_on_npz(str(npz_path))
        t_run = time.perf_counter() - t0

        t_debug = 0.0
        if cfg.get("project", {}).get("debug", False):
            t0 = time.perf_counter()
            io_utils.uois_debug_save(run_result)
            t_debug = time.perf_counter() - t0

        t0 = time.perf_counter()
        annotated_rgb = io_utils.save_rgb_with_segment_ids(run_result)
        t_rgb_annotated_save = time.perf_counter() - t0

        t0 = time.perf_counter()
        paths = io_utils.uois_save_cgn_format(
            rgb=run_result["rgb"],
            xyz=run_result["xyz"],
            seg=run_result["seg"],
            source_npz=run_result["data"],
        )
        t_cgn_save = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start

        return {
            "status": "ok",
            "mask_shape": list(run_result["seg"].shape),
            "time": {
                "run_model_sec": round(t_run, 4),
                "debug_save_sec": round(t_debug, 4),
                "rgb_annotated_save_sec": round(t_rgb_annotated_save, 4),
                "cgn_save_sec": round(t_cgn_save, 4),
                "total_endpoint_sec": round(t_total, 4),
            },
            "result_path": paths["cgn_npz"],
            "rgb_annotated_path": annotated_rgb["rgb_annotated_path"],
            "dsn_config": run_result["dsn_config"],
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key in input npz: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))