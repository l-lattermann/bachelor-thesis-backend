from pathlib import Path
import os
import yaml

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from service_core import run_uois_on_npz, load_model
import data_io as io_utils  

app = FastAPI()

CONFIG_PATH = "/app/config.yaml"
cfg = None

def load_config() -> dict:
    """Load service configuration from YAML file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@app.on_event("startup")
def startup_event():
    """Load config and model on service startup."""
    global cfg
    cfg = load_config()
    load_model()
    print("============================")
    print("===     UOIS SERVICE      ==")
    print("============================")


class UOISRequest(BaseModel):
    """Request body for UOIS prediction."""
    npz_path: str
    stem: str = "uois"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "uois"}


@app.post("/predict")
def predict(req: UOISRequest) -> dict:
    global cfg

    try:
        npz_path = Path(req.npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Input file not found: {req.npz_path}")

        run_result = run_uois_on_npz(str(npz_path))

        if cfg.get("project", {}).get("debug", False):
            io_utils.uois_debug_save(run_result)

        paths = io_utils.uois_save_cgn_format(
            rgb=run_result["rgb"],
            xyz=run_result["xyz"],
            seg=run_result["seg"]
        )

        response = {
            "status": "ok",
            "mask_shape": list(run_result["seg"].shape),
            "time": run_result.get("time"),
            "cgn_npz": paths.get("cgn_npz"),
            "dsn_config": run_result.get("dsn_config"),
        }

        return response

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key in input npz: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))