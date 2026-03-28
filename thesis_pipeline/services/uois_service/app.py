from pathlib import Path
import yaml

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

    print("============================", flush=True)
    print("===     UOIS SERVICE      ==", flush=True)
    print("============================", flush=True)


class UOISRequest(BaseModel):
    npz_path: str


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
            seg=run_result["seg"],
            source_npz=run_result["data"],
        )

        return {
            "status": "ok",
            "mask_shape": list(run_result["seg"].shape),
            "time": run_result["time"],
            "cgn_npz": paths["cgn_npz"],
            "dsn_config": run_result["dsn_config"],
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key in input npz: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))