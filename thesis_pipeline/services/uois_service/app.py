from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml

from service_core import run_uois_on_npy, load_model, save_results

app = FastAPI()

CONFIG_PATH = "/app/config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@app.on_event("startup")
def startup_event():
    load_model()
    print("============================")
    print("===     UOIS SERVICE      ==")
    print("============================")


class UOISRequest(BaseModel):
    npy_path: str
    stem: str = "uois"


@app.get("/health")
def health():
    return {"status": "ok", "service": "uois"}


@app.post("/predict")
def predict(req: UOISRequest):
    try:
        if not Path(req.npy_path).exists():
            raise FileNotFoundError(f"Input file not found: {req.npy_path}")

        run_result = run_uois_on_npy(req.npy_path)
        paths = save_results(run_result, stem=req.stem)

        response = {
            "status": "ok",
            "mask_path": paths["mask_npy"],
            "mask_shape": list(run_result["seg"].shape),
            "time": run_result["time"],
            "cgn_npy_path": paths["cgn_npy"],
            "dsn_config": run_result["dsn_config"]
        }
        
        if "vis_dir" in paths:
            response["vis_dir"] = paths["vis_dir"]

        if "metrics" in paths:
            response["metrics"] = paths["metrics"]

        return response

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key in input npy: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))