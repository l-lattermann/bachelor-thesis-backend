from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
import numpy as np
import io
import tempfile

from service_core import run_contact_graspnet, load_model

app = FastAPI()


@app.on_event("startup")
def startup_event():
    load_model()
    print("============================", flush=True)
    print("===     CGN SERVICE       ==", flush=True)
    print("============================", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "service": "contact_graspnet"}


@app.post("/inference")
async def inference(request: Request):
    try:
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty request body")

        result = run_contact_graspnet_bytes(raw)

        buf = io.BytesIO()
        np.savez(
            buf,
            pred_grasps_cam=result["pred_grasps_cam"],
            scores=result["scores"],
            contact_pts=result["contact_pts"],
            gripper_openings=result["gripper_openings"],
        )
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="application/octet-stream",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_contact_graspnet_bytes(raw: bytes):
    with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
        tmp.write(raw)
        tmp.flush()
        return run_contact_graspnet(tmp.name)