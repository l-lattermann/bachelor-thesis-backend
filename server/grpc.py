

# main.py
import os
from time import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from grpc_client.minimal_solution import RcCubeGrpcClient
from point_cloud_utils.ply_converter.ply_to_npy import inspect_pts_cols, load_pts_cols, convert_ply_to_uois_npy
from point_cloud_utils.check_npy_structure.inspect_npy import inspect_npy

app = FastAPI(title="RC Cube gRPC Test Service")

# Keep a single client instance (reused across requests)
rc_client: Optional[RcCubeGrpcClient] = None


@app.on_event("startup")
def startup():
    # --- GRPC CLIENT ---
    global rc_client
    print("[SERVICE] startup: creating gRPC client")
    rc_client = RcCubeGrpcClient(rc_cube_ip="172.27.5.9:50051")
    print("[SERVICE] startup: gRPC client ready")

@app.on_event("shutdown")
def shutdown():
    print("[SERVICE] shutdown")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/test/grpc")
def test_grpc():
    """
    Minimal test: call the gRPC stream once and verify we receive at least one ImageSet.
    """
    if rc_client is None:
        return JSONResponse(status_code=500, content={"ok": False, "error": "rc_client not initialized"})

    t0 = time.time()
    try:
        # If your class already has something like test_connection(), use it:
        ok = rc_client.test_connection(timeout_sec=3.0) if hasattr(rc_client, "test_connection") else True
        dt = time.time() - t0
        return {"ok": bool(ok), "latency_s": dt}
    except Exception as e:
        dt = time.time() - t0
        return JSONResponse(status_code=500, content={"ok": False, "latency_s": dt, "error": repr(e)})



@app.post("/infer/ply")
def get_ply(
    timeout_s: float = 20.0,
    max_points: int = 500_000,
    out_dir: str = "output",
    out_npy: str = "output/grpc/example_for_uois.npy",
    z_min: float = 1e-6,
    z_max: float = 10.0,
    seg_id: int = 1,                 # use 1 so label isn't all background
    out_size_w: int = 640,           # UOIS resolution (recommended)
    out_size_h: int = 480,
):

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)

    # NOTE: Your client expects output_dir; you were passing a file-like name.
    # Keep this consistent with your client's behavior.
    out_ply_dir = os.path.join(out_dir, "rc_cube_mesh.ply")
    os.makedirs(out_ply_dir, exist_ok=True)

    t0 = time()
    ply_bytes, path, cam = rc_client.get_point_cloud_ply(
        output_dir=out_ply_dir,
        timeout=timeout_s,
        max_points=max_points,
    )
    dt = time() - t0

    if path is None or not os.path.exists(path):
        raise HTTPException(status_code=501, detail="PLY file was not written / path missing.")

    size = os.path.getsize(path)

    # Optional debug in server logs
    print("CAM =", cam)
    print({
        "ok": True,
        "latency_s": float(dt),
        "saved_to": str(path),
        "mesh_type": str(type(ply_bytes)),
        "mesh_dtype": str(ply_bytes.dtype) if hasattr(ply_bytes, "dtype") else None,
    })

    print("loading:", path)
    pts, cols = load_pts_cols(path)

    d_uois, out_path = convert_ply_to_uois_npy(
        ply_path=path,
        out_npy_path=out_npy,
        cam=cam,
        z_min=z_min,
        z_max=z_max,
        seg_id=seg_id,
        out_size=(out_size_w, out_size_h),
    )

    # Return only JSON-serializable metadata (not the dict itself)
    return {
        "ok": True,
        "latency_s": float(dt),
        "ply_saved_to": str(path),
        "ply_bytes": int(size),
        "npy_saved_to": str(out_path),
        "cam": {
            "width": int(cam["width"]),
            "height": int(cam["height"]),
            "fx": float(cam["fx"]),
            "fy": float(cam["fy"]),
            "cx": float(cam["cx"]),
            "cy": float(cam["cy"]),
        },
        "uois_shape": {
            "rgb": list(d_uois["rgb"].shape),
            "xyz": list(d_uois["xyz"].shape),
            "label": list(d_uois["label"].shape),
        },
        "npy_bytes"
    }
