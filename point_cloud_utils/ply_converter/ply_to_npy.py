import argparse
import numpy as np
from plyfile import PlyData
import io
import numpy as np




def load_pts_cols_from_bytes(ply_bytes: bytes):
    ply = PlyData.read(io.BytesIO(ply_bytes))
    v = ply["vertex"].data
    names = v.dtype.names or ()

    pts = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

    cols = None
    if {"diffuse_red","diffuse_green","diffuse_blue"}.issubset(names):
        cols = np.vstack([v["diffuse_red"], v["diffuse_green"], v["diffuse_blue"]]).T

    if cols is not None:
        cols = np.asarray(cols)
        if np.issubdtype(cols.dtype, np.floating) and cols.max() <= 1.0 + 1e-6:
            cols = (cols * 255.0).round()
        cols = np.clip(cols, 0, 255).astype(np.uint8)

    return pts, cols

def convert_pts_to_uois_dict(
    pts: np.ndarray,
    cols: np.ndarray | None,
    cam: dict,
    z_min: float = 1e-6,
    z_max: float = 10.0,
    seg_id: int = 1,
    background_label: int = 0,
    out_size: tuple[int, int] | None = (640, 480),  # (W,H); set None to keep cam resolution
) -> dict:
    """
    Convert point cloud arrays directly to UOIS dict:
      returns {"rgb": (H,W,3) uint8, "xyz": (H,W,3) float32, "label": (H,W) int32}

    - pts: (N,3) float32 (camera frame)
    - cols: (N,3) uint8 or None
    - cam: {"width","height","fx","fy","cx","cy"} for the ORIGINAL image size
    - out_size: if not None, project into this resolution by scaling intrinsics accordingly
    """
    # ---- scale intrinsics if projecting to a different output resolution ----
    if out_size is not None:
        out_w, out_h = out_size
        sx = float(out_w) / float(cam["width"])
        sy = float(out_h) / float(cam["height"])
        W, H = int(out_w), int(out_h)
        fx = float(cam["fx"]) * sx
        fy = float(cam["fy"]) * sy
        cx = float(cam["cx"]) * sx
        cy = float(cam["cy"]) * sy
    else:
        W, H = int(cam["width"]), int(cam["height"])
        fx = float(cam["fx"])
        fy = float(cam["fy"])
        cx = float(cam["cx"])
        cy = float(cam["cy"])

    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts must be (N,3); got {pts.shape}")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > z_min) & (z < z_max)

    x, y, z = x[valid], y[valid], z[valid]
    if cols is not None:
        cols = np.asarray(cols)
        cols = cols[valid]

    if z.size == 0:
        raise ValueError("No points left after z_min/z_max filtering.")

    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, x, y, z = u[in_img], v[in_img], x[in_img], y[in_img], z[in_img]
    if cols is not None:
        cols = cols[in_img]

    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    xyz = np.zeros((H, W, 3), dtype=np.float32)
    label = np.full((H, W), background_label, dtype=np.int32)

    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    for i in range(z.shape[0]):
        ui, vi, zi = u[i], v[i], z[i]
        if zi < zbuf[vi, ui]:
            zbuf[vi, ui] = zi
            xyz[vi, ui, 0] = x[i]
            xyz[vi, ui, 1] = y[i]
            xyz[vi, ui, 2] = zi
            label[vi, ui] = seg_id
            if cols is not None:
                rgb[vi, ui] = cols[i]

    return {"rgb": rgb, "xyz": xyz, "label": label}





def load_pts_cols(ply_path: str):
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    names = v.dtype.names or ()

    pts = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

    cols = None
    if {"red", "green", "blue"}.issubset(names):
        cols = np.vstack([v["red"], v["green"], v["blue"]]).T
    elif {"r", "g", "b"}.issubset(names):
        cols = np.vstack([v["r"], v["g"], v["b"]]).T
    elif {"diffuse_red", "diffuse_green", "diffuse_blue"}.issubset(names):
        cols = np.vstack([v["diffuse_red"], v["diffuse_green"], v["diffuse_blue"]]).T

    if cols is not None:
        cols = np.asarray(cols)
        if np.issubdtype(cols.dtype, np.floating) and cols.max() <= 1.0 + 1e-6:
            cols = (cols * 255.0).round()
        cols = np.clip(cols, 0, 255).astype(np.uint8)

    return pts, cols

def inspect_pts_cols(pts, cols):
    print("---- POINTS ----")
    print("type:", type(pts))
    print("dtype:", pts.dtype if hasattr(pts, "dtype") else None)
    print("shape:", pts.shape if hasattr(pts, "shape") else None)

    if isinstance(pts, np.ndarray) and pts.size > 0:
        print("min:", pts.min(axis=0))
        print("max:", pts.max(axis=0))
        print("mean:", pts.mean(axis=0))
        print("first 5 rows:\n", pts[:5])

    print("\n---- COLORS ----")
    if cols is None:
        print("cols is None")
        return

    print("type:", type(cols))
    print("dtype:", cols.dtype if hasattr(cols, "dtype") else None)
    print("shape:", cols.shape if hasattr(cols, "shape") else None)

    if isinstance(cols, np.ndarray) and cols.size > 0:
        print("min:", cols.min(axis=0))
        print("max:", cols.max(axis=0))
        print("first 5 rows:\n", cols[:5])



def scale_cam(cam: dict, out_w: int, out_h: int) -> dict:
    """
    Scale intrinsics to a new resolution (same camera, resized image).
    """
    in_w = float(cam["width"])
    in_h = float(cam["height"])
    sx = float(out_w) / in_w
    sy = float(out_h) / in_h

    return {
        "width": int(out_w),
        "height": int(out_h),
        "fx": float(cam["fx"]) * sx,
        "fy": float(cam["fy"]) * sy,
        "cx": float(cam["cx"]) * sx,
        "cy": float(cam["cy"]) * sy,
    }


def ply_to_uois_dict_with_cam(
    ply_path: str,
    cam: dict,
    z_min: float = 1e-6,
    z_max: float = 10.0,
    seg_id: int = 1,
    background_label: int = 0,
    out_size: tuple[int, int] | None = (640, 480),  # (W,H); set None to keep cam resolution
) -> dict:
    """
    Direct: PLY -> UOIS dict
      returns {"rgb": (H,W,3) uint8, "xyz": (H,W,3) float32, "label": (H,W) int32}

    If out_size is provided, the projection is done at that resolution by scaling intrinsics.
    """
    if out_size is not None:
        out_w, out_h = out_size
        cam = scale_cam(cam, out_w, out_h)

    W = int(cam["width"])
    H = int(cam["height"])
    fx = float(cam["fx"])
    fy = float(cam["fy"])
    cx = float(cam["cx"])
    cy = float(cam["cy"])

    pts, cols = load_pts_cols(ply_path)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > z_min) & (z < z_max)

    x, y, z = x[valid], y[valid], z[valid]
    if cols is not None:
        cols = cols[valid]

    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, x, y, z = u[in_img], v[in_img], x[in_img], y[in_img], z[in_img]
    if cols is not None:
        cols = cols[in_img]

    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    xyz = np.zeros((H, W, 3), dtype=np.float32)
    label = np.full((H, W), background_label, dtype=np.int32)

    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    for i in range(z.shape[0]):
        ui, vi, zi = u[i], v[i], z[i]
        if zi < zbuf[vi, ui]:
            zbuf[vi, ui] = zi
            xyz[vi, ui, 0] = x[i]
            xyz[vi, ui, 1] = y[i]
            xyz[vi, ui, 2] = zi
            label[vi, ui] = seg_id
            if cols is not None:
                rgb[vi, ui] = cols[i]

    return {"rgb": rgb, "xyz": xyz, "label": label}


def convert_ply_file_to_uois_npy_file(
    ply_path: str,
    out_npy_path: str,
    cam: dict,
    z_min: float = 1e-6,
    z_max: float = 10.0,
    seg_id: int = 1,
    background_label: int = 0,
    out_size: tuple[int, int] | None = (640, 480),  # (W,H)
):
    d = ply_to_uois_dict_with_cam(
        ply_path=ply_path,
        cam=cam,
        z_min=z_min,
        z_max=z_max,
        seg_id=seg_id,
        background_label=background_label,
        out_size=out_size,
    )
    np.save(out_npy_path, d)
    return d, out_npy_path

