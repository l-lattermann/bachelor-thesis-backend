import os
from pathlib import Path
import cv2
import numpy as np
import yaml

import uois.src.data_augmentation as data_augmentation
import uois.src.util.utilities as util_

CONFIG_PATH = Path("/app/config.yaml")
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)


def load_uois_input(npz_path: str):
    """
    Load UOIS input from .npz file and prepare tensor batch for model.
    """
    data = np.load(npz_path)
    rgb, xyz, label = data["rgb"], data["xyz"], data.get("label")

    rgb_std = data_augmentation.standardize_image(rgb)
    batch = {
        "rgb": data_augmentation.array_to_tensor(rgb_std[None, ...]),
        "xyz": data_augmentation.array_to_tensor(xyz[None, ...]),
    }

    return data, batch, rgb, xyz, label


def load_camera_yaml(path: str = "/shared/pipeline_io/cam.yaml"):
    """
    Load camera intrinsics from a YAML file. Returns None if file not found.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("camera_intrinsics", None)


def uois_debug_save(run_result: dict):
    """
    Save debug visualizations of UOIS run results.
    """
    debug_out_dir = Path(CFG.get("paths", {}).get("output_debug", "/shared/debug")) / "uois"
    debug_out_dir.mkdir(parents=True, exist_ok=True)

    seg, xyz, label, batch = run_result["seg"], run_result["xyz"], run_result["label"], run_result["batch"]

    if label is not None:
        num_objs = int(max(seg.max(), label.max()) + 1)
        pred_vis, gt_vis = util_.get_color_mask(seg, nc=num_objs), util_.get_color_mask(label, nc=num_objs)
        rgb = util_.torch_to_numpy(batch["rgb"].cpu(), is_standardized_image=True)[0].astype(np.uint8)

        img_no = CFG.get("uois", {}).get("img_numbers_to_save", [0, 1, 2, 3, 4])
        for i, img in enumerate([rgb, xyz[..., 2].astype(np.float32), pred_vis, gt_vis]):
            if i in img_no:
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(debug_out_dir / f"uois_{i}.png"), img)


def convert_npz_uois_to_cgn(rgb: np.ndarray, xyz: np.ndarray, seg: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> dict:
    """
    Convert UOIS-style npz data to CGN-style dictionary.
    """
    rgb = rgb.astype(np.uint8)

    depth = xyz[..., 2].astype(np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0

    seg = seg.astype(np.int32).copy()
    seg[seg < 0] = 0

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    out = {"rgb": rgb, "depth": depth, "K": K, "seg": seg}
    return out


def uois_save_cgn_format(rgb: np.ndarray, xyz: np.ndarray, seg: np.ndarray):
    """
    Save UOIS output in CGN-compatible npz format.
    """
    out_dir = Path(CFG.get("paths", {}).get("pipeline_file_share", "/shared/pipeline_file_share"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = load_camera_yaml()
    if cam is not None:
        fx, fy, cx, cy = cam["fx_px"], cam["fy_px"], cam["cx_px"], cam["cy_px"]
    else:
        H, W = rgb.shape[:2]
        fx = float(CFG["contact_graspnet"]["camera"]["fallback_fx"])
        fy = float(CFG["contact_graspnet"]["camera"]["fallback_fy"])
        cx = W / 2.0 if CFG["contact_graspnet"]["camera"]["fallback_cx"] is None else float(CFG["contact_graspnet"]["camera"]["fallback_cx"])
        cy = H / 2.0 if CFG["contact_graspnet"]["camera"]["fallback_cy"] is None else float(CFG["contact_graspnet"]["camera"]["fallback_cy"])

    out_path = out_dir / "uois_output.npz"

    # Convert to CGN format
    cgn_data = convert_npz_uois_to_cgn(rgb, xyz, seg, fx, fy, cx, cy)

    # Save the data
    np.savez(out_path, **cgn_data)

    return {"cgn_npz": str(out_path)}