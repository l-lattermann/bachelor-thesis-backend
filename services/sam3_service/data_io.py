from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_left_rgb(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_rc_cube_npz(npz_path: str):
    data = np.load(npz_path)

    rgb = data["rgb"]
    xyz = data["xyz"]

    fx = float(data["fx"])
    fy = float(data["fy"])
    cx = float(data["cx"])
    cy = float(data["cy"])
    width = int(data["width"])
    height = int(data["height"])

    cam = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "K": np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }

    return rgb, xyz, cam


def masks_to_segmentation(masks, image_shape):
    h, w = image_shape[:2]
    seg = np.zeros((h, w), dtype=np.int32)

    for i, mask in enumerate(masks, start=1):
        seg[mask.astype(bool)] = i

    return seg


def convert_to_output_npz(
    rgb: np.ndarray,
    xyz: np.ndarray,
    seg: np.ndarray,
    cam: dict,
) -> dict:
    rgb = rgb.astype(np.uint8)

    depth = xyz[..., 2].astype(np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0

    return {
        "rgb": rgb,
        "depth": depth,
        "K": cam["K"].astype(np.float32),
        "seg": seg.astype(np.int32),
        "fx": np.float32(cam["fx"]),
        "fy": np.float32(cam["fy"]),
        "cx": np.float32(cam["cx"]),
        "cy": np.float32(cam["cy"]),
        "width": np.int32(cam["width"]),
        "height": np.int32(cam["height"]),
    }


def save_output_npz(output_path, rgb, xyz, seg, cam):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = convert_to_output_npz(
        rgb=rgb,
        xyz=xyz,
        seg=seg,
        cam=cam,
    )

    np.savez(output_path, **out)
    return str(output_path)


def save_debug_plot(image, state, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    from sam3.visualization_utils import plot_results
    plot_results(image, state)
    plt.savefig(output_path, bbox_inches="tight", dpi=160)
    plt.close()

    return str(output_path)