from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


def load_left_rgb(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_rc_cube_npz(npz_path):
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
        "K": cam["K"],
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


def generate_bright_colors(n):
    base = [
        (255, 99, 132),
        (255, 159, 64),
        (255, 205, 86),
        (75, 192, 192),
        (54, 162, 235),
        (153, 102, 255),
        (255, 102, 255),
        (0, 255, 127),
        (255, 0, 127),
        (0, 255, 255),
    ]
    return [base[i % len(base)] for i in range(n)]


def get_label_position(mask_u8):
    ys, xs = np.where(mask_u8 > 0)

    if len(xs) == 0 or len(ys) == 0:
        return 0, 0

    cx = int(xs.mean())
    cy = int(ys.mean())

    return cx, cy


def save_annotated_masks_outline(image, masks, output_path=None, show=False):
    vis = image.copy().astype(np.uint8)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    colors = generate_bright_colors(len(masks))

    for idx, (mask, color) in enumerate(zip(masks, colors), start=1):
        if isinstance(mask, dict):
            mask_u8 = mask["segmentation"].astype(np.uint8)
        else:
            mask_u8 = mask.astype(np.uint8)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, thickness=2)

        cx, cy = get_label_position(mask_u8)

        ax.text(
            cx,
            cy,
            str(idx),
            color=np.array(color) / 255.0,
            fontsize=16,
            ha="center",
            va="center",
            path_effects=[
                path_effects.Stroke(linewidth=1, foreground="black"),
                path_effects.Normal(),
            ],
        )

    ax.imshow(vis)
    plt.axis("off")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=200)

    if show:
        plt.show()

    plt.close(fig)

    return str(output_path) if output_path is not None else None

def save_annotated_masks_overlay(image, masks, output_path=None, show=False, alpha=0.45):
    vis = image.copy().astype(np.uint8)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    colors = generate_bright_colors(len(masks))

    for mask, color in zip(masks, colors):
        if isinstance(mask, dict):
            mask_u8 = mask["segmentation"].astype(np.uint8)
        else:
            mask_u8 = mask.astype(np.uint8)

        color_arr = np.array(color, dtype=np.uint8)

        vis[mask_u8 > 0] = (
            (1.0 - alpha) * vis[mask_u8 > 0] + alpha * color_arr
        ).astype(np.uint8)

    ax.imshow(vis)
    plt.axis("off")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=200)

    if show:
        plt.show()

    plt.close(fig)

    return str(output_path) if output_path is not None else None