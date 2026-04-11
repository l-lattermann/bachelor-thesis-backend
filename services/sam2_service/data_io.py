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


def get_label_position(mask_u8):
    ys, xs = np.where(mask_u8 > 0)

    if len(xs) == 0 or len(ys) == 0:
        return 0, 0

    cx = int(xs.mean())
    cy = int(ys.mean())

    return cx, cy

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

def get_contrast_color(mean_rgb, idx):
    distinct_bright_colors_rgb = [
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (255, 255, 0),  # yellow
        (0, 255, 255),  # cyan
        (255, 0, 255),  # magenta
        (255, 128, 0),  # orange
    ]

    mean_rgb = np.array(mean_rgb, dtype=np.float32)

    dists = []
    for color in distinct_bright_colors_rgb:
        color_arr = np.array(color, dtype=np.float32)
        dist = np.linalg.norm(color_arr - mean_rgb)
        dists.append((dist, color))

    dists.sort(key=lambda x: x[0], reverse=True)

    top_k = min(3, len(dists))
    best_colors = [color for _, color in dists[:top_k]]

    return best_colors[(idx - 1) % top_k]


def save_annotated_masks_outline(image, masks, output_path=None):
    vis = image.copy().astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 10))

    for idx, mask in enumerate(masks, start=1):
        if isinstance(mask, dict):
            mask_u8 = mask["segmentation"].astype(np.uint8)
        else:
            mask_u8 = mask.astype(np.uint8)

        pixels = vis[mask_u8 > 0]
        if len(pixels) == 0:
            continue

        mean_rgb = np.median(pixels, axis=0)
        outline_color = get_contrast_color(mean_rgb, idx)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, outline_color, thickness=1)

        cx, cy = get_label_position(mask_u8)

        text_color = np.array(outline_color) / 255.0

        ax.text(
            cx,
            cy,
            str(idx),
            color=text_color,
            fontsize=7,
            ha="center",
            va="center",
            path_effects=[
                path_effects.Stroke(linewidth=2, foreground="black"),
                path_effects.Normal(),
            ],
        )

    ax.imshow(vis)
    ax.axis("off")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=200)

    plt.close(fig)

    return str(output_path) if output_path is not None else None

def save_annotated_masks_overlay(image, masks, output_path=None, alpha=0.45):
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

    plt.close(fig)

    return str(output_path) if output_path is not None else None