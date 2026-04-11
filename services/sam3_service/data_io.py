from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_left_rgb(image_path: str):
    return Image.open(image_path).convert("RGB")


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


def resize_masks_to_shape(masks, image_shape):
    h, w = image_shape[:2]
    out = []

    for mask in masks:
        mask = np.asarray(mask).astype(np.uint8)

        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        out.append(mask.astype(np.uint8))

    return out


def masks_to_segmentation(masks, image):
    if hasattr(image, "size") and not hasattr(image, "shape"):
        w, h = image.size
        image_shape = (h, w)
    else:
        image_shape = image.shape[:2]

    h, w = image_shape
    seg = np.zeros((h, w), dtype=np.int32)

    resized_masks = []

    for mask in masks:
        mask = np.asarray(mask).astype(np.uint8)

        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        resized_masks.append(mask)

    for i, mask in enumerate(resized_masks, start=1):
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


def save_annotated_png(image, masks, boxes, scores, output_path, pad_bb=5, label_size=5):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(image, "size") and not hasattr(image, "shape"):
        image = np.array(image)

    vis = image.copy().astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(vis)

    boxes = np.asarray(boxes) if boxes is not None else np.empty((0, 4), dtype=np.float32)
    scores = np.asarray(scores) if scores is not None else np.empty((0,), dtype=np.float32)

    colors = [
        "#ff3b30", "#34c759", "#007aff", "#ffcc00",
        "#af52de", "#ff9500", "#5ac8fa", "#ff2d55",
    ]

    h, w = vis.shape[:2]

    for i, mask in enumerate(masks, start=1):
        mask = np.asarray(mask).astype(np.uint8)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        if i - 1 < len(boxes):
            x1, y1, x2, y2 = boxes[i - 1]
        else:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        x1 = max(0, x1 - pad_bb)
        y1 = max(0, y1 - pad_bb)
        x2 = min(w - 1, x2 + pad_bb)
        y2 = min(h - 1, y2 + pad_bb)

        color = colors[(i - 1) % len(colors)]
        score = float(scores[i - 1]) if i - 1 < len(scores) else None

        label = f"id:{i}"
        if score is not None:
            label += f"  c:{score:.2f}"

        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            max(2, y1 - 6),
            label,
            fontsize=label_size,
            color="black",
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor=color,
                linewidth=1,
                alpha=0.95,
            ),
        )

    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=160)
    plt.close(fig)

    return str(output_path)