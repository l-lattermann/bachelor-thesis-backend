import cv2
import numpy as np
import yaml
from pathlib import Path

import uois.src.data_augmentation as data_augmentation
import uois.src.util.utilities as util_

CONFIG_PATH = Path("/app/config.yaml")
with CONFIG_PATH.open("r") as f:
    CFG = yaml.safe_load(f)


def load_uois_input(npz_path: str):
    data = np.load(npz_path)

    rgb = data["rgb"]
    xyz = data["xyz"]
    label = data["label"] if "label" in data.files else None

    rgb_std = data_augmentation.standardize_image(rgb)
    batch = {
        "rgb": data_augmentation.array_to_tensor(rgb_std[None, ...]),
        "xyz": data_augmentation.array_to_tensor(xyz[None, ...]),
    }

    return data, batch, rgb, xyz, label


def remap_segmentation_order(mask, row_tol: int = 40):
    mask = mask.astype(np.int32).copy()
    mask[mask < 0] = 0

    segments = []

    for seg_id in np.unique(mask):
        if seg_id == 0:
            continue

        ys, xs = np.where(mask == seg_id)
        if len(xs) == 0:
            continue

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        row_key = int(round(cy / row_tol))

        segments.append({
            "seg_id": int(seg_id),
            "cx": cx,
            "cy": cy,
            "row_key": row_key,
        })

    segments.sort(key=lambda s: (-s["row_key"], s["cx"]))

    id_map = {seg["seg_id"]: i + 1 for i, seg in enumerate(segments)}

    new_mask = mask.copy()
    for old_id, new_id in id_map.items():
        new_mask[mask == old_id] = new_id

    return new_mask, id_map


def draw_segment_ids(img, mask):
    out = img.copy()

    for seg_id in np.unique(mask):
        if seg_id == 0:
            continue

        ys, xs = np.where(mask == seg_id)
        if len(xs) == 0:
            continue

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        text = str(int(seg_id))

        cv2.putText(
            out,
            text,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            out,
            text,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    return out


def uois_debug_save(run_result: dict):
    debug_out_dir = Path(CFG.get("paths", {}).get("output_debug", "/shared/debug")) / "uois"
    debug_out_dir.mkdir(parents=True, exist_ok=True)

    seg = run_result["seg"]
    xyz = run_result["xyz"]
    label = run_result["label"]
    batch = run_result["batch"]

    seg, _ = remap_segmentation_order(seg)

    if label is not None:
        label, _ = remap_segmentation_order(label)

    num_objs = int(seg.max() + 1) if label is None else int(max(seg.max(), label.max()) + 1)
    pred_vis = util_.get_color_mask(seg, nc=num_objs)
    gt_vis = util_.get_color_mask(label, nc=num_objs) if label is not None else None
    rgb = util_.torch_to_numpy(batch["rgb"].cpu(), is_standardized_image=True)[0].astype(np.uint8)

    rgb = draw_segment_ids(rgb, seg)
    pred_vis = draw_segment_ids(pred_vis, seg)

    if gt_vis is not None:
        gt_vis = draw_segment_ids(gt_vis, label)

    img_no = CFG.get("uois", {}).get("img_numbers_to_save", [0, 1, 2, 3, 4])
    debug_imgs = [
        rgb,
        xyz[..., 2].astype(np.float32),
        pred_vis,
    ]

    if gt_vis is not None:
        debug_imgs.append(gt_vis)

    for i, img in enumerate(debug_imgs):
        if i not in img_no:
            continue
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(debug_out_dir / f"uois_{i}.png"), img)

def save_rgb_with_segment_ids(run_result: dict) -> dict:
    out_dir = Path(CFG.get("paths", {}).get("pipeline_file_share", "/shared/pipeline_io"))
    out_dir.mkdir(parents=True, exist_ok=True)

    seg = run_result["seg"]
    batch = run_result["batch"]

    seg, _ = remap_segmentation_order(seg)

    rgb = util_.torch_to_numpy(
        batch["rgb"].cpu(),
        is_standardized_image=True
    )[0].astype(np.uint8)

    rgb_annotated = draw_segment_ids(rgb, seg)

    out_path = out_dir / "uois_rgb_annotated.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb_annotated, cv2.COLOR_RGB2BGR))

    return {"rgb_annotated_path": str(out_path)}


def _extract_camera_from_npz(data) -> dict:
    fx = float(data["fx"])
    fy = float(data["fy"])
    cx = float(data["cx"])
    cy = float(data["cy"])

    width = int(data["width"]) if "width" in data.files else int(data["rgb"].shape[1])
    height = int(data["height"]) if "height" in data.files else int(data["rgb"].shape[0])

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return {
        "K": K,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
    }


def convert_uois_to_cgn(
    rgb: np.ndarray,
    xyz: np.ndarray,
    seg: np.ndarray,
    cam: dict,
) -> dict:
    rgb = rgb.astype(np.uint8)

    depth = xyz[..., 2].astype(np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0

    seg, _ = remap_segmentation_order(seg)

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


def uois_save_cgn_format(rgb: np.ndarray, xyz: np.ndarray, seg: np.ndarray, source_npz):
    out_dir = Path(CFG.get("paths", {}).get("pipeline_file_share", "/shared/pipeline_io"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = _extract_camera_from_npz(source_npz)
    out_path = out_dir / "uois_output.npz"

    cgn_data = convert_uois_to_cgn(
        rgb=rgb,
        xyz=xyz,
        seg=seg,
        cam=cam,
    )

    np.savez(out_path, **cgn_data)
    return {"cgn_npz": str(out_path)}