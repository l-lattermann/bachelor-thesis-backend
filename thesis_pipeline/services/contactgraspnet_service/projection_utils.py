from pathlib import Path
from typing import Dict

import cv2
import numpy as np


def get_gripper_wireframe(
    opening: float,
    finger_len: float = 0.058,
    palm_depth: float = 0.02,
    tail_len: float = 0.02,
):
    """
    Build a CGN-like 7-point gripper wireframe in the local grasp frame.

    Local convention:
    - x: gripper opening direction
    - z: forward direction towards the fingers
    - origin: grasp center / wrist reference

    The point ordering mimics the original CGN wireframe style more closely.
    """
    half_open = float(opening) / 2.0

    mid_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    left_root = np.array([-half_open, 0.0, 0.0], dtype=np.float32)
    right_root = np.array([half_open, 0.0, 0.0], dtype=np.float32)

    left_tip = np.array([-half_open, 0.0, finger_len], dtype=np.float32)
    right_tip = np.array([half_open, 0.0, finger_len], dtype=np.float32)

    palm_back = np.array([0.0, 0.0, -palm_depth], dtype=np.float32)
    rear_marker = np.array([0.0, 0.0, -palm_depth - tail_len], dtype=np.float32)

    pts = np.array(
        [
            palm_back,    # 0 back / wrist side
            mid_point,    # 1 center front of palm
            left_root,    # 2 left finger root
            left_tip,     # 3 left finger tip
            left_root,    # 4 repeated to break polyline like CGN
            right_root,   # 5 right finger root
            right_tip,    # 6 right finger tip
        ],
        dtype=np.float32,
    )

    lines = np.array(
        [
            [0, 1],  # palm axis
            [1, 2],  # left branch
            [2, 3],  # left finger
            [4, 5],  # bridge left root -> right root
            [5, 6],  # right finger
        ],
        dtype=np.int32,
    )

    return pts, lines


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def project_points(K: np.ndarray, points_3d: np.ndarray):
    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    valid = Z > 1e-6
    pts_2d = np.full((len(points_3d), 2), np.nan, dtype=np.float32)

    if np.any(valid):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        pts_2d[valid, 0] = fx * X[valid] / Z[valid] + cx
        pts_2d[valid, 1] = fy * Y[valid] / Z[valid] + cy

    return pts_2d, valid


def _normalize_grasps(grasps: np.ndarray) -> np.ndarray:
    grasps = np.asarray(grasps)
    if grasps.size == 0:
        return np.empty((0, 4, 4), dtype=np.float32)
    if grasps.ndim == 2 and grasps.shape == (4, 4):
        return grasps[None, ...]
    return grasps


def _normalize_openings(openings, n_expected: int, default_opening: float = 0.08) -> np.ndarray:
    arr = np.asarray(openings)
    if arr.ndim == 0 or arr.size == 0:
        return np.full((n_expected,), default_opening, dtype=np.float32)

    arr = np.atleast_1d(arr).astype(np.float32)
    if len(arr) != n_expected:
        return np.full((n_expected,), default_opening, dtype=np.float32)

    return arr


def _to_bgr_if_needed(rgb: np.ndarray) -> np.ndarray:
    if rgb is None:
        raise ValueError("RGB image is None.")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}.")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _draw_text_transparent(
    image: np.ndarray,
    text: str,
    origin: tuple,
    font_scale: float = 0.5,
    color=(0, 255, 0),
    thickness: int = 1,
):
    x, y = origin
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_projected_grasp_with_label(
    image: np.ndarray,
    K: np.ndarray,
    grasp_T: np.ndarray,
    opening: float,
    label_text: str,
    color=(0, 0, 255),
    thickness: int = 1,
    finger_len: float = 0.058,
    palm_depth: float = 0.02,
    tail_len: float = 0.02,
    draw_label: bool = False,
):
    img = image.copy()

    pts_local, lines = get_gripper_wireframe(
        opening=opening,
        finger_len=finger_len,
        palm_depth=palm_depth,
        tail_len=tail_len,
    )

    pts_cam = transform_points(grasp_T, pts_local)
    pts_2d, valid = project_points(K, pts_cam)

    for i0, i1 in lines:
        if not (valid[i0] and valid[i1]):
            continue

        p0 = pts_2d[i0]
        p1 = pts_2d[i1]

        if np.any(np.isnan(p0)) or np.any(np.isnan(p1)):
            continue

        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))

        cv2.line(img, (x0, y0), (x1, y1), color, thickness, lineType=cv2.LINE_AA)

    if draw_label:
        # place label further behind the gripper along negative local z
        anchor_local = np.array([[0.0, 0.0, -palm_depth - tail_len - 0.04]], dtype=np.float32)
        anchor_cam = transform_points(grasp_T, anchor_local)
        anchor_2d, valid_anchor = project_points(K, anchor_cam)

        if valid_anchor[0]:
            p = anchor_2d[0]
            if not np.any(np.isnan(p)):
                tx = int(round(p[0]))
                ty = int(round(p[1]))
                _draw_text_transparent(
                    img,
                    text=label_text,
                    origin=(tx, ty),
                    font_scale=0.45,
                    color=(0, 255, 0),
                    thickness=1,
                )

    return img


def save_projected_grasp_overlay(
    rgb: np.ndarray,
    K: np.ndarray,
    pred_grasps_cam: Dict,
    scores: Dict,
    gripper_openings: Dict,
    draw_default_opening: bool = True,
    default_opening: float = 0.10,
    draw_confidence: bool = True,
    output_path: str = "/shared/pipeline_io/cgn_output.png",
):
    if rgb is None:
        return ""

    if K is None:
        raise ValueError("Camera intrinsics K are required for grasp projection.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = _to_bgr_if_needed(rgb)

    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
    ]

    global_idx = 1

    for key in pred_grasps_cam:
        grasps_k = _normalize_grasps(pred_grasps_cam[key])
        if len(grasps_k) == 0:
            continue

        scores_k = np.atleast_1d(scores.get(key, np.array([]))).astype(np.float32)

        if draw_default_opening:
            openings_k = np.full((len(grasps_k),), float(default_opening), dtype=np.float32)
        else:
            openings_k = _normalize_openings(
                gripper_openings.get(key, np.array([])),
                n_expected=len(grasps_k),
                default_opening=default_opening,
            )

        if len(scores_k) != len(grasps_k):
            scores_k = np.zeros((len(grasps_k),), dtype=np.float32)

        for i in range(len(grasps_k)):
            score_str = f"{scores_k[i]:.2f}".replace(".", ",")
            label_text = f"{global_idx} (c: {score_str})"
            color = palette[(global_idx - 1) % len(palette)]

            img = draw_projected_grasp_with_label(
                image=img,
                K=K,
                grasp_T=grasps_k[i],
                opening=float(openings_k[i]),
                label_text=label_text,
                color=color,
                thickness=2,
                draw_label=draw_confidence,
            )
            global_idx += 1

    cv2.imwrite(str(out_path), img)
    return str(out_path)