from pathlib import Path
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import mesh_utils


COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (128, 0, 255),
    (0, 128, 255),
    (180, 105, 255),
]


def _ensure_uint8_rgb(rgb):
    if rgb is None:
        return None
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _project_xyz(points_cam, K):
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    valid = z > 1e-6
    u = np.full(len(points_cam), np.nan, dtype=np.float32)
    v = np.full(len(points_cam), np.nan, dtype=np.float32)

    u[valid] = K[0, 0] * x[valid] / z[valid] + K[0, 2]
    v[valid] = K[1, 1] * y[valid] / z[valid] + K[1, 2]
    return u, v, valid


def _project_point(point_3d, K):
    pt = np.asarray(point_3d, dtype=np.float32).reshape(3)
    x, y, z = pt
    if z <= 1e-6:
        return None
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    if np.isnan(u) or np.isnan(v):
        return None
    return int(round(u)), int(round(v))


def _save_rgb(output_path, rgb):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return str(out_path)


def _make_scene_depth(depth, H, W):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (H, W):
        raise ValueError(f"depth must have shape {(H, W)}, got {depth.shape}")
    depth = depth.copy()
    depth[~np.isfinite(depth)] = np.inf
    depth[depth <= 0] = np.inf
    return depth


def _get_top_grasp(scores):
    top_key = None
    top_idx = None
    top_score = -np.inf

    for key, value in scores.items():
        value = np.asarray(value).reshape(-1)
        if len(value) == 0:
            continue
        idx = int(np.argmax(value))
        score = float(value[idx])
        if score > top_score:
            top_key, top_idx, top_score = key, idx, score

    return top_key, top_idx



def save_projected_grasp_overlay(
    rgb,
    K,
    pred_grasps_cam,
    scores,
    segmap,
    object_id,
    depth,
    gripper_openings=None,
    draw_grip_opening_bigger=True,
    increase_grip_opening_by=0.02,
    gripper_line_width=2,
    number_line_width=2,
    output_path="/shared/pipeline_io/cgn_output.png",
):
    if rgb is None:
        return ""
    if K is None:
        raise ValueError("Camera intrinsics K are required.")
    if segmap is None:
        raise ValueError("segmap is required for occlusion handling.")
    if object_id is None:
        raise ValueError("object_id is required for occlusion handling.")
    if depth is None:
        raise ValueError("depth is required for occlusion handling.")
    if gripper_openings is None:
        raise ValueError("gripper_openings is None")

    rgb = _ensure_uint8_rgb(rgb)
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    H, W = rgb.shape[:2]
    scene_depth = _make_scene_depth(depth, H, W)

    gripper = mesh_utils.create_gripper("panda")
    cp = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()

    left_finger_vec = cp[3].astype(np.float32) - cp[1].astype(np.float32)
    right_finger_vec = cp[4].astype(np.float32) - cp[2].astype(np.float32)

    template = np.array(
        [
            np.zeros(3, dtype=np.float32),
            0.5 * (cp[1] + cp[2]),
            cp[1].astype(np.float32),
            cp[3].astype(np.float32),
            cp[1].astype(np.float32),
            cp[2].astype(np.float32),
            cp[4].astype(np.float32),
        ],
        dtype=np.float32,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.axis("off")

    label_points_xy = []
    top_key, top_idx = _get_top_grasp(scores)

    def project_grasp(grasp_T, opening):
        pts = template.copy()
        half_opening = float(opening) / 2.0

        pts[2, 0] = -half_opening
        pts[4, 0] = -half_opening
        pts[5, 0] = +half_opening
        pts[3] = pts[2] + left_finger_vec
        pts[6] = pts[5] + right_finger_vec
        pts[1] = 0.5 * (pts[2] + pts[5])

        pts_cam = pts @ grasp_T[:3, :3].T
        pts_cam += grasp_T[:3, 3]
        u, v, valid = _project_xyz(pts_cam, K)
        return pts_cam, u, v, valid

    def is_visible(px, py, pz, margin=0.005):
        if px < 0 or px >= W or py < 0 or py >= H:
            return False
        z = scene_depth[py, px]
        return not np.isfinite(z) or pz <= z + margin

    def first_visible_on_finger(base_pt, tip_pt, margin=0.002):
        u0, v0, z0 = base_pt
        u1, v1, z1 = tip_pt

        length = int(max(abs(u1 - u0), abs(v1 - v0)))
        steps = max(length * 2, 2)

        for t in np.linspace(0.0, 1.0, steps):
            u = (1 - t) * u0 + t * u1
            v = (1 - t) * v0 + t * v1
            z = (1 - t) * z0 + t * z1
            if is_visible(int(round(u)), int(round(v)), z, margin):
                return u, v, z
        return None

    def get_grasp_color(rgb_img, seg_img, obj_id, idx):
        color_list_bgr = [
            (0, 0, 255),    # red
            (0, 255, 0),    # green
            (0, 255, 255),  # yellow
            (255, 255, 0),  # cyan
            (255, 0, 255),  # magenta
            (255, 128, 0),  # orange
        ]

        mask = np.asarray(seg_img) == int(obj_id)
        pixels = np.asarray(rgb_img)[mask]

        if len(pixels) == 0:
            remaining = color_list_bgr[1:] if len(color_list_bgr) > 1 else color_list_bgr
            return remaining[idx % len(remaining)]

        pixels = pixels.reshape(-1, 3).astype(np.uint8)
        colors, counts = np.unique(pixels, axis=0, return_counts=True)
        mode_rgb = colors[np.argmax(counts)]
        mode_bgr = np.array([mode_rgb[2], mode_rgb[1], mode_rgb[0]], dtype=np.float32)

        dists = []
        for color in color_list_bgr:
            color_arr = np.array(color, dtype=np.float32)
            dists.append((np.linalg.norm(color_arr - mode_bgr), color))

        dists.sort(key=lambda x: x[0])
        remaining = [color for _, color in dists[1:]]
        return remaining[idx % len(remaining)]

    def draw_line(p0, p1, color_bgr):
        color_rgb = np.array(color_bgr[::-1], dtype=np.float32) / 255.0
        linewidth = max(1.0, float(gripper_line_width) * 0.75)

        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=color_rgb,
            linewidth=linewidth,
            solid_capstyle="round",
            path_effects=[
                path_effects.Stroke(linewidth=linewidth + 1.2, foreground="black"),
                path_effects.Normal(),
            ],
        )

    def draw_label(u, v, valid, label, color_bgr, is_top):
        if not (valid[0] and valid[1]):
            return
        if np.isnan([u[0], v[0], u[1], v[1]]).any():
            return

        p0 = np.array([u[0], v[0]], dtype=np.float32)
        p1 = np.array([u[1], v[1]], dtype=np.float32)
        direction = p1 - p0
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0], dtype=np.float32)

        text_xy = tuple(np.round(p0 - 18.0 * direction + np.array([0.0, -6.0])).astype(int))
        label_points_xy.append(text_xy)

        text = f"{label}*" if is_top else str(label)
        text_color = np.array(color_bgr[::-1], dtype=np.float32) / 255.0

        ax.text(
            text_xy[0],
            text_xy[1],
            text,
            color=text_color,
            fontsize=max(7, int(number_line_width * 4)),
            ha="center",
            va="center",
            path_effects=[
                path_effects.Stroke(linewidth=2, foreground="black"),
                path_effects.Normal(),
            ],
        )

    def draw_grasp(grasp_T, opening, color_bgr, label, is_top):
        pts_cam, u, v, valid = project_grasp(grasp_T, opening)

        for i0, i1 in [(0, 1), (1, 2), (4, 5)]:
            if valid[i0] and valid[i1] and not np.isnan([u[i0], v[i0], u[i1], v[i1]]).any():
                draw_line(
                    (u[i0], v[i0], pts_cam[i0, 2]),
                    (u[i1], v[i1], pts_cam[i1, 2]),
                    color_bgr,
                )

        for i0, i1 in [(2, 3), (5, 6)]:
            if not (valid[i0] and valid[i1]):
                continue
            if np.isnan([u[i0], v[i0], u[i1], v[i1]]).any():
                continue

            start = first_visible_on_finger(
                (u[i0], v[i0], pts_cam[i0, 2]),
                (u[i1], v[i1], pts_cam[i1, 2]),
            )
            if start is not None:
                draw_line(start, (u[i1], v[i1], pts_cam[i1, 2]), color_bgr)

        draw_label(u, v, valid, label, color_bgr, is_top)

    label = 1
    for key, grasps in pred_grasps_cam.items():
        grasps = np.asarray(grasps)
        if len(grasps) == 0:
            continue

        openings = np.asarray(gripper_openings[key], dtype=np.float32)
        if draw_grip_opening_bigger:
            openings = openings + increase_grip_opening_by

        for i, (grasp_T, opening) in enumerate(zip(grasps, openings)):
            color_bgr = get_grasp_color(rgb, segmap, object_id, i)
            draw_grasp(
                grasp_T,
                opening,
                color_bgr,
                label,
                key == top_key and i == top_idx,
            )
            label += 1

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)

    full_path = str(out_path)
    cropped_path = str(out_path.with_name(out_path.stem + "_cropped" + out_path.suffix))

    annotated_rgb = cv2.cvtColor(cv2.imread(str(out_path)), cv2.COLOR_BGR2RGB)

    h, w = annotated_rgb.shape[:2]
    h_orig, w_orig = rgb.shape[:2]

    scale_x = w / w_orig
    scale_y = h / h_orig

    mask = np.asarray(segmap) == int(object_id)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError(f"No pixels found for object_id={object_id}")

    x_min = int(np.floor(xs.min() * scale_x))
    x_max = int(np.ceil(xs.max() * scale_x))
    y_min = int(np.floor(ys.min() * scale_y))
    y_max = int(np.ceil(ys.max() * scale_y))

    if label_points_xy is not None and len(label_points_xy) > 0:
        pts = np.asarray(label_points_xy, dtype=np.float32).reshape(-1, 2)
        pts = pts[np.isfinite(pts).all(axis=1)]

        if len(pts) > 0:
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y

            x_min = min(x_min, int(np.floor(pts[:, 0].min())))
            x_max = max(x_max, int(np.ceil(pts[:, 0].max())))
            y_min = min(y_min, int(np.floor(pts[:, 1].min())))
            y_max = max(y_max, int(np.ceil(pts[:, 1].max())))

    pad = 40
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w - 1, x_max + pad)
    y_max = min(h - 1, y_max + pad)

    cropped_rgb = annotated_rgb[y_min:y_max + 1, x_min:x_max + 1].copy()
    cv2.imwrite(str(cropped_path), cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))

    return {
        "annotated_full_size": full_path,
        "annotated_cropped": cropped_path,
    }


def save_grasp_score_heatmap(
    rgb,
    K,
    contact_pts,
    pred_grasps_cam,
    scores,
    output_dir="/shared/debug/cgn/heatmap",
):
    if rgb is None:
        return ""
    if K is None:
        raise ValueError("Camera intrinsics K are required.")

    rgb = _ensure_uint8_rgb(rgb)
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"heatmap_{time.strftime('%Y%m%d_%H%M%S')}.png"

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    all_scores = []
    for value in scores.values():
        all_scores.extend(np.asarray(value).reshape(-1).tolist())

    if len(all_scores) == 0:
        cv2.imwrite(str(out_path), img)
        return str(out_path)

    all_scores = np.asarray(all_scores, dtype=np.float32)
    s_min = float(all_scores.min())
    s_max = float(all_scores.max())

    def score_to_color(score):
        t = 1.0 if s_max - s_min < 1e-8 else float((score - s_min) / (s_max - s_min))
        return (0, int(round(255 * t)), int(round(255 * (1.0 - t))))

    for key in contact_pts:
        pts = np.asarray(contact_pts[key], dtype=np.float32)
        grasps = np.asarray(pred_grasps_cam[key], dtype=np.float32)
        key_scores = np.asarray(scores[key]).reshape(-1)

        if pts.ndim != 2 or pts.shape[1] != 3:
            print(f"Skipping key {key}: unexpected contact_pts shape {pts.shape}")
            continue
        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
            print(f"Skipping key {key}: unexpected pred_grasps_cam shape {grasps.shape}")
            continue

        n = min(len(pts), len(grasps), len(key_scores))
        for pt, grasp_T, score in zip(pts[:n], grasps[:n], key_scores[:n]):
            contact_uv = _project_point(pt, K)
            base_uv = _project_point(grasp_T[:3, 3], K)
            if contact_uv is None or base_uv is None:
                continue

            cu, cv = contact_uv
            bu, bv = base_uv
            if not (0 <= cu < img.shape[1] and 0 <= cv < img.shape[0]):
                continue
            if not (0 <= bu < img.shape[1] and 0 <= bv < img.shape[0]):
                continue

            color = score_to_color(float(score))
            cv2.line(img, (cu, cv), (bu, bv), color, 1, lineType=cv2.LINE_AA)
            cv2.circle(img, (cu, cv), 2, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (bu, bv), 2, color, -1, lineType=cv2.LINE_AA)

    legend_w, legend_h = 160, 18
    x0, y0 = 20, img.shape[0] - 40

    for i in range(legend_w):
        t = i / max(legend_w - 1, 1)
        color = score_to_color(s_min + t * (s_max - s_min))
        cv2.line(img, (x0 + i, y0), (x0 + i, y0 + legend_h), color, 1)

    cv2.rectangle(img, (x0, y0), (x0 + legend_w, y0 + legend_h), (255, 255, 255), 1)
    cv2.putText(img, f"{s_min:.3f}", (x0, y0 + legend_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"{s_max:.3f}", (x0 + legend_w - 50, y0 + legend_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(img, "low", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(img, "high", (x0 + legend_w - 35, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    return str(out_path)