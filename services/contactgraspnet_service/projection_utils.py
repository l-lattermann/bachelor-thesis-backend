from pathlib import Path
import cv2
import numpy as np
import time 

import mesh_utils


#def save_projected_grasp_overlay(
#    rgb,
#    K,
#    pred_grasps_cam,
#    gripper_openings=None,
#    draw_grip_opening_bigger=True,
#    increase_grip_opening_by=0.02,
#    gripper_line_width=2,
#    number_line_width=2,
#    output_path="/shared/pipeline_io/cgn_output.png",
#):
#
#
#    if rgb is None:
#        return ""
#
#    if K is None:
#        raise ValueError("Camera intrinsics K are required.")
#
#    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
#
#    if rgb.dtype != np.uint8:
#        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
#
#    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#
#    gripper = mesh_utils.create_gripper("panda")
#    cp = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
#    mid = 0.5 * (cp[1] + cp[2])
#
#    grasp_line_template = np.array(
#        [
#            np.zeros(3, dtype=np.float32),
#            mid.astype(np.float32),
#            cp[1].astype(np.float32),
#            cp[3].astype(np.float32),
#            cp[1].astype(np.float32),
#            cp[2].astype(np.float32),
#            cp[4].astype(np.float32),
#        ],
#        dtype=np.float32,
#    )
#
#    lines = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]
#
#    colors = [
#        (0, 255, 0),
#        (0, 0, 255),
#        (0, 255, 255),
#        (255, 0, 255),
#        (255, 255, 0),
#        (0, 165, 255),
#        (128, 0, 255),
#        (0, 128, 255),
#        (180, 105, 255),
#    ]
#
#    def project_points(grasp_T, opening):
#        pts = grasp_line_template.copy()
#        pts[2:, 0] = np.sign(grasp_line_template[2:, 0]) * float(opening) / 2.0
#
#        pts_cam = pts @ grasp_T[:3, :3].T
#        pts_cam += grasp_T[:3, 3]
#
#        x = pts_cam[:, 0]
#        y = pts_cam[:, 1]
#        z = pts_cam[:, 2]
#
#        valid = z > 1e-6
#        u = np.full(len(pts_cam), np.nan, dtype=np.float32)
#        v = np.full(len(pts_cam), np.nan, dtype=np.float32)
#
#        u[valid] = K[0, 0] * x[valid] / z[valid] + K[0, 2]
#        v[valid] = K[1, 1] * y[valid] / z[valid] + K[1, 2]
#
#        return u, v, valid
#
#    def draw_label(u, v, valid, label, color):
#        if not (valid[0] and valid[1]):
#            return
#        if np.isnan(u[0]) or np.isnan(v[0]) or np.isnan(u[1]) or np.isnan(v[1]):
#            return
#
#        p0 = np.array([u[0], v[0]], dtype=np.float32)
#        p1 = np.array([u[1], v[1]], dtype=np.float32)
#
#        direction = p1 - p0
#        norm = np.linalg.norm(direction)
#        if norm > 1e-6:
#            direction /= norm
#        else:
#            direction = np.array([1.0, 0.0], dtype=np.float32)
#
#        text_pos = p0 - 18.0 * direction + np.array([0.0, -6.0], dtype=np.float32)
#        text_xy = (int(round(text_pos[0])), int(round(text_pos[1])))
#
#        cv2.putText(
#            img,
#            str(label),
#            text_xy,
#            cv2.FONT_HERSHEY_SIMPLEX,
#            0.55,
#            color,
#            number_line_width,
#            lineType=cv2.LINE_AA,
#        )
#
#    def draw_grasp(grasp_T, opening, color, label):
#        u, v, valid = project_points(grasp_T, opening)
#
#        for i0, i1 in lines:
#            if not (valid[i0] and valid[i1]):
#                continue
#            if np.isnan(u[i0]) or np.isnan(v[i0]) or np.isnan(u[i1]) or np.isnan(v[i1]):
#                continue
#
#            p0 = (int(round(u[i0])), int(round(v[i0])))
#            p1 = (int(round(u[i1])), int(round(v[i1])))
#            cv2.line(img, p0, p1, color, gripper_line_width, lineType=cv2.LINE_AA)
#
#        draw_label(u, v, valid, label, color)
#
#    label = 1
#
#    for key in pred_grasps_cam:
#        grasps = np.asarray(pred_grasps_cam[key])
#        if len(grasps) == 0:
#            continue
#
#        if gripper_openings is None:
#            raise ValueError(f"Gripper_openings is None")
#
#        elif draw_grip_opening_bigger:
#            openings = np.asarray(gripper_openings[key], dtype=np.float32) + increase_grip_opening_by
#        else:
#            openings = np.asarray(gripper_openings[key], dtype=np.float32)
#
#        for i, (grasp_T, opening) in enumerate(zip(grasps, openings)):
#            color = colors[i % len(colors)]
#            draw_grasp(grasp_T, opening, color, label)
#            label += 1
#
#    out_path = Path(output_path)
#    out_path.parent.mkdir(parents=True, exist_ok=True)
#    cv2.imwrite(str(out_path), img)
#    return str(out_path)   

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

    K = np.asarray(K, dtype=np.float32).reshape(3, 3)

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    H, W = rgb.shape[:2]

    gripper = mesh_utils.create_gripper("panda")
    cp = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid = 0.5 * (cp[1] + cp[2])

    grasp_line_template = np.array(
        [
            np.zeros(3, dtype=np.float32),
            mid.astype(np.float32),
            cp[1].astype(np.float32),
            cp[3].astype(np.float32),
            cp[1].astype(np.float32),
            cp[2].astype(np.float32),
            cp[4].astype(np.float32),
        ],
        dtype=np.float32,
    )

    colors = [
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

    depth = np.asarray(depth, dtype=np.float32)

    if depth.shape != (H, W):
        raise ValueError(f"depth must have shape {(H, W)}, got {depth.shape}")

    # ganze Szene als Occluder benutzen
    scene_depth = depth.copy()

    # ungültige Werte raus
    scene_depth[~np.isfinite(scene_depth)] = np.inf
    scene_depth[scene_depth <= 0] = np.inf

    left_finger_vec = cp[3].astype(np.float32) - cp[1].astype(np.float32)
    right_finger_vec = cp[4].astype(np.float32) - cp[2].astype(np.float32)

    def project_points(grasp_T, opening):
        pts = grasp_line_template.copy()

        half_opening = float(opening) / 2.0

        pts[2, 0] = -half_opening
        pts[4, 0] = -half_opening
        pts[5, 0] = +half_opening

        pts[3] = pts[2] + left_finger_vec
        pts[6] = pts[5] + right_finger_vec
        pts[1] = 0.5 * (pts[2] + pts[5])

        pts_cam = pts @ grasp_T[:3, :3].T
        pts_cam += grasp_T[:3, 3]

        x = pts_cam[:, 0]
        y = pts_cam[:, 1]
        z = pts_cam[:, 2]

        valid = z > 1e-6
        u = np.full(len(pts_cam), np.nan, dtype=np.float32)
        v = np.full(len(pts_cam), np.nan, dtype=np.float32)

        u[valid] = K[0, 0] * x[valid] / z[valid] + K[0, 2]
        v[valid] = K[1, 1] * y[valid] / z[valid] + K[1, 2]

        return pts_cam, u, v, valid

    def is_visible(px, py, pz, depth_margin=0.005):
        if px < 0 or px >= W or py < 0 or py >= H:
            return False

        scene_z = scene_depth[py, px]

        if not np.isfinite(scene_z):
            return True

        return pz <= (scene_z + depth_margin)

    def find_first_visible_point_on_finger(base_uvz, tip_uvz, depth_margin=0.002):
        u0, v0, z0 = base_uvz
        u1, v1, z1 = tip_uvz

        length = int(max(abs(u1 - u0), abs(v1 - v0)))
        steps = max(length * 2, 2)

        for t in np.linspace(0.0, 1.0, steps):
            u = (1.0 - t) * u0 + t * u1
            v = (1.0 - t) * v0 + t * v1
            z = (1.0 - t) * z0 + t * z1

            px = int(round(u))
            py = int(round(v))

            if is_visible(px, py, z, depth_margin=depth_margin):
                return (u, v, z)

        return None

    def draw_normal_line(p0_uvz, p1_uvz, color):
        u0, v0, _ = p0_uvz
        u1, v1, _ = p1_uvz

        p0 = (int(round(u0)), int(round(v0)))
        p1 = (int(round(u1)), int(round(v1)))

        cv2.line(img, p0, p1, color, gripper_line_width, lineType=cv2.LINE_AA)

    def draw_label(u, v, valid, label, color, is_top_grasp=False):
        if not (valid[0] and valid[1]):
            return
        if np.isnan(u[0]) or np.isnan(v[0]) or np.isnan(u[1]) or np.isnan(v[1]):
            return

        p0 = np.array([u[0], v[0]], dtype=np.float32)
        p1 = np.array([u[1], v[1]], dtype=np.float32)

        direction = p1 - p0
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0], dtype=np.float32)

        text_pos = p0 - 18.0 * direction + np.array([0.0, -6.0], dtype=np.float32)
        text_xy = (int(round(text_pos[0])), int(round(text_pos[1])))

        text = f"{label}*" if is_top_grasp else str(label)

        cv2.putText(
            img,
            text,
            text_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            number_line_width,
            lineType=cv2.LINE_AA,
        )

    def draw_grasp(grasp_T, opening, color, label, is_top_grasp=False):
        pts_cam, u, v, valid = project_points(grasp_T, opening)

        normal_lines = [(0, 1), (1, 2), (4, 5)]
        finger_lines = [(2, 3), (5, 6)]

        for i0, i1 in normal_lines:
            if not (valid[i0] and valid[i1]):
                continue
            if np.isnan(u[i0]) or np.isnan(v[i0]) or np.isnan(u[i1]) or np.isnan(v[i1]):
                continue

            p0 = (u[i0], v[i0], pts_cam[i0, 2])
            p1 = (u[i1], v[i1], pts_cam[i1, 2])
            draw_normal_line(p0, p1, color)

        for i0, i1 in finger_lines:
            if not (valid[i0] and valid[i1]):
                continue
            if np.isnan(u[i0]) or np.isnan(v[i0]) or np.isnan(u[i1]) or np.isnan(v[i1]):
                continue

            base_pt = (u[i0], v[i0], pts_cam[i0, 2])
            tip_pt = (u[i1], v[i1], pts_cam[i1, 2])

            visible_start = find_first_visible_point_on_finger(base_pt, tip_pt)

            if visible_start is None:
                continue

            draw_normal_line(visible_start, tip_pt, color)

        draw_label(u, v, valid, label, color, is_top_grasp=is_top_grasp)

    top_key = None
    top_local_idx = None
    top_score = -np.inf

    for key in pred_grasps_cam:
        if key not in scores:
            continue
        key_scores = np.asarray(scores[key]).reshape(-1)
        if len(key_scores) == 0:
            continue

        local_idx = int(np.argmax(key_scores))
        local_score = float(key_scores[local_idx])

        if local_score > top_score:
            top_score = local_score
            top_key = key
            top_local_idx = local_idx

    label = 1

    for key in pred_grasps_cam:
        grasps = np.asarray(pred_grasps_cam[key])
        if len(grasps) == 0:
            continue

        if gripper_openings is None:
            raise ValueError("gripper_openings is None")

        if draw_grip_opening_bigger:
            openings = np.asarray(gripper_openings[key], dtype=np.float32) + increase_grip_opening_by
        else:
            openings = np.asarray(gripper_openings[key], dtype=np.float32)

        for i, (grasp_T, opening) in enumerate(zip(grasps, openings)):
            color = colors[i % len(colors)]
            is_top_grasp = (key == top_key and i == top_local_idx)
            draw_grasp(grasp_T, opening, color, label, is_top_grasp=is_top_grasp)
            label += 1

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return str(out_path)


def save_grasp_score_heatmap(
    rgb,
    K,
    contact_pts,
    pred_grasps_cam,
    scores,
    output_dir="/shared/debug/cgn/heatmap",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"heatmap_{timestamp}.png"

    if rgb is None:
        return ""

    if K is None:
        raise ValueError("Camera intrinsics K are required.")

    K = np.asarray(K, dtype=np.float32).reshape(3, 3)

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    all_scores = []
    for key in scores:
        s = np.asarray(scores[key]).reshape(-1)
        if len(s) > 0:
            all_scores.extend(s.tolist())

    if len(all_scores) == 0:
        cv2.imwrite(str(out_path), img)
        return str(out_path)

    all_scores = np.asarray(all_scores, dtype=np.float32)
    s_min = float(all_scores.min())
    s_max = float(all_scores.max())

    def score_to_color(score):
        if s_max - s_min < 1e-8:
            t = 1.0
        else:
            t = float((score - s_min) / (s_max - s_min))

        r = int(round(255 * (1.0 - t)))
        g = int(round(255 * t))
        return (0, g, r)  # BGR: red low, green high

    def project_point(pt):
        pt = np.asarray(pt, dtype=np.float32).reshape(3)
        x, y, z = pt

        if z <= 1e-6:
            return None

        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]

        if np.isnan(u) or np.isnan(v):
            return None

        return int(round(u)), int(round(v))

    for key in contact_pts:
        pts = np.asarray(contact_pts[key], dtype=np.float32)
        grasps = np.asarray(pred_grasps_cam[key], dtype=np.float32)
        s = np.asarray(scores[key]).reshape(-1)

        if pts.ndim != 2 or pts.shape[1] != 3:
            print(f"Skipping key {key}: unexpected contact_pts shape {pts.shape}")
            continue

        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
            print(f"Skipping key {key}: unexpected pred_grasps_cam shape {grasps.shape}")
            continue

        n = min(len(pts), len(grasps), len(s))

        for pt, grasp_T, score in zip(pts[:n], grasps[:n], s[:n]):
            contact_uv = project_point(pt)
            base_uv = project_point(grasp_T[:3, 3])

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


    legend_w = 160
    legend_h = 18
    x0 = 20
    y0 = img.shape[0] - 40

    for i in range(legend_w):
        t = i / max(legend_w - 1, 1)
        score = s_min + t * (s_max - s_min)
        color = score_to_color(score)
        cv2.line(img, (x0 + i, y0), (x0 + i, y0 + legend_h), color, 1)

    cv2.rectangle(img, (x0, y0), (x0 + legend_w, y0 + legend_h), (255, 255, 255), 1)

    cv2.putText(
        img,
        f"{s_min:.3f}",
        (x0, y0 + legend_h + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"{s_max:.3f}",
        (x0 + legend_w - 50, y0 + legend_h + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "low",
        (x0, y0 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "high",
        (x0 + legend_w - 35, y0 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )

    cv2.imwrite(str(out_path), img)
    return str(out_path)