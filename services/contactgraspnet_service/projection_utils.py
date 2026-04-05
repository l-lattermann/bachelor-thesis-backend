from pathlib import Path

import cv2
import numpy as np

import mesh_utils


def save_projected_grasp_overlay(
    rgb,
    K,
    pred_grasps_cam,
    gripper_openings=None,
    draw_default_opening=True,
    default_opening=0.10,
    gripper_line_width=2,
    number_line_width=2,
    output_path="/shared/pipeline_io/cgn_output.png",
):


    if rgb is None:
        return ""

    if K is None:
        raise ValueError("Camera intrinsics K are required.")

    K = np.asarray(K, dtype=np.float32).reshape(3, 3)

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

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

    lines = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]

    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 165, 255),
        (128, 0, 255),
        (0, 128, 255),
        (180, 105, 255),
    ]

    def project_points(grasp_T, opening):
        pts = grasp_line_template.copy()
        pts[2:, 0] = np.sign(grasp_line_template[2:, 0]) * float(opening) / 2.0

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

        return u, v, valid

    def draw_label(u, v, valid, label, color):
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

        cv2.putText(
            img,
            str(label),
            text_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            number_line_width,
            lineType=cv2.LINE_AA,
        )

    def draw_grasp(grasp_T, opening, color, label):
        u, v, valid = project_points(grasp_T, opening)

        for i0, i1 in lines:
            if not (valid[i0] and valid[i1]):
                continue
            if np.isnan(u[i0]) or np.isnan(v[i0]) or np.isnan(u[i1]) or np.isnan(v[i1]):
                continue

            p0 = (int(round(u[i0])), int(round(v[i0])))
            p1 = (int(round(u[i1])), int(round(v[i1])))
            cv2.line(img, p0, p1, color, gripper_line_width, lineType=cv2.LINE_AA)

        draw_label(u, v, valid, label, color)

    label = 1

    for key in pred_grasps_cam:
        grasps = np.asarray(pred_grasps_cam[key])
        if len(grasps) == 0:
            continue

        if draw_default_opening or gripper_openings is None:
            openings = np.full(len(grasps), default_opening, dtype=np.float32)
        else:
            openings = np.asarray(gripper_openings[key], dtype=np.float32)

        for i, (grasp_T, opening) in enumerate(zip(grasps, openings)):
            color = colors[i % len(colors)]
            draw_grasp(grasp_T, opening, color, label)
            label += 1

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return str(out_path)