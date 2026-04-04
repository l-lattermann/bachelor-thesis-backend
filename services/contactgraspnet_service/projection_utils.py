from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import mesh_utils


def save_projected_grasp_overlay(
    rgb,
    K,
    pred_grasps_cam,
    scores,
    gripper_openings,
    draw_default_opening=True,
    default_opening=0.10,
    draw_confidence=True,
    output_path="/shared/pipeline_io/cgn_output.png",
):
    if rgb is None:
        return ""

    if K is None:
        raise ValueError("Camera intrinsics K are required for grasp projection.")

    K = np.asarray(K, dtype=np.float32).reshape(3, 3)

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    gripper = mesh_utils.create_gripper("panda")
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5 * (gripper_control_points[1, :] + gripper_control_points[2, :])

    grasp_line_template = np.array(
        [
            np.zeros((3,), dtype=np.float32),
            mid_point.astype(np.float32),
            gripper_control_points[1].astype(np.float32),
            gripper_control_points[3].astype(np.float32),
            gripper_control_points[1].astype(np.float32),
            gripper_control_points[2].astype(np.float32),
            gripper_control_points[4].astype(np.float32),
        ],
        dtype=np.float32,
    )

    lines = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]

    def get_projected_gripper_points(grasp_T, opening):
        pts_local = grasp_line_template.copy()
        pts_local[2:, 0] = np.sign(grasp_line_template[2:, 0]) * float(opening) / 2.0

        pts_cam = pts_local @ grasp_T[:3, :3].T
        pts_cam += grasp_T[:3, 3]

        X = pts_cam[:, 0]
        Y = pts_cam[:, 1]
        Z = pts_cam[:, 2]

        valid = Z > 1e-6
        u = np.full(len(pts_cam), np.nan, dtype=np.float32)
        v = np.full(len(pts_cam), np.nan, dtype=np.float32)

        u[valid] = K[0, 0] * X[valid] / Z[valid] + K[0, 2]
        v[valid] = K[1, 1] * Y[valid] / Z[valid] + K[1, 2]

        return u, v, valid

    def project_grasp(img, grasp_T, opening, color, thickness):
        u, v, valid = get_projected_gripper_points(grasp_T, opening)

        for i0, i1 in lines:
            if not (valid[i0] and valid[i1]):
                continue
            if np.isnan(u[i0]) or np.isnan(v[i0]) or np.isnan(u[i1]) or np.isnan(v[i1]):
                continue

            p0 = (int(round(u[i0])), int(round(v[i0])))
            p1 = (int(round(u[i1])), int(round(v[i1])))
            cv2.line(img, p0, p1, color, thickness, lineType=cv2.LINE_AA)

    def mpl_to_bgr(c):
        return tuple((np.array(c[:3])[::-1] * 255).astype(np.uint8).tolist())

    def score_to_color(score):
        if score > 0.8:
            return (0.0, 1.0, 0.0)   # green
        elif score > 0.5:
            return (1.0, 1.0, 0.0)   # yellow
        else:
            return (1.0, 0.0, 0.0)   # red

    cmap = plt.get_cmap("tab20")

    keys = list(pred_grasps_cam.keys())
    if len(keys) == 0:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        return str(out_path)

    for k in pred_grasps_cam:
        if not np.any(pred_grasps_cam[k]):
            continue

        grasps_k = pred_grasps_cam[k]
        scores_k = scores[k]

        gripper_openings_k = (
            np.ones(len(grasps_k), dtype=np.float32) * default_opening
            if draw_default_opening or gripper_openings is None
            else np.asarray(gripper_openings[k], dtype=np.float32)
        )

        n = len(grasps_k)

        if len(keys) > 1:
            grasp_colors = [
                mpl_to_bgr(cmap(i % cmap.N)[:3])
                for i in range(n)
            ]

            for grasp_T, opening, color in zip(grasps_k, gripper_openings_k, grasp_colors):
                project_grasp(img, grasp_T, opening, color, 1)

            best_idx = int(np.argmax(scores_k))
            best_color = mpl_to_bgr(score_to_color(float(scores_k[best_idx])))
            project_grasp(img, grasps_k[best_idx], gripper_openings_k[best_idx], best_color, 2)

        else:
            for i, (grasp_T, opening, score) in enumerate(zip(grasps_k, gripper_openings_k, scores_k)):
                if draw_confidence:
                    color = mpl_to_bgr(score_to_color(float(score)))
                else:
                    color = mpl_to_bgr(cmap(i % cmap.N)[:3])

                project_grasp(img, grasp_T, opening, color, 1)

            best_idx = int(np.argmax(scores_k))
            project_grasp(img, grasps_k[best_idx], gripper_openings_k[best_idx], (255, 255, 255), 2)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return str(out_path)