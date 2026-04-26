from pathlib import Path
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import mesh_utils

def _ensure_uint8_rgb(rgb):
    if rgb is None:
        return None
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb

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

    def sscore_to_color(score):
        t = 1.0 if s_max - s_min < 1e-8 else float((score - s_min) / (s_max - s_min))
        t = 1.0 - t        # invert → high = green
        t = t**2           # optional: emphasize high scores

        t = int(t * 255)
        color = cv2.applyColorMap(np.array([[t]], dtype=np.uint8), cv2.COLORMAP_TURBO)
        return tuple(int(c) for c in color[0,0])

    def score_to_color(score):
        t = 1.0 if s_max - s_min < 1e-8 else float((score - s_min) / (s_max - s_min))
        
        if t >= 0.4:
            return (0, 255, 0)   # green (high)
        else:
            return (0, 0, 255)   # red (low)

    for key in contact_pts:
        pts = np.asarray(contact_pts[key], dtype=np.float32)
        grasps = np.asarray(pred_grasps_cam[key], dtype=np.float32)
        key_scores = np.asarray(scores[key]).reshape(-1)

        if pts.ndim != 2 or pts.shape[1] != 3:
            continue
        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
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
            #cv2.line(img, (cu, cv), (bu, bv), color, 1, lineType=cv2.LINE_AA)
            cv2.circle(img, (cu, cv), 7, color, -1, lineType=cv2.LINE_AA)
            #cv2.circle(img, (bu, bv), 2, color, -1, lineType=cv2.LINE_AA)
            
    s = 2  # scale
    fs, th = 0.45*s, 2

    legend_w, legend_h = 160*s, 18*s
    x0, y0 = 20*s, img.shape[0] - 40*s

    # title centered above scale
    label = "Grasp Score"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)

    cv2.putText(
        img,
        label,
        (x0 + (legend_w - text_w)//2, y0 - 20*s),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        (255,255,255),
        th,
        cv2.LINE_AA
    )

    for i in range(legend_w):
        t = i / max(legend_w - 1, 1)
        color = score_to_color(s_min + t * (s_max - s_min))
        cv2.line(img, (x0 + i, y0), (x0 + i, y0 + legend_h), color, 1)

    cv2.rectangle(img, (x0, y0), (x0 + legend_w, y0 + legend_h), (255, 255, 255), 2)

    
    cv2.putText(img, f"{s_min:.3f}", (x0, y0 + legend_h + 18*s), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th, cv2.LINE_AA)
    cv2.putText(img, f"{s_max:.3f}", (x0 + legend_w - 50*s, y0 + legend_h + 18*s), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th, cv2.LINE_AA)
    cv2.putText(img, "low",  (x0, y0 - 8*s),  cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th, cv2.LINE_AA)
    cv2.putText(img, "high", (x0 + legend_w - 35*s, y0 - 8*s), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    return str(out_path)