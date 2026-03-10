import numpy as np

def disparity_to_uois_dict(disp_px, rgb, cam, disp_params, seg_id=1, background_label=0):
    H, W = int(cam["height"]), int(cam["width"])
    fx, fy = float(cam["fx"]), float(cam["fy"])
    cx, cy = float(cam["cx"]), float(cam["cy"])
    
    scale = float(disp_params["scale"])
    offset = float(disp_params["offset"])
    invalid = float(disp_params["invalid"])
    baseline = float(disp_params["baseline_m"])

    # disparity pixels -> disparity in px
    d = disp_px.astype(np.float32) * scale + offset
    valid = np.isfinite(d) & (d > 0) & (disp_px != invalid)

    # Z = fx * B / disparity
    z = np.zeros((H, W), dtype=np.float32)
    z[valid] = (fx * baseline) / d[valid]

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    x = np.zeros((H, W), dtype=np.float32)
    y = np.zeros((H, W), dtype=np.float32)
    x[valid] = (uu[valid] - cx) * z[valid] / fx
    y[valid] = (vv[valid] - cy) * z[valid] / fy

    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)

    if rgb is None:
        rgb_out = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        # ensure shape matches camera
        rgb_out = rgb
        if rgb_out.shape[:2] != (H, W):
            rgb_out = cv2.resize(rgb_out, (W, H), interpolation=cv2.INTER_NEAREST)
        rgb_out = rgb_out.astype(np.uint8)

    label = np.full((H, W), background_label, dtype=np.int32)
    label[valid] = seg_id

    return {"rgb": rgb_out, "xyz": xyz, "label": label}