import numpy as np
import cv2



def load_images_to_pxl(path_disp_img, path_conf_img, path_left_img):
    """
    Loads disparity (PNG) and RGB image.

    Returns:
        disp_px (H, W) float32
        rgb     (H, W, 3) uint8 (RGB)
    """

    # --- load disparity (keep original depth, e.g. 16-bit) ---
    disp_px = cv2.imread(path_disp_img, cv2.IMREAD_UNCHANGED)
    if disp_px is None:
        raise ValueError(f"Failed to load disparity image: {path_disp_img}")

    # Load confidence
    conf = cv2.imread(path_conf_img, cv2.IMREAD_UNCHANGED)

    # ensure single channel
    if disp_px.ndim == 3:
        disp_px = cv2.cvtColor(disp_px, cv2.COLOR_BGR2GRAY)

    disp_px = disp_px.astype(np.float32)

    # --- load RGB image ---
    rgb = cv2.imread(path_left_img, cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError(f"Failed to load RGB image: {path_left_img}")

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    return disp_px, conf, rgb


def disparity_to_uois_dict(
    disp_px,
    rgb,
    cam,
    disp_params,
    conf=None,
    conf_thr=128,
    seg_id=1,
    background_label=0
):
    H, W = int(cam["height"]), int(cam["width"])
    fx, fy = float(cam["fx"]), float(cam["fy"])
    cx, cy = float(cam["cx"]), float(cam["cy"])
    
    scale = float(disp_params["scale"])
    offset = float(disp_params["offset"])
    invalid = float(disp_params["invalid"])
    baseline = float(disp_params["baseline_m"])

    # --- disparity ---
    d = disp_px.astype(np.float32) * scale + offset

    valid = np.isfinite(d) & (d > 0) & (disp_px != invalid)

    # --- NEW: confidence filtering ---
    if conf is not None:
        if conf.shape != disp_px.shape:
            conf = cv2.resize(conf, (W, H), interpolation=cv2.INTER_NEAREST)
        valid = valid & (conf > conf_thr)

    # --- depth ---
    z = np.zeros((H, W), dtype=np.float32)
    z[valid] = (fx * baseline) / d[valid]

    # --- backprojection ---
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    x = np.zeros((H, W), dtype=np.float32)
    y = np.zeros((H, W), dtype=np.float32)
    x[valid] = (uu[valid] - cx) * z[valid] / fx
    y[valid] = (vv[valid] - cy) * z[valid] / fy

    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)

    # --- rgb ---
    if rgb is None:
        rgb_out = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        rgb_out = rgb
        if rgb_out.shape[:2] != (H, W):
            rgb_out = cv2.resize(rgb_out, (W, H), interpolation=cv2.INTER_NEAREST)
        rgb_out = rgb_out.astype(np.uint8)

    # --- label ---
    label = np.full((H, W), background_label, dtype=np.int32)
    label[valid] = seg_id

    return {"rgb": rgb_out, "xyz": xyz, "label": label}


def save_uois_npy(data_dict, out_path):
    """
    Saves {"rgb": ..., "xyz": ..., "label": ...} as .npy
    """
    np.save(out_path, data_dict, allow_pickle=True)


def make_disp_visable(path_in, path_out):
    disp = cv2.imread(path_in, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # normalize to 0–255 for visualization
    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)

    disp_vis = disp_norm.astype(np.uint8)
    cv2.imwrite(path_out, disp_vis)

def save_rgb_disp_overlay(path_disp, path_rgb, path_out, alpha=0.6):
    # load disparity and normalize
    disp = cv2.imread(path_disp, cv2.IMREAD_UNCHANGED).astype(np.float32)
    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # apply colormap for better visibility
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

    # load rgb (convert BGR → RGB for consistency, then back for cv2)
    rgb = cv2.imread(path_rgb, cv2.IMREAD_COLOR)
    
    # resize if needed
    if rgb.shape[:2] != disp_color.shape[:2]:
        rgb = cv2.resize(rgb, (disp_color.shape[1], disp_color.shape[0]))

    # blend
    overlay = cv2.addWeighted(rgb, alpha, disp_color, 1 - alpha, 0)

    cv2.imwrite(path_out, overlay)