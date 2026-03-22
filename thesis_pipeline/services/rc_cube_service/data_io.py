import numpy as np
import os
import cv2
from plyfile import PlyData, PlyElement
import yaml


def save_uois_dict_to_ply(uois_dict, ply_path, use_rgb=True, mask_valid=True):
    xyz = uois_dict["xyz"].reshape(-1, 3)
    rgb = uois_dict.get("rgb")

    if rgb is not None:
        rgb = rgb.reshape(-1, 3).astype(np.uint8)

    if mask_valid:
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
        xyz = xyz[valid]
        if rgb is not None:
            rgb = rgb[valid]

    out_dir = os.path.dirname(ply_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if use_rgb and rgb is not None:
        verts = np.empty(
            len(xyz),
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        verts["x"] = xyz[:, 0]
        verts["y"] = xyz[:, 1]
        verts["z"] = xyz[:, 2]
        verts["red"] = rgb[:, 0]
        verts["green"] = rgb[:, 1]
        verts["blue"] = rgb[:, 2]
    else:
        verts = np.empty(
            len(xyz),
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
            ],
        )
        verts["x"] = xyz[:, 0]
        verts["y"] = xyz[:, 1]
        verts["z"] = xyz[:, 2]

    ply = PlyData([PlyElement.describe(verts, "vertex")], text=False)
    ply.write(ply_path)

    return ply_path

def disparity_to_uois_dict(
    disp_arr,
    left_rgb,
    cam,
    disp_params,
    conf=None,
    conf_thr=128,
    seg_id=1,
    background_label=0,
):
    # disparity resolution = target resolution
    H, W = disp_arr.shape[:2]

    # original camera/image resolution
    H_cam = int(cam["height"])
    W_cam = int(cam["width"])

    # scale intrinsics from left-image resolution down to disparity resolution
    sx = W / W_cam
    sy = H / H_cam

    fx = float(cam["fx"]) * sx
    fy = float(cam["fy"]) * sy
    cx = float(cam["cx"]) * sx
    cy = float(cam["cy"]) * sy

    scale = float(disp_params["scale"])
    offset = float(disp_params["offset"])
    invalid = float(disp_params["invalid"])
    baseline = float(disp_params["baseline_m"])

    if conf is not None and conf.shape != (H, W):
        conf = cv2.resize(conf, (W, H), interpolation=cv2.INTER_NEAREST)

    # disparity
    d = disp_arr.astype(np.float32) * scale + offset
    valid = np.isfinite(d) & (d > 0) & (disp_arr != invalid)

    if conf is not None:
        valid = valid & (conf > conf_thr)

    # depth
    z = np.zeros((H, W), dtype=np.float32)
    z[valid] = (fx * baseline) / d[valid]

    # backprojection
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    x = np.zeros((H, W), dtype=np.float32)
    y = np.zeros((H, W), dtype=np.float32)
    x[valid] = (uu[valid] - cx) * z[valid] / fx
    y[valid] = (vv[valid] - cy) * z[valid] / fy

    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)

    # only scale left image down to disparity size
    rgb_out = left_rgb
    if rgb_out.shape[:2] != (H, W):
        rgb_out = cv2.resize(rgb_out, (W, H), interpolation=cv2.INTER_NEAREST)
    rgb_out = rgb_out.astype(np.uint8)

    # label
    label = np.full((H, W), background_label, dtype=np.int32)
    label[valid] = seg_id

    return {
        "rgb": rgb_out,
        "xyz": xyz,
        "label": label,
    }

def save_uois_npy(uois_dict, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        rgb=uois_dict["rgb"],
        xyz=uois_dict["xyz"],
        label=uois_dict["label"],
    )
    return out_path

def save_rc_cube_output(left_rgb, disp_arr, cam=None, disp_params=None, base_dir="/shared/input"):
    os.makedirs(base_dir, exist_ok=True)

    left_png = os.path.join(base_dir, "left.png")
    disp_png = os.path.join(base_dir, "disparity.png")
    cam_yaml = os.path.join(base_dir, "cam.yaml")
    disp_yaml = os.path.join(base_dir, "disp_params.yaml")

    cv2.imwrite(left_png, cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))

    disp_vis = disp_arr.astype(np.float32)
    mask = disp_vis > 0

    if mask.any():
        min_val = disp_vis[mask].min()
        max_val = disp_vis[mask].max()

        if max_val > min_val:
            disp_vis = (disp_vis - min_val) / (max_val - min_val) * 255.0
        else:
            disp_vis[:] = 0

    disp_vis = disp_vis.astype(np.uint8)
    cv2.imwrite(disp_png, disp_vis)

    if cam is not None:
        cam_struct = {
            "camera_intrinsics": cam
        }
        with open(cam_yaml, "w") as f:
            yaml.safe_dump(cam_struct, f, sort_keys=False)

    if disp_params is not None:
        disp_struct = {
            "disparity": disp_params
        }
        with open(disp_yaml, "w") as f:
            yaml.safe_dump(disp_struct, f, sort_keys=False)

    return {
        "left": left_png,
        "disparity": disp_png,
        "cam": cam_yaml if cam else None,
        "disp_params": disp_yaml if disp_params else None,
    }