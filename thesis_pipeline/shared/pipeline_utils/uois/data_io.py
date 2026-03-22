import os
import cv2
import numpy as np



def save_imgs(img_batch, path, img_no):
    os.makedirs(path, exist_ok=True)

    for set_name, img_set in img_batch.items():
        for i, img in enumerate(img_set):
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            print("SET NAME = ", set_name)
            if i in img_no:
                out_path = os.path.join(path, f"{set_name}_{i}.png")
                print("Writing:", out_path)

                cv2.imwrite(out_path, img)

def depth_to_xyz(depth, K, depth_scale=1.0):
    """
    depth: (H, W)
    K: (3, 3)
    depth_scale: multiply depth by this (e.g. 0.001 if depth is in mm)
    """
    depth = depth.astype(np.float32) * depth_scale

    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=-1).astype(np.float32)  # (H, W, 3)


def resize_rgb_depth_and_K(rgb, depth, K, out_w=640, out_h=480):
    """
    Resizes rgb+depth and scales K accordingly.
    """
    in_h, in_w = rgb.shape[:2]
    if depth.shape[:2] != (in_h, in_w):
        raise ValueError(f"rgb and depth size mismatch: rgb={rgb.shape[:2]} depth={depth.shape[:2]}")

    sx = out_w / in_w
    sy = out_h / in_h

    rgb_r = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    depth_r = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    K2 = K.astype(np.float32).copy()
    K2[0, 0] *= sx  # fx
    K2[1, 1] *= sy  # fy
    K2[0, 2] *= sx  # cx
    K2[1, 2] *= sy  # cy

    return rgb_r, depth_r, K2


def convert_cgn_npy_to_uois(
    npy_dict,
    depth_scale=1.0,
    keep_seg=False,
    out_size=(640, 480),
    add_dummy_label=True,
):
    """
    CGN-style: {'rgb','depth','K','seg'}
    -> UOIS-style: {'rgb','xyz'} (+ optional 'seg', + optional dummy 'label')

    out_size: (out_w, out_h). If None, keep original resolution.
    """
    required = {"rgb", "depth", "K"}
    if not required.issubset(npy_dict.keys()):
        raise KeyError(f"Missing keys: {required - set(npy_dict.keys())}")

    rgb = npy_dict["rgb"]
    depth = npy_dict["depth"]
    K = npy_dict["K"]

    # Resize BEFORE computing xyz
    if out_size is not None:
        out_w, out_h = out_size
        if rgb.shape[:2] != (out_h, out_w):
            rgb, depth, K = resize_rgb_depth_and_K(
                rgb, depth, K, out_w=out_w, out_h=out_h
            )

    xyz = depth_to_xyz(depth, K, depth_scale=depth_scale)

    out = {
        "rgb": rgb,
        "xyz": xyz,
    }

    # Optional CGN segmentation
    if keep_seg and "seg" in npy_dict:
        seg = npy_dict["seg"]
        if out_size is not None and seg.shape[:2] != (out_h, out_w):
            seg = cv2.resize(seg, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        out["seg"] = seg.astype(np.int32)

    # -------------------------------------------------
    # DUMMY LABEL (FOR INFERENCE-ONLY SCRIPTS)
    # Can be removed later without side effects
    # -------------------------------------------------
    if add_dummy_label:
        H, W = rgb.shape[:2]
        out["label"] = np.zeros((H, W), dtype=np.int32)

    return out


def uois_to_contactgraspnet(rgb, xyz, seg, out_npy, fx, fy, cx, cy):
    """
    rgb: uint8 (H,W,3) RGB
    xyz: float32 (H,W,3) in camera coords (meters); depth = xyz[...,2]
    seg: (H,W) integer labels (background 0)
    """
    rgb = rgb.astype(np.uint8)

    depth = xyz[..., 2].astype(np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0

    seg = seg.astype(np.int32).copy()
    seg[seg < 0] = 0

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    #rgb = rgb[..., ::-1]  # RGB -> BGR
    out = {"rgb": rgb, "depth": depth, "K": K, "seg": seg}

    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, out)

    return out
