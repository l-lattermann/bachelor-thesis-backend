from pathlib import Path
import os
import cv2
import numpy as np
import yaml


def disparity_to_uois_dict(
    disp_arr,
    left_rgb,
    cam,
    disp_params,
    conf=None,
    conf_thr=32,
    seg_id=1,
    background_label=0,
):
    H, W = disp_arr.shape[:2]

    H_cam = int(cam["height"])
    W_cam = int(cam["width"])

    sx = W / W_cam
    sy = H / H_cam

    cam = cam.copy()
    cam["fx"] *= sx
    cam["fy"] *= sy
    cam["cx"] *= sx
    cam["cy"] *= sy
    cam["width"] = W
    cam["height"] = H

    fx = float(cam["fx"])
    fy = float(cam["fy"])
    cx = float(cam["cx"])
    cy = float(cam["cy"])

    scale = float(disp_params["scale"])
    offset = float(disp_params["offset"])
    invalid = float(disp_params["invalid"])
    baseline = float(disp_params["baseline_m"])

    if conf is not None and conf.shape != (H, W):
        conf = cv2.resize(conf, (W, H), interpolation=cv2.INTER_NEAREST)

    d = disp_arr.astype(np.float32) * scale + offset
    valid = np.isfinite(d) & (d > 0) & (disp_arr != invalid)

    if conf is not None:
        valid = valid & (conf > conf_thr)

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

    rgb_out = left_rgb
    if rgb_out.shape[:2] != (H, W):
        rgb_out = cv2.resize(rgb_out, (W, H), interpolation=cv2.INTER_NEAREST)
    rgb_out = rgb_out.astype(np.uint8)

    label = np.full((H, W), background_label, dtype=np.int32)
    label[valid] = seg_id

    return {
        "rgb": rgb_out,
        "xyz": xyz,
        "label": label,
        "fx": np.float32(fx),
        "fy": np.float32(fy),
        "cx": np.float32(cx),
        "cy": np.float32(cy),
        "width": np.int32(W),
        "height": np.int32(H),
    }


def process_rc_cube_output(
    left_rgb,
    disp_arr,
    cam,
    disp_params,
    out_dir="/shared/pipeline_io",
    debug=False,
    debug_base_dir="/shared/debug",
):
    os.makedirs(out_dir, exist_ok=True)

    uois_dict = disparity_to_uois_dict(
        disp_arr=disp_arr,
        left_rgb=left_rgb,
        cam=cam,
        disp_params=disp_params,
        conf=None,
        seg_id=1,
        background_label=0,
    )

    npz_path = os.path.join(out_dir, "rc_cube_output.npz")
    np.savez_compressed(
        npz_path,
        rgb=uois_dict["rgb"],
        xyz=uois_dict["xyz"],
        label=uois_dict["label"],
        fx=uois_dict["fx"],
        fy=uois_dict["fy"],
        cx=uois_dict["cx"],
        cy=uois_dict["cy"],
        width=uois_dict["width"],
        height=uois_dict["height"],
    )

    result = {
        "rc_out_npz": npz_path,
    }

    if debug:
        os.makedirs(debug_base_dir, exist_ok=True)

        left_png = os.path.join(debug_base_dir, "left.png")
        disp_png = os.path.join(debug_base_dir, "disparity.png")
        cam_yaml = os.path.join(debug_base_dir, "cam.yaml")
        disp_yaml = os.path.join(debug_base_dir, "disp_params.yaml")

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

        scaled_cam = {
            "fx": float(uois_dict["fx"]),
            "fy": float(uois_dict["fy"]),
            "cx": float(uois_dict["cx"]),
            "cy": float(uois_dict["cy"]),
            "width": int(uois_dict["width"]),
            "height": int(uois_dict["height"]),
        }

        with open(cam_yaml, "w") as f:
            yaml.safe_dump({"camera_intrinsics": scaled_cam}, f, sort_keys=False)

        with open(disp_yaml, "w") as f:
            yaml.safe_dump({"disparity": disp_params}, f, sort_keys=False)

        result.update({
            "left": left_png,
            "disparity": disp_png,
            "cam": cam_yaml,
            "disp_params": disp_yaml,
            "debug": {
                "left_rgb_shape": list(left_rgb.shape),
                "rgb_min": int(left_rgb.min()),
                "rgb_max": int(left_rgb.max()),
                "rgb_mean": float(left_rgb.mean()),
                "disp_shape": list(disp_arr.shape),
                "cam_hw": [int(uois_dict["height"]), int(uois_dict["width"])],
            },
        })

    return result


def load_rc_cube_mock(folder: str):
    folder = Path(folder)

    left_img_path = next(folder.glob("*_left_*.png"))
    disp_img_path = next(folder.glob("*_disparity_*.png"))
    left_param_path = next(folder.glob("*_left_*_param.txt"))
    disp_param_path = next(folder.glob("*_disparity_*_param.txt"))

    left_rgb = cv2.imread(str(left_img_path))
    left_rgb = cv2.cvtColor(left_rgb, cv2.COLOR_BGR2RGB)

    disp_arr = cv2.imread(str(disp_img_path), cv2.IMREAD_UNCHANGED)

    def parse_param_file(path):
        data = {}
        with open(path, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    data[k.strip()] = v.strip()
        return data

    left_params = parse_param_file(left_param_path)
    disp_params_raw = parse_param_file(disp_param_path)

    A = left_params["camera.A"].strip("[]").replace(";", " ").split()

    cam = {
        "width": int(left_params["camera.width"]),
        "height": int(left_params["camera.height"]),
        "fx": float(A[0]),
        "fy": float(A[4]),
        "cx": float(A[2]),
        "cy": float(A[5]),
    }

    disp_params = {
        "scale": float(disp_params_raw["disp.scale"]),
        "offset": float(disp_params_raw["disp.offset"]),
        "invalid": float(disp_params_raw["disp.inv"]),
        "baseline_m": float(disp_params_raw["t"]),
        "delta_d": None,
        "encoding": "mono16",
    }

    return left_rgb, disp_arr, cam, disp_params