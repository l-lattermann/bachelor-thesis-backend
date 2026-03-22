from pathlib import Path
from time import time
import numpy as np
import yaml
import os

import uois.src.data_augmentation as data_augmentation
import uois.src.evaluation as evaluation
import uois.src.segmentation as segmentation
import uois.src.util.utilities as util_

import data_io as io_utils


CONFIG_PATH = Path("/app/config.yaml")

with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

dsn_config = CFG["uois"]["dsn_config"]
rrn_config = CFG["uois"]["rrn_config"]
uois3d_config = dict(CFG["uois"]["uois3d_config"])

checkpoint_dir = Path(CFG["uois"]["checkpoint_dir"])
dsn_ckpt = checkpoint_dir / CFG["uois"]["checkpoints"]["dsn"]
rrn_ckpt = checkpoint_dir / CFG["uois"]["checkpoints"]["rrn"]

output_dir = Path(CFG["paths"]["output_dir"])
_model = None


def build_model():
    config = dict(uois3d_config)
    config["final_close_morphology"] = "TableTop_v5" in str(rrn_ckpt)

    return segmentation.UOISNet3D(
        config,
        str(dsn_ckpt),
        dsn_config,
        str(rrn_ckpt),
        rrn_config,
    )


def load_model():
    global _model
    if _model is None:
        _model = build_model()
    return _model

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


def load_uois_input(npy_path: str):
    data = np.load(npy_path)
    rgb = data["rgb"]
    xyz = data["xyz"]
    label = data["label"] if "label" in data else None

    if set(data.keys()) == {"rgb", "depth", "K", "seg"}:
        data = io_utils.convert_cgn_npy_to_uois(data, depth_scale=1.0, out_size=(640, 480))

    rgb = data["rgb"]
    xyz = data["xyz"]
    label = data.get("label")

    rgb_std = data_augmentation.standardize_image(rgb)

    batch = {
        "rgb": data_augmentation.array_to_tensor(rgb_std[None, ...]),
        "xyz": data_augmentation.array_to_tensor(xyz[None, ...]),
    }

    return data, batch, rgb, xyz, label


def run_uois_on_npy(npy_path: str):
    model = load_model()
    data, batch, rgb, xyz, label = load_uois_input(npy_path)

    start = time()
    fg_masks, center_offsets, initial_masks, seg_masks = model.run_on_batch(batch)
    elapsed = time() - start

    seg = seg_masks.cpu().numpy()[0]
    fg = fg_masks.cpu().numpy()[0]
    center = center_offsets.cpu().numpy().transpose(0, 2, 3, 1)[0]
    initial = initial_masks.cpu().numpy()[0]

    return {
        "seg": seg,
        "fg_masks": fg,
        "center_offsets": center,
        "initial_masks": initial,
        "time": elapsed,
        "rgb": rgb,
        "xyz": xyz,
        "label": label,
        "batch": batch,
        "dsn_config": dsn_config,
    }

def load_camera_yaml(path="/shared/input/cam.yaml"):
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return data.get("camera_intrinsics", None)

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


def save_results(run_result: dict, stem: str = "uois"):
    output_dir.mkdir(parents=True, exist_ok=True)

    seg = run_result["seg"]
    rgb = run_result["rgb"]
    xyz = run_result["xyz"]
    label = run_result["label"]
    batch = run_result["batch"]

    paths = {
        "mask_npy": str(output_dir / f"{stem}_mask.npy"),
    }
    np.save(paths["mask_npy"], seg)

    if label is not None:
        num_objs = int(max(seg.max(), label.max()) + 1)
        pred_vis = util_.get_color_mask(seg, nc=num_objs)
        gt_vis = util_.get_color_mask(label, nc=num_objs)
        rgb_uint8 = util_.torch_to_numpy(batch["rgb"].cpu(), is_standardized_image=True)[0].astype(np.uint8)

        img_batch = {
            f"{stem}.npy": [rgb_uint8, xyz[..., 2].astype(np.float32), pred_vis, gt_vis]
        }

        vis_dir = output_dir / f"{stem}_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        img_no = CFG["uois"]["img_numbers_to_save"]
        save_imgs(img_batch, str(vis_dir), img_no)
        paths["vis_dir"] = str(vis_dir)

        metrics = evaluation.multilabel_metrics(seg, label)
        paths["metrics"] = metrics

    # --- load camera ---
    cam = load_camera_yaml()

    if cam is not None:
        fx = cam["fx_px"]
        fy = cam["fy_px"]
        cx = cam["cx_px"]
        cy = cam["cy_px"]
    else:
        # fallback
        H, W = rgb.shape[:2]
        fx = float(CFG["contact_graspnet"]["camera"]["fallback_fx"])
        fy = float(CFG["contact_graspnet"]["camera"]["fallback_fy"])
        cx = W / 2.0 if CFG["contact_graspnet"]["camera"]["fallback_cx"] is None else float(CFG["contact_graspnet"]["camera"]["fallback_cx"])
        cy = H / 2.0 if CFG["contact_graspnet"]["camera"]["fallback_cy"] is None else float(CFG["contact_graspnet"]["camera"]["fallback_cy"])

    # --- CGN export ---
    cgn_npy = output_dir / f"{stem}_cgn.npy"
    io_utils.uois_to_contactgraspnet(
        rgb=rgb.astype(np.uint8),
        xyz=xyz.astype(np.float32),
        seg=seg,
        out_npy=str(cgn_npy),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )
    paths["cgn_npy"] = str(cgn_npy)

    return paths