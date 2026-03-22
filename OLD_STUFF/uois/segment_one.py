#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from time import time
from pathlib import Path

import numpy as np

import src.data_augmentation as data_augmentation
import src.segmentation as segmentation
import src.evaluation as evaluation
import src.util.utilities as util_
import src.util.custom_utils as cu


# =========================
# 0) PICK ONE INPUT HERE
# =========================
EXAMPLE_NPY = Path.cwd() / "uois" / "example_images" / "OCID_image_0.npy"  # <-- change


# =========================
# 1) CONFIGS
# =========================
dsn_config = {
    "feature_dim": 64,
    "max_GMS_iters": 10,
    "epsilon": 0.05,
    "sigma": 0.02,
    "num_seeds": 200,
    "subsample_factor": 5,
    "min_pixels_thresh": 500,
    "tau": 15.0,
}

rrn_config = {
    "feature_dim": 64,
    "img_H": 224,
    "img_W": 224,
    "use_coordconv": False,
}

uois3d_config = {
    "padding_percentage": 0.25,
    "use_open_close_morphology": True,
    "open_close_morphology_ksize": 9,
    "use_largest_connected_component": True,
}

checkpoint_dir = Path("uois/checkpoints")
dsn_filename = checkpoint_dir / "DepthSeedingNetwork_3D_TOD_checkpoint.pth"
rrn_filename = checkpoint_dir / "RRN_OID_checkpoint.pth"
uois3d_config["final_close_morphology"] = "TableTop_v5" in str(rrn_filename)

uois_net_3d = segmentation.UOISNet3D(
    uois3d_config,
    str(dsn_filename),
    dsn_config,
    str(rrn_filename),
    rrn_config,
)


# =========================
# 2) LOAD SINGLE EXAMPLE
# =========================
if not EXAMPLE_NPY.exists():
    raise FileNotFoundError(f"Example file not found: {EXAMPLE_NPY}")

d = np.load(str(EXAMPLE_NPY), allow_pickle=True, encoding="bytes").item()

# If it’s a Contact-GraspNet-style dict, convert it to UOIS format
if set(d.keys()) == {"rgb", "depth", "K", "seg"}:
    d = cu.convert_cgn_npy_to_uois(d, depth_scale=1.0, out_size=(640, 480))

# Expected UOIS keys: rgb, xyz, label (and maybe others)
rgb_img = d["rgb"]
xyz_img = d["xyz"]
label_img = d["label"]

rgb_std = data_augmentation.standardize_image(rgb_img)

batch = {
    "rgb": data_augmentation.array_to_tensor(rgb_std[None, ...]),   # (1,H,W,3) -> tensor
    "xyz": data_augmentation.array_to_tensor(xyz_img[None, ...]),   # (1,H,W,3) -> tensor
}

print(f"Running single image: {EXAMPLE_NPY.name}")


# =========================
# 3) RUN INFERENCE
# =========================
st_time = time()
fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)
total_time = time() - st_time

print(f"Total time: {total_time:.3f}s")
print(f"FPS: {1.0 / total_time:.3f}")

seg_masks_np = seg_masks.cpu().numpy()[0]          # (H,W)
fg_masks_np = fg_masks.cpu().numpy()[0]
center_offsets_np = center_offsets.cpu().numpy().transpose(0, 2, 3, 1)[0]
initial_masks_np = initial_masks.cpu().numpy()[0]


# =========================
# 4) EVAL + SAVE OUTPUTS
# =========================
eval_metrics = evaluation.multilabel_metrics(seg_masks_np, label_img)
print("Metrics:")
print(eval_metrics)

# Prepare visualization outputs
rgb_uint8 = util_.torch_to_numpy(batch["rgb"].cpu(), is_standardized_image=True)[0].astype(np.uint8)
depth = xyz_img[..., 2].astype(np.float32)

num_objs = int(max(np.unique(seg_masks_np).max(), np.unique(label_img).max()) + 1)
seg_mask_plot = util_.get_color_mask(seg_masks_np, nc=num_objs)
gt_mask_plot = util_.get_color_mask(label_img, nc=num_objs)

img_batch = {
    EXAMPLE_NPY.name: [rgb_uint8, depth, seg_mask_plot, gt_mask_plot]
}

out_vis_dir = Path("output/uois_single")
out_vis_dir.mkdir(parents=True, exist_ok=True)
cu.save_imgs(img_batch, str(out_vis_dir))

print(f"Saved visualizations to: {out_vis_dir}")


# =========================
# 5) OPTIONAL: EXPORT CGN-FORMAT .npy
# =========================
out_cgn_dir = Path("contact_graspnet/test_data_npy_from_uois")
out_cgn_dir.mkdir(parents=True, exist_ok=True)

H, W = rgb_uint8.shape[:2]
fx, fy = 525.0, 525.0
cx, cy = W / 2.0, H / 2.0

stem = EXAMPLE_NPY.stem
out_npy = out_cgn_dir / f"{stem}.npy"

cu.uois_to_contactgraspnet(
    rgb=rgb_uint8,
    xyz=xyz_img.astype(np.float32),
    seg=seg_masks_np,
    out_npy=str(out_npy),
    fx=fx, fy=fy, cx=cx, cy=cy,
)

print(f"Exported Contact-GraspNet npy to: {out_npy}")