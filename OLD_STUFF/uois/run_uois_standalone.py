#!/usr/bin/env python3

import os
from pathlib import Path
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import src.data_augmentation as data_augmentation
import src.evaluation as evaluation
import src.segmentation as segmentation
import src.util.custom_utils as cu
import src.util.utilities as util_


# =========================
# Config
# =========================
CHECKPOINT_DIR = Path("checkpoints")
EXAMPLE_DIR = Path("example_images")
OUT_IMG_DIR = Path("output/uois")
OUT_CGN_DIR = Path("contact_graspnet/test_data_npy_from_uois")

DSN_CONFIG = {
    "feature_dim": 64,
    "max_GMS_iters": 10,
    "epsilon": 0.05,
    "sigma": 0.02,
    "num_seeds": 200,
    "subsample_factor": 5,
    "min_pixels_thresh": 500,
    "tau": 15.0,
}

RRN_CONFIG = {
    "feature_dim": 64,
    "img_H": 224,
    "img_W": 224,
    "use_coordconv": False,
}

UOIS3D_CONFIG = {
    "padding_percentage": 0.25,
    "use_open_close_morphology": True,
    "open_close_morphology_ksize": 9,
    "use_largest_connected_component": True,
}


# =========================
# Helpers
# =========================
def build_model(
    dsn_ckpt,
    rrn_ckpt,
    dsn_config,
    rrn_config,
    uois3d_config,
):
    config = dict(uois3d_config)
    config["final_close_morphology"] = "TableTop_v5" in str(rrn_ckpt)

    return segmentation.UOISNet3D(
        config,
        str(dsn_ckpt),
        dsn_config,
        str(rrn_ckpt),
        rrn_config,
    )


def load_npy_dataset(folder: Path):
    npy_files = sorted(folder.glob("*.npz"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {folder}")

    rgb_imgs, xyz_imgs, label_imgs, names = [], [], [], []

    for path in npy_files:
        d = np.load(path)
        rgb = d["rgb"]
        xyz = d["xyz"]
        label = d["label"] if "label" in d else None

        if set(d.keys()) == {"rgb", "depth", "K", "seg"}:
            d = cu.convert_cgn_npy_to_uois(d, depth_scale=1.0, out_size=(640, 480))

        rgb_imgs.append(data_augmentation.standardize_image(d["rgb"]))
        xyz_imgs.append(d["xyz"])
        label_imgs.append(d["label"])
        names.append(path.name)

    batch = {
        "rgb": data_augmentation.array_to_tensor(np.stack(rgb_imgs).astype(np.float32)),
        "xyz": data_augmentation.array_to_tensor(np.stack(xyz_imgs).astype(np.float32)),
    }

    return batch, np.stack(xyz_imgs), np.stack(label_imgs), names


def run_inference(model, batch):
    start = time()
    fg_masks, center_offsets, initial_masks, seg_masks = model.run_on_batch(batch)
    elapsed = time() - start

    return {
        "seg_masks": seg_masks.cpu().numpy(),
        "fg_masks": fg_masks.cpu().numpy(),
        "center_offsets": center_offsets.cpu().numpy().transpose(0, 2, 3, 1),
        "initial_masks": initial_masks.cpu().numpy(),
        "time": elapsed,
    }


def save_results(batch, xyz_imgs, label_imgs, seg_masks, names, img_no):
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CGN_DIR.mkdir(parents=True, exist_ok=True)

    rgb_imgs = util_.torch_to_numpy(batch["rgb"].cpu(), is_standardized_image=True)
    H, W = rgb_imgs.shape[1:3]
    fx, fy = 525.0, 525.0
    cx, cy = W / 2.0, H / 2.0

    img_batch = {}

    for i, name in enumerate(names):
        rgb = rgb_imgs[i].astype(np.uint8)
        xyz = xyz_imgs[i].astype(np.float32)

        num_objs = int(max(seg_masks[i].max(), label_imgs[i].max()) + 1)
        pred_vis = util_.get_color_mask(seg_masks[i], nc=num_objs)
        gt_vis = util_.get_color_mask(label_imgs[i], nc=num_objs)

        img_batch[name] = [rgb, xyz[..., 2], pred_vis, gt_vis]

        metrics = evaluation.multilabel_metrics(seg_masks[i], label_imgs[i])
        print(f"\n{name}")
        print(metrics)

        cu.uois_to_contactgraspnet(
            rgb=rgb,
            xyz=xyz,
            seg=seg_masks[i],
            out_npy=str(OUT_CGN_DIR / f"{Path(name).stem}.npy"),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

    cu.save_imgs(img_batch, str(OUT_IMG_DIR), img_no)


# =========================
# Main
# =========================
def main():

    base_dsn = {
        "feature_dim": 64,
        "max_GMS_iters": 10,
        "epsilon": 0.10,
        "sigma": 0.04,
        "num_seeds": 400,
        "subsample_factor": 5,
        "min_pixels_thresh": 500,
        "tau": 15.0,
    }

    base_rrn = {
        "feature_dim": 64,
        "img_H": 224,
        "img_W": 224,
        "use_coordconv": False,
    }

    base_uois = {
        "padding_percentage": 0.25,
        "use_open_close_morphology": True,
        "open_close_morphology_ksize": 9,
        "use_largest_connected_component": True,
    }

    # -------- PARAM GRID --------
    runs = [
        # Baseline
        {"name": "baseline"},
       
    # VERY GOOD RESULTS
    # current best reference
    {"name": "aggressive_merge", "epsilon": 0.10, "sigma": 0.04, "min_px": 200, "lcc": False, "num_seeds": 400},

    # I) high epsilon + moderate sigma (merge across depth noise)
    {"name": "depth_merge", "epsilon": 0.12, "sigma": 0.03, "min_px": 200, "lcc": False, "num_seeds": 400},

    # J) stabilize planar objects (your case: top-down boxes)
    {"name": "planar_bias", "epsilon": 0.11, "sigma": 0.025, "min_px": 300, "lcc": True, "num_seeds": 400},

    # K) aggressive but keep biggest component (avoid splits)
    {"name": "aggressive_lcc", "epsilon": 0.10, "sigma": 0.04, "min_px": 200, "lcc": True, "num_seeds": 400},

    # L) extreme all (boundary test)
    {"name": "max_merge", "epsilon": 0.15, "sigma": 0.06, "min_px": 100, "lcc": False, "num_seeds": 400},
    ]

    # -------- LOAD DATA ONCE --------
    batch, xyz_imgs, label_imgs, names = load_npy_dataset(EXAMPLE_DIR)
    print(f"Loaded {len(names)} image(s)")

    for run in runs:
        print(f"\n===== RUN: {run['name']} =====")
        run_names = []
        for name in names:
            run_names.append(run['name'] + "_" + name)


        # copy configs
        dsn = dict(base_dsn)
        uois = dict(base_uois)

        # apply overrides
        for k, v in run.items():
            if k == "name":
                continue
            if k in dsn:
                dsn[k] = v
            elif k in uois:
                uois[k] = v

        if "sub" in run:
            dsn["subsample_factor"] = run["sub"]

        if "llc" in run:
            uois["use_largest_connected_component"] = run["lcc"]

        # build model
        model = build_model(
            dsn_ckpt=CHECKPOINT_DIR / "DepthSeedingNetwork_3D_TOD_checkpoint.pth",
            rrn_ckpt=CHECKPOINT_DIR / "RRN_OID_checkpoint.pth",
            dsn_config=dsn,
            rrn_config=base_rrn,
            uois3d_config=uois,
        )

        # run inference
        results = run_inference(model, batch)

        print(f"time: {results['time']:.3f}s | FPS: {len(names)/results['time']:.3f}")

        # save per run
        save_results(
            batch,
            xyz_imgs,
            label_imgs,
            results["seg_masks"],
            run_names,
            img_no=[2] # 0-3
        )


if __name__ == "__main__":
    main()