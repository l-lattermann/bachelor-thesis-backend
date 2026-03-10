import os 
from time import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import numpy as np
import io
from plyfile import PlyData
import requests
import sys

import uois.src.data_augmentation as data_augmentation
import uois.src.segmentation as segmentation
import uois.src.evaluation as evaluation
import uois.src.util.utilities as util_
import uois.src.util.custom_utils as cu

from grpc_client.minimal_solution import RcCubeGrpcClient
from point_cloud_utils.ply_converter.ply_to_npy import inspect_pts_cols, load_pts_cols, convert_ply_file_to_uois_npy_file, convert_pts_to_uois_dict, load_pts_cols_from_bytes
from point_cloud_utils.check_npy_structure.inspect_npy import inspect_npy
from point_cloud_utils.disparity_to_npy.dispartiy_to_npy import disparity_to_uois_dict

USE_RC_CLIENT = False
USE_DISPARITY = True

if USE_RC_CLIENT:
    t0 = time()
    test_file = "GRPC fetched image"
    rc_client = RcCubeGrpcClient(rc_cube_ip="172.27.5.9:50051")
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    if USE_DISPARITY:
        out_dir = os.path.join(out_dir, "rc_cube_disparity")
        disp_px, rgb, cam, disp_params = rc_client.get_disparity_img(out_dir)
        uois_dict = disparity_to_uois_dict(disp_px, rgb, cam, disp_params, seg_id=1, background_label=0) 
        out_dir = os.path.join(out_dir, "example_for_uois.npy")
        print("Fetching from RC cube duration: ", time()-t0)
        np.save(out_dir, uois_dict)

        npy = uois_dict #TODO: rename all npy to uois_dict
        out_npy = out_dir #TODO: Also unify this


    else:
        timeout_s = 20
        max_points = 500000
        out_ply = os.path.join(out_dir, "rc_cube_mesh")

        t0 = time()

        ply_bytes, path, cam = rc_client.get_point_cloud_ply(
            output_dir=out_ply,
            timeout=timeout_s,
            max_points=max_points,
        )
        print("CAM = ", cam)
        dt = time() - t0

        pts, cols = load_pts_cols_from_bytes(ply_bytes)
        print("cols is None?", cols is None)
        if cols is not None:
            print("cols dtype/shape/min/max:", cols.dtype, cols.shape, cols.min(), cols.max())
        npy = convert_pts_to_uois_dict(
            pts=pts,
            cols=cols,
            cam=cam,
        )

        inspect_npy(array=npy)
        out_npy_path = "output/pipeline_test_npy/test.npy"
        os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)
        np.save(out_npy_path, npy)


else:
    # load from test file
    test_file = "uois/example_images/OSD_image_0.npy"
    npy = np.load(test_file, allow_pickle=True, encoding='bytes').item()
    print(f"====== {test_file} ======")
    print("TYPE: ", type(npy))
    print("KEYS: ", npy.keys())



# 1) CONFIGS

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

# Expected UOIS keys: rgb, xyz, label (and maybe others)
rgb_img = npy["rgb"]
xyz_img = npy["xyz"]
label_img = npy["label"]

rgb_std = data_augmentation.standardize_image(rgb_img)

batch = {
    "rgb": data_augmentation.array_to_tensor(rgb_std[None, ...]),   # (1,H,W,3) -> tensor
    "xyz": data_augmentation.array_to_tensor(xyz_img[None, ...]),   # (1,H,W,3) -> tensor
}

print(f"Running single image: {test_file}")


# 3) RUN INFERENCE

st_time = time()
fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)
total_time = time() - st_time

print(f"Total time: {total_time:.3f}s")
print(f"FPS: {1.0 / total_time:.3f}")

seg_masks_np = seg_masks.cpu().numpy()[0]          # (H,W)
fg_masks_np = fg_masks.cpu().numpy()[0]
center_offsets_np = center_offsets.cpu().numpy().transpose(0, 2, 3, 1)[0]
initial_masks_np = initial_masks.cpu().numpy()[0]


# 4) EVAL + SAVE OUTPUTS

eval_metrics = evaluation.multilabel_metrics(seg_masks_np, label_img)
print("Metrics:")
print(eval_metrics)

path = Path(test_file)
# Prepare visualization outputs
rgb_uint8 = util_.torch_to_numpy(batch["rgb"].cpu(), is_standardized_image=True)[0].astype(np.uint8)
depth = xyz_img[..., 2].astype(np.float32)

num_objs = int(max(np.unique(seg_masks_np).max(), np.unique(label_img).max()) + 1)
seg_mask_plot = util_.get_color_mask(seg_masks_np, nc=num_objs)
gt_mask_plot = util_.get_color_mask(label_img, nc=num_objs)

img_batch = {
    path.name: [rgb_uint8, depth, seg_mask_plot, gt_mask_plot]
}

out_vis_dir = Path("output/uois_single")
out_vis_dir.mkdir(parents=True, exist_ok=True)
cu.save_imgs(img_batch, str(out_vis_dir))

print(f"Saved visualizations to: {out_vis_dir}")



# 5) OPTIONAL: EXPORT CGN-FORMAT .npy

out_cgn_dir = Path("output/test_data_npy_from_uois")
out_cgn_dir.mkdir(parents=True, exist_ok=True)

if USE_RC_CLIENT:
    H, W = rgb_uint8.shape[:2]
    scale_x = W / cam["width"]
    scale_y = H / cam["height"]

    fx = cam["fx"] * scale_x
    fy = cam["fy"] * scale_y
    cx = cam["cx"] * scale_x
    cy = cam["cy"] * scale_y
else:
    H, W = rgb_uint8.shape[:2]
    fx = 525.0
    fy = 525.0
    cx = W / 2.0
    cy = H / 2.0

    out_npy = out_cgn_dir    #TODO: Also unify this

out = cu.uois_to_contactgraspnet(
    rgb=rgb_uint8,
    xyz=xyz_img.astype(np.float32),
    seg=seg_masks_np,
    out_npy=str(out_npy),
    fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
)

print(f"Exported Contact-GraspNet npy to: {out_npy}")


def send_cgn_dict_to_server(out_dict, url):
    buf = io.BytesIO()
    np.save(buf, out_dict, allow_pickle=True)
    buf.seek(0)

    r = requests.post(
        url,
        data=buf.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
        timeout=300
    )
    r.raise_for_status()

    return np.load(io.BytesIO(r.content), allow_pickle=True)

# Example
result = send_cgn_dict_to_server(
    out,
    #"http://localhost:8000/inference"
    #"http://localhost:8000/inference?z_min=0&z_max=3&filter_grasps=true&segmap_id=4"
    "http://localhost:8000/inference?z_min=0.0&z_max=5.1"
)

print(result.files)
g = result["pred_grasps_cam"].item()
s = result["scores"].item()
c = result["contact_pts"].item()

print(type(g))                 # likely dict
print("object ids:", list(g.keys()))

total_grasps = sum(len(v) for v in g.values())
print("num objects:", len(g))
print("total grasps:", total_grasps)

np.savez(
    "output/gcn/predictions_test.npz",
    pred_grasps_cam=result["pred_grasps_cam"],
    scores=result["scores"],
    contact_pts=result["contact_pts"],
)
