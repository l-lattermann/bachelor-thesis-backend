import io
import os
from pathlib import Path
from time import time

import cv2
import numpy as np
import requests

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import uois.src.data_augmentation as data_augmentation
import uois.src.evaluation as evaluation
import uois.src.segmentation as segmentation
import uois.src.util.custom_utils as cu
import uois.src.util.utilities as util_
from grpc_client.minimal_solution import RcCubeGrpcClient
from point_cloud_utils.check_npy_structure.inspect_npy import inspect_npy
from point_cloud_utils.disparity_to_npy.dispartiy_to_npy import disparity_to_uois_dict
from point_cloud_utils.ply_converter.ply_to_npy import (
    convert_pts_to_uois_dict,
    load_pts_cols_from_bytes,
)

USE_RC_CLIENT = True
USE_DISPARITY = True
RC_CUBE_IP = "172.27.48.156:50051"
CGN_URL = "http://localhost:8000/inference?z_min=0.0&z_max=5.1"

OUTPUT_DIR = Path("output")
RC_DISPARITY_DIR = OUTPUT_DIR / "rc_cube_disparity"
PIPELINE_TEST_DIR = OUTPUT_DIR / "pipeline_test_npy"
UOIS_VIS_DIR = OUTPUT_DIR / "uois_single"
CGN_EXPORT_DIR = OUTPUT_DIR / "test_data_npy_from_uois"
CGN_RESULT_DIR = OUTPUT_DIR / "gcn"


def align_rgb_and_cam_to_disparity(rgb: np.ndarray, disp_px: np.ndarray, cam: dict):
    print("before resize:")
    print("disp shape:", disp_px.shape)
    print("rgb shape:", rgb.shape)
    print("cam width/height:", cam["width"], cam["height"])

    if rgb.shape[:2] != disp_px.shape[:2]:
        orig_w = cam["width"]
        orig_h = cam["height"]

        new_h, new_w = disp_px.shape[:2]
        sx = new_w / orig_w
        sy = new_h / orig_h

        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        cam = cam.copy()
        cam["fx"] *= sx
        cam["fy"] *= sy
        cam["cx"] *= sx
        cam["cy"] *= sy
        cam["width"] = new_w
        cam["height"] = new_h

    print("after resize:")
    print("disp shape:", disp_px.shape)
    print("rgb shape:", rgb.shape)
    print("cam width/height:", cam["width"], cam["height"])

    assert rgb.shape[:2] == disp_px.shape[:2], "RGB and disparity must match in resolution"
    assert cam["width"] == disp_px.shape[1]
    assert cam["height"] == disp_px.shape[0]

    return rgb, cam


def load_input_data():
    if USE_RC_CLIENT:
        t0 = time()
        rc_client = RcCubeGrpcClient(rc_cube_ip=RC_CUBE_IP)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if USE_DISPARITY:
            RC_DISPARITY_DIR.mkdir(parents=True, exist_ok=True)

            disp_px, rgb, cam, disp_params = rc_client.get_disparity_img(str(RC_DISPARITY_DIR))
            rgb, cam = align_rgb_and_cam_to_disparity(rgb, disp_px, cam)

            uois_dict = disparity_to_uois_dict(
                disp_px,
                rgb,
                cam,
                disp_params,
                seg_id=1,
                background_label=0,
            )

            out_npy = RC_DISPARITY_DIR / "example_for_uois.npy"
            np.save(out_npy, uois_dict)

            print("Fetching from RC cube duration:", time() - t0)
            return {
                "test_file": "GRPC fetched image",
                "npy": uois_dict,
                "out_npy": out_npy,
                "cam": cam,
            }

        timeout_s = 20
        max_points = 500000
        out_ply = OUTPUT_DIR / "rc_cube_mesh"

        t0 = time()
        ply_bytes, _, cam = rc_client.get_point_cloud_ply(
            output_dir=str(out_ply),
            timeout=timeout_s,
            max_points=max_points,
        )
        print("Fetching point cloud duration:", time() - t0)
        print("CAM =", cam)

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

        out_npy = PIPELINE_TEST_DIR / "test.npy"
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, npy)

        return {
            "test_file": "GRPC fetched point cloud",
            "npy": npy,
            "out_npy": out_npy,
            "cam": cam,
        }

    test_file = "uois/example_images/OSD_image_0.npy"
    npy = np.load(test_file, allow_pickle=True, encoding="bytes").item()

    print(f"====== {test_file} ======")
    print("TYPE:", type(npy))
    print("KEYS:", npy.keys())

    return {
        "test_file": test_file,
        "npy": npy,
        "out_npy": None,
        "cam": None,
    }


def build_uois_model():
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

    return segmentation.UOISNet3D(
        uois3d_config,
        str(dsn_filename),
        dsn_config,
        str(rrn_filename),
        rrn_config,
    )


def run_uois_inference(uois_net_3d, npy: dict, test_file: str):
    rgb_img = npy["rgb"]
    xyz_img = npy["xyz"]
    label_img = npy["label"]

    rgb_std = data_augmentation.standardize_image(rgb_img)
    batch = {
        "rgb": data_augmentation.array_to_tensor(rgb_std[None, ...]),
        "xyz": data_augmentation.array_to_tensor(xyz_img[None, ...]),
    }

    print(f"Running single image: {test_file}")

    st_time = time()
    fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)
    total_time = time() - st_time

    print(f"Total time: {total_time:.3f}s")
    print(f"FPS: {1.0 / total_time:.3f}")

    seg_masks_np = seg_masks.cpu().numpy()[0]
    fg_masks_np = fg_masks.cpu().numpy()[0]
    center_offsets_np = center_offsets.cpu().numpy().transpose(0, 2, 3, 1)[0]
    initial_masks_np = initial_masks.cpu().numpy()[0]

    return {
        "batch": batch,
        "rgb_img": rgb_img,
        "xyz_img": xyz_img,
        "label_img": label_img,
        "seg_masks_np": seg_masks_np,
        "fg_masks_np": fg_masks_np,
        "center_offsets_np": center_offsets_np,
        "initial_masks_np": initial_masks_np,
    }


def save_uois_outputs(test_file: str, uois_result: dict):
    label_img = uois_result["label_img"]
    seg_masks_np = uois_result["seg_masks_np"]
    batch = uois_result["batch"]
    xyz_img = uois_result["xyz_img"]

    eval_metrics = evaluation.multilabel_metrics(seg_masks_np, label_img)
    print("Metrics:")
    print(eval_metrics)

    path = Path(test_file)
    rgb_uint8 = util_.torch_to_numpy(batch["rgb"].cpu(), is_standardized_image=True)[0].astype(np.uint8)
    depth = xyz_img[..., 2].astype(np.float32)

    num_objs = int(max(np.unique(seg_masks_np).max(), np.unique(label_img).max()) + 1)
    seg_mask_plot = util_.get_color_mask(seg_masks_np, nc=num_objs)
    gt_mask_plot = util_.get_color_mask(label_img, nc=num_objs)

    img_batch = {
        path.name: [rgb_uint8, depth, seg_mask_plot, gt_mask_plot]
    }

    UOIS_VIS_DIR.mkdir(parents=True, exist_ok=True)
    cu.save_imgs(img_batch, str(UOIS_VIS_DIR), [0, 2])
    print(f"Saved visualizations to: {UOIS_VIS_DIR}")

    return rgb_uint8


def export_to_contact_graspnet(rgb_uint8: np.ndarray, xyz_img: np.ndarray, seg_masks_np: np.ndarray, cam: dict, out_npy: Path):
    CGN_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    if cam is not None:
        fx = float(cam["fx"])
        fy = float(cam["fy"])
        cx = float(cam["cx"])
        cy = float(cam["cy"])
    else:
        h, w = rgb_uint8.shape[:2]
        fx = 525.0
        fy = 525.0
        cx = w / 2.0
        cy = h / 2.0

    out = cu.uois_to_contactgraspnet(
        rgb=rgb_uint8,
        xyz=xyz_img.astype(np.float32),
        seg=seg_masks_np,
        out_npy=str(out_npy),
        fx=fx,
        fy=fy,
        cx=cy if False else cx,
        cy=cy,
    )

    print(f"Exported Contact-GraspNet npy to: {out_npy}")
    return out


def send_cgn_dict_to_server(out_dict: dict, url: str):
    buf = io.BytesIO()
    np.save(buf, out_dict, allow_pickle=True)
    buf.seek(0)

    response = requests.post(
        url,
        data=buf.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
        timeout=300,
    )
    response.raise_for_status()

    return np.load(io.BytesIO(response.content), allow_pickle=True)


def save_cgn_results(result):
    print(result.files)

    grasps = result["pred_grasps_cam"].item()
    scores = result["scores"].item()
    contacts = result["contact_pts"].item()

    print(type(grasps))
    print("object ids:", list(grasps.keys()))

    total_grasps = sum(len(v) for v in grasps.values())
    print("num objects:", len(grasps))
    print("total grasps:", total_grasps)

    CGN_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        CGN_RESULT_DIR / "predictions_test.npz",
        pred_grasps_cam=result["pred_grasps_cam"],
        scores=result["scores"],
        contact_pts=result["contact_pts"],
    )


def main():
    input_data = load_input_data()
    test_file = input_data["test_file"]
    npy = input_data["npy"]
    cam = input_data["cam"]

    uois_net_3d = build_uois_model()
    uois_result = run_uois_inference(uois_net_3d, npy, test_file)
    rgb_uint8 = save_uois_outputs(test_file, uois_result)

    out_npy = CGN_EXPORT_DIR / "example_for_contact_graspnet.npy"
    out = export_to_contact_graspnet(
        rgb_uint8=rgb_uint8,
        xyz_img=uois_result["xyz_img"],
        seg_masks_np=uois_result["seg_masks_np"],
        cam=cam,
        out_npy=out_npy,
    )

    result = send_cgn_dict_to_server(out, CGN_URL)
    save_cgn_results(result)


if __name__ == "__main__":
    main()