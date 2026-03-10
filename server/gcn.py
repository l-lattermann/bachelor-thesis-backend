# main.py
import os
from time import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

import io
import sys
import argparse
import numpy as np
import time
import glob
import cv2


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from contact_graspnet.contact_graspnet import config_utils
from contact_graspnet.contact_graspnet.data import regularize_pc_point_count, depth2pc, load_available_input_data
from contact_graspnet.contact_graspnet.contact_grasp_estimator import GraspEstimator
from contact_graspnet.contact_graspnet.visualization_utils import visualize_grasps, show_image
#from contact_graspnet.contact_graspnet.visualization_utils import visualize_grasps, show_image

app = FastAPI(title="Contact Grasp Net Service")

# Keep a single client instance (reused across requests)
grasp_estimator: Optional[GraspEstimator] = None



@app.on_event("startup")
def startup():

    global grasp_estimator, sess

    print("Starting Contact-GraspNet...")

    ckpt_dir = "contact_graspnet/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001"

    global_config = config_utils.load_config(
        ckpt_dir,
        batch_size=1,
        arg_configs=[]
    )

    # fix wrong model path
    if global_config["MODEL"]["model"] == "contact_graspnet":
        global_config["MODEL"]["model"] = "contact_graspnet.contact_graspnet.contact_graspnet"

    # ---- Build model ----
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    saver = tf.train.Saver(save_relative_paths=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    grasp_estimator.load_weights(sess, saver, ckpt_dir, mode='test')

    print("Model loaded.")

    # ---- Minimal warmup (single dummy forward pass) ----
    dummy_pc = np.random.uniform(
        low=-0.1, high=0.1,
        size=(2048, 3)
    ).astype(np.float32)

    print("Running warmup forward pass...")
    _ = grasp_estimator.predict_scene_grasps(
        sess,
        dummy_pc,
        pc_segments={},
        local_regions=False,
        filter_grasps=False,
        forward_passes=1
    )

    print("Warmup complete.")   

@app.on_event("shutdown")
def shutdown():
    print("[SERVICE] shutdown")


@app.get("/health")
def health():
    return {"status": "ok"}




def infer_from_cgn_npy_bytes(
    npy_bytes: bytes,
    ckpt_dir: str = "checkpoints/scene_test_2048_bs3_hor_sigma_001",
    np_path: str = "test_data/7.npy",
    png_path: str = "",
    K = None,
    z_range: list = [0.0, 4.0],
    local_regions: bool = False,
    filter_grasps: bool = False,
    skip_border_objects: bool = False,
    forward_passes: int = 1,
    segmap_id: int = 0,
    arg_configs: list = None,
):
    global grasp_estimator, sess

    # ---- Load dict from bytes ----
    print("\n[DEBUG] Loading npy bytes...")
    data = np.load(io.BytesIO(npy_bytes), allow_pickle=True).item()

    print("[DEBUG] Keys in input dict:", list(data.keys()))

    depth = data["depth"]
    cam_K = data["K"]
    segmap = data.get("seg", None)
    rgb = data.get("rgb", None)

    pc_segments = {}
    pc_full, pc_colors = None, None
        
    if segmap is None and (local_regions or filter_grasps):
        raise ValueError('Need segmentation map to extract local regions or filter grasps')

    if pc_full is None:
        print('Converting depth to point cloud(s)...')
        pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
            depth, cam_K,
            segmap=segmap, rgb=rgb,
            segmap_id=segmap_id,
            skip_border_objects=skip_border_objects,
            z_range=z_range
        )

    print('Generating Grasps...')
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
                                                                                        local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  

    # Save results
    np.savez(
        'results/predictions_.npz',
        pred_grasps_cam=pred_grasps_cam, 
        scores=scores, 
        contact_pts=contact_pts
        )


    # Visualize results          
    show_image(rgb, segmap)
    visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
    
    print("[DEBUG] Inference complete.\n")


    return {
        "grasps": pred_grasps_cam,
        "scores": scores,
        "contacts": contact_pts,
    }

@app.post("/inference")
async def infer(
        request: Request,
        z_min: float = 0.0,
        z_max: float = 4.0,
        filter_grasps: bool = False,
        segmap_id: int = 0,
    ):
    npy_bytes = await request.body()

    print("segmap_id = ", segmap_id)
    print("filter_graps = ", filter_grasps)

    z_range = [z_min, z_max]
    result = infer_from_cgn_npy_bytes(
        npy_bytes=npy_bytes, 
        z_range=z_range, 
        segmap_id=segmap_id, 
        filter_grasps=filter_grasps
        )

    buf = io.BytesIO()
    np.savez(
        buf,
        pred_grasps_cam=result["grasps"],
        scores=result["scores"],
        contact_pts=result["contacts"],
    )
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream"
    )