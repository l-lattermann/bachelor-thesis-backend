import yaml
import numpy as np
import tensorflow as tf

from contact_graspnet.contact_grasp_estimator import GraspEstimator
from contact_graspnet import config_utils

_model = None
_sess = None
_cfg = None


def load_config():
    global _cfg
    if _cfg is None:
        with open("/app/config.yaml", "r") as f:
            _cfg = yaml.safe_load(f)["contact_graspnet"]
    return _cfg


def load_model():
    global _model, _sess

    if _model is None or _sess is None:
        cfg = load_config()
        ckpt_dir = cfg["checkpoint_dir"]

        model_cfg = config_utils.load_config(ckpt_dir)
        estimator = GraspEstimator(model_cfg)
        estimator.build_network()

        sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver(save_relative_paths=True)
        estimator.load_weights(sess, saver, ckpt_dir, mode="test")

        _model = estimator
        _sess = sess

    return _model, _sess


def run_contact_graspnet(npy_path: str):
    cfg = load_config()
    model, sess = load_model()

    pred_cfg = cfg["prediction"]
    sel_cfg = cfg["selection"]

    data = np.load(npy_path, allow_pickle=True)

    xyz = data["xyz"]
    seg = data["seg"] if "seg" in data.files else None

    # z filtering
    z = xyz[..., 2]
    valid = np.isfinite(z) & (z >= pred_cfg["z_min"]) & (z <= pred_cfg["z_max"])
    xyz = xyz.copy()
    xyz[~valid] = 0.0

    # full point cloud
    flat_xyz = xyz.reshape(-1, 3)
    valid_xyz = np.isfinite(flat_xyz).all(axis=1) & (np.linalg.norm(flat_xyz, axis=1) > 0)
    pc_full = flat_xyz[valid_xyz]

    # segmented object point clouds
    pc_segments = {}
    if seg is not None:
        flat_seg = seg.reshape(-1)

        segmap_id = pred_cfg.get("segmap_id")
        if segmap_id is not None:
            flat_seg = np.where(flat_seg == segmap_id, flat_seg, 0)

        for obj_id in np.unique(flat_seg):
            if obj_id == 0:
                continue
            mask = (flat_seg == obj_id) & valid_xyz
            pts = flat_xyz[mask]
            if len(pts) > 0:
                pc_segments[int(obj_id)] = pts

    pred_grasps_cam, scores, contact_pts, gripper_openings = model.predict_scene_grasps(
        sess,
        pc_full,
        pc_segments=pc_segments,
        local_regions=pred_cfg["local_regions"],
        filter_grasps=pred_cfg["filter_grasps"],
        forward_passes=pred_cfg["forward_passes"],
    )

    # optional post-selection
    selected_grasps = {}
    selected_scores = {}
    selected_contacts = {}
    selected_openings = {}

    for k in pred_grasps_cam.keys():
        if len(contact_pts[k]) == 0 or len(scores[k]) == 0:
            selected_grasps[k] = np.array([])
            selected_scores[k] = np.array([])
            selected_contacts[k] = np.array([])
            selected_openings[k] = np.array([])
            continue

        idx = model.select_grasps(
            contact_pts[k],
            scores[k],
            max_farthest_points=sel_cfg["max_farthest_points"],
            num_grasps=sel_cfg["num_grasps"],
            first_thres=sel_cfg["first_thres"],
            second_thres=sel_cfg["second_thres"],
            with_replacement=sel_cfg["with_replacement"],
        )

        selected_grasps[k] = pred_grasps_cam[k][idx]
        selected_scores[k] = scores[k][idx]
        selected_contacts[k] = contact_pts[k][idx]
        selected_openings[k] = gripper_openings[k][idx] if len(gripper_openings[k]) > 0 else np.array([])

    return {
        "pred_grasps_cam": selected_grasps,
        "scores": selected_scores,
        "contact_pts": selected_contacts,
        "gripper_openings": selected_openings,
    }