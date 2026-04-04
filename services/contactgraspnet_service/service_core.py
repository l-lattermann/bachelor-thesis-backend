from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf
import yaml
from typing import Optional

from contact_graspnet.contact_graspnet import config_utils
from contact_graspnet.contact_graspnet.contact_grasp_estimator import GraspEstimator

tf.disable_eager_execution()

_MODEL = None
_SESS = None

CONFIG_PATH = Path("/app/config.yaml")
with CONFIG_PATH.open("r") as f:
    CFG = yaml.safe_load(f)

CGN_CFG = CFG["contact_graspnet"]


def _override_checkpoint_config(model_cfg: dict, cgn_cfg: dict) -> dict:
    """
    Start from the checkpoint config and override only the runtime-relevant
    parameters from /app/config.yaml.
    """
    selection_cfg = cgn_cfg.get("selection", {})
    prediction_cfg = cgn_cfg.get("prediction", {})

    model_cfg["MODEL"]["model"] = "contact_graspnet.contact_graspnet.contact_graspnet"

    gripper_width = selection_cfg.get("gripper_width")
    if gripper_width is not None:
        model_cfg.setdefault("DATA", {})
        model_cfg["DATA"]["gripper_width"] = float(gripper_width)

    model_cfg.setdefault("TEST", {})

    if "max_farthest_points" in selection_cfg:
        model_cfg["TEST"]["max_farthest_points"] = int(selection_cfg["max_farthest_points"])

    if "first_thres" in selection_cfg:
        model_cfg["TEST"]["first_thres"] = float(selection_cfg["first_thres"])

    if "second_thres" in selection_cfg:
        model_cfg["TEST"]["second_thres"] = float(selection_cfg["second_thres"])

    if "with_replacement" in selection_cfg:
        model_cfg["TEST"]["with_replacement"] = bool(selection_cfg["with_replacement"])

    if "forward_passes" in prediction_cfg:
        model_cfg["TEST"]["num_samples"] = int(prediction_cfg["forward_passes"])

    return model_cfg


def load_model():
    global _MODEL, _SESS

    if _MODEL is not None and _SESS is not None:
        return _MODEL, _SESS

    ckpt_dir = CGN_CFG["checkpoint_dir"]

    model_cfg = config_utils.load_config(
        ckpt_dir,
        batch_size=1,
        arg_configs=[],
    )
    model_cfg = _override_checkpoint_config(model_cfg, CGN_CFG)

    estimator = GraspEstimator(model_cfg)
    estimator.build_network()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    sess = tf.Session(config=tf_config)

    saver = tf.train.Saver(save_relative_paths=True)
    estimator.load_weights(sess, saver, ckpt_dir, mode="test")

    dummy_pc = np.random.uniform(
        low=-0.1,
        high=0.1,
        size=(2048, 3),
    ).astype(np.float32)

    estimator.predict_scene_grasps(
        sess,
        dummy_pc,
        pc_segments={},
        local_regions=False,
        filter_grasps=False,
        forward_passes=1,
    )

    _MODEL = estimator
    _SESS = sess
    return _MODEL, _SESS


def _load_npz_inputs(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)

    rgb = data["rgb"] if "rgb" in data.files else None
    depth = data["depth"] if "depth" in data.files else None
    K = data["K"] if "K" in data.files else None
    seg = data["seg"] if "seg" in data.files else None

    if depth is None or K is None:
        raise ValueError(
            f"Unsupported NPZ format. Expected at least 'depth' and 'K'. "
            f"Available keys: {data.files}"
        )

    return rgb, depth, K, seg




def run_contact_graspnet(npz_path: str, object_id: Optional[int] = None):
    model, sess = load_model()

    pred_cfg = CGN_CFG["prediction"]

    rgb, depth, K, seg = _load_npz_inputs(npz_path)

    pc_full, pc_segments, pc_colors = model.extract_point_clouds(
        depth=depth,
        K=K,
        segmap=seg,
        rgb=rgb,
        z_range=[pred_cfg["z_min"], pred_cfg["z_max"]],
        segmap_id=object_id if object_id is not None else None,
        skip_border_objects=pred_cfg.get("skip_border_objects", False),
        margin_px=pred_cfg.get("margin_px", 5),
    )

    # if no object_id → ignore segments completely
    if object_id is None:
        pc_segments = []

    use_local_regions = pred_cfg.get("local_regions", False) and len(pc_segments) > 0
    use_filter_grasps = pred_cfg.get("filter_grasps", False) and len(pc_segments) > 0

    pred_grasps_cam, scores, contact_pts, gripper_openings = model.predict_scene_grasps(
        sess,
        pc_full,
        pc_segments=pc_segments,
        local_regions=use_local_regions,
        filter_grasps=use_filter_grasps,
        forward_passes=pred_cfg.get("forward_passes", 1),
    )

    return {
        "pred_grasps_cam": pred_grasps_cam,
        "scores": scores,
        "contact_pts": contact_pts,
        "gripper_openings": gripper_openings,
        "pc_full": pc_full,
        "segmap": seg,
        "rgb": rgb,
        "K": K,
        "pc_colors": pc_colors,
    }