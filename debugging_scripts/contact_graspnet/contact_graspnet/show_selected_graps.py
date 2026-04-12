import os
import sys
import argparse
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

from visualization_utils import visualize_grasps, show_image


def _unpack(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
        if x.shape == ():
            return x.item()
        if len(x) == 1:
            return x[0]
    return x


def load_saved_cgn_output(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)

    pred_grasps_cam = _unpack(data["pred_grasps_cam"])
    scores = _unpack(data["scores"])
    gripper_openings = _unpack(data["gripper_openings"])
    pc_full = _unpack(data["pc_full"])
    segmap = _unpack(data["segmap"])
    rgb = _unpack(data["rgb"])
    pc_colors = _unpack(data["pc_colors"])

    return pred_grasps_cam, scores, gripper_openings, pc_full, segmap, rgb, pc_colors


def visualize_saved_output(npz_path: str):
    print("Loading", npz_path)

    pred_grasps_cam, scores, gripper_openings, pc_full, segmap, rgb, pc_colors = load_saved_cgn_output(npz_path)

    show_image(rgb, segmap)
    print("Visualizing...takes time")
    visualize_grasps(
        pc_full,
        pred_grasps_cam,
        scores,
        plot_opencv_cam=True,
        pc_colors=pc_colors,
        gripper_openings=gripper_openings,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz_path",
        required=True,
        help="Path to saved selected CGN output npz",
    )
    FLAGS = parser.parse_args()

    print("pid:", os.getpid())
    visualize_saved_output(FLAGS.npz_path)