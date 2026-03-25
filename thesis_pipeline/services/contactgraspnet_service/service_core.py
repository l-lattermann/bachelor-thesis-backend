from pathlib import Path
import os
import yaml
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

from contact_graspnet.contact_graspnet.contact_grasp_estimator import GraspEstimator
from contact_graspnet.contact_graspnet import config_utils




_model = None
_sess = None

CONFIG_PATH = Path("/app/config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

CGN_CFG = cfg["contact_graspnet"]


def load_model():
    global _model, _sess

    if _model is None or _sess is None:
        ckpt_dir = CGN_CFG["checkpoint_dir"]

        model_cfg = config_utils.load_config(
            ckpt_dir,
            batch_size=1,
            arg_configs=[]
        )

        if model_cfg["MODEL"]["model"] == "contact_graspnet":
            model_cfg["MODEL"]["model"] = "contact_graspnet.contact_graspnet.contact_graspnet"

        estimator = GraspEstimator(model_cfg)
        estimator.build_network()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        sess = tf.Session(config=config)

        saver = tf.train.Saver(save_relative_paths=True)
        estimator.load_weights(sess, saver, ckpt_dir, mode="test")

        dummy_pc = np.random.uniform(
            low=-0.1, high=0.1,
            size=(2048, 3)
        ).astype(np.float32)

        _ = estimator.predict_scene_grasps(
            sess,
            dummy_pc,
            pc_segments={},
            local_regions=False,
            filter_grasps=False,
            forward_passes=1
        )

        _model = estimator
        _sess = sess

    return _model, _sess


def run_contact_graspnet(npz_path: str):
    model, sess = load_model()

    pred_cfg = CGN_CFG["prediction"]
    sel_cfg = CGN_CFG["selection"]

    data = np.load(npz_path, allow_pickle=True)

    rgb = data["rgb"] if "rgb" in data.files else None
    depth = data["depth"] if "depth" in data.files else None
    K = data["K"] if "K" in data.files else None
    seg = data["seg"] if "seg" in data.files else None

    if depth is None or K is None:
        raise ValueError(f"Unsupported NPZ format. Expected at least 'depth' and 'K'. Available keys: {data.files}")

    segmap_id = pred_cfg.get("segmap_id", 0)

    pc_full, pc_segments, pc_colors = model.extract_point_clouds(
        depth=depth,
        K=K,
        segmap=seg,
        rgb=rgb,
        z_range=[pred_cfg["z_min"], pred_cfg["z_max"]],
        segmap_id=segmap_id,
        skip_border_objects=pred_cfg.get("skip_border_objects", False),
        margin_px=pred_cfg.get("margin_px", 5),
    )

    use_local_regions = pred_cfg["local_regions"] and len(pc_segments) > 0
    use_filter_grasps = pred_cfg["filter_grasps"] and len(pc_segments) > 0
    print(f"use_local_regions: {use_local_regions}")
    print(f"use_filter_grasps: {use_filter_grasps}")


    pred_grasps_cam, scores, contact_pts, gripper_openings = model.predict_scene_grasps(
        sess,
        pc_full,
        pc_segments=pc_segments,
        local_regions=use_local_regions,
        filter_grasps=use_filter_grasps,
        forward_passes=pred_cfg["forward_passes"],
    )

    # post-selection
    selected_grasps = {}
    selected_scores = {}
    selected_contacts = {}
    selected_openings = {}

    for k in pred_grasps_cam.keys():
        pts_k = np.atleast_2d(contact_pts[k])
        scores_k = np.atleast_1d(scores[k])
        grasps_k = np.asarray(pred_grasps_cam[k])
        openings_k = np.atleast_1d(gripper_openings[k]) if len(np.asarray(gripper_openings[k]).shape) > 0 else np.array([])

        if len(pts_k) == 0 or len(scores_k) == 0:
            selected_grasps[k] = np.array([])
            selected_scores[k] = np.array([])
            selected_contacts[k] = np.array([])
            selected_openings[k] = np.array([])
            continue

        idx = model.select_grasps(
            pts_k,
            scores_k,
            max_farthest_points=sel_cfg["max_farthest_points"],
            num_grasps=sel_cfg["num_grasps"],
            first_thres=sel_cfg["first_thres"],
            second_thres=sel_cfg["second_thres"],
            with_replacement=sel_cfg["with_replacement"],
        )

        selected_grasps[k] = grasps_k[idx]
        selected_scores[k] = scores_k[idx]
        selected_contacts[k] = pts_k[idx]
        selected_openings[k] = openings_k[idx] if len(openings_k) > 0 else np.array([])

    return {
        "pred_grasps_cam": selected_grasps,
        "scores": selected_scores,
        "contact_pts": selected_contacts,
        "gripper_openings": selected_openings,
        "pc_full": pc_full,
        "segmap": seg,
        "rgb": rgb,
        "pc_colors": pc_colors,
    }


def save_segmap_image(rgb, segmap, out_path):
    plt.figure(figsize=(8, 6))

    if rgb is not None:
        plt.imshow(rgb)

    if segmap is not None:
        cmap = plt.get_cmap("rainbow")
        cmap.set_under(alpha=0.0)
        plt.imshow(segmap, cmap=cmap, alpha=0.5, vmin=0.0001)

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_coordinates(t, r, tube_radius=0.005):
    mlab.plot3d([t[0], t[0] + 0.2 * r[0, 0]], [t[1], t[1] + 0.2 * r[1, 0]], [t[2], t[2] + 0.2 * r[2, 0]], color=(1, 0, 0), tube_radius=tube_radius, opacity=1)
    mlab.plot3d([t[0], t[0] + 0.2 * r[0, 1]], [t[1], t[1] + 0.2 * r[1, 1]], [t[2], t[2] + 0.2 * r[2, 1]], color=(0, 1, 0), tube_radius=tube_radius, opacity=1)
    mlab.plot3d([t[0], t[0] + 0.2 * r[0, 2]], [t[1], t[1] + 0.2 * r[1, 2]], [t[2], t[2] + 0.2 * r[2, 2]], color=(0, 0, 1), tube_radius=tube_radius, opacity=1)


def draw_pc_with_colors(pc, pc_colors=None, single_color=(0.3, 0.3, 0.3), mode='2dsquare', scale_factor=0.0018):
    if pc_colors is None:
        mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=single_color, scale_factor=scale_factor, mode=mode)
    else:
        def create_8bit_rgb_lut():
            xl = np.mgrid[0:256, 0:256, 0:256]
            lut = np.vstack((
                xl[0].reshape(1, 256**3),
                xl[1].reshape(1, 256**3),
                xl[2].reshape(1, 256**3),
                255 * np.ones((1, 256**3))
            )).T
            return lut.astype("int32")

        pc_colors = np.asarray(pc_colors).astype(np.int32)
        scalars = pc_colors[:, 0] * 256**2 + pc_colors[:, 1] * 256 + pc_colors[:, 2]
        rgb_lut = create_8bit_rgb_lut()
        points_mlab = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], scalars, mode=mode, scale_factor=scale_factor)
        points_mlab.glyph.scale_mode = "scale_by_vector"
        points_mlab.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, rgb_lut.shape[0])
        points_mlab.module_manager.scalar_lut_manager.lut.number_of_colors = rgb_lut.shape[0]
        points_mlab.module_manager.scalar_lut_manager.lut.table = rgb_lut


def draw_grasps(grasps, gripper_openings, color=(0, 1., 0), colors=None, tube_radius=0.0008):
    all_pts = []
    connections = []
    index = 0
    N = 7

    for i, (g, g_opening) in enumerate(zip(grasps, gripper_openings)):
        half_open = float(g_opening) / 2.0

        grasp_line_plot = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.03],
            [-half_open, 0.0, 0.03],
            [-half_open, 0.0, 0.06],
            [half_open, 0.0, 0.03],
            [half_open, 0.0, 0.06],
            [0.0, 0.02, 0.015],
        ], dtype=np.float32)

        pts = np.matmul(grasp_line_plot, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)

        all_pts.append(pts)
        connections.append(np.array([
            [index + 0, index + 1],
            [index + 1, index + 2],
            [index + 2, index + 3],
            [index + 1, index + 4],
            [index + 4, index + 5],
            [index + 0, index + 6],
        ]))
        index += N

    if len(all_pts) == 0:
        return

    all_pts = np.vstack(all_pts)
    connections = np.vstack(connections)
    src = mlab.pipeline.scalar_scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2])
    src.mlab_source.dataset.lines = connections
    src.update()
    lines = mlab.pipeline.tube(src, tube_radius=tube_radius, tube_sides=12)
    mlab.pipeline.surface(lines, color=color, opacity=1.0)


def visualize_grasps_offscreen(
    full_pc,
    pred_grasps_cam,
    scores,
    save_path,
    plot_opencv_cam=False,
    pc_colors=None,
    gripper_openings=None,
    gripper_width=0.08,
):
    mlab.options.offscreen = True

    cm1 = plt.get_cmap("rainbow")
    cm2 = plt.get_cmap("gist_rainbow")

    fig = mlab.figure(size=(1400, 1000), bgcolor=(1, 1, 1))
    mlab.view(azimuth=180, elevation=180, distance=0.2)

    draw_pc_with_colors(full_pc, pc_colors)

    colors = [cm1(1.0 * i / max(len(pred_grasps_cam), 1))[:3] for i in range(len(pred_grasps_cam))]
    colors2 = {k: cm2(0.5 * np.max(scores[k]))[:3] for k in pred_grasps_cam if np.any(pred_grasps_cam[k])}

    if plot_opencv_cam:
        plot_coordinates(np.zeros(3,), np.eye(3, 3))

    for i, k in enumerate(pred_grasps_cam):
        grasps_k = np.asarray(pred_grasps_cam[k])
        scores_k = np.atleast_1d(scores[k])

        if grasps_k.size == 0 or scores_k.size == 0:
            continue

        if grasps_k.ndim == 2 and grasps_k.shape == (4, 4):
            grasps_k = grasps_k[np.newaxis, ...]

        if gripper_openings is None or k not in gripper_openings:
            gripper_openings_k = np.ones(len(grasps_k)) * gripper_width
        else:
            gripper_openings_k = np.atleast_1d(gripper_openings[k])
            if gripper_openings_k.ndim == 0:
                gripper_openings_k = gripper_openings_k[None]
            if len(gripper_openings_k) != len(grasps_k):
                gripper_openings_k = np.ones(len(grasps_k)) * gripper_width

        if len(pred_grasps_cam) > 1:
            draw_grasps(
                grasps_k,
                gripper_openings=gripper_openings_k,
                color=colors[i],
                tube_radius=0.0008,
            )

            best_idx = int(np.argmax(scores_k))
            draw_grasps(
                [grasps_k[best_idx]],
                gripper_openings=[gripper_openings_k[best_idx]],
                color=colors2[k],
                tube_radius=0.0025,
            )
        else:
            draw_grasps(
                grasps_k,
                gripper_openings=gripper_openings_k,
                color=(1, 0, 0),
                tube_radius=0.0008,
            )

    mlab.savefig(str(save_path), size=(1400, 1000))
    mlab.close(fig)


def save_cgn_output_image(pc_full, pred_grasps_cam, scores, segmap=None, rgb=None, pc_colors=None, gripper_openings=None):
    base_path = cfg["paths"]["pipeline_file_share"]
    out_dir = Path(base_path) / "cgn_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)

    grasp_path = out_dir / f"cgn_out_grasps_{timestamp}.png"
    seg_path = out_dir / f"cgn_out_segmap_{timestamp}.png" if segmap is not None else None

    visualize_grasps_offscreen(
        full_pc=pc_full,
        pred_grasps_cam=pred_grasps_cam,
        scores=scores,
        save_path=grasp_path,
        plot_opencv_cam=False,
        pc_colors=pc_colors,
        gripper_openings=gripper_openings,
        gripper_width=0.08,
    )

    if segmap is not None:
        save_segmap_image(rgb, segmap, seg_path)
        segmap_img = str(seg_path)
    else:
        segmap_img = None

    return {
        "grasps_img": str(grasp_path),
        "segmap_img": segmap_img,
    }