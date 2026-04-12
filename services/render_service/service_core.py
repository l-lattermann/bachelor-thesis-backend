from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import queue
import threading

RENDERER = None
RENDER_THREAD = None
RENDER_QUEUE = queue.Queue()
RENDER_READY = threading.Event()
RENDER_ERROR = None



def render_worker(runtime_cfg):
    global RENDERER, RENDER_ERROR

    try:
        width = int(runtime_cfg["width"])
        height = int(runtime_cfg["height"])

        RENDERER = rendering.OffscreenRenderer(width, height)
        RENDERER.scene.set_background(runtime_cfg["background_rgba"])
        RENDER_READY.set()
    except Exception as e:
        RENDER_ERROR = e
        RENDER_READY.set()
        return

    while True:
        job = RENDER_QUEUE.get()
        if job is None:
            RENDER_QUEUE.task_done()
            break

        try:
            _render_view_internal(
                renderer=RENDERER,
                objects=job["objects"],
                labels=job["labels"],
                grasp_centers=job["grasp_centers"],
                out_path=job["out_path"],
                eye_dir=job["eye_dir"],
                up=job["up"],
                dist=job["dist"],
                fov=job["fov"],
                runtime_cfg=job["runtime_cfg"],
            )
            job["result"]["value"] = True
        except Exception as e:
            job["result"]["error"] = e
        finally:
            job["event"].set()
            RENDER_QUEUE.task_done()


def start_render_worker(runtime_cfg):
    global RENDER_THREAD, RENDER_ERROR

    if RENDER_THREAD is not None:
        return

    RENDER_THREAD = threading.Thread(
        target=render_worker,
        args=(runtime_cfg,),
        daemon=True,
    )
    RENDER_THREAD.start()

    RENDER_READY.wait(timeout=30)

    if not RENDER_READY.is_set():
        raise RuntimeError("Render worker startup timed out.")

    if RENDER_ERROR is not None:
        raise RuntimeError(f"Render worker failed to initialize: {RENDER_ERROR}")


def validate_config(cfg: dict) -> None:
    required = (
        "width",
        "height",
        "max_points",
        "gripper_width",
        "gripper_opening_offset",
        "finger_len",
        "point_size",
        "line_radius",
        "label_offset",
        "approach_offset",
        "label_left_shift_px",
        "label_font_scale",
        "label_text_thickness",
        "label_outline_thickness",
        "background_rgba",
        "colormap",
        "output_dir",
        "left_view_filename",
        "right_view_filename",
        "top_eye_dir",
        "top_up",
        "top_dist",
        "top_fov",
        "front_eye_dir",
        "front_up",
        "front_dist",
        "front_fov",
    )

    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing config key: {key}")


def _unpack(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
        if x.shape == ():
            return x.item()
        if len(x) == 1:
            return x[0]
    return x


def load_saved_cgn_output(npz_path: str):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    required = ("pred_grasps_cam", "scores", "gripper_openings", "pc_full")
    for key in required:
        if key not in data:
            raise ValueError(f"Missing key in NPZ: {key}")

    pred_grasps_cam = _unpack(data["pred_grasps_cam"])
    scores = _unpack(data["scores"])
    gripper_openings = _unpack(data["gripper_openings"])
    pc_full = _unpack(data["pc_full"])
    pc_colors = _unpack(data["pc_colors"]) if "pc_colors" in data else None

    return pred_grasps_cam, scores, gripper_openings, pc_full, pc_colors


def create_cylinder_mesh(p0, p1, radius, color):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)

    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length < 1e-8:
        return None

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    mesh.compute_vertex_normals()

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    direction = vec / length

    v = np.cross(z_axis, direction)
    c = np.dot(z_axis, direction)

    if np.linalg.norm(v) < 1e-8:
        if c < 0:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.array([1.0, 0.0, 0.0], dtype=np.float64) * np.pi
            )
        else:
            R = np.eye(3, dtype=np.float64)
    else:
        vx = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])
        R = np.eye(3, dtype=np.float64) + vx + vx @ vx * (1.0 / (1.0 + c))

    mesh.rotate(R, center=np.zeros(3, dtype=np.float64))
    mesh.translate((p0 + p1) / 2.0)
    mesh.paint_uniform_color(color)
    return mesh


def create_grasp_meshes(T, opening, color, runtime_cfg, cam_pose=None):
    half = float(opening) / 2.0
    approach_offset = float(runtime_cfg["approach_offset"])
    finger_len = float(runtime_cfg["finger_len"])
    line_radius = float(runtime_cfg["line_radius"])
    label_offset = float(runtime_cfg["label_offset"])

    p0 = np.array([0.0, 0.0, approach_offset], dtype=np.float64)
    p1 = np.array([-half, 0.0, approach_offset], dtype=np.float64)
    p2 = np.array([half, 0.0, approach_offset], dtype=np.float64)
    p3 = p1 + np.array([0.0, 0.0, finger_len], dtype=np.float64)
    p4 = p2 + np.array([0.0, 0.0, finger_len], dtype=np.float64)

    pts = np.stack([p0, p1, p2, p3, p4], axis=0)
    pts = pts @ T[:3, :3].T + T[:3, 3]

    if cam_pose is not None:
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        pts = (pts_h @ cam_pose.T)[:, :3]

    segments = [(0, 1), (0, 2), (1, 3), (2, 4)]
    meshes = []

    for i0, i1 in segments:
        mesh = create_cylinder_mesh(pts[i0], pts[i1], line_radius, color)
        if mesh is not None:
            meshes.append(mesh)

    approach_direction = pts[3] - pts[1]
    norm = np.linalg.norm(approach_direction)
    if norm < 1e-8:
        approach_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        approach_direction = approach_direction / norm

    label_pos = pts[0] + label_offset * approach_direction
    return meshes, label_pos


def normalize_colors(pc_colors):
    if pc_colors is None:
        return None
    pc_colors = np.asarray(pc_colors, dtype=np.float32)
    if pc_colors.size == 0:
        return None
    if pc_colors.max() > 1.0:
        pc_colors = pc_colors / 255.0
    return pc_colors


def build_scene_objects(pc_full, pc_colors, pred_grasps_cam, scores, gripper_openings, runtime_cfg):
    objects = []
    labels = []
    grasp_centers = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_full.astype(np.float64))

    pc_colors = normalize_colors(pc_colors)
    if pc_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pc_colors.astype(np.float64))

    objects.append(("pcd", pcd, "pcd"))

    cm = plt.get_cmap(runtime_cfg["colormap"])
    all_grasps = []
    cam_pose = np.eye(4, dtype=np.float64)

    for k in pred_grasps_cam.keys():
        grasps = np.asarray(pred_grasps_cam[k])
        if len(grasps) == 0:
            continue

        if gripper_openings is None:
            openings = np.ones(len(grasps), dtype=np.float32) * float(runtime_cfg["gripper_width"])
        else:
            openings = np.asarray(gripper_openings[k], dtype=np.float32).reshape(-1)
            openings = openings + float(runtime_cfg["gripper_opening_offset"])

        for T, o in zip(grasps, openings):
            all_grasps.append((T, o))

    n_total = max(len(all_grasps), 1)

    mesh_id = 0
    for grasp_idx, (T, o) in enumerate(all_grasps, start=1):
        color = cm((grasp_idx - 1) / n_total)[:3]

        meshes, label_pos = create_grasp_meshes(
            T=T,
            opening=o,
            color=color,
            runtime_cfg=runtime_cfg,
            cam_pose=cam_pose,
        )

        for mesh in meshes:
            objects.append((f"grasp_{mesh_id}", mesh, "mesh"))
            mesh_id += 1

        labels.append({
            "text": str(grasp_idx),
            "pos": label_pos,
            "color": tuple(int(255 * c) for c in color),
        })

        grasp_centers.append(T[:3, 3])

    grasp_centers = np.asarray(grasp_centers, dtype=np.float64)
    return objects, labels, grasp_centers


def add_objects_to_offscreen_scene(renderer, objects, runtime_cfg):
    pcd_mat = rendering.MaterialRecord()
    pcd_mat.shader = "defaultUnlit"
    pcd_mat.point_size = float(runtime_cfg["point_size"])

    mesh_mat = rendering.MaterialRecord()
    mesh_mat.shader = "defaultLit"

    for name, obj, obj_type in objects:
        if obj_type == "pcd":
            renderer.scene.add_geometry(name, obj, pcd_mat)
        else:
            renderer.scene.add_geometry(name, obj, mesh_mat)


def look_at_view_matrix(eye, center, up):
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z = eye - center
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-8:
        raise ValueError("eye and center must not be identical.")
    z = z / z_norm

    x = np.cross(up, z)
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-8:
        raise ValueError("up vector must not be parallel to viewing direction.")
    x = x / x_norm

    y = np.cross(z, x)

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = x
    view[1, :3] = y
    view[2, :3] = z
    view[0, 3] = -np.dot(x, eye)
    view[1, 3] = -np.dot(y, eye)
    view[2, 3] = -np.dot(z, eye)
    return view


def project_point(point, eye, center, up, width, height, fov_deg):
    point = np.asarray(point, dtype=np.float64)
    view = look_at_view_matrix(eye, center, up)

    p = np.ones(4, dtype=np.float64)
    p[:3] = point
    cam = view @ p

    z_cam = cam[2]
    if z_cam >= 0:
        return None

    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    aspect = width / height

    x_ndc = (cam[0] * f / aspect) / (-z_cam)
    y_ndc = (cam[1] * f) / (-z_cam)

    if abs(x_ndc) > 1.2 or abs(y_ndc) > 1.2:
        return None

    u = (x_ndc + 1.0) * 0.5 * width
    v = (1.0 - (y_ndc + 1.0) * 0.5) * height
    return np.array([u, v], dtype=np.float64)


def draw_labels_on_image(img_rgb, labels, eye, center, up, width, height, fov_deg, runtime_cfg):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    left_shift_px = int(runtime_cfg["label_left_shift_px"])
    font_scale = float(runtime_cfg["label_font_scale"])
    text_thickness = int(runtime_cfg["label_text_thickness"])
    outline_thickness = int(runtime_cfg["label_outline_thickness"])

    for item in labels:
        uv = project_point(item["pos"], eye, center, up, width, height, fov_deg)
        if uv is None:
            continue

        u = int(round(uv[0])) - left_shift_px
        v = int(round(uv[1]))

        text = item["text"]
        color = tuple(int(c) for c in item["color"][::-1])

        cv2.putText(
            img_bgr,
            text,
            (u, v),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            outline_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img_bgr,
            text,
            (u, v),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            text_thickness,
            lineType=cv2.LINE_AA,
        )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _render_view_internal(renderer, objects, labels, grasp_centers, out_path, eye_dir, up, dist, fov, runtime_cfg):
    width = int(runtime_cfg["width"])
    height = int(runtime_cfg["height"])

    renderer.scene.clear_geometry()
    renderer.scene.set_background(runtime_cfg["background_rgba"])

    add_objects_to_offscreen_scene(renderer, objects, runtime_cfg)

    center = np.mean(grasp_centers, axis=0).astype(np.float32)
    eye_dir = np.asarray(eye_dir, dtype=np.float32)

    norm = np.linalg.norm(eye_dir)
    if norm < 1e-8:
        raise ValueError("eye_dir must not be zero.")
    eye_dir = eye_dir / norm

    eye = center + eye_dir * float(dist)

    renderer.setup_camera(
        float(fov),
        center.tolist(),
        eye.tolist(),
        np.asarray(up, dtype=np.float32).tolist(),
    )

    img = renderer.render_to_image()
    img_np = np.asarray(img)
    img_np = draw_labels_on_image(
        img_rgb=img_np,
        labels=labels,
        eye=eye,
        center=center,
        up=up,
        width=width,
        height=height,
        fov_deg=float(fov),
        runtime_cfg=runtime_cfg,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(out_path), o3d.geometry.Image(img_np))

def render_view(objects, labels, grasp_centers, out_path, eye_dir, up, dist, fov, runtime_cfg):
    if RENDER_THREAD is None:
        raise RuntimeError("Render worker is not started.")

    if not RENDER_THREAD.is_alive():
        raise RuntimeError("Render worker is not alive.")

    done = threading.Event()
    result = {"value": None, "error": None}

    RENDER_QUEUE.put({
        "objects": objects,
        "labels": labels,
        "grasp_centers": grasp_centers,
        "out_path": out_path,
        "eye_dir": eye_dir,
        "up": up,
        "dist": dist,
        "fov": fov,
        "runtime_cfg": runtime_cfg,
        "event": done,
        "result": result,
    })

    finished = done.wait(timeout=120)
    if not finished:
        raise RuntimeError("Render job timed out.")

    if result["error"] is not None:
        raise result["error"]

    return result["value"]


def render_from_npz(npz_path: str, runtime_cfg: dict) -> dict:
    pred_grasps_cam, scores, gripper_openings, pc_full, pc_colors = load_saved_cgn_output(npz_path)

    pc_full = np.asarray(pc_full, dtype=np.float32)
    if pc_full.ndim != 2 or pc_full.shape[1] != 3:
        raise ValueError(f"pc_full must have shape (N, 3), got {pc_full.shape}")

    max_points = int(runtime_cfg["max_points"])
    if len(pc_full) > max_points:
        idx = np.random.choice(len(pc_full), max_points, replace=False)
        pc_full = pc_full[idx]
        if pc_colors is not None:
            pc_colors = np.asarray(pc_colors)[idx]

    objects, labels, grasp_centers = build_scene_objects(
        pc_full=pc_full,
        pc_colors=pc_colors,
        pred_grasps_cam=pred_grasps_cam,
        scores=scores,
        gripper_openings=gripper_openings,
        runtime_cfg=runtime_cfg,
    )

    if len(grasp_centers) == 0:
        raise ValueError("No grasps found in pred_grasps_cam.")

    output_dir = Path(runtime_cfg["output_dir"])
    top_out_path = output_dir / runtime_cfg["left_view_filename"]
    front_out_path = output_dir / runtime_cfg["right_view_filename"]

    render_view(
        objects=objects,
        labels=labels,
        grasp_centers=grasp_centers,
        out_path=top_out_path,
        eye_dir=runtime_cfg["top_eye_dir"],
        up=runtime_cfg["top_up"],
        dist=runtime_cfg["top_dist"],
        fov=runtime_cfg["top_fov"],
        runtime_cfg=runtime_cfg,
    )

    render_view(
        objects=objects,
        labels=labels,
        grasp_centers=grasp_centers,
        out_path=front_out_path,
        eye_dir=runtime_cfg["front_eye_dir"],
        up=runtime_cfg["front_up"],
        dist=runtime_cfg["front_dist"],
        fov=runtime_cfg["front_fov"],
        runtime_cfg=runtime_cfg,
    )

    return {
        "status": "ok",
        "npz_path": str(npz_path),
        "top_render_path": str(top_out_path),
        "front_render_path": str(front_out_path),
    }

