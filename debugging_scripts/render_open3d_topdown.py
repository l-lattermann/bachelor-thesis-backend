import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
import cv2


# ===================== CONFIG =====================
NPZ_PATH = "/home/ubuntu/bachelor-thesis-backend/shared/debug/cgn/selected_grasps.npz"

TOP_OUT_PATH = "/home/ubuntu/bachelor-thesis-backend/debugging_scripts/grasps_topdown.png"
FRONT_OUT_PATH = "/home/ubuntu/bachelor-thesis-backend/debugging_scripts/grasps_frontal.png"

WIDTH = 2000
HEIGHT = 1500
MAX_POINTS = 1200000
GRIPPER_WIDTH = 0.08
FINGER_LEN = 0.05
POINT_SIZE = 13.0
LINE_RADIUS = 0.003
LABEL_OFFSET=-0.02 # m

SHOW_INTERACTIVE = False
# =================================================


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
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
        else:
            R = np.eye(3)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

    mesh.rotate(R, center=np.zeros(3))
    mesh.translate((p0 + p1) / 2.0)
    mesh.paint_uniform_color(color)
    return mesh


def create_grasp_meshes(T, opening, color, cam_pose=None, finger_len=0.05, radius=0.0015, label_offset=3):
    half = float(opening) / 2.0
    approach_offset = 0.05

    p0 = np.array([0.0, 0.0, approach_offset], dtype=np.float64)    # wrist middle            p1__p0__p2
    p1 = np.array([-half, 0.0, approach_offset], dtype=np.float64)  # wrist outer "left"       |     |    
    p2 = np.array([ half, 0.0, approach_offset], dtype=np.float64)  # wrist outer "right"      |     |    
    p3 = p1 + np.array([0.0, 0.0, finger_len], dtype=np.float64)    # finger "left"            |     |
    p4 = p2 + np.array([0.0, 0.0, finger_len], dtype=np.float64)    # finger right           p3|     |p4

    pts = np.stack([p0, p1, p2, p3, p4], axis=0)
    pts = pts @ T[:3, :3].T + T[:3, 3]

    if cam_pose is not None:
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        pts = (pts_h @ cam_pose.T)[:, :3]

    segments = [(0, 1), (0, 2), (1, 3), (2, 4)]
    meshes = []

    for i0, i1 in segments:
        mesh = create_cylinder_mesh(pts[i0], pts[i1], radius, color)
        if mesh is not None:
            meshes.append(mesh)

    approach_direction = pts[3] - pts[1]
    approach_direction = approach_direction / np.linalg.norm(approach_direction)
    label_pos = pts[0] + label_offset * approach_direction

    return meshes, label_pos


def normalize_colors(pc_colors):
    if pc_colors is None:
        return None
    pc_colors = np.asarray(pc_colors, dtype=np.float32)
    if pc_colors.max() > 1.0:
        pc_colors = pc_colors / 255.0
    return pc_colors


def build_scene_objects(pc_full, pc_colors, pred_grasps_cam, scores, gripper_openings, label_offset):
    objects = []
    labels = []
    grasp_centers = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_full.astype(np.float64))

    pc_colors = normalize_colors(pc_colors)
    if pc_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pc_colors.astype(np.float64))

    objects.append(("pcd", pcd, "pcd"))

    cm = plt.get_cmap("rainbow")

    all_grasps = []
    cam_pose = np.eye(4)

    for k in pred_grasps_cam.keys():
        grasps = np.asarray(pred_grasps_cam[k])
        if len(grasps) == 0:
            continue

        openings = (
            np.ones(len(grasps), dtype=np.float32) * GRIPPER_WIDTH
            if gripper_openings is None
            else np.asarray(gripper_openings[k], dtype=np.float32).reshape(-1)
        )

        for T, o in zip(grasps, openings):
            all_grasps.append((T, o))

    n_total = max(len(all_grasps), 1)

    mesh_id = 0
    for grasp_idx, (T, o) in enumerate(all_grasps, start=1):
        color = cm((grasp_idx - 1) / n_total)[:3]

        meshes, label_pos = create_grasp_meshes(
            T,
            o,
            color,
            cam_pose=cam_pose,
            finger_len=FINGER_LEN,
            radius=LINE_RADIUS,
            label_offset=label_offset,
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


def add_objects_to_offscreen_scene(renderer, objects):
    pcd_mat = rendering.MaterialRecord()
    pcd_mat.shader = "defaultUnlit"
    pcd_mat.point_size = POINT_SIZE

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
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
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

def draw_labels_on_image(img_rgb, labels, eye, center, up, width, height, fov_deg=60.0):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    for item in labels:
        uv = project_point(item["pos"], eye, center, up, width, height, fov_deg)
        if uv is None:
            continue

        u = int(round(uv[0])) - 20   # light left shift
        v = int(round(uv[1]))

        text = item["text"]
        color = tuple(int(c) for c in item["color"][::-1])  # RGB -> BGR

        cv2.putText(
            img_bgr,
            text,
            (u, v),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            10,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img_bgr,
            text,
            (u, v),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            color,
            6,
            lineType=cv2.LINE_AA,
        )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def render_view(objects, labels, grasp_centers, out_path, eye_dir, up, dist=0.35, fov=40.0):
    renderer = rendering.OffscreenRenderer(WIDTH, HEIGHT)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    add_objects_to_offscreen_scene(renderer, objects)

    center = np.mean(grasp_centers, axis=0).astype(np.float32)
    eye_dir = np.asarray(eye_dir, dtype=np.float32)
    eye_dir = eye_dir / np.linalg.norm(eye_dir)
    eye = center + eye_dir * dist

    renderer.setup_camera(fov, center.tolist(), eye.tolist(), up.tolist())

    img = renderer.render_to_image()
    img_np = np.asarray(img)
    img_np = draw_labels_on_image(img_np, labels, eye, center, up, WIDTH, HEIGHT, fov)

    o3d.io.write_image(out_path, o3d.geometry.Image(img_np))

def show_interactive(objects):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1400, height=1000, visible=True)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = POINT_SIZE

    for _, obj, _ in objects:
        vis.add_geometry(obj)

    vis.run()
    vis.destroy_window()


def render():
    print("Loading:", NPZ_PATH)

    pred_grasps_cam, scores, gripper_openings, pc_full, pc_colors = load_saved_cgn_output(NPZ_PATH)

    if gripper_openings is not None:
        for k, v in gripper_openings.items():
            gripper_openings[k] = np.asarray(v, dtype=np.float32) + 0.02

    pc_full = np.asarray(pc_full, dtype=np.float32)
    if pc_full.ndim != 2 or pc_full.shape[1] != 3:
        raise ValueError(f"pc_full must have shape (N, 3), got {pc_full.shape}")

    if len(pc_full) > MAX_POINTS:
        idx = np.random.choice(len(pc_full), MAX_POINTS, replace=False)
        pc_full = pc_full[idx]
        if pc_colors is not None:
            pc_colors = np.asarray(pc_colors)[idx]

    center = pc_full.mean(axis=0)
    min_bound = pc_full.min(axis=0)
    max_bound = pc_full.max(axis=0)
    extent = np.max(max_bound - min_bound)
    if extent <= 1e-6:
        extent = 1.0

    objects, labels, grasp_centers = build_scene_objects(
        pc_full, pc_colors, pred_grasps_cam, scores, gripper_openings, LABEL_OFFSET
    )

    render_view(
        objects,
        labels,
        grasp_centers,
        TOP_OUT_PATH,
        eye_dir=np.array([0.8, 1.2, -1.2], dtype=np.float32),
        up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        dist=0.8,
        fov=40.0,
    )

    render_view(
        objects,
        labels,
        grasp_centers,
        FRONT_OUT_PATH,
        eye_dir=np.array([-0.8, 1.2, -1.4], dtype=np.float32),
        up=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        dist=0.6,
        fov=45.0,
    )
    print("Saved:", FRONT_OUT_PATH)

    if SHOW_INTERACTIVE:
        show_interactive(objects)


render()