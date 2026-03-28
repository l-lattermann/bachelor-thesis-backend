import sys
import numpy as np
import open3d as o3d


npz_path = sys.argv[1] if len(sys.argv) > 1 else "uois_output.npz"

print("Loading:", npz_path)
data = np.load(npz_path, allow_pickle=True)
print("npz files:", data.files)

rgb = data["rgb"] if "rgb" in data.files else None
xyz = data["xyz"] if "xyz" in data.files else None
depth = data["depth"] if "depth" in data.files else None
K = data["K"] if "K" in data.files else None
seg = data["seg"] if "seg" in data.files else None
label = data["label"] if "label" in data.files else None

# support both naming styles
mask = seg if seg is not None else label

# case 1: already has xyz (e.g. rc_cube_output.npz)
if xyz is not None:
    xyz = xyz.astype(np.float32)

# case 2: only has depth + K (e.g. uois_output.npz)
elif depth is not None and K is not None:
    depth = depth.astype(np.float32)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    xyz = np.stack([x, y, z], axis=-1)

else:
    raise ValueError("Need either 'xyz' or ('depth' and 'K') in npz file.")

pts = xyz.reshape(-1, 3)
valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
pts = pts[valid]

print("\n=== POINT CLOUD DEBUG ===")
print("num valid points:", len(pts))
print("PointCloud extent:", pts.max(axis=0) - pts.min(axis=0))
print("Min:", pts.min(axis=0))
print("Max:", pts.max(axis=0))
print("=========================\n")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

# prefer true RGB colors if present
if rgb is not None:
    rgb_flat = rgb.reshape(-1, 3)[valid].astype(np.float32) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(rgb_flat)

# otherwise use seg/label colors
elif mask is not None:
    mask_flat = mask.reshape(-1)[valid]
    unique_ids = np.unique(mask_flat)

    rng = np.random.default_rng(42)
    color_map = {sid: rng.random(3) for sid in unique_ids}
    color_map[0] = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    colors = np.array([color_map[sid] for sid in mask_flat], dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud: {npz_path}")