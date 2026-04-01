import numpy as np
import open3d as o3d

RAW_PATH = "/home/ubuntu/bachelor-thesis-backend/thesis_pipeline/shared/pipeline_io/rc_cube_output.npz"
SEG_PATH = "/home/ubuntu/bachelor-thesis-backend/thesis_pipeline/shared/pipeline_io/uois_output.npz"



def debug_stats(name, pts, depth=None, K=None, mask=None):
    print(f"\n=== DEBUG: {name} ===")

    print("num points:", len(pts))
    print("min:", pts.min(axis=0))
    print("max:", pts.max(axis=0))
    print("extent:", pts.max(axis=0) - pts.min(axis=0))
    print("mean:", pts.mean(axis=0))
    print("std:", pts.std(axis=0))

    z = pts[:, 2]
    print("z range:", z.min(), "→", z.max())
    print("z mean/std:", z.mean(), z.std())

    # depth sanity
    if depth is not None:
        print("\n-- DEPTH --")
        print("shape:", depth.shape)
        print("min/max:", np.nanmin(depth), np.nanmax(depth))
        print("mean/std:", np.nanmean(depth), np.nanstd(depth))

    # intrinsics sanity
    if K is not None:
        print("\n-- INTRINSICS --")
        print(K)

    # segmentation sanity
    if mask is not None:
        print("\n-- SEGMENTATION --")
        unique = np.unique(mask)
        print("num segments:", len(unique))
        print("ids:", unique[:20], "...")

        counts = {i: np.sum(mask == i) for i in unique[:10]}
        print("pixel counts (sample):", counts)

    print("====================\n")


def load_pcd(path):
    data = np.load(path, allow_pickle=True)

    rgb = data["rgb"] if "rgb" in data.files else None
    xyz = data["xyz"] if "xyz" in data.files else None
    depth = data["depth"] if "depth" in data.files else None
    K = data["K"] if "K" in data.files else None
    seg = data["seg"] if "seg" in data.files else None
    label = data["label"] if "label" in data.files else None
    mask = seg if seg is not None else label

    if xyz is None:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        xyz = np.stack([x, y, z], axis=-1)

    pts = xyz.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
    pts = pts[valid]

    # DEBUG
    debug_stats(
        name=path,
        pts=pts,
        depth=depth,
        K=K,
        mask=mask
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    return pcd, rgb, mask, valid


def apply_rgb(pcd, rgb, valid):
    if rgb is None:
        return pcd

    colors = rgb.reshape(-1, 3)[valid].astype(np.float32) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def color_mask(pcd, mask, valid):
    if mask is None:
        return pcd

    mask = mask.reshape(-1)[valid]
    unique = np.unique(mask)

    rng = np.random.default_rng(42)
    cmap = {i: rng.random(3) for i in unique}
    cmap[0] = [0.5, 0.5, 0.5]

    colors = np.array([cmap[i] for i in mask])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# --- LOAD ---
pcd_raw, rgb_raw, _, valid_raw = load_pcd(RAW_PATH)
pcd_seg, _, mask_seg, valid_seg = load_pcd(SEG_PATH)

# --- COLOR ---
pcd_raw = apply_rgb(pcd_raw, rgb_raw, valid_raw)
pcd_seg = color_mask(pcd_seg, mask_seg, valid_seg)

# --- VIS ---
o3d.visualization.draw_geometries([pcd_raw], window_name="RAW (RGB)")
o3d.visualization.draw_geometries([pcd_seg], window_name="SEGMENTED (MASK)")