from pathlib import Path

import numpy as np
import open3d as o3d


RAW_PATH = "/home/ubuntu/bachelor-thesis-backend/shared/pipeline_io/rc_cube_output.npz"
SEG_PATH = "/home/ubuntu/bachelor-thesis-backend/debugging_scripts/sam_output_3d_segmented.npz"

OCID_COMPARE_PATH = "/home/ubuntu/bachelor-thesis-backend/OCID-dataset/ARID20/table/top/seq04/pcd/result_2018-08-21-12-15-27.pcd"
OCID_NPZ_OUT = "/home/ubuntu/bachelor-thesis-backend/debugging_scripts/OCID_image_0.npz"
OCID_NPZ_SURE_GOOD = "/home/ubuntu/bachelor-thesis-backend/debugging_scripts/OCID_image_0.npy"

def debug_stats(name, pts, depth=None, K=None, mask=None):
    print(f"\n=== DEBUG: {name} ===")
    print("num points:", len(pts))
    print("min:", np.nanmin(pts, axis=0))
    print("max:", np.nanmax(pts, axis=0))
    print("extent:", np.nanmax(pts, axis=0) - np.nanmin(pts, axis=0))
    print("mean:", np.nanmean(pts, axis=0))
    print("std:", np.nanstd(pts, axis=0))

    z = pts[:, 2]
    print("z range:", np.nanmin(z), "→", np.nanmax(z))
    print("z mean/std:", np.nanmean(z), np.nanstd(z))

    if depth is not None:
        print("\n-- DEPTH --")
        print("shape:", depth.shape)
        print("min/max:", np.nanmin(depth), np.nanmax(depth))
        print("mean/std:", np.nanmean(depth), np.nanstd(depth))

    if K is not None:
        print("\n-- INTRINSICS --")
        print(K)

    if mask is not None:
        print("\n-- SEGMENTATION --")
        unique = np.unique(mask)
        print("num segments:", len(unique))
        print("ids:", unique[:20], "...")
        counts = {int(i): int(np.sum(mask == i)) for i in unique[:10]}
        print("pixel counts (sample):", counts)

    print("====================\n")


def build_K_from_scalar_fields(data):
    fx = float(data["fx"])
    fy = float(data["fy"])
    cx = float(data["cx"])
    cy = float(data["cy"])
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def reconstruct_xyz_from_depth(depth, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=-1).astype(np.float32)


def load_npz_contents(path):
    data = np.load(path, allow_pickle=True)

    rgb = data["rgb"] if "rgb" in data.files else None
    xyz = data["xyz"] if "xyz" in data.files else None
    depth = data["depth"] if "depth" in data.files else None

    if "K" in data.files:
        K = data["K"]
    elif all(k in data.files for k in ["fx", "fy", "cx", "cy"]):
        K = build_K_from_scalar_fields(data)
    else:
        K = None

    seg = data["seg"] if "seg" in data.files else None
    label = data["label"] if "label" in data.files else None
    mask = seg if seg is not None else label

    if xyz is None and depth is not None and K is not None:
        xyz = reconstruct_xyz_from_depth(depth, K)

    if xyz is None:
        raise ValueError(f"{path}: need xyz or (depth + K/fx/fy/cx/cy)")

    return xyz, rgb, depth, K, mask


def load_npz_pointcloud(path):
    xyz, rgb, depth, K, mask = load_npz_contents(path)

    pts = xyz.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
    pts_valid = pts[valid]

    debug_stats(
        name=path,
        pts=pts_valid,
        depth=depth,
        K=K,
        mask=mask,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_valid)

    return pcd, rgb, mask, valid, xyz


def load_ocid_pcd(path, width=640, height=480):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OCID pcd not found: {path}")

    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)

    if len(pts) != width * height:
        raise ValueError(
            f"Point count {len(pts)} does not match {width}x{height}. "
            "Check whether this pcd is organized."
        )

    valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
    pts_valid = pts[valid]

    pcd_clean = o3d.geometry.PointCloud()
    pcd_clean.points = o3d.utility.Vector3dVector(pts_valid)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)[valid]
        pcd_clean.colors = o3d.utility.Vector3dVector(colors)

    debug_stats(
        name=str(path),
        pts=pts_valid,
    )

    return pcd_clean


def apply_rgb(pcd, rgb, valid):
    if rgb is None:
        return pcd

    colors = rgb.reshape(-1, 3)[valid].astype(np.float32) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def color_mask(pcd, mask, valid):
    if mask is None:
        return pcd

    mask_flat = mask.reshape(-1)[valid]
    unique = np.unique(mask_flat)

    rng = np.random.default_rng(42)
    cmap = {int(i): rng.random(3) for i in unique}
    cmap[0] = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    colors = np.array([cmap[int(i)] for i in mask_flat], dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def ocid_pcd_to_npz(
    pcd_path,
    output_path,
    width=640,
    height=480,
    fx=575.0,
    fy=575.0,
    cx=320.0,
    cy=240.0,
):
    """
    Save OCID .pcd in the same format as your RC/UOIS debug npz:
    rgb, xyz, label, fx, fy, cx, cy, width, height
    """
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts = np.asarray(pcd.points, dtype=np.float32)

    if len(pts) != width * height:
        raise ValueError(
            f"Point count {len(pts)} does not match {width}x{height}. "
            "Check resolution or whether pcd is organized."
        )

    xyz = pts.reshape(height, width, 3)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)
        rgb = (colors.reshape(height, width, 3) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)

    label = np.zeros((height, width), dtype=np.uint16)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        rgb=rgb,
        xyz=xyz.astype(np.float32),
        label=label,
        fx=np.float32(fx),
        fy=np.float32(fy),
        cx=np.float32(cx),
        cy=np.float32(cy),
        width=np.int32(width),
        height=np.int32(height),
    )

    print(f"Saved OCID → NPZ: {output_path}")
    print("rgb shape:", rgb.shape)
    print("xyz shape:", xyz.shape)
    print("xyz z range:", float(np.nanmin(xyz[:, :, 2])), "→", float(np.nanmax(xyz[:, :, 2])))

def visualize_npz_contents(path):
    xyz, rgb, depth, K, mask = load_npz_contents(path)

    pts = xyz.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
    pts_valid = pts[valid]

    print(f"\n=== VISUALIZE NPZ: {path} ===")
    print("xyz shape:", xyz.shape)
    print("rgb shape:", None if rgb is None else rgb.shape)
    print("depth shape:", None if depth is None else depth.shape)
    print("mask shape:", None if mask is None else mask.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_valid)

    if rgb is not None:
        colors = rgb.reshape(-1, 3)[valid].astype(np.float32) / 255.0
        pcd_rgb = o3d.geometry.PointCloud(pcd)
        pcd_rgb.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd_rgb], window_name=f"{Path(path).name} - RGB")


from pathlib import Path

import numpy as np


def _load_sample(path):
    path = Path(path)

    if path.suffix == ".npy":
        data = np.load(path, allow_pickle=True, encoding="bytes")
        if hasattr(data, "item"):
            data = data.item()
        if not isinstance(data, dict):
            raise ValueError(f"{path}: .npy does not contain a dict-like sample")
        return data

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        sample = {k: data[k] for k in data.files}

        if "K" not in sample and all(k in sample for k in ["fx", "fy", "cx", "cy"]):
            sample["K"] = np.array(
                [
                    [float(sample["fx"]), 0.0, float(sample["cx"])],
                    [0.0, float(sample["fy"]), float(sample["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

        if "xyz" not in sample and "depth" in sample and "K" in sample:
            depth = sample["depth"].astype(np.float32)
            K = sample["K"].astype(np.float32)

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            h, w = depth.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))

            z = depth
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            sample["xyz"] = np.stack([x, y, z], axis=-1).astype(np.float32)

        return sample

    raise ValueError(f"Unsupported file type: {path.suffix}")


def analyze_uois_sample(path):
    path = Path(path)
    d = _load_sample(path)

    rgb = d.get("rgb")
    xyz = d.get("xyz")
    depth = d.get("depth")
    label = d.get("label")
    seg = d.get("seg")
    mask = seg if seg is not None else label
    K = d.get("K")

    print(f"\n=== ANALYZE: {path} ===")

    if rgb is not None:
        print("\n--- RGB ---")
        print("shape:", rgb.shape)
        print("dtype:", rgb.dtype)
        print("min/max:", rgb.min(), rgb.max())
        print("mean/std:", float(rgb.mean()), float(rgb.std()))

        for c, name in enumerate(["R", "G", "B"]):
            ch = rgb[..., c]
            print(
                f"{name}: min={ch.min()}, max={ch.max()}, "
                f"mean={float(ch.mean()):.3f}, std={float(ch.std()):.3f}"
            )

    if depth is not None:
        print("\n--- DEPTH ---")
        print("shape:", depth.shape)
        print("dtype:", depth.dtype)
        print("min/max:", float(np.nanmin(depth)), float(np.nanmax(depth)))
        print("mean/std:", float(np.nanmean(depth)), float(np.nanstd(depth)))

    if K is not None:
        print("\n--- INTRINSICS ---")
        print(K)

    if xyz is not None:
        print("\n--- XYZ ---")
        print("shape:", xyz.shape)
        print("dtype:", xyz.dtype)

        finite_mask = np.isfinite(xyz).all(axis=2)
        valid_mask = finite_mask & (xyz[..., 2] > 0)

        print("finite ratio:", float(finite_mask.mean()))
        print("valid z>0 ratio:", float(valid_mask.mean()))

        if valid_mask.any():
            pts = xyz[valid_mask]
            z = pts[:, 2]

            print("min:", pts.min(axis=0))
            print("max:", pts.max(axis=0))
            print("extent:", pts.max(axis=0) - pts.min(axis=0))
            print("mean:", pts.mean(axis=0))
            print("std:", pts.std(axis=0))

            print("z min/max:", float(z.min()), float(z.max()))
            print("z mean/std:", float(z.mean()), float(z.std()))
            print("z percentiles:", np.percentile(z, [1, 5, 25, 50, 75, 95, 99]))

            ys, xs = np.where(valid_mask)
            print(
                "valid bbox (x_min, y_min, x_max, y_max):",
                (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            )

    if mask is not None:
        print("\n--- LABEL / SEG ---")
        print("shape:", mask.shape)
        print("dtype:", mask.dtype)

        unique, counts = np.unique(mask, return_counts=True)
        print("num labels:", len(unique))
        print("labels:", unique[:30], "..." if len(unique) > 30 else "")

        pairs = sorted(
            [(int(u), int(c)) for u, c in zip(unique, counts)],
            key=lambda x: x[1],
            reverse=True,
        )
        print("largest label counts:", pairs[:15])
        print("foreground ratio:", float((mask > 0).mean()))

    if rgb is not None and xyz is not None:
        print("\n--- CONSISTENCY ---")
        print("rgb/xyz spatial match:", rgb.shape[:2] == xyz.shape[:2])

    if mask is not None and xyz is not None:
        print("mask/xyz spatial match:", mask.shape[:2] == xyz.shape[:2])

    print("===========================\n")
    return d


# --- SAVE OCID AS CGN/UOIS-STYLE NPZ ---
ocid_pcd_to_npz(
    pcd_path=OCID_COMPARE_PATH,
    output_path=OCID_NPZ_OUT,
)

def set_labels_zero(npz_path):
    data = np.load(npz_path)

    label = np.zeros(
        (int(data["height"]), int(data["width"])),
        dtype=np.int32
    )

    np.savez_compressed(
        npz_path,
        rgb=data["rgb"],
        xyz=data["xyz"],
        label=label,
        fx=data["fx"],
        fy=data["fy"],
        cx=data["cx"],
        cy=data["cy"],
        width=data["width"],
        height=data["height"],
    )

def visualize_xyz_with_mask_colors(path):
    xyz, _, _, _, mask = load_npz_contents(path)

    if mask is None:
        raise ValueError(f"{path}: no seg/label mask found")

    pts = xyz.reshape(-1, 3)
    mask_flat = mask.reshape(-1)

    valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
    pts_valid = pts[valid]
    mask_valid = mask_flat[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_valid)

    unique = np.unique(mask_valid)

    rng = np.random.default_rng(42)
    cmap = {int(i): rng.random(3) for i in unique}
    cmap[0] = np.array([0.2, 0.2, 0.2], dtype=np.float32)

    colors = np.array([cmap[int(i)] for i in mask_valid], dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"{Path(path).name} - XYZ mask colors",
    )

# --- LOAD DEBUG CLOUDS ---
pcd_raw, rgb_raw, _, valid_raw, _ = load_npz_pointcloud(RAW_PATH)
pcd_seg, _, mask_seg, valid_seg, _ = load_npz_pointcloud(OCID_NPZ_OUT)
#pcd_ocid = load_ocid_pcd(OCID_COMPARE_PATH)

# --- COLOR ---
pcd_raw = apply_rgb(pcd_raw, rgb_raw, valid_raw)
pcd_seg = color_mask(pcd_seg, mask_seg, valid_seg)

# -- ANALYZE ---
analyze_uois_sample(RAW_PATH)
#analyze_uois_sample(SEG_PATH)
analyze_uois_sample(OCID_NPZ_OUT)
#analyze_uois_sample(OCID_NPZ_SURE_GOOD)

# --- VIS ---
#o3d.visualization.draw_geometries([pcd_raw], window_name="RAW (RGB)")
#o3d.visualization.draw_geometries([pcd_seg], window_name="SEGMENTED (MASK)")
#o3d.visualization.draw_geometries([pcd_ocid], window_name="OCID PCD")

visualize_npz_contents(RAW_PATH)
visualize_npz_contents(SEG_PATH)
visualize_xyz_with_mask_colors(SEG_PATH)
#visualize_npz_contents(OCID_NPZ_OUT)

set_labels_zero(RAW_PATH)