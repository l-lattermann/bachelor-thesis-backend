import argparse
import numpy as np
import open3d as o3d


def main(file_path: str, only_labeled: bool = False):
    d = np.load(file_path, allow_pickle=True).item()

    # UOIS format
    rgb_img = d["rgb"]    # (H,W,3) uint8
    xyz_img = d["xyz"]    # (H,W,3) float32
    label_img = d["label"]  # (H,W) int32

    H, W, _ = xyz_img.shape

    xyz = xyz_img.reshape(-1, 3).astype(np.float32)
    rgb = rgb_img.reshape(-1, 3).astype(np.float32)
    labels = label_img.reshape(-1).astype(np.int32)

    # Valid points: finite and z>0
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)

    # Optionally show only labeled (foreground) points
    if only_labeled:
        valid = valid & (labels > 0)

    pts = xyz[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Colors (normalize to 0..1 for Open3D)
    if rgb.size > 0:
        cols = rgb[valid]
        if cols.max() > 1.0:
            cols /= 255.0
        cols = np.clip(cols, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(cols)

    print(f"Loaded: {file_path}")
    print(f"Image size: {W}x{H}")
    print(f"Points shown: {pts.shape[0]} (only_labeled={only_labeled})")
    print(f"Unique labels (sample): {np.unique(labels)[:20]}")

    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--only-labeled", action="store_true", help="show only label > 0 points")
    args = parser.parse_args()
    main(args.file, only_labeled=args.only_labeled)