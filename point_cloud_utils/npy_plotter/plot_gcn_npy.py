import argparse
import numpy as np
import open3d as o3d


def depth_to_points(depth, K):
    # depth: (H,W) in meters (or mm; see note below)
    # K: 3x3 intrinsics
    depth = depth.astype(np.float32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = depth.shape

    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
    return pts[valid], valid


def main(file_path: str):
    d = np.load(file_path, allow_pickle=True).item()
    rgb, depth, K = d["rgb"], d["depth"], d["K"]

    pts, valid = depth_to_points(depth, K)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # optional colors if rgb exists
    if rgb is not None:
        rgb = rgb.reshape(-1, 3)[valid].astype(np.float32)
        if rgb.max() > 1.0:
            rgb /= 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    main(args.file)