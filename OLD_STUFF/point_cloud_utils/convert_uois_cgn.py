import numpy as np
import os

def uois_to_contactgraspnet(
    in_npy,
    out_npy,
    fx, fy, cx, cy
):
    d = np.load(in_npy, allow_pickle=True).item()

    rgb = d["rgb"].astype(np.uint8)              # (H,W,3)
    xyz = d["xyz"].astype(np.float32)            # (H,W,3)
    label = d["label"].astype(np.int32)           # (H,W)

    # depth = z channel (meters)
    depth = xyz[..., 2].copy()
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0

    # segmentation
    seg = label.copy()
    seg[seg < 0] = 0

    # camera intrinsics
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    out = {
        "rgb": rgb,
        "depth": depth.astype(np.float32),
        "K": K,
        "seg": seg.astype(np.int32),
    }

    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, out)

    print(f"Saved CGN-format file: {out_npy}")
    print("RGB:", rgb.shape, rgb.dtype)
    print("Depth:", depth.shape, depth.dtype, "min/max:", depth.min(), depth.max())
    print("Seg unique labels:", np.unique(seg))



    uois_to_contactgraspnet(
    in_npy="uois/example_images/OCID_image_0.npy",
    out_npy="contact_graspnet/test_data_npy/OCID_image_0.npy",
    fx=912.72,
    fy=912.74,
    cx=649.00,
    cy=363.25,
)