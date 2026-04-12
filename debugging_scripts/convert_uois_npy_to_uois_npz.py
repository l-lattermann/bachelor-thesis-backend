from pathlib import Path
import numpy as np

INPUT_PATH = "/home/ubuntu/bachelor-thesis-backend/debugging_scripts/uois/example_images/OCID_image_0.npy"
OUTPUT_PATH = "/home/ubuntu/bachelor-thesis-backend/shared/debug/rc_cube/rc_cube_mock_full_output/OCID_image_0.npz"


def convert_npy_to_npz(input_path, output_path):
    d = np.load(input_path, allow_pickle=True, encoding="bytes").item()

    rgb = d["rgb"].astype(np.uint8)
    xyz = d["xyz"].astype(np.float32)
    label = d["label"].astype(np.int32)

    height, width = rgb.shape[:2]

    # Try to get intrinsics if available, otherwise fallback
    if "K" in d:
        K = d["K"]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
    else:
        # fallback (OCID typical)
        fx, fy = 575.0, 575.0
        cx, cy = width / 2.0, height / 2.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    label = np.zeros((height, width), dtype=np.int32)
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

    print(f"Saved: {output_path}")
    print("rgb:", rgb.shape, rgb.dtype)
    print("xyz:", xyz.shape, xyz.dtype)
    print("label:", label.shape, label.dtype)
    print("fx fy cx cy:", fx, fy, cx, cy)


if __name__ == "__main__":
    convert_npy_to_npz(INPUT_PATH, OUTPUT_PATH)