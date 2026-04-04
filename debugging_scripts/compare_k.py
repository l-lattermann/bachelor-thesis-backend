import numpy as np

data = np.load("rc_cube_output.npz", allow_pickle=True)

xyz = data["xyz"].astype(np.float32)
K = data["K"].astype(np.float32)

depth = xyz[..., 2].copy()
depth[~np.isfinite(depth)] = 0.0
depth[depth < 0] = 0.0

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

h, w = depth.shape
u, v = np.meshgrid(np.arange(w), np.arange(h))

z = depth
x = (u - cx) * z / fx
y = (v - cy) * z / fy
xyz_reproj = np.stack([x, y, z], axis=-1)

valid = np.isfinite(xyz).all(axis=-1) & np.isfinite(xyz_reproj).all(axis=-1) & (xyz[..., 2] > 0)
diff = np.linalg.norm(xyz[valid] - xyz_reproj[valid], axis=1)

print("mean diff:", diff.mean())
print("median diff:", np.median(diff))
print("max diff:", diff.max())