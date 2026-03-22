# NPY Point Cloud Viewer (Open3D) — Personal Notes

## Purpose
Load a Contact-GraspNet-style `.npy` file and visualize the reconstructed point cloud using **Open3D**.

The `.npy` is expected to contain:
- `rgb`   : (H, W, 3) uint8 **BGR** (CGN convention)
- `depth` : (H, W) float32 (meters)
- `K`     : (3, 3) camera intrinsics

This script:
- Back-projects depth → 3D points using `K`
- Attaches colors to points
- Displays the point cloud

---

## Input
- `--file`  
  Path to a `.npy` file containing a dict with keys: `rgb`, `depth`, `K`.

Expected structure:
```python
{
  "rgb":   (H, W, 3) uint8   # BGR (CGN convention)
  "depth": (H, W) float32
  "K":     (3, 3) float32
}
```

## Color convention (important)

* Contact-GraspNet .npy files store BGR

* Open3D expects RGB

This script assumes RGB.
If your .npy is CGN-style (BGR), swap channels before visualization:


## CLI usage

```python
python plot_npy_pointcloud.py --file skiboots.npy
```