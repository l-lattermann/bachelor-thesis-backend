# PLY → Contact-GraspNet NPY Converter (Personal Notes)

## Purpose
Convert a colored point cloud `.ply` into a Contact-GraspNet-style `.npy` file containing:

- `rgb`  : projected color image (stored as **BGR** for CGN compatibility)
- `depth`: projected depth image (meters)
- `K`    : camera intrinsics matrix (3×3)
- `seg`  : segmentation map (currently constant `seg_id` everywhere)

This is useful to feed custom RC-Cube / Meshlab PLY point clouds into Contact-GraspNet inference.

---

## Input
### Required
- `--ply`  
  Path to a `.ply` file with per-vertex fields:
  - `x, y, z`
  - `diffuse_red, diffuse_green, diffuse_blue`

### Assumptions
- Points are already in the camera coordinate frame (x right, y down, z forward).
- Units are meters (depth values will be saved directly from `z`).
- Colors exist as vertex attributes (`diffuse_*`).

---

## Output
### File written
- `--out`  
  A `.npy` containing a Python dict (pickled object):

```python
{
  "rgb":   (H, W, 3) uint8   # stored as BGR
  "depth": (H, W) float32    # meters
  "K":     (3, 3) float32
  "seg":   (H, W) int32
}
```

## Important: color order

```python
rgb = rgb[..., [2, 1, 0]]
```

## CLI usage

```python
python ply_to_npy.py \
  --ply scene.ply \
  --out scene.npy \
  --mode organized \
  --depth-unit mm_u16



python ply_to_npy.py \
  --ply scene.ply \
  --out scene.npy \
  --mode project
```