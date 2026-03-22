PIPELINE DATA FLOW


[1] RC_CUBE OUTPUT  (/shared/input/)

left.png
disparity.png
cam.yaml
disp_params.yaml

cam.yaml

camera_intrinsics:
  width: int
  height: int
  fx: float
  fy: float
  cx: float
  cy: float

disp_params.yaml

disparity:
  scale: float
  offset: float
  invalid: float
  baseline_m: float
  delta_d: float
  encoding: str



[1.5] RC_CUBE → UOIS CONVERSION (in-memory)

INPUT:
  left_rgb   : uint8   (H_cam, W_cam, 3)
  disp_arr   : uint16/float (H, W)
  cam        : dict (intrinsics, original resolution)
  disp_params: dict (stereo params)
  conf       : optional (H, W)

PROCESS:
  - scale intrinsics to disparity resolution
  - disparity → depth:
      z = (fx * baseline) / d
  - backproject:
      x = (u - cx) * z / fx
      y = (v - cy) * z / fy
  - build xyz:
      xyz (H, W, 3)
  - resize rgb → (H, W)
  - build label mask:
      label (H, W)

OUTPUT (uois_dict):
  rgb   : uint8   (H, W, 3)
  xyz   : float32 (H, W, 3)
  label : int32   (H, W)



[1.6] SAVED UOIS INPUT (.npz)

keys:
  rgb   : uint8   (H, W, 3)
  xyz   : float32 (H, W, 3)
  label : int32   (H, W)

NOTE:
- "label" = binary mask (your seg_id vs background)
- no multi-instance segmentation yet (UOIS will replace this)



[2] UOIS INPUT (.npz)

keys:
  rgb   : uint8   (H, W, 3)
  xyz   : float32 (H, W, 3)
  seg   : int32   (H, W)        # optional (if GT or precomputed)
  label : int32   (H, W)        # optional (for evaluation)



[3] UOIS OUTPUT (/shared/output/)

uois_input_mask.npy        → seg (H, W)
uois_input_vis/            → debug images
uois_input_cgn.npy/.npz    → CGN input

uois_input_cgn.npz
--
keys:
  rgb : uint8   (H, W, 3)
  xyz : float32 (H, W, 3)
  seg : int32   (H, W)      # instance ids



[4] CGN INTERNAL FORMAT
-
pc_full:
  float32 (N, 3)

pc_segments:
  dict[int → float32 (Mi, 3)]
  # one point cloud per object id



[5] CGN OUTPUT (.npz response)

keys:
  pred_grasps_cam:
    dict[int → float32 (K, 4, 4)]
    # grasp poses (SE3 matrices)

  scores:
    dict[int → float32 (K,)]
    # grasp confidence

  contact_pts:
    dict[int → float32 (K, 3)]
    # contact points in camera frame

  gripper_openings:
    dict[int → float32 (K,)]
    # predicted gripper width



FLOW SUMMARY
============
RC_CUBE
  → images + disparity + intrinsics

→ UOIS (.npz)
  rgb + xyz (+ optional seg/label)

→ UOIS OUTPUT
  seg mask + CGN npz

→ CGN
  point cloud + segments

→ CGN OUTPUT
  grasps + scores + contacts + openings