RENDER SERVICE

Test Request:
curl -X POST http://localhost:8005/render \
  -H "Content-Type: application/json" \
  -d '{
    "npz_path": "/shared/debug/cgn/selected_grasps.npz"
  }'  | jq

Input:
- npz_path: NPZ file containing Contact-GraspNet output (grasps, point cloud, optional colors)

Output:
- top_render_path: path to top-down rendered image
- front_render_path: path to frontal rendered image
- num_points: number of points used for rendering (after subsampling)
- num_grasps: number of rendered grasps

Notes:
- all parameters (rendering, camera views, output paths) are defined in config.yaml
- config is loaded once at startup and passed to service_core
- supports large point clouds via max_points subsampling
- uses Open3D offscreen renderer (no GUI required)
- labels are projected in image space after rendering
- code is mounted via docker-compose → fast rebuild (no COPY of source)
- requires libgl1, libglib2.0-0, libgomp1 inside container
- input NPZ must contain: pred_grasps_cam, scores, gripper_openings, pc_full (optional pc_colors)