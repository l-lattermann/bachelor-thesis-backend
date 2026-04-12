SAM3 SERVICE

Test Request:
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "npz_path": "/shared/pipeline_io/rc_cube_pointcloud.npz",
    "image_path": "/shared/pipeline_io/left.png",
    "text_prompt": "red box"
  }' | jq

Input:
- npz_path: RC Cube NPZ containing rgb, xyz and camera parameters
- image_path: left RGB image
- text_prompt: object description for SAM3

Output:
- result_path: NPZ with segmentation
- rgb_annotated_path: annotated image
- mask_count: number of masks
- box_count: number of boxes
- scores: confidence scores

Notes:
- model is loaded at startup
- HF_TOKEN must be set
- image is loaded as PIL internally
- multiple masks possible, selection handled later in orchestrator