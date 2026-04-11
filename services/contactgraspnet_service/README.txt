CONTACT GRASPNET SERVICE

Test Request:
curl -X POST http://localhost:8003/inference \
  -H "Content-Type: application/json" \
  -d '{
    "npz_path": "/shared/pipeline_io/sam_output.npz",
    "object_id": 1
  }'

Input:
- npz_path: segmented NPZ from SAM3 (includes rgb, depth, segmap, K)
- object_id: optional mask label to restrict grasping

Output:
- annotated_full_size: full image with grasps
- annotated_cropped: zoomed grasp visualization
- heatmap_path: grasp score heatmap
- num_grasps: number of selected grasps

Notes:
- requires SAM3 output with segmentation
- model loads from checkpoint_dir at startup
- DBSCAN + score filtering used for grasp selection
- outputs multiple grasp candidates, final selection done later by LLM