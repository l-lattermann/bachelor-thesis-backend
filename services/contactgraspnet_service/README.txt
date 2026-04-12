CONTACT GRASPNET SERVICE

Test Request:
curl -X POST http://localhost:8003/inference \
  -H "Content-Type: application/json" \
  -d '{
    "npz_path": "/shared/pipeline_io/sam_output.npz",
    "object_id": 1
  }' | jq

Input:
- npz_path: segmented NPZ from SAM3 (includes rgb, depth, segmap, K)
- object_id: optional mask label to restrict grasping

Output:
- sel_grasps_npz: NPZ with selected grasp candidates and point cloud data for downstream rendering
- heatmap_path: grasp score heatmap
- num_grasps: number of selected grasps

Notes:
- requires SAM3 output with segmentation
- model loads from checkpoint_dir at startup
- DBSCAN + score filtering used for grasp selection
- no image annotation is produced here anymore
- grasp visualization is fully handled later by the render service
- outputs multiple grasp candidates, final selection done later by the LLM