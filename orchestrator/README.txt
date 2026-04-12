ORCHESTRATOR SERVICE

Test Request:
curl -sS -X POST http://localhost:8000/run_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "object_query": "red box"
  }' | jq

Input:
- object_query: text description of the target object (e.g. "red box")

Output:
- raw_pointcloud: path to raw point cloud from RC Cube
- segmented_pointcloud: segmented point cloud from SAM3
- segmentation_image: annotated segmentation image
- selected_grasps_npz: NPZ with selected grasp candidates from Contact-GraspNet
- heatmap_path: optional grasp heatmap path
- render_top_image: top rendered grasp visualization
- render_front_image: front rendered grasp visualization
- selected_object_id: selected object label
- mask_count: number of detected objects
- grasp_candidates_after_cgn: number of grasp candidates after Contact-GraspNet
- llm_grasp_response: final grasp selection from LLM
- timings: per-service runtimes

Notes:
- pipeline: RC Cube → SAM3 → (LLM object selection if needed) → Contact-GraspNet → Render Service → LLM grasp selection
- mask_count > 1 → LLM resolves object ambiguity
- mask_count == 1 → goes directly to Contact-GraspNet
- LLM grasp selection uses the renderer output images (top + front view)
- all services must be running
- first request can be slower due to model loading
- errors include step + traceback