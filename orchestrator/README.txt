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
- rgb_annotated_img: segmentation image
- mask_count: number of detected objects
- selected_object_id: selected object label
- grasp_annotation_full_img: full grasp visualization
- grasp_annotation_zoomed_img: zoomed grasp visualization
- llm_grasp_response: final grasp selection
- timings: per-service runtimes

Notes:
- pipeline: RC → SAM3 → (LLM optional) → CGN → LLM
- mask_count > 1 → LLM resolves object ambiguity
- mask_count == 1 → goes directly to CGN
- all services must be running
- first request can be slower due to model loading
- errors include step + traceback