curl -X POST http://localhost:8002/inference \
  -H "Content-Type: application/json" \
  -d '{
    "npz_path": "/shared/pipeline_io/sam_segmented_output.npz",
    "object_id": 11
  }'