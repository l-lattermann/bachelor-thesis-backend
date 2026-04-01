curl -X POST http://localhost:8004/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_name": "select_grasp",
    "image_path": "/shared/pipeline_io/cgn_output.png"
  }'