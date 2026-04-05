curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "npz_path": "/shared/pipeline_io/rc_cube_output.npz",
    "image_path": "/shared/pipeline_io/left.png"
  }'