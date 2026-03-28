curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "npz_path": "/shared/pipeline_io/rc_cube_output.npz"
  }'