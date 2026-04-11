curl -sS -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{
    "npz_path": "/shared/pipeline_io/rc_cube_pointcloud.npz",
    "image_path": "/shared/pipeline_io/left.png"
  }'