# RC-Cube / SC Client README

## 1. Required GUI Setup (CRITICAL)

Before running the pipeline, configure the RC / reception GUI:

1. Open **Depth Image Settings**
2. Set **Acquisition Mode → Single**
3. Click **Acquire**
4. Switch back to **Acquisition Mode → Continuous**
5. Keep GUI running
6. Then start Docker + services

> If this is not done, **disparity will timeout while left image works**

---

## 2. Architecture

Services:
- `rc_cube_service` → fetch left + disparity
- `uois_service` → segmentation
- `contactgraspnet_service` → grasps
- `orchestrator` → pipeline control

Paths:
- `/shared/input`
- `/shared/output`

Host output:
- `/home/ubuntu/Desktop/pipeline_output`

---

## 3. RC-Cube Client

### Class
`RcCubeGrpcClient`

### Config (config.yaml)
- `rc_cube.ip`
- `rc_cube.timeout_sec`
- `rc_cube.max_message_size_mb`

---

## 4. Methods

### `__init__(config_path)`
Loads:
- IP
- timeout
- max message size

---

### `get_disparity_and_left()`

Fetches:
- left image
- disparity image

Request:
```python
pb2.ImageSetRequest(
    left_enabled=True,
    right_enabled=False,
    disparity_enabled=True,
    disparity_error_enabled=False,
    confidence_enabled=False,
    mesh_enabled=False,
    color=True,
)
```

Returns:
- `left_rgb` → (H,W,3) uint8
- `disp_arr` → (H,W) uint16/uint8
- `cam` → intrinsics
- `disp_params`

---

### `test_disparity()`

Debug method:
- prints request params
- prints available fields
- detects missing disparity

Use when:
- left works
- disparity fails

---

## 5. Output Saving

Function:
`save_rc_cube_output(...)`

Writes:
- `left.png`
- `disparity.png` (normalized)
- `cam.txt`
- `disp_params.txt`

Location:
```
/shared/input/
```

---

## 6. API Endpoints

### Health
```bash
curl http://localhost:8003/health
```

---

### Fetch data
```bash
curl -X POST http://localhost:8003/fetch_disparity_and_left
```

---

## 7. Docker

```bash
docker compose build rc_cube
docker compose up -d rc_cube
docker compose logs -f rc_cube
```

---

## 8. Correct Execution Order

1. GUI → Single → Acquire → Continuous
2. Start Docker
3. `/health`
4. `/fetch_disparity_and_left`

---

## 9. Troubleshooting

### Left OK, disparity fails

Causes:
- GUI not initialized
- pipeline not active
- timeout too low
- wrong IP

Fix:
- repeat GUI sequence
- increase timeout
- test `/test_disparity`

---

### Import errors

Fix:
- ensure `pipeline_utils` mounted
- correct import:
```python
from pipeline_utils.rc_cube.save_output import save_rc_cube_output
```

---

### Files not visible

Check:
- docker volumes
- `/shared/input`
- Desktop mount

---

## 10. Minimal Config

```yaml
rc_cube:
  ip: 172.27.48.156:50051
  timeout_sec: 60.0
  max_message_size_mb: 300
  save_output: true

pipeline:
  use_rc_cube: true
```

---

## Summary

Capabilities:
- fetch left + disparity
- debug stream
- save outputs
- integrate into pipeline

Critical:
**Always initialize GUI (Single → Acquire → Continuous) before running services**
