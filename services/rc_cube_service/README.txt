RC CUBE SERVICE

Test Request:
curl -X POST http://localhost:8001/fetch_disparity_and_left

Input:
- no request body
- uses RC_CUBE_SOCKET_ADRESS from environment
- alternatively mock modes from config.yaml

Output:
- rc_out_npz: path to generated NPZ file
- left_png_path: path to left RGB image
- debug: additional RC Cube output

Notes:
- can run with live RC Cube or mock data
- generates point cloud + left image for downstream pipeline
- if mock_rc_cube_full is enabled, returns precomputed output directly