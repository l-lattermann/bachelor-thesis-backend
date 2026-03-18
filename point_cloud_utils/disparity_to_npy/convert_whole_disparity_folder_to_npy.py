from pathlib import Path
import re
import numpy as np
import dispartiy_to_npy as converter



input_root = Path("input")
output_root = Path("output")
output_root.mkdir(exist_ok=True)

def first_match(root, a, b):
    return next(iter(root.glob(a)), None) or next(iter(root.glob(b)), None)

for root in input_root.iterdir():
    if not root.is_dir():
        continue

    path_disp_img  = first_match(root, "*_disparity_0_0.png", "*_disparity_00_00.png")
    path_left_img  = first_match(root, "*_left_0_0.png", "*_left_00_00.png")
    path_conf_img  = first_match(root, "*_confidence_0_0.png", "*_confidence_00_00.png")
    path_param_txt = first_match(root, "*_disparity_0_0_param.txt", "*_disparity_00_00_param.txt")

    if not all([path_disp_img, path_left_img, path_conf_img, path_param_txt]):
        print(f"skip {root.name}: missing files")
        continue

    # base robust for both 0_0 and 00_00
    base = re.sub(r"_disparity_(0_0|00_00)$", "", path_disp_img.stem)

    cam, disp_params = {}, {}

    with open(path_param_txt) as f:
        lines = f.readlines()

    # parse camera.A
    A_line = next(l for l in lines if l.startswith("camera.A"))
    A_vals = re.findall(r"[-+]?\d*\.\d+|\d+", A_line)
    A = np.array(A_vals, dtype=float).reshape(3, 3)

    cam["fx"] = A[0, 0]
    cam["fy"] = A[1, 1]
    cam["cx"] = A[0, 2]
    cam["cy"] = A[1, 2]

    # parse key=value
    for l in lines:
        if "=" not in l:
            continue
        k, v = l.strip().split("=", 1)

        try:
            v = float(v)
        except:
            continue

        if k == "camera.width":
            cam["width"] = int(v)
        elif k == "camera.height":
            cam["height"] = int(v)
        elif k == "disp.scale":
            disp_params["scale"] = v
        elif k == "disp.offset":
            disp_params["offset"] = v
        elif k == "disp.inv":
            disp_params["invalid"] = v
        elif k == "t":
            disp_params["baseline_m"] = v

    disp_px, conf, rgb = converter.load_images_to_pxl(
        path_disp_img, path_conf_img, path_left_img
    )

    for thr in [0, 32, 64]:
        data_dict = converter.disparity_to_uois_dict(
            disp_px,
            rgb,
            cam,
            disp_params,
            conf=conf,
            conf_thr=thr
        )
        out_path = output_root / f"{base}_conf={thr}.npy"
        converter.save_uois_npy(data_dict, out_path)

    vis_out_path = output_root / f"{base}_disp_visable.png"
    converter.make_disp_visable(path_disp_img, vis_out_path)

    rgb_vis_out_path = output_root / f"{base}_disp_visable_rgb.png"
    converter.save_rgb_disp_overlay(
        path_disp_img, path_left_img, rgb_vis_out_path, alpha=0.5
    )

    print(f"done: {root.name}")














# 
# ocid_testdata_path = "test_ocid_npy/OCID_image_0.npy"
# ocid_data = np.load(ocid_testdata_path, allow_pickle=True).item()
# 
# # --- helper ---
# def print_array_stats(name, arr):
#     arr = np.asarray(arr)
#     print(f"\n{name}")
#     print(f"  shape: {arr.shape}")
#     print(f"  dtype: {arr.dtype}")
# 
#     if np.issubdtype(arr.dtype, np.number):
#         finite = np.isfinite(arr)
#         print(f"  finite ratio: {finite.mean():.4f}")
# 
#         if finite.any():
#             vals = arr[finite]
#             print(f"  min/max: {vals.min():.6f} / {vals.max():.6f}")
#             print(f"  mean/std: {vals.mean():.6f} / {vals.std():.6f}")
#             print(f"  percentiles [1,5,50,95,99]: {np.percentile(vals, [1,5,50,95,99])}")
# 
# def compare_rgb(my_rgb, ref_rgb):
#     print("\n=== RGB comparison ===")
#     print_array_stats("my rgb", my_rgb)
#     print_array_stats("ocid rgb", ref_rgb)
# 
#     my_rgb_f = my_rgb.astype(np.float32)
#     ref_rgb_f = ref_rgb.astype(np.float32)
# 
#     print("  my channel means:", my_rgb_f.mean(axis=(0, 1)))
#     print("  ocid channel means:", ref_rgb_f.mean(axis=(0, 1)))
#     print("  my channel stds :", my_rgb_f.std(axis=(0, 1)))
#     print("  ocid channel stds :", ref_rgb_f.std(axis=(0, 1)))
# 
# def compare_xyz(my_xyz, ref_xyz):
#     print("\n=== XYZ comparison ===")
#     print_array_stats("my xyz", my_xyz)
#     print_array_stats("ocid xyz", ref_xyz)
# 
#     my_z = my_xyz[:, :, 2]
#     ref_z = ref_xyz[:, :, 2]
# 
#     my_valid = np.isfinite(my_z) & (my_z > 0)
#     ref_valid = np.isfinite(ref_z) & (ref_z > 0)
# 
#     print(f"  my z valid ratio: {my_valid.mean():.4f}")
#     print(f"  ocid z valid ratio: {ref_valid.mean():.4f}")
# 
#     if my_valid.any():
#         print(f"  my z percentiles [1,5,50,95,99]: {np.percentile(my_z[my_valid], [1,5,50,95,99])}")
#     if ref_valid.any():
#         print(f"  ocid z percentiles [1,5,50,95,99]: {np.percentile(ref_z[ref_valid], [1,5,50,95,99])}")
# 
# def compare_label(my_label, ref_label):
#     print("\n=== LABEL comparison ===")
#     print_array_stats("my label", my_label)
#     print_array_stats("ocid label", ref_label)
# 
#     my_ids, my_counts = np.unique(my_label, return_counts=True)
#     ref_ids, ref_counts = np.unique(ref_label, return_counts=True)
# 
#     print("  my unique labels:", list(zip(my_ids.tolist(), my_counts.tolist()))[:20])
#     print("  ocid unique labels:", list(zip(ref_ids.tolist(), ref_counts.tolist()))[:20])
# 
#     print(f"  my foreground ratio: {(my_label != 0).mean():.4f}")
#     print(f"  ocid foreground ratio: {(ref_label != 0).mean():.4f}")
# 
# 
# # --- compare ---
# print("\n==============================")
# print("COMPARE MY UOIS INPUT VS OCID")
# print("==============================")
# 
# compare_rgb(data_dict["rgb"], ocid_data["rgb"])
# compare_xyz(data_dict["xyz"], ocid_data["xyz"])
# compare_label(data_dict["label"], ocid_data["label"])
# 