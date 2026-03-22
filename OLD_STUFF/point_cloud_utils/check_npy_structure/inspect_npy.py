#!/usr/bin/env python3
import argparse
import os
import numpy as np

def _dtype_shape(x):
    try:
        return f"dtype={x.dtype}, shape={x.shape}, ndim={x.ndim}"
    except Exception:
        return f"type={type(x)}"

def _stats(x, name, max_items=10):
    if not isinstance(x, np.ndarray):
        print(f"  {name}: (non-ndarray) {type(x)} -> {x}")
        return

    print(f"  {name}: {_dtype_shape(x)}")
    if x.size == 0:
        print("    empty")
        return

    # show a few values
    flat = x.ravel()
    sample = flat[:max_items]
    print(f"    head[{min(max_items, flat.size)}]: {sample}")

    # numeric stats
    if np.issubdtype(x.dtype, np.number):
        finite = np.isfinite(x) if np.issubdtype(x.dtype, np.floating) else np.ones(x.shape, dtype=bool)
        finite_count = int(finite.sum()) if finite.size else 0
        total = x.size
        print(f"    finite: {finite_count}/{total}")

        if np.issubdtype(x.dtype, np.floating) and finite_count > 0:
            xf = x[finite]
            print(f"    min/max: {xf.min()} / {xf.max()}")
            print(f"    mean/std: {xf.mean()} / {xf.std()}")
            nan_count = int(np.isnan(x).sum())
            inf_count = int(np.isinf(x).sum())
            print(f"    NaN/Inf: {nan_count} / {inf_count}")
        else:
            try:
                print(f"    min/max: {x.min()} / {x.max()}")
            except Exception:
                pass

def inspect_npy(path: str=None, array=None):
    if path != None:
        print(f"== Loading: {path}")
        arr = np.load(path, allow_pickle=True)
    else:
        arr = array

    # np.save(dict) -> 0-d object array, item() is dict
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        obj = arr.item()
    else:
        obj = arr

    print(f"== Top-level: {type(obj)}")
    if isinstance(obj, dict):
        print(f"== Dict keys ({len(obj)}): {list(obj.keys())}")
        for k in obj.keys():
            print(f"\n-- Key: {k}")
            _stats(obj[k], k)

        # common conventions quick checks
        if "rgb" in obj:
            rgb = obj["rgb"]
            if isinstance(rgb, np.ndarray):
                print("\n== RGB checks")
                print(f"  expected HxWx3 uint8; got {rgb.shape}, {rgb.dtype}")
        if "depth" in obj:
            depth = obj["depth"]
            if isinstance(depth, np.ndarray):
                print("\n== Depth checks")
                print(f"  dtype={depth.dtype}, shape={depth.shape}")
                if depth.dtype == np.uint16:
                    nz = (depth > 0).mean()
                    print(f"  nonzero fraction: {nz:.4f} (uint16 often mm)")
                elif np.issubdtype(depth.dtype, np.floating):
                    nz = (depth > 0).mean()
                    print(f"  nonzero fraction: {nz:.4f} (float often meters)")
        if "K" in obj:
            K = obj["K"]
            if isinstance(K, np.ndarray):
                print("\n== Intrinsics K")
                print(K)
        if "seg" in obj:
            seg = obj["seg"]
            if isinstance(seg, np.ndarray):
                print("\n== Seg checks")
                uniq = np.unique(seg)
                print(f"  unique labels (up to 50): {uniq[:50]} (count={len(uniq)})")

    elif isinstance(obj, np.ndarray):
        print(f"== Array: {_dtype_shape(obj)}")
        _stats(obj, "array")

        # heuristics for point clouds
        if obj.ndim == 2 and obj.shape[1] in (3, 4, 6, 7):
            print("\n== Heuristic: NxC point cloud")
            print("  columns likely:")
            if obj.shape[1] == 3:
                print("    [x, y, z]")
            elif obj.shape[1] == 4:
                print("    [x, y, z, w/intensity]")
            elif obj.shape[1] == 6:
                print("    [x, y, z, r, g, b] (r,g,b may be 0..255 or 0..1)")
            elif obj.shape[1] == 7:
                print("    [x, y, z, r, g, b, something]")
            if np.issubdtype(obj.dtype, np.floating):
                nan_rows = np.isnan(obj).any(axis=1).sum()
                inf_rows = np.isinf(obj).any(axis=1).sum()
                print(f"  rows with NaN: {nan_rows}, rows with Inf: {inf_rows}")
            # if rgb is float 0..1
            if obj.shape[1] >= 6:
                rgb = obj[:, 3:6]
                if np.issubdtype(rgb.dtype, np.floating):
                    print(f"  rgb float max={np.nanmax(rgb):.6f} (<=1 suggests normalized colors)")
                else:
                    print(f"  rgb int min/max={rgb.min()} / {rgb.max()}")

        # organized cloud guess: (H,W,3) etc
        if obj.ndim == 3 and obj.shape[2] in (3, 4, 6):
            print("\n== Heuristic: organized image-like array (HxWxC)")
            print(f"  H={obj.shape[0]}, W={obj.shape[1]}, C={obj.shape[2]}")

    else:
        print(f"== Unsupported top-level type: {type(obj)}")
        print(obj)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", help="Path to .npy file")
    args = ap.parse_args()

    if not os.path.exists(args.f):
        raise SystemExit(f"File not found: {args.f}")

    inspect_npy(args.f)

if __name__ == "__main__":
    main()