import time
import cv2
import numpy as np
import torch
from ultralytics import SAM


MODEL = None



def load_model(checkpoint_path: str, device: str = "cuda", warmup: bool = True):
    global MODEL

    MODEL = SAM(checkpoint_path).to(device)

    print("=== SAM LOAD ===")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("requested device:", device)

    try:
        p = next(MODEL.model.parameters())
        print("model parameter device:", p.device)
    except Exception as e:
        print("could not inspect model device:", e)

    if torch.cuda.is_available():
        print("torch cuda device count:", torch.cuda.device_count())
        print("torch current device:", torch.cuda.current_device())
        print("torch device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    if warmup:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for _ in range(2):
            _ = MODEL(dummy, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("warmup done")




def generate_masks(image: np.ndarray):
    if MODEL is None:
        raise RuntimeError("Model not loaded")

    print("image shape:", image.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    results = MODEL(image, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    masks = []
    for r in results:
        if r.masks is None:
            continue
        data = r.masks.data.cpu().numpy()
        print("masks in result:", len(data))
        masks.extend((m > 0.5).astype(np.uint8) for m in data)

    print(f"MODEL(...) time: {t1 - t0:.3f}s")
    print("num raw masks:", len(masks))
    return masks


def touches_border(mask: np.ndarray):
    return mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()


def filter_masks(masks, min_area=200, max_area_ratio=0.5):
    out = []
    for m in masks:
        area = int(m.sum())
        if area < min_area:
            continue
        if area / (m.shape[0] * m.shape[1]) > max_area_ratio:
            continue
        if touches_border(m):
            continue
        out.append(m.astype(np.uint8))
    return out


def sort_masks_for_output(masks):
    return sorted(masks, key=lambda m: int(m.sum()), reverse=True)


def resize_masks_to_shape(masks, target_hw):
    h, w = target_hw
    return [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in masks]


def masks_to_segmentation(masks, image_shape):
    h, w = image_shape[:2]
    seg = np.zeros((h, w), dtype=np.int32)
    for i, m in enumerate(masks, start=1):
        seg[m.astype(bool)] = i
    return seg


def run_sam_pipeline(image: np.ndarray, target_hw, min_area=200, max_area_ratio=0.5):
    t0 = time.perf_counter()

    t1 = time.perf_counter()
    raw = generate_masks(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    filtered = filter_masks(raw, min_area, max_area_ratio)
    t3 = time.perf_counter()

    ordered = sort_masks_for_output(filtered)
    t4 = time.perf_counter()

    resized = resize_masks_to_shape(ordered, target_hw)
    t5 = time.perf_counter()

    seg = masks_to_segmentation(resized, (*target_hw, 3))
    t6 = time.perf_counter()

    print("\n=== SAM TIMING ===")
    print(f"inference: {(t2 - t1):.3f}s")
    print(f"filter:    {(t3 - t2):.3f}s")
    print(f"sort:      {(t4 - t3):.3f}s")
    print(f"resize:    {(t5 - t4):.3f}s")
    print(f"seg:       {(t6 - t5):.3f}s")
    print(f"total:     {(t6 - t0):.3f}s")

    return {
        "raw_masks": raw,
        "filtered_masks": filtered,
        "ordered_masks_fullres": ordered,
        "ordered_masks_resized": resized,
        "seg": seg,
    }