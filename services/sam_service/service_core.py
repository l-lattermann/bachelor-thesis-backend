import time
import cv2
import numpy as np
import torch
from ultralytics import SAM


MODEL = None



def load_model(checkpoint_path: str, device: str = "cuda", warmup: bool = True):
    global MODEL

    MODEL = SAM(checkpoint_path).to(device)

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
    global MODEL
    
    if MODEL is None:
        raise RuntimeError("Model not loaded")

    results = MODEL(image, verbose=False)

    masks = []
    for r in results:
        if r.masks is None:
            continue
        data = r.masks.data.cpu().numpy()
        print("masks in result:", len(data))
        masks.extend((m > 0.5).astype(np.uint8) for m in data)

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
    raw = generate_masks(image) 
    filtered = filter_masks(raw, min_area, max_area_ratio)
    ordered = sort_masks_for_output(filtered)
    resized = resize_masks_to_shape(ordered, target_hw)
    seg = masks_to_segmentation(resized, (*target_hw, 3))

    return {
        "raw_masks": raw,
        "filtered_masks": filtered,
        "ordered_masks_fullres": ordered,
        "ordered_masks_resized": resized,
        "seg": seg,
    }