import numpy as np
import torch
from huggingface_hub import login

MODEL = None
PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    hf_token: str = None,
    device: str = "cuda",
    confidence_threshold: float = 0.5,
    warmup: bool = False,
    warmup_image=None,
):
    global MODEL, PROCESSOR, DEVICE

    if MODEL is not None and PROCESSOR is not None:
        return

    DEVICE = device

    if hf_token:
        login(token=hf_token)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    MODEL = build_sam3_image_model()
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    PROCESSOR = Sam3Processor(MODEL, confidence_threshold=confidence_threshold)

    if warmup and warmup_image is not None:
        _ = generate_masks(warmup_image, "object", confidence_threshold)
        print("warmup done")


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _extract(output: dict):
    masks = _to_numpy(output.get("masks"))
    boxes = _to_numpy(output.get("boxes"))
    scores = _to_numpy(output.get("scores"))

    if masks is None:
        return [], boxes, scores

    masks = np.asarray(masks)[:, 0, ...]  # (N,1,H,W) -> (N,H,W)

    out_masks = [(m > 0.5).astype(np.uint8) for m in masks]

    return out_masks, boxes, scores


def generate_masks(image, text_prompt: str, confidence_threshold: float = None):
    global MODEL, PROCESSOR

    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("Model not loaded")

    if confidence_threshold is not None:
        PROCESSOR.confidence_threshold = confidence_threshold

    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                state = PROCESSOR.set_image(image)
                PROCESSOR.reset_all_prompts(state)
                output = PROCESSOR.set_text_prompt(state=state, prompt=text_prompt)
        else:
            state = PROCESSOR.set_image(image)
            PROCESSOR.reset_all_prompts(state)
            output = PROCESSOR.set_text_prompt(state=state, prompt=text_prompt)

    masks, boxes, scores = _extract(output)

    return {
        "state": state,
        "masks": masks,
        "boxes": boxes,
        "scores": scores,
        "mask_count": len(masks),
    }