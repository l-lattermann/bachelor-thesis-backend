from pathlib import Path
from time import time
import yaml

import uois.src.segmentation as segmentation
from data_io import load_uois_input  

CONFIG_PATH = Path("/app/config.yaml")
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

dsn_config = CFG["uois"]["dsn_config"]
rrn_config = CFG["uois"]["rrn_config"]
uois3d_config = dict(CFG["uois"]["uois3d_config"])

checkpoint_dir = Path(CFG["uois"]["checkpoint_dir"])
dsn_ckpt = checkpoint_dir / CFG["uois"]["checkpoints"]["dsn"]
rrn_ckpt = checkpoint_dir / CFG["uois"]["checkpoints"]["rrn"]

output_dir = Path(CFG["paths"]["pipeline_file_share"])
_model = None


def build_model() -> segmentation.UOISNet3D:
    """
    Build the UOIS 3D model with DSN and RRN checkpoints.
    """
    config = dict(uois3d_config)
    config["final_close_morphology"] = "TableTop_v5" in str(rrn_ckpt)

    model = segmentation.UOISNet3D(
        config,
        str(dsn_ckpt),
        dsn_config,
        str(rrn_ckpt),
        rrn_config,
    )
    return model


def load_model() -> segmentation.UOISNet3D:
    """
    Lazy load the global UOIS model.
    """
    global _model
    if _model is None:
        _model = build_model()
    return _model


def run_uois_on_npz(npz_path: str) -> dict:
    """
    Run UOIS segmentation on a given .npz file.
    """
    model = load_model()
    data, batch, rgb, xyz, label = load_uois_input(npz_path)

    start = time()
    fg_masks, center_offsets, initial_masks, seg_masks = model.run_on_batch(batch)
    elapsed = time() - start

    seg = seg_masks.cpu().numpy()[0]
    fg = fg_masks.cpu().numpy()[0]
    center = center_offsets.cpu().numpy().transpose(0, 2, 3, 1)[0]
    initial = initial_masks.cpu().numpy()[0]

    return {
        "data": data,
        "seg": seg,
        "fg_masks": fg,
        "center_offsets": center,
        "initial_masks": initial,
        "time": elapsed,
        "rgb": rgb,
        "xyz": xyz,
        "label": label,
        "batch": batch,
        "dsn_config": dsn_config,
    }