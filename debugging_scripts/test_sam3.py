import csv
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from huggingface_hub import login

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR / "sam3"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    login(token="REMOVED_HF_TOKEN")


from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMG_PATH = Path("/home/ubuntu/bachelor-thesis-backend/shared/pipeline_io/left.png")
OUT_DIR = IMG_PATH.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS_PER_CONFIG = 30
WARMUP_RUNS = 3

# This is NOT true "all objects in image" from pure image-only discovery.
# SAM3 here is used as promptable segmentation, so object listing depends on the prompt vocabulary you provide.
CANDIDATE_PROMPTS = [
    "table",
    "cloth",
    "screwdriver",
    "mouse",
    "battery",
    "can",
    "box",
    "open box",
    "sponge",
    "cable",
    "wire",
    "pack",
    "tissue pack",
    "carton",
    "lid",
    "container",
    "tray",
    "ring",
    "bracelet",
    "tool",
]


# Add or remove variants here.
# Keep build_kwargs limited to arguments that the actual function supports in your local repo.
# The script will filter unsupported keys automatically.
MODEL_CONFIGS = [
    {
        "name": "base_fp16_thr050",
        "build_kwargs": {},
        "processor_kwargs": {"confidence_threshold": 0.50},
        "use_autocast": True,
        "autocast_dtype": "float16",
    },
    {
        "name": "base_fp16_thr030",
        "build_kwargs": {},
        "processor_kwargs": {"confidence_threshold": 0.30},
        "use_autocast": True,
        "autocast_dtype": "float16",
    },
    {
        "name": "base_fp16_thr070",
        "build_kwargs": {},
        "processor_kwargs": {"confidence_threshold": 0.70},
        "use_autocast": True,
        "autocast_dtype": "float16",
    },
]

def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return value[:180]

def short_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))

def get_callable_params(fn):
    return list(inspect.signature(fn).parameters.keys())

def filter_kwargs(fn, kwargs):
    valid = set(get_callable_params(fn))
    return {k: v for k, v in kwargs.items() if k in valid}

def save_signature_report():
    report = {
        "build_sam3_image_model_signature": str(inspect.signature(build_sam3_image_model)),
        "sam3_processor_init_signature": str(inspect.signature(Sam3Processor.__init__)),
        "device": DEVICE,
        "img_path": str(IMG_PATH),
    }
    with open(OUT_DIR / "available_params.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

def sync_cuda():
    if DEVICE == "cuda":
        torch.cuda.synchronize()

def get_autocast_context(use_autocast: bool, autocast_dtype: str | None):
    if DEVICE != "cuda" or not use_autocast:
        return torch.autocast(device_type="cpu", enabled=False)

    if autocast_dtype == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if autocast_dtype == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    return torch.autocast(device_type="cuda", enabled=False)

def load_model_and_processor(config: dict):
    build_kwargs = filter_kwargs(build_sam3_image_model, config.get("build_kwargs", {}))
    processor_kwargs = filter_kwargs(Sam3Processor.__init__, config.get("processor_kwargs", {}))

    print(f"Loading config: {config['name']}")
    print("build_kwargs:", build_kwargs)
    print("processor_kwargs:", processor_kwargs)

    model = build_sam3_image_model(**build_kwargs)
    model = model.to(DEVICE)
    model.eval()

    processor = Sam3Processor(model, **processor_kwargs)
    return model, processor, build_kwargs, processor_kwargs

def extract_count(output: dict) -> int:
    if output is None:
        return 0

    for key in ("scores", "boxes", "masks"):
        value = output.get(key)
        if value is None:
            continue
        if hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])
        try:
            return len(value)
        except Exception:
            pass
    return 0

def extract_scores(output: dict):
    scores = output.get("scores")
    if scores is None:
        return []
    if torch.is_tensor(scores):
        return scores.detach().float().cpu().tolist()
    return list(scores)

def detect_objects(processor, image, config):
    found = []
    details = []

    with torch.inference_mode():
        with get_autocast_context(config["use_autocast"], config["autocast_dtype"]):
            base_state = processor.set_image(image)

            for prompt in CANDIDATE_PROMPTS:
                processor.reset_all_prompts(base_state)
                out = processor.set_text_prompt(state=base_state, prompt=prompt)
                count = extract_count(out)
                scores = extract_scores(out)

                info = {
                    "prompt": prompt,
                    "count": count,
                    "scores": scores,
                    "max_score": max(scores) if scores else None,
                }
                details.append(info)

                if count > 0:
                    found.append(prompt)

    return found, details

def run_single_prompt(processor, image, prompt, config):
    sync_cuda()
    t0 = time.perf_counter()

    with torch.inference_mode():
        with get_autocast_context(config["use_autocast"], config["autocast_dtype"]):
            state = processor.set_image(image)
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=prompt)

    sync_cuda()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return state, latency_ms

def save_overlay(image, state, filename_stem):
    plt.figure(figsize=(10, 8))
    plot_results(image, state)
    out_path = OUT_DIR / f"{filename_stem}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()
    return out_path

def summarize_latencies(values):
    if not values:
        return {}
    vals = sorted(values)
    n = len(vals)
    return {
        "n": n,
        "avg_ms": sum(vals) / n,
        "min_ms": vals[0],
        "max_ms": vals[-1],
        "p50_ms": vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2,
        "p90_ms": vals[min(n - 1, int(0.9 * (n - 1)))],
        "p95_ms": vals[min(n - 1, int(0.95 * (n - 1)))],
    }

def main():
    if not IMG_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")

    save_signature_report()

    image = Image.open(IMG_PATH).convert("RGB")

    csv_rows = []
    config_summaries = []

    for config in MODEL_CONFIGS:
        model, processor, build_kwargs, processor_kwargs = load_model_and_processor(config)

        found_prompts, discovery_details = detect_objects(processor, image, config)

        discovery_path = OUT_DIR / f"{sanitize_filename(config['name'])}_discovery.json"
        with open(discovery_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config_name": config["name"],
                    "build_kwargs": build_kwargs,
                    "processor_kwargs": processor_kwargs,
                    "found_prompts": found_prompts,
                    "discovery_details": discovery_details,
                },
                f,
                indent=2,
            )

        print(f"[{config['name']}] found prompts:", found_prompts if found_prompts else "none")

        prompts_for_benchmark = found_prompts if found_prompts else ["shoe"]

        # Warmup
        for i in range(WARMUP_RUNS):
            prompt = prompts_for_benchmark[i % len(prompts_for_benchmark)]
            _state, _lat = run_single_prompt(processor, image, prompt, config)

        latencies = []

        for i in range(RUNS_PER_CONFIG):
            prompt = prompts_for_benchmark[i % len(prompts_for_benchmark)]
            state, latency_ms = run_single_prompt(processor, image, prompt, config)
            latencies.append(latency_ms)

            scores = extract_scores(state)
            obj_count = extract_count(state)

            file_stem = sanitize_filename(
                f"{config['name']}_run{i+1:02d}_prompt-{prompt}_objs-{obj_count}_ms-{latency_ms:.2f}"
            )
            img_out = save_overlay(image, state, file_stem)

            csv_rows.append(
                {
                    "config_name": config["name"],
                    "run_index": i + 1,
                    "prompt": prompt,
                    "latency_ms": round(latency_ms, 4),
                    "object_count": obj_count,
                    "scores": short_json({"scores": scores}),
                    "image_output": str(img_out),
                    "build_kwargs": short_json(build_kwargs),
                    "processor_kwargs": short_json(processor_kwargs),
                    "use_autocast": config["use_autocast"],
                    "autocast_dtype": config["autocast_dtype"],
                }
            )

        summary = summarize_latencies(latencies)
        summary_record = {
            "config_name": config["name"],
            "build_kwargs": build_kwargs,
            "processor_kwargs": processor_kwargs,
            "use_autocast": config["use_autocast"],
            "autocast_dtype": config["autocast_dtype"],
            "found_prompts": found_prompts,
            **summary,
        }
        config_summaries.append(summary_record)

        del processor
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    csv_path = OUT_DIR / "benchmark_runs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        if csv_rows:
            writer.writeheader()
            writer.writerows(csv_rows)

    summary_path = OUT_DIR / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(config_summaries, f, indent=2)

    print("\n=== SUMMARY ===")
    for s in config_summaries:
        print(
            f"{s['config_name']}: "
            f"avg={s.get('avg_ms', float('nan')):.2f} ms, "
            f"p50={s.get('p50_ms', float('nan')):.2f} ms, "
            f"p95={s.get('p95_ms', float('nan')):.2f} ms, "
            f"min={s.get('min_ms', float('nan')):.2f} ms, "
            f"max={s.get('max_ms', float('nan')):.2f} ms, "
            f"found={s.get('found_prompts', [])}"
        )

    print("\nSaved to:")
    print(OUT_DIR)

if __name__ == "__main__":
    main()