from pathlib import Path

import numpy as np
import yaml

import service_core_standalone as cs
from grasp_selection import process_contact_graspnet_result
from projection_utils import save_projected_grasp_overlay
from contact_graspnet.contact_graspnet.visualization_utils import (
    show_image,
    visualize_grasps,
)

SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_DIR.parents[1]

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
INPUT_PATH = PROJECT_ROOT / "shared" / "pipeline_io" / "uois_output.npz"
OUTPUT_DIR = PROJECT_ROOT / "shared" / "pipeline_io" / "cgn_standalone_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def print_selected_grasps(result: dict):
    print("\n=== SELECTED GRASPS DEBUG ===", flush=True)

    for key in result["pred_grasps_cam"]:
        grasps_k = np.asarray(result["pred_grasps_cam"][key])
        scores_k = np.atleast_1d(result["scores"][key])
        contacts_k = np.atleast_2d(result["contact_pts"][key])
        openings_k = np.atleast_1d(result["gripper_openings"][key])

        if len(grasps_k) == 0:
            print(f"[key={key}] no grasps", flush=True)
            continue

        for i in range(len(grasps_k)):
            x, y, z = contacts_k[i]
            score = scores_k[i]
            opening = openings_k[i] if len(openings_k) > 0 else -1.0

            print(
                f"[key={key}] grasp {i + 1}: "
                f"score={score:.3f}, "
                f"opening={opening:.3f} m, "
                f"contact=({x:.3f}, {y:.3f}, {z:.3f})",
                flush=True,
            )

    print("================================\n", flush=True)


def main():
    cfg = load_config()

    print("==========================", flush=True)
    print("=== CGN TEST RUN START ===", flush=True)
    print("==========================", flush=True)
    print(f"Config: {CONFIG_PATH}", flush=True)
    print(f"Input:  {INPUT_PATH}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    cs.load_model()
    result = cs.run_contact_graspnet(str(INPUT_PATH))

    print("Befor selection")
    for key in result["gripper_openings"]:
        arr = np.asarray(result["gripper_openings"][key])
        print("SELECTED key:", key)
        print("SELECTED openings:", arr)

    if cfg.get("project", {}).get("debug", False):
        show_image(result.get("rgb"), result.get("segmap"))
        visualize_grasps(
            result["pc_full"],
            result["pred_grasps_cam"],
            result["scores"],
            plot_opencv_cam=True,
            pc_colors=result.get("pc_colors"),
            gripper_openings=None,
            gripper_width=cfg["contact_graspnet"]["selection"].get("gripper_width", 0.08),
        )
        print("[DEBUG] Raw CGN inference visualization complete.", flush=True)

    result = process_contact_graspnet_result(
        result=result,
        sel_cfg=cfg["contact_graspnet"]["selection"],
    )
    print("After selection")
    for key in result["gripper_openings"]:
        arr = np.asarray(result["gripper_openings"][key])
        print("SELECTED key:", key)
        print("SELECTED openings:", arr)

    print_selected_grasps(result)

    overlay_path = save_projected_grasp_overlay(
        rgb=result.get("rgb"),
        K=result.get("K"),
        pred_grasps_cam=result["pred_grasps_cam"],
        scores=result["scores"],
        gripper_openings=result["gripper_openings"],
        output_path=str(OUTPUT_DIR / "cgn_output.png"),
        debug=cfg.get("project", {}).get("debug", False),
    )

    npz_out_path = OUTPUT_DIR / "cgn_result.npz"
    np.savez(
        npz_out_path,
        pred_grasps_cam=result["pred_grasps_cam"],
        scores=result["scores"],
        contact_pts=result["contact_pts"],
        gripper_openings=result["gripper_openings"],
    )

    print("Saved files:", flush=True)
    print(f"  Overlay: {overlay_path}", flush=True)
    print(f"  Result:  {npz_out_path}", flush=True)
    print("\n=== CGN TEST RUN DONE ===", flush=True)


if __name__ == "__main__":
    main()