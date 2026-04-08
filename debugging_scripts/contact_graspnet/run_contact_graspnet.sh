#!/bin/bash

set -e

BASE_DIR=/home/ubuntu/bachelor-thesis-backend/debugging_scripts/contact_graspnet/
NP_PATH=/home/ubuntu/bachelor-thesis-backend/shared/pipeline_io/sam_segmented_output.npz
CKPT_DIR=/home/ubuntu/bachelor-thesis-backend/debugging_scripts/contact_graspnet/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001

cd "$BASE_DIR"


echo "Running Contact-GraspNet..."
echo "Checkpoint: $CKPT_DIR"
echo "Input: $NP_PATH"


python contact_graspnet/inference.py \
  --ckpt_dir "$CKPT_DIR" \
  --np_path "$NP_PATH" \
  --local_regions \
  --filter_grasps \
  --segmap_id 1


