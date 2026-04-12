#!/bin/bash

set -e

BASE_DIR=/home/ubuntu/bachelor-thesis-backend/debugging_scripts/contact_graspnet/
NPZ_PATH=/home/ubuntu/bachelor-thesis-backend/shared/debug/cgn/selected_grasps.npz

cd "$BASE_DIR"

echo "Visualizing saved CGN output..."
echo "Input: $NPZ_PATH"

ls -lh $NPZ_PATH

python contact_graspnet/show_selected_graps.py \
  --npz_path "$NPZ_PATH"