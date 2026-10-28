#!/bin/bash
cd "$(dirname "$0")/.."
export FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false
for deg in 45 135 270; do
  roll=$((deg*4))
  echo "=== ${deg} deg (roll ${roll}) $(date)"
  RA_ROLL=$roll RA_OUT=results/fs_rotation_all_${deg}.npy .venv-jax/bin/python -m graphcast_sae.gridlock.rotation_all 2>&1 | grep -v -i "warning\|^WARNING\|xla_bridge"
  echo "=== rc=${PIPESTATUS[0]}"
done
echo ALLDONE
