#!/bin/bash
cd "$(dirname "$0")/.."
echo "$(date -u +%H:%M:%S) START four grid-locked features vs matched controls (10 arms, 6 ICs, 48 h)"
GR_GROUPS=results/matched_controls_six.json \
GR_ARMS=f407,ctrl_f407,f2585,ctrl_f2585,f2989,ctrl_f2989,f3535,ctrl_f3535,floor \
GR_ICS=2020-01-10,2020-03-12,2020-05-11,2020-07-10,2020-09-08,2020-11-07 \
GR_S=8 GR_OUT=results/fs_matched_rmse_six.npy GR_STATUS=results/matched_rmse_six_status.txt \
FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0 \
.venv-jax/bin/python -m graphcast_sae.gridlock.global_rmse_ablate
echo "$(date -u +%H:%M:%S) rc=$?"
