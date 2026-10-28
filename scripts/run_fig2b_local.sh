#!/bin/bash
# Fig. 2b on one local GPU: convection feature 2401 alone on Ida and Haishen, the J1 gains
# (alpha = 0, 0.5, 1.25, 1.5, 2, 2.5, 3) plus the unmodified forecast, deterministic GPU ops.
#   GRAPHCAST_PARAMS=<checkpoint.npz> GRAPHCAST_ASSETS=<dir with stats/> PY=<jax python> \
#       bash scripts/run_fig2b_local.sh
cd "${REPO:-$(dirname "$0")/..}"   # REPO lets a copy of this script run from elsewhere
mkdir -p out
export FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
# the 0.25 deg / 37-level forward needs ~26 GB; on a smaller card oversubscribe via CUDA unified memory
export TF_FORCE_UNIFIED_MEMORY=${TF_FORCE_UNIFIED_MEMORY:-1} XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-3.0}
G=0,0.5,1.25,1.5,2,2.5,3
ARMS=baseline
for g in ${G//,/ }; do ARMS=$ARMS,gain-$g; done
MECH_ALLOW_OVERLAP=1 MECH_NAME=fig2_2401 MECH_FEATS=2401 MECH_GAINS=$G MECH_ARMS=$ARMS \
  ${PY:-python} -m graphcast_sae.storms.skill_conv_run ida2021 haishen2020
echo "rc=$?"
