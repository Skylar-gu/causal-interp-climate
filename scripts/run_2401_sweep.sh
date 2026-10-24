#!/bin/bash
# Gain sweep for the single convection feature 2401, same arms as J1 (scripts/rev_j1.sbatch):
# against the in-box control matched to 2401 (results/inbox_control_2401.json),
# deterministic GPU ops, the seven developing storms in one invocation. Single GPU, run from anywhere.
#   GRAPHCAST_PARAMS=<checkpoint.npz> GRAPHCAST_ASSETS=<release dir with stats/> PY=<jax python> \
#       bash scripts/run_2401_sweep.sh
cd "$(dirname "$0")/.."
mkdir -p out
export FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
# The 0.25 deg / 37-level forward needs ~26 GB; on a smaller card let JAX oversubscribe through CUDA
# unified memory (spills to host RAM; slower, same numbers). Harmless on a card that fits.
export TF_FORCE_UNIFIED_MEMORY=${TF_FORCE_UNIFIED_MEMORY:-1} XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-3.0}
# 11 arms per storm: the unsteered baseline, seven gains, and the in-box control at alpha = 0, 2, 3
# (gain-1 is the baseline itself and is not rerun). Seven developing storms; no non-developer.
export MECH_ALLOW_OVERLAP=1 MECH_NAME=suff_2401 MECH_FEATS=2401 \
       MECH_GAINS=0,0.5,1.25,1.5,2,2.5,3 MECH_CTL_ALL_GAINS=1 \
       MECH_ARMS=baseline,gain-0,gain-0.5,gain-1.25,gain-1.5,gain-2,gain-2.5,gain-3,rand-gain-0,rand-gain-2,rand-gain-3 \
       MECH_INBOX_CTL=results/inbox_control_2401.json
${PY:-python} -m graphcast_sae.storms.skill_conv_run ida2021 michael2018 haishen2020 goni2020 haiyan2013 patricia2015 wilma2005 \
    2>&1 | tee -a out/suff_2401.log
echo "rc=${PIPESTATUS[0]}"
