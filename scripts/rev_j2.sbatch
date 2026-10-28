#!/bin/bash
#SBATCH --account=<your-allocation>
#SBATCH -J rev-j2
#SBATCH --gpus=h100:1 -c 16 --mem=120G -t 02:59:00
#SBATCH --array=0-7
#SBATCH -o out/rev_j2_%A_%a.log
# J2 direction baselines (prereg amendment 2): SAE direction vs 5 random and 1 diff-of-means
# direction at matched per-node magnitude, g in {1.5,2,3}, all arms of a storm in one invocation.
source "$SLURM_SUBMIT_DIR/scripts/rev_env.sh"   # submit from the repository root
STORMS=(ida2021 michael2018 haishen2020 goni2020 haiyan2013 patricia2015 wilma2005 nondev2013)
export MECH_ALLOW_OVERLAP=1 MECH_NAME=dirs_j2 MECH_GAINS=1.5,2,3 MECH_DIRS=1
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
$PY -m graphcast_sae.storms.skill_conv_run ${STORMS[$SLURM_ARRAY_TASK_ID]}
echo "rc=$?"
