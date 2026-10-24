#!/bin/bash
#SBATCH --account=<your-allocation>
#SBATCH -J rev-j1
#SBATCH --gpus=h100:1 -c 16 --mem=120G -t 02:59:00
#SBATCH --array=0-7
#SBATCH -o out/rev_j1_%A_%a.log
# J1 (+J3) controlled gain sweep. One task per STORM; every arm of a storm in one invocation
# (array parallelism is per storm, never per arm).
# Launch task 0 (Ida) alone first: it is the timing probe and sanity gates 2-3.
source "$SLURM_SUBMIT_DIR/scripts/rev_env.sh"   # submit from the repository root
STORMS=(ida2021 michael2018 haishen2020 goni2020 haiyan2013 patricia2015 wilma2005 nondev2013)
export MECH_ALLOW_OVERLAP=1 MECH_NAME=suff_ctl MECH_GAINS=0,0.5,1,1.25,1.5,2,2.5,3 MECH_FIELDS=1 \
       MECH_CTL_ALL_GAINS=1 MECH_PRE=1 \
       MECH_INBOX_CTL=results/inbox_control_skill_conv_storms.json
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
$PY -m graphcast_sae.storms.skill_conv_run ${STORMS[$SLURM_ARRAY_TASK_ID]}
echo "rc=$?"
