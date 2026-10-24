#!/bin/bash
#SBATCH --account=<your-allocation>
#SBATCH -J rev-2401
#SBATCH --gpus=h100:1 -c 16 --mem=120G -t 01:30:00
#SBATCH --array=0-6
#SBATCH -o out/rev_2401_%A_%a.log
# Gain sweep for the single convection feature 2401 (Fig. 2b and the convection numbers in the
# text). Same operator, gains, determinism and in-box matching as J1 (rev_j1.sbatch), with the
# control drawn for 2401 alone (results/inbox_control_2401.json) and run at alpha = 0, 2, 3.
# One task per storm; every arm of a storm in one invocation. Submit from the repository root.
source "$SLURM_SUBMIT_DIR/scripts/rev_env.sh"
STORMS=(ida2021 michael2018 haishen2020 goni2020 haiyan2013 patricia2015 wilma2005)
export MECH_ALLOW_OVERLAP=1 MECH_NAME=suff_2401 MECH_FEATS=2401 \
       MECH_GAINS=0,0.5,1.25,1.5,2,2.5,3 MECH_CTL_ALL_GAINS=1 \
       MECH_ARMS=baseline,gain-0,gain-0.5,gain-1.25,gain-1.5,gain-2,gain-2.5,gain-3,rand-gain-0,rand-gain-2,rand-gain-3 \
       MECH_INBOX_CTL=results/inbox_control_2401.json
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
$PY -m graphcast_sae.storms.skill_conv_run ${STORMS[$SLURM_ARRAY_TASK_ID]}
echo "rc=$?"
