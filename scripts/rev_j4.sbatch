#!/bin/bash
#SBATCH --account=<your-allocation>
#SBATCH -J rev-j4
#SBATCH --gpus=h100:1 -c 16 --mem=120G -t 02:00:00
#SBATCH --array=0-1
#SBATCH -o out/rev_j4_%A_%a.log
# J4 determinism replicate: baseline x5 and conv-normal x5 as separate arms, ONE invocation.
source "$SLURM_SUBMIT_DIR/scripts/rev_env.sh"   # submit from the repository root
STORMS=(ida2021 patricia2015)
export MECH_ALLOW_OVERLAP=1 MECH_NAME=determinism_j4 MECH_ARMS=baseline,conv-normal MECH_REPLICATES=5 \
       MECH_REPL_ARMS=baseline,conv-normal \
       MECH_INBOX_CTL=results/inbox_control_skill_conv_storms.json
$PY -m graphcast_sae.storms.skill_conv_run ${STORMS[$SLURM_ARRAY_TASK_ID]}
echo "rc=$?"
