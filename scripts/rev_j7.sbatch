#!/bin/bash
#SBATCH --account=<your-allocation>
#SBATCH -J rev-j7
#SBATCH --gpus=h100:1 -c 16 --mem=120G -t 02:59:00
#SBATCH --array=0-4
#SBATCH -o out/rev_j7_%A_%a.log
# J7 Southern Hemisphere gain sweep, J1 protocol with the SH in-box controls.
source "$SLURM_SUBMIT_DIR/scripts/rev_env.sh"   # submit from the repository root
STORMS=(winston2016 harold2020 marcus2018 veronica2019 fantala2016)
export SKILL_STORMS=skill_sh_storms MECH_ALLOW_OVERLAP=1 \
       MECH_NAME=sh_suff_ctl MECH_GAINS=0,0.5,1,1.25,1.5,2,2.5,3 MECH_FIELDS=1 \
       MECH_CTL_ALL_GAINS=1 MECH_PRE=1 \
       MECH_INBOX_CTL=results/inbox_control_skill_sh_storms.json
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
$PY -m graphcast_sae.storms.skill_conv_run ${STORMS[$SLURM_ARRAY_TASK_ID]}
echo "rc=$?"
