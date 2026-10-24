#!/bin/bash
#SBATCH --account=<your-allocation>
#SBATCH -J rev-fig2
#SBATCH --gpus=h100:1 -c 16 --mem=120G -t 01:30:00
#SBATCH --array=0-13
#SBATCH -o out/rev_fig2_%A_%a.log
# Single-feature gain sweeps for Fig. 2b (convection, 2401) and 2c (low-level vorticity, 3316).
# Same operator, gains and determinism as J1 (rev_j1.sbatch). Each feature has its own control,
# matched per storm on core activation (graphcast_sae/storms/core_control_single.py ->
# results/core_control_<feature>.json), run at every gain. Tasks 0-6: 2401, tasks 7-13: 3316,
# one storm per task, every arm of a storm in one invocation. Submit from the repository root.
source "$SLURM_SUBMIT_DIR/scripts/rev_env.sh"
STORMS=(ida2021 michael2018 haishen2020 goni2020 haiyan2013 patricia2015 wilma2005)
FEATS=(2401 3316)
FEAT=${FEATS[$((SLURM_ARRAY_TASK_ID / 7))]}
STORM=${STORMS[$((SLURM_ARRAY_TASK_ID % 7))]}
G=0,0.5,1.25,1.5,2,2.5,3
ARMS=baseline
for g in ${G//,/ }; do ARMS=$ARMS,gain-$g,rand-gain-$g; done
export MECH_ALLOW_OVERLAP=1 MECH_NAME=fig2_$FEAT MECH_FEATS=$FEAT MECH_GAINS=$G MECH_CTL_ALL_GAINS=1 \
       MECH_ARMS=$ARMS MECH_INBOX_CTL=results/core_control_$FEAT.json
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
echo "feature $FEAT storm $STORM arms $ARMS"
$PY -m graphcast_sae.storms.skill_conv_run $STORM
echo "rc=$?"
