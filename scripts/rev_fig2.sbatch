#!/bin/bash
#SBATCH --account=<your-allocation>
#SBATCH -J rev-fig2
#SBATCH --gpus=h100:1 -c 16 --mem=120G -t 01:15:00
#SBATCH --array=0-55
#SBATCH -o out/rev_fig2_%A_%a.log
# Single-feature gain sweeps for Fig. 2b (convection, 2401) and 2c (low-level vorticity, 3316).
# Same operator, gains and determinism as J1 (rev_j1.sbatch). Each feature has three controls,
# matched per storm on core activation (graphcast_sae/storms/core_control_single.py ->
# results/core_control_<feature>_c<k>.json); each control is scaled on its own at every gain and
# the control curve is their mean. Task = feature x storm x run, run 0 = the feature, runs 1-3 =
# controls c0-c2; every run carries the baseline so each is differenced within itself.
# Submit from the repository root.
source "$SLURM_SUBMIT_DIR/scripts/rev_env.sh"
STORMS=(ida2021 michael2018 haishen2020 goni2020 haiyan2013 patricia2015 wilma2005)
FEATS=(2401 3316)
T=$SLURM_ARRAY_TASK_ID
FEAT=${FEATS[$((T / 28))]}
STORM=${STORMS[$((T % 28 / 4))]}
RUN=$((T % 4))
G=0,0.5,1.25,1.5,2,2.5,3
if [ $RUN -eq 0 ]; then
  NAME=fig2_$FEAT; PREFIX=gain-; CTL=results/core_control_${FEAT}_c0.json
else
  NAME=fig2_${FEAT}_c$((RUN - 1)); PREFIX=rand-gain-; CTL=results/core_control_${FEAT}_c$((RUN - 1)).json
fi
ARMS=baseline
for g in ${G//,/ }; do ARMS=$ARMS,$PREFIX$g; done
export MECH_ALLOW_OVERLAP=1 MECH_NAME=$NAME MECH_FEATS=$FEAT MECH_GAINS=$G MECH_CTL_ALL_GAINS=1 \
       MECH_ARMS=$ARMS MECH_INBOX_CTL=$CTL
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
echo "feature $FEAT storm $STORM run $NAME arms $ARMS"
$PY -m graphcast_sae.storms.skill_conv_run $STORM
echo "rc=$?"
