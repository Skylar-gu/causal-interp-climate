#!/bin/bash
#SBATCH --account=<your-allocation>
#SBATCH -J rev-g2det
#SBATCH --gpus=h100:1 -c 16 --mem=120G -t 01:00:00
#SBATCH -o out/rev_gate2_det_%j.log
# Gate-2 diagnostic (J1 Ida run 23047447: gain-1 != baseline by 0.11 hPa). Same arms with XLA
# deterministic ops: if gain-1 == baseline bit-exactly here, the patch is a true no-op and the
# J1 discrepancy is GPU scatter nondeterminism, which J4 then sizes.
source "$SLURM_SUBMIT_DIR/scripts/rev_env.sh"   # submit from the repository root
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export MECH_ALLOW_OVERLAP=1 MECH_NAME=gate2_det MECH_GAINS=0,1 MECH_ARMS=baseline,gain-0,gain-1 \
       MECH_REPLICATES=2 MECH_REPL_ARMS=baseline \
       MECH_INBOX_CTL=results/inbox_control_skill_conv_storms.json
$PY -m graphcast_sae.storms.skill_conv_run ida2021
echo "rc=$?"
