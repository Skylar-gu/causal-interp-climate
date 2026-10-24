#!/bin/bash
#SBATCH --account=<your-allocation>
#SBATCH -J rev-j6
#SBATCH --gpus=h100:1 -c 16 --mem=120G -t 02:00:00
#SBATCH -o out/rev_j6_%j.log
# J6: feature 2954 zero vs mean ablation, against its matched control, one invocation.
source "$SLURM_SUBMIT_DIR/scripts/rev_env.sh"   # submit from the repository root
export GR_GROUPS=results/j6_groups.json GR_ARMS=f2954,ctrl_f2954,mean_f2954,ctrl_mean_f2954 \
       GR_ICS=2020-01-10,2020-03-12,2020-05-11,2020-07-10,2020-09-08,2020-11-07 \
       GR_S=8 GR_OUT=results/fs_j6_ablation_type.npy GR_STATUS=results/j6_status.txt \
       XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
$PY -m graphcast_sae.gridlock.global_rmse_ablate
echo "rc=$?"
