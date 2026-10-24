# Common environment for the camera-ready revision jobs on Rorqual (source, don't run).
# Alliance's PYTHONPATH injects a _manylinux override and its own site-packages; the venv is a
# plain PyPI build of env/requirements-jax-freeze.txt, so keep them out.
unset PYTHONPATH PIP_CONFIG_FILE
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export GC_SCRATCH=${GC_SCRATCH:?set GC_SCRATCH to the scratch root (graphcast_sae/paths.py)}
export FS_ERA5_LOCAL=$GC_SCRATCH/era5_wb2.zarr   # graphcast_sae/extraction/stage_era5_local.py
export FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export CC_CLUSTER=${CC_CLUSTER:-rorqual}
PY=$REPO/.venv-jax/bin/python
cd $REPO
echo "$(date -u +%FT%TZ) host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
