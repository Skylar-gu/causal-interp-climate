#!/bin/bash
# One-time setup for the GPU jobs on an Alliance cluster (Rorqual). Run on a LOGIN node from the
# repository root (compute nodes have no internet):
#     GC_SCRATCH=/lustre10/scratch/$USER/gc_flagship bash scripts/setup_rorqual.sh
# Builds .venv-jax (Python 3.11 + DeepMind GraphCast at the pinned commit + JAX), downloads the
# GraphCast 0.25 deg / 37-level checkpoint and its normalisation stats, and stages the ERA5 frames
# the storm jobs read. Every step is skipped if its output already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GC_SCRATCH:?set GC_SCRATCH, e.g. /lustre10/scratch/$USER/gc_flagship}"
mkdir -p "$GC_SCRATCH"
# Alliance's PYTHONPATH and pip config point at its own wheelhouse; this env is plain PyPI.
unset PYTHONPATH PIP_CONFIG_FILE

# 1. Python 3.11 (a private miniforge; the cluster modules are not used)
MF=${MF:-$GC_SCRATCH/miniforge3}
if [ ! -x "$MF/envs/py311/bin/python" ]; then
  [ -x "$MF/bin/conda" ] || { curl -L -o /tmp/mf.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/mf.sh -b -p "$MF"; }
  "$MF/bin/conda" create -y -q -p "$MF/envs/py311" python=3.11
fi

# 2. GraphCast + JAX env (cartopy is plotting-only and does not build here: skipped)
if [ ! -x .venv-jax/bin/python ]; then
  "$MF/envs/py311/bin/python" -m venv .venv-jax
  .venv-jax/bin/pip install -q --upgrade pip
  grep -v '^cartopy' env/requirements-jax.txt > /tmp/req-jax.txt
  .venv-jax/bin/pip install -q -r /tmp/req-jax.txt
  .venv-jax/bin/pip install -q "jax[cuda12]==0.10.2"
fi
.venv-jax/bin/python -c "import graphcast, jax, haiku; print('graphcast env OK, jax', jax.__version__)"

# 3. GraphCast checkpoint + normalisation stats (public, no auth), at the paths graphcast_sae/paths.py expects
B=https://storage.googleapis.com/dm_graphcast/graphcast
CK=graphcast_sae/weights/graphcast_flagship_0p25_37lev.npz
[ -s "$CK" ] || curl -L -o "$CK" \
  "$B/params/GraphCast%20-%20ERA5%201979-2017%20-%20resolution%200.25%20-%20pressure%20levels%2037%20-%20mesh%202to6%20-%20precipitation%20input%20and%20output.npz"
echo "c3ed5ba70a44691d52a2b3cb6771a88f  $CK" | md5sum -c -
mkdir -p assets/stats
for f in mean_by_level stddev_by_level diffs_stddev_by_level; do
  [ -s assets/stats/$f.nc ] || curl -L -o assets/stats/$f.nc "$B/stats/$f.nc"
done

# 4. ERA5 frames for the seven storms (+ analogue dates), as a sparse local copy of the WB2 zarr
.venv-jax/bin/python -m graphcast_sae.extraction.stage_era5_local --out "$GC_SCRATCH/era5_wb2.zarr" --jobs conv

echo "setup done. Next: sed -i 's/<your-allocation>/<account>/' scripts/rev_fig2.sbatch; mkdir -p out; sbatch scripts/rev_fig2.sbatch"
