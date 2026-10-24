# Environments

**Two Python environments. They must not be mixed** — the JAX/GraphCast stack pins
`numpy>=2.4` and Python 3.11; the analysis stack runs Python 3.9 with `numpy 2.0` and
`tigramite`, which do not coexist.

| env | interpreter on the origin box | used by |
|---|---|---|
| **JAX / GraphCast** | `~/miniforge3/envs/graphcast/bin/python` (3.11.15) | `graphcast_sae/*`, `figures/main_claims/fig2_interventions.py`, anything running a GraphCast forward pass |
| **torch / analysis** | `.venv-analysis` (3.9, below) | CPU-only scripts in `graphcast_sae/` whose docstring says `python3` |

Throughout the docs these are written `$PY_JAX` and `$PY_TORCH`.

## Files here

- `requirements-jax.txt` / `requirements-torch.txt` — **curated**: the packages the code
  actually imports, at the exact versions the published results were produced with. Use
  these.


## JAX / GraphCast env

```bash
conda create -n graphcast python=3.11 -y && conda activate graphcast
pip install -r env/requirements-jax.txt
# GPU wheels for JAX are not in the curated file (they depend on your CUDA):
pip install --upgrade "jax[cuda12]==0.10.2"
python -c "import jax; print(jax.devices())"     # must show a GPU
```

**Origin box for reference:** NVIDIA driver 595.71.05, `nvidia-cudnn-cu12 9.24.0.43`,
`nvidia-cublas-cu12 12.9.2.10`. GraphCast is installed straight from the DeepMind repo at
a pinned commit (see the curated file) — the released PyPI package is not equivalent.

## torch / analysis env

```bash
python3.9 -m venv .venv-analysis && source .venv-analysis/bin/activate
pip install -r env/requirements-torch.txt
python -c "import torch; print(torch.__version__)"
```

`torch` is only needed to load the `.pt` SAE (`graphcast_sae/common/fs_common.py`); a CPU-only wheel is fine.

## Non-Python assets

Neither environment file covers these; fetch them separately (see `HANDOFF_GPU.md` §1.2 at tag `pre-camera-ready`):

- **GraphCast weights + normalisation stats** → `assets/` from the public
  `gs://dm_graphcast` bucket (no auth).
- **ERA5**: `graphcast_sae/extraction/stage_era5_local.py` stages the WeatherBench 2 frames a job needs to local
  zarr (`FS_ERA5_LOCAL`, set in `scripts/rev_env.sh`) for compute nodes without internet.
