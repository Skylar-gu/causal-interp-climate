"""Every location the code touches, resolved in one place.

Nothing else in the package may hardcode a machine path. Import what you need:

    from graphcast_sae.paths import REPO_ROOT, RESULTS, SCRATCH, SAE_WEIGHTS

Shipped with the repository (no configuration needed)
    REPO_ROOT        this checkout
    RESULTS          results/   curated result files the paper's figures read
    DATA             data/      mesh geometry, per-feature footprints, land-sea mask
    DOCS             docs/      pre-registrations and cited notes
    FIGURES          figures/   figure builders and their PDFs
    WEIGHTS          graphcast_sae/weights/  the published SAE (npz) + its config

Not shipped, configurable through environment variables
    GRAPHCAST_PARAMS the DeepMind GraphCast 0.25 deg / 37-level checkpoint
                     (default: WEIGHTS/graphcast_flagship_0p25_37lev.npz)
    GRAPHCAST_ASSETS the DeepMind release's `params/` + `stats/` directory
                     (default: REPO_ROOT/assets)
    GC_SCRATCH       root for every regenerable dump: the i.i.d. layer-8 activation
                     dump, the encoded catalog, pooled trajectories, extraction
                     status files (default: REPO_ROOT/scratch, git-ignored)

Everything under SCRATCH is produced by a script in this package; each child below
names the script that writes it.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DATA = REPO_ROOT / "data"
DOCS = REPO_ROOT / "docs"
FIGURES = REPO_ROOT / "figures"
OUT = REPO_ROOT / "out"                       # crash-safe status files of long runs
CANDIDATES = REPO_ROOT / "candidates"         # pool / catalog files (regenerable, not shipped)

WEIGHTS = REPO_ROOT / "graphcast_sae" / "weights"
SAE_WEIGHTS = WEIGHTS / "sae_k32_lat4096_lay08.npz"
SAE_CONFIG = WEIGHTS / "sae_config.json"
GRAPHCAST_PARAMS = Path(os.environ.get("GRAPHCAST_PARAMS",
                                       WEIGHTS / "graphcast_flagship_0p25_37lev.npz"))
ASSETS = Path(os.environ.get("GRAPHCAST_ASSETS", REPO_ROOT / "assets"))

SCRATCH = Path(os.environ.get("GC_SCRATCH", REPO_ROOT / "scratch"))
IID_DUMP = SCRATCH / "fs_iid_dump.npy"        # extraction/extract_iid_dump.py (6.7 GB)
IID_META = SCRATCH / "fs_iid_meta.json"       # extraction/extract_iid_dump.py
FS_CATALOG = SCRATCH / "fs_catalog.npz"       # extraction/fs_catalog.py
FS_MODES = SCRATCH / "fs_modes.npz"           # legacy/fs_retry1_communities.py
ACTS_DIR = SCRATCH / "fs_acts"                # extraction/fs_extract.py
MESH_DEGREE = SCRATCH / "flagship_m6_degree.npy"   # M6 node degree (mesh_speed_limit geometry)
POOLED_DIR = SCRATCH / "pooled"               # obsgraph/extract_traj_flag2.py --dump-pooled
MINI_ACTS_DIR = SCRATCH / "mini_acts"         # extraction/mini_extract_wb2.py --out-dir

MESH_GEOM = DATA / "mesh_2to6_geom.npy"       # shipped: M6 mesh node lat/lon/xyz

# WeatherBench2 public bucket (anonymous gcsfs); streamed, never mirrored locally.
WB2_ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr"


def ensure_dirs():
    """Create the writable roots a run needs (idempotent)."""
    for d in (RESULTS, OUT, SCRATCH):
        d.mkdir(parents=True, exist_ok=True)
