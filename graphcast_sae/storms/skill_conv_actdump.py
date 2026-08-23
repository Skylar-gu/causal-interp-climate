"""Dump raw layer-8 activations inside the storm box, every step of a baseline rollout.

Why this exists. `results/skill/fields_conv` stores node-level SAE *codes* for four
features; the geometry test in `docs/notes/result_local_aggregate_2026_08_21.md` needs the raw
512-dim activations, which nothing on disk carries for the named storms. That test therefore
ran on TC-located surrogate boxes from the IID dump, which are real cyclones but are single
initial conditions -- t = 0 of a forecast, not +54 h with the edit already running. This
script closes that gap.

Baseline arm only, no patch, so it changes nothing and reproduces the committed baseline
trajectory. Per storm it writes (H, n_box, 512) float16 plus the box node coordinates, the
per-step tracked centre and the per-step min-MSLP, which is what the mid-rollout PCA needs.

Size: 16 x 839 x 512 x 2 B = 13 MB per storm, ~95 MB for the battery.

    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \

Paper: Appendix app:mesh (raw in-box activations for the local-aggregate test)
Inputs: results/skill/actdump (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/skill/actdump/act_<storm>.npz
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.storms.skill_conv_actdump [storm ...]
"""
import os
import sys
import time

os.environ["FS_DEVICE"] = "gpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax.numpy as jnp
import xarray as xr

import graphcast_sae.common.fs_common as fc
from graphcast import rollout
import graphcast_sae.storms.skill_conv_run as R                    # build_io, numpyify, box_phys, box_fields
import graphcast_sae.common.skill_conv_storms as S

OUT = fc.ROOT / "results/skill/actdump"
OUT.mkdir(parents=True, exist_ok=True)
H = S.H

def main():
    only = sys.argv[1:] or None
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]
    mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    R.tc = tc                                  # build_io reads the task config through R
    rf, _ = fc.build_apply_cond(mc, tc, stats, sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    zeroF = jnp.zeros(sae.n_features, jnp.float32)
    zeroN = jnp.zeros(len(mlat), jnp.float32)
    print(f"model+SAE loaded; H={H}; storms={list(S.STORMS)}", flush=True)

    for name, cfg in S.STORMS.items():
        if only and name not in only:
            continue
        fpath = OUT / f"act_{name}.npz"
        if fpath.exists():
            print(f"[{name}] exists, skip", flush=True)
            continue
        box = cfg["box"]
        inbox = ((mlat >= box["lat"][0]) & (mlat <= box["lat"][1])
                 & (mlon >= box["lon"][0]) & (mlon <= box["lon"][1]))
        print(f"[{name}] box={int(inbox.sum())} mesh nodes, IC {cfg['ic']}", flush=True)
        t0 = time.time()

        inp, tgt, frc = R.build_io(cfg["ic"], H, tc)
        tct = tgt.time.isel(time=slice(0, 1))
        for cc in ("datetime",):
            if cc in tgt.coords:
                tgt = tgt.drop_vars(cc)
            if cc in frc.coords:
                frc = frc.drop_vars(cc)

        cur = inp
        acts, mslp, clat, clon = [], [], [], []
        patch = (zeroF, zeroF, zeroN)
        for h in range(H):
            ct = tgt.isel(time=slice(h, h + 1)).assign_coords(time=tct)
            cf = frc.isel(time=slice(h, h + 1)).assign_coords(time=tct)
            p, a = apply(cur, ct, cf, patch)
            A = np.asarray(a, np.float32).reshape(-1, fc.D_IN)
            acts.append(A[inbox].astype(np.float16))
            pn = R.numpyify(p)
            fld, _, _ = R.box_fields(pn, box)
            mslp.append(float(np.nanmin(fld["mslp"])))
            clat.append(fld["clat"])
            clon.append(fld["clon"])
            if h < H - 1:
                cur = (rollout._get_next_inputs(cur, xr.merge([pn, cf]))
                       .assign_coords(time=cur.coords["time"]))

        np.savez_compressed(
            fpath,
            act=np.stack(acts),                       # (H, n_box, 512) float16
            box_lat=mlat[inbox].astype(np.float32),
            box_lon=mlon[inbox].astype(np.float32),
            mslp_min=np.array(mslp, np.float32),
            clat=np.array(clat, np.float32),
            clon=np.array(clon, np.float32),
            center=np.array(cfg["center"], np.float32),
            ic=str(cfg["ic"]),
        )
        print(f"[{name}] -> {fpath.name}  {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
