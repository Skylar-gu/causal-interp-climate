"""Do the suspected artifacts damage the forecast INSIDE THEIR OWN FOOTPRINT?

Global RMSE averages over the sphere, so a feature that damages its own neighbourhood
intensely can vanish in it. The global run answered "not removable at globally-detectable
scale"; this asks the narrower question the global average cannot: is there anywhere the
ablation HELPS?

Same rollout, same paired 8 ICs, same fields. The only change is that RMSE is accumulated
over a set of MASKS as well as globally, so every arm is scored inside each footprint, inside
its complement, and over the whole grid at once. Masks come from footprint_masks.py at a
60 km halo -- half the 112 km level-6 mesh spacing, the tightest radius that still tiles
without holes.

WHAT THE COVERAGE ALREADY SAYS, before any arm runs:

    mask            area      note
    mesh_locked    22.2%      the coarse skeleton spans the globe; "local" is not available
    ctrl_mesh      33.9%
    scatter_blob   68.4%      union of 127 wide features; near-vacuous as a group mask
    ctrl_blob      86.9%
    f2075           6.9%      the literal bowtie, 3,436 nodes in 68 components
    f2235          16.6%      polar mesh-convergence bands
    f656            0.4%      213 active nodes in 213 components -- maximally scattered

So the GROUP masks cannot deliver a local test: mesh_locked is spread thinly over a fifth of
the planet by construction, and scatter_blob's union covers two thirds. Only the SINGLE
features are local enough for "its own footprint" to mean anything, which is why single-feature
arms are carried here and were not in the global run.

    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \

Paper: Sec. 3, grid-locked-feature ablations (demo notebook part 4)
Inputs: results/fs_footprint_masks.npz (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: out/global_rmse_status.txt; results/fs_footprint_rmse.npy
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.gridlock.footprint_rmse
"""
import dataclasses
import json
import os
import sys
import time

os.environ["FS_DEVICE"] = "gpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import xarray as xr
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

S = int(os.environ.get("GR_S", "20"))                 # 20 x 6 h = 120 h
ICS = os.environ.get("GR_ICS", "2020-01-10,2020-03-12,2020-05-11,2020-07-10,"
                                "2020-09-08,2020-11-07,2021-02-05,2021-06-15").split(",")
OUT = fc.ROOT / os.environ.get("GR_OUT", "results/fs_footprint_rmse.npy")
MASKS = np.load(fc.ROOT / "results/fs_footprint_masks.npz")
MK = [k for k in MASKS.files if k not in ("lat", "lon")]
STATUS = fc.ROOT / os.environ.get("GR_STATUS", "out/global_rmse_status.txt")
G0 = 9.80665

# WeatherBench2 headline fields: (name, variable, level or None)
FIELDS = [("z500", "geopotential", 500), ("t850", "temperature", 850),
          ("t2m", "2m_temperature", None), ("u850", "u_component_of_wind", 850),
          ("v850", "v_component_of_wind", 850), ("q700", "specific_humidity", 700),
          ("msl", "mean_sea_level_pressure", None)]

def log(m):
    print(m, flush=True)
    with open(STATUS, "a") as f:
        f.write(m + "\n")

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def build_io(t0, tc, s):
    blk, times, statics = fc.load_block(np.datetime64(t0), nframes=s + 2)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims:
            w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS:
        w[v] = statics[v]
    w = w.assign_coords(datetime=(("batch", "time"),
                                  times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*s}h"), **dataclasses.asdict(tc))

def grab(ds, var, lev):
    d = ds[var].isel(batch=0, time=0)
    if lev is not None:
        d = d.sel(level=lev)
    a = np.asarray(d.transpose("lat", "lon").values, np.float64)
    return a / G0 if var == "geopotential" else a

def main():
    open(STATUS, "w").close()
    GF = os.environ.get("GR_GROUPS", "/tmp/artifact_groups.json")
    G = json.load(open(GF))
    order = os.environ.get("GR_ARMS", "")
    keys = order.split(",") if order else list(G)
    ARMS = [("baseline", None)] + [(k, G[k]) for k in keys]
    log(f"GLOBAL RMSE ABLATION  S={S} ({6*S} h)  ICs={len(ICS)}  arms={len(ARMS)}")
    for k, v in ARMS:
        log(f"   {k:<14} {'-' if v is None else str(len(v)) + ' features'}")

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply_fn = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)
    patches = {k: (noop if v is None else fc.coef_patch(sae, v, -1.0))
               for k, v in ARMS}

    # PER-IC storage, not a running mean. Every arm sees the SAME initial conditions, so the
    # comparison is PAIRED: for a given IC the arm and its control forecast identical weather
    # and their difference isolates the ablation from "was this a hard week". Keeping only the
    # mean throws that structure away and leaves a number with no testable uncertainty.
    # (nIC, lead, field, region) where region indexes: 0 = global, then each mask, then each
    # mask's COMPLEMENT. The complement matters: an arm that helps inside a footprint while
    # hurting outside it is the artifact signature, and one number cannot show that.
    REG = ["global"] + MK + [f"~{k}" for k in MK]
    log(f"regions: {REG}")
    wgrid = {}
    acc = {k: np.full((len(ICS), S, len(FIELDS), len(REG)), np.nan) for k, _ in ARMS}
    n = 0
    for ii, ic in enumerate(ICS):
        t0 = time.time()
        inp, tgt, frc = build_io(ic, tc, S)
        lat = np.asarray(tgt.lat.values, np.float64)
        w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, tgt.lon.size))
        if not wgrid:
            for k in MK:
                m = np.asarray(MASKS[k], bool)
                wgrid[k] = w * m
                wgrid[f"~{k}"] = w * (~m)
            wgrid["global"] = w
            for r in REG:
                log(f"   {r:<16} weight share {wgrid[r].sum()/w.sum():.4f}")
        for name, _ in ARMS:
            cur, p = inp, patches[name]
            for s in range(S):
                tg = tgt.isel(time=slice(s, s + 1)).assign_coords(time=tgt.time[:1])
                fr = frc.isel(time=slice(s, s + 1)).assign_coords(time=frc.time[:1])
                pr = numpyify(apply_fn(cur, tg, fr, p)[0])    # patch EVERY step; apply returns (preds, acts)
                truth = tgt.isel(time=slice(s, s + 1))
                for fi, (nm, var, lev) in enumerate(FIELDS):
                    d = grab(pr, var, lev) - grab(truth, var, lev)
                    d2 = d * d
                    for ri, r in enumerate(REG):
                        ww = wgrid[r]
                        acc[name][ii, s, fi, ri] = np.sqrt((ww * d2).sum() / ww.sum())
                cur = rollout._get_next_inputs(cur, xr.merge([pr, fr])).assign_coords(
                    time=inp.coords["time"])
        n += 1
        log(f"  IC {ic} done ({time.time()-t0:.0f}s)  {n}/{len(ICS)}")
        np.save(OUT, dict(acc=acc, n=n,
                          fields=[f[0] for f in FIELDS], S=S, ics=ICS[:n],
                          arms=[a[0] for a in ARMS], groups=G), allow_pickle=True)

    log(f"\n-> {OUT}  (score with footprint_rmse_analyze.py)")

if __name__ == "__main__":
    main()
