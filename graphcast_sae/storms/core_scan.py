"""Per-feature activation INSIDE the storm core, for all 4,096 features, on every storm.

WHY. Across sixteen concept groups, spearman(deepening lost, core activation) = +0.844.
Every group costing ~0% has exactly zero activation within 300 km of the centre; every group
costing a lot puts 90-94% of its in-box firing there. The random controls sit at zero too, so
the treatment-vs-control contrast is largely asking "does this group fire in the eyewall"
rather than "is this the right mechanism".

The control that separates the two must fire in the core AT THE SAME RATE and be a different
concept. Nothing on disk can draw it: the runs store node activations only for their own
group plus the control, so core firing is unknown for the other ~4,090 features.

This measures it. One unpatched rollout per storm to +48 h, capture the layer-8 codes, and
sum each feature's activation within 300 km of the MSLP minimum. Output is a (storm, 4096)
matrix that answers two things at once:
  1. do ANY non-ascent features fire in the core as hard as the convection group does?
  2. if so, which -- so a core-matched control can be drawn and ablated.
If the answer to (1) is no, that is itself the finding: the dictionary may simply not contain
a core-localised alternative, in which case the specificity question is not answerable with
this SAE and must be reported as such rather than as a passing control.

    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \

Paper: Table tab:mechanism-interventions (input to the core-matched control)
Inputs: GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/fs_core_scan.npy
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.storms.core_scan
"""
import dataclasses
import importlib
import os
import sys
import time

os.environ.setdefault("FS_DEVICE", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import xarray as xr
import jax.numpy as jnp
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

S = importlib.import_module("graphcast_sae.common." + os.environ.get("SKILL_STORMS", "skill_conv_storms"))
MID = int(os.environ.get("CS_MID", "7"))            # +48 h
CORE_KM = float(os.environ.get("CS_CORE_KM", "300"))
OUT = fc.ROOT / os.environ.get("CS_OUT", "results/fs_core_scan.npy")
R = 6371.0

def gc_km(la, lo, la0, lo0):
    d = (np.asarray(lo, float) - lo0 + 180) % 360 - 180
    return R * np.arccos(np.clip(
        np.sin(np.deg2rad(la)) * np.sin(np.deg2rad(la0)) +
        np.cos(np.deg2rad(la)) * np.cos(np.deg2rad(la0)) * np.cos(np.deg2rad(d)), -1, 1))

def main():
    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply_fn = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    g = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = np.asarray(g["lat"], float)
    mlon = np.where(np.asarray(g["lon"], float) > 180,
                    np.asarray(g["lon"], float) - 360, np.asarray(g["lon"], float))

    core = {}
    for name, cfg in S.STORMS.items():
        t0 = time.time()
        blk, times, statics = fc.load_block(np.datetime64(cfg["ic"]), nframes=MID + 3)
        w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
        for v in list(w.data_vars):
            if "time" in w[v].dims:
                w[v] = w[v].expand_dims("batch")
        for v in fc.STATIC_VARS:
            w[v] = statics[v]
        w = w.assign_coords(datetime=(("batch", "time"),
                                      times[None, :].astype("datetime64[ns]")))
        inp, tgt, frc = data_utils.extract_inputs_targets_forcings(
            w, target_lead_times=slice("6h", f"{6*(MID+1)}h"), **dataclasses.asdict(tc))
        cur = inp
        C = None
        for s in range(MID + 1):
            tg = tgt.isel(time=slice(s, s + 1)).assign_coords(time=tgt.time[:1])
            fr = frc.isel(time=slice(s, s + 1)).assign_coords(time=frc.time[:1])
            pr, acts = apply_fn(cur, tg, fr, noop)
            pr = xr.Dataset({v: (pr[v].dims, np.asarray(pr[v].values)) for v in pr.data_vars},
                            coords={k: pr.coords[k] for k in pr.coords})
            if s == MID:
                X = jnp.asarray(np.asarray(acts, np.float32).reshape(-1, fc.D_IN))
                C = np.asarray(sae.codes(X))               # (n_mesh, 4096)
                m = pr["mean_sea_level_pressure"].isel(batch=0, time=0)
                m = m.sel(lat=slice(*cfg["box"]["lat"]))
                lo = S.norm_lon(cfg["box"]["lon"])
                m = m.sel(lon=slice(lo[0], lo[1]))
                a = np.asarray(m.transpose("lat", "lon").values, float)
                j, i = np.unravel_index(int(np.nanargmin(a)), a.shape)
                clat = float(np.asarray(m.lat.values)[j])
                clon = float(np.asarray(m.lon.values)[i])
                break
            cur = rollout._get_next_inputs(cur, xr.merge([pr, fr])).assign_coords(
                time=inp.coords["time"])
        disk = gc_km(mlat, np.where(mlon < 0, mlon + 360, mlon), clat, clon) < CORE_KM
        core[name] = dict(core=C[disk].sum(0).astype(np.float32),
                          allnode=C.sum(0).astype(np.float32),
                          n_disk=int(disk.sum()), clat=clat, clon=clon)
        print(f"  {name:<14} centre ({clat:+.1f},{clon:.1f})  {int(disk.sum())} core nodes  "
              f"({time.time()-t0:.0f}s)", flush=True)

    np.save(OUT, dict(core=core, core_km=CORE_KM, mid=MID,
                      conv=S.CONV, tc=S.TC), allow_pickle=True)
    print(f"\n-> {OUT}")

if __name__ == "__main__":
    main()
