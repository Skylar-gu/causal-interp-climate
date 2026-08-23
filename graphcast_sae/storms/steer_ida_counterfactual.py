"""Counterfactual convection over Hurricane Ida — restore to NORMAL, not zero, and locally.

All interventions are confined to a ~1500 km disk around Ida's track (the physical genesis scale),
and differ only in what the convection features are set to INSIDE that disk:
  baseline   : untouched                              -> the storm forms
  zero-local : convection capped at 0 (removed)       -> no convection at all, locally
  normal     : convection capped at its NORMAL level  -> keep the background, strip Ida's excess
                (normal level measured from quiet, no-storm late-August analog years)
If 'normal' still suppresses the cyclone feature -> Ida's OWN anomalous convection is necessary.
If 'normal' looks like baseline -> the background sufficed; the excess was incidental.

Paper: Fig. 5 and Table tab:mechanism-interventions (the restore-to-normal counterfactual every storm run reuses)
Inputs: GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/fs_ida_counterfactual.npy
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.storms.steer_ida_counterfactual
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout
from graphcast_sae.common.signature_physics import gc_km

TC = 3243; CONV = [2401, 2067, 3174]
IC = "2021-08-26"; H = 8
CENTER = (22.0, -84.0); RADIUS_KM = 1500.0                      # Ida's track, genesis-scale disk
BOX = dict(lat=(10, 33), lon=(-98, -58))
ANALOGS = ["2014-08-27", "2013-08-27", "2009-08-27", "2006-08-27", "2015-08-27"]

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def build_io(t0, H, task_config):
    blk, times, st = fc.load_block(np.datetime64(t0), nframes=2 + H)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = st[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*H}h"), **dataclasses.asdict(task_config))

def main():
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]; mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    nmask = (gc_km(mlat, mlon, CENTER[0], CENTER[1]) < RADIUS_KM).astype(np.float32)
    inbox = (mlat >= BOX["lat"][0]) & (mlat <= BOX["lat"][1]) & (mlon >= BOX["lon"][0]) & (mlon <= BOX["lon"][1])
    print(f"disk: {int(nmask.sum())} mesh nodes within {RADIUS_KM:.0f} km of {CENTER}", flush=True)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply_cond(mc, tc, stats, sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)

    def codes_at(t0):
        blk = fc.load_block(np.datetime64(t0), nframes=fc.INPUT_WINDOW)
        inp, tg, fr = fc.build_batch_inputs([blk], 0, tc)
        z = jnp.zeros(sae.n_features, jnp.float32)
        _, acts = apply(inp, tg, fr, (z, z, np.zeros(len(mlat), np.float32)))    # fsel=0 -> exact model
        X = np.asarray(acts, np.float32).reshape(-1, fc.D_IN)
        return np.asarray(sae.codes(jnp.asarray(X)))

    # ---- NORMAL reference: convection level in quiet, no-storm late-Augusts ----
    print("measuring normal convection from no-storm analogs (skip any with a storm in the box):", flush=True)
    acc = {f: [] for f in CONV}
    for a in ANALOGS:
        c = codes_at(a); storm = c[inbox, TC].sum()
        if storm > 20: print(f"  {a}: TC={storm:.0f} — storm present, SKIP", flush=True); continue
        for f in CONV:
            v = c[nmask.astype(bool), f]; acc[f].extend(v[v > 0].tolist())
        print(f"  {a}: TC={storm:.0f} — quiet, used", flush=True)
    ftarget = np.zeros(sae.n_features, np.float32)
    for f in CONV:
        ftarget[f] = np.mean(acc[f]) if acc[f] else 0.0
    print(f"normal convection level (mean active) per feature: {[round(ftarget[f],2) for f in CONV]}", flush=True)

    fsel = np.zeros(sae.n_features, np.float32); fsel[CONV] = 1.0
    zero = np.zeros(sae.n_features, np.float32)

    # ---- three localized arms, persistent through the rollout ----
    inp, tgt, frc = build_io(IC, H, tc)
    tct = tgt.time.isel(time=slice(0, 1))
    for cc in ("datetime",):
        if cc in tgt.coords: tgt = tgt.drop_vars(cc)
        if cc in frc.coords: frc = frc.drop_vars(cc)

    def tcbox(a):
        X = jnp.asarray(np.asarray(a, np.float32).reshape(-1, fc.D_IN))
        return float(np.asarray(sae.codes(X))[inbox, TC].sum())

    def roll(patch):
        cur = inp; tr = []
        for h in range(H):
            ct = tgt.isel(time=slice(h, h+1)).assign_coords(time=tct); cf = frc.isel(time=slice(h, h+1)).assign_coords(time=tct)
            p, a = apply(cur, ct, cf, patch); tr.append(tcbox(a)); p = numpyify(p)
            if h < H-1: cur = rollout._get_next_inputs(cur, xr.merge([p, cf])).assign_coords(time=cur.coords["time"])
        return np.array(tr)

    arms = {"baseline": (zero, zero, np.zeros(len(mlat), np.float32)),
            "zero-local": (fsel, zero, nmask),
            "normal": (fsel, jnp.asarray(ftarget), nmask)}
    print(f"\n{'arm':>12}  genesis curve (TC feature in box, +6h..+48h)   TC@48h  vs base", flush=True)
    res = {}
    base48 = None
    for name, patch in arms.items():
        tr = roll((jnp.asarray(patch[0]), jnp.asarray(patch[1]), jnp.asarray(patch[2]))); res[name] = tr
        if name == "baseline": base48 = tr[-1]
        print(f"  {name:>12}  {np.array2string(tr,precision=0)}   {tr[-1]:>5.0f}  {100*(tr[-1]-base48)/max(base48,1):>+5.0f}%", flush=True)
    np.save(fc.ROOT / "results/fs_ida_counterfactual.npy", dict(res=res, ftarget=ftarget[CONV], conv=CONV,
            center=CENTER, radius=RADIUS_KM, analogs=ANALOGS), allow_pickle=True)
    print("\nInterpretation: normal~=baseline -> background convection sufficed; normal<<baseline -> Ida's own excess convection is necessary.")
    print("-> results/fs_ida_counterfactual.npy")

if __name__ == "__main__":
    main()
