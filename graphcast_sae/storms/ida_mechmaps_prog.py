"""Per-mechanism PROGRESSION maps over Ida: mechanism activation | cyclone feature at +12/+24/+36/+48 h.

Fork of ida_mechmaps.py with two changes for the paper figure rebuild:
  - wider capture box (zoomed out: 0-40N, 110-40W instead of 8-36N, 100-55W)
  - the TC feature 3243 is captured at FOUR leads (+12/+24/+36/+48 h) for the
    baseline and for each boosted run, so the figure can show the storm's
    spin-up as a time-lapse rather than a before/after pair.

Paper: Fig. 5, Ida dial-up (figures/paper_fig_ida_dialup.py)
Inputs: GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/fs_ida_mechmaps_prog<MECHMAPS_TAG>.npy (shipped: _v2 = calibrated groups, "" = the 39c8e9b groups)
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.storms.ida_mechmaps_prog
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

TC = 3243; IC = "2021-08-26"; H = 8; DOSE = 1.0
MECH = {"convection": [2401, 2067, 3174], "moisture": [3501, 845],
        "vorticity": [3861, 2514, 2089], "shear": [1996, 2349, 744]}
REG = dict(lat=(0, 40), lon=(-110, -40))
# Calibrated-group override (2026-08-29): MECHMAPS_GROUPS='{"convection":[...],...}' and MECHMAPS_TAG='_v2'
# write results/fs_ida_mechmaps_prog<TAG>.npy. The shipped _v2 file used
#   {"convection":[2401,2067,3174],"moisture":[2415,3780,1829],"vorticity":[2089,2514,3316],"shear":[1996,3460]}
import json as _json
TAG = os.environ.get("MECHMAPS_TAG", "")
if os.environ.get("MECHMAPS_GROUPS"): MECH = {k: [int(f) for f in v] for k, v in _json.loads(os.environ["MECHMAPS_GROUPS"]).items()}
assert TC not in [f for g in MECH.values() for f in g], "outcome feature inside a mechanism group"
CAP = [1, 3, 5, 7]          # capture leads: (h+1)*6 h = +12/+24/+36/+48

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def main():
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]; mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    node = (mlat >= REG["lat"][0]) & (mlat <= REG["lat"][1]) & (mlon >= REG["lon"][0]) & (mlon <= REG["lon"][1])

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    blk, times, st = fc.load_block(np.datetime64(IC), nframes=2 + H)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = st[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    inp, tgt, frc = data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*H}h"), **dataclasses.asdict(tc))
    tct = tgt.time.isel(time=slice(0, 1))
    for c in ("datetime",):
        if c in tgt.coords: tgt = tgt.drop_vars(c)
        if c in frc.coords: frc = frc.drop_vars(c)

    def roll(patch, want):
        """roll; return ({lead_h: {feat: node map}} for leads in CAP, {feat: map} at h==4)."""
        cur = inp; frames = {}; mid = None
        for h in range(H):
            t0 = time.time()
            ct = tgt.isel(time=slice(h, h+1)).assign_coords(time=tct); cf = frc.isel(time=slice(h, h+1)).assign_coords(time=tct)
            p, a = apply(cur, ct, cf, patch)
            X = jnp.asarray(np.asarray(a, np.float32).reshape(-1, fc.D_IN)); code = np.asarray(sae.codes(X))
            fr = {f: code[node, f] for f in want}
            if h == 4: mid = fr
            if h in CAP: frames[(h + 1) * 6] = fr
            p = numpyify(p)
            if h < H-1: cur = rollout._get_next_inputs(cur, xr.merge([p, cf])).assign_coords(time=cur.coords["time"])
            print(f"    step {h+1}/{H}  {time.time()-t0:.0f}s", flush=True)
        return frames, mid

    allfeats = [TC] + [f for g in MECH.values() for f in g]
    t0 = time.time()
    frames, base_mid = roll(noop, allfeats)
    out = {"mlat": mlat[node], "mlon": mlon[node], "reg": REG, "mech": MECH, "tc": TC,
           "ic": IC, "dose": DOSE, "leads": sorted(frames),
           "tc_base": {lead: fr[TC] for lead, fr in frames.items()},
           "mech_map": {m: sum(base_mid[f] for f in g) for m, g in MECH.items()}}
    print(f"baseline done {time.time()-t0:.0f}s; TC base +48h sum {out['tc_base'][48].sum():.0f}", flush=True)
    for m, g in MECH.items():
        frames, _ = roll(fc.coef_patch(sae, g, DOSE), [TC])
        out[f"tc_{m}"] = {lead: fr[TC] for lead, fr in frames.items()}
        print(f"  dose {m}: TC +48h sum {out[f'tc_{m}'][48].sum():.0f} (base {out['tc_base'][48].sum():.0f})", flush=True)
    np.save(fc.ROOT / f"results/fs_ida_mechmaps_prog{TAG}.npy", out, allow_pickle=True)
    print(f"-> results/fs_ida_mechmaps_prog{TAG}.npy", flush=True)

if __name__ == "__main__":
    main()
