"""Causal feature interactions — do features compose additively, or interact?

For causally-connected pairs (i,j) from the interventional graph, compare the forecast response to
  dose i alone,  dose j alone,  dose BOTH   (z500 at t+12h, vs baseline).
Interaction = || Δ_both - (Δ_i + Δ_j) ||  /  || Δ_i + Δ_j ||   (cos-lat weighted, global).
~0 = additive (a linear feature model); large = the model's feature computation is non-additive
(real compositional structure). Sign of the projection says synergy (>sum) vs redundancy (<sum).

Paper: not in the paper; kept for provenance only
Inputs: results/fs_cgraph.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/fs_interact.npy
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.legacy.steer_interact
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

GAMMA = 1.0
ICS = ["2020-01-05", "2018-09-10"]
NPAIR = 14
HOR = 2                                                        # measure at t+12h

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def build_io(t0, tc, H):
    blk, times, statics = fc.load_block(np.datetime64(t0), nframes=2 + H)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = statics[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*H}h"), **dataclasses.asdict(tc))

def roll_z500(apply, sae, inp, tgt, frc, patch0, H):
    """impulse patch0 at t0, noop after; return z500 at the final lead."""
    noop = fc.noop_patch(sae); tct = tgt.time.isel(time=slice(0, 1))
    for c in ("datetime",):
        if c in tgt.coords: tgt = tgt.drop_vars(c)
        if c in frc.coords: frc = frc.drop_vars(c)
    cur = inp; z = None
    for h in range(H):
        ct = tgt.isel(time=slice(h, h+1)).assign_coords(time=tct); cf = frc.isel(time=slice(h, h+1)).assign_coords(time=tct)
        preds = numpyify(apply(cur, ct, cf, patch0 if h == 0 else noop)[0])
        z = np.asarray(preds["geopotential"].isel(batch=0, time=0).sel(level=500).values) / 9.81
        if h < H-1: cur = rollout._get_next_inputs(cur, xr.merge([preds, cf])).assign_coords(time=cur.coords["time"])
    return z

def main():
    g = np.load(fc.ROOT / "results/fs_cgraph.npy", allow_pickle=True).item()
    A, feats, labels = g["A"], g["feats"], g["labels"]
    F = len(feats)
    # top causally-connected pairs (unordered, dedup), by |A_ij|
    edges = sorted([(r, c, abs(A[r, c])) for r in range(F) for c in range(F) if r != c],
                   key=lambda x: -x[2])
    pairs, seen = [], set()
    for r, c, _ in edges:
        key = frozenset((int(feats[r]), int(feats[c])))
        if key in seen: continue
        seen.add(key); pairs.append((int(feats[r]), int(feats[c])))
        if len(pairs) >= NPAIR: break
    print(f"testing {len(pairs)} causally-connected pairs over {len(ICS)} ICs", flush=True)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    lat = np.asarray(fc.load_block(np.datetime64(ICS[0]))[0]["lat"].values)
    cosl = np.cos(np.radians(lat))[:, None]

    def wnorm(d): return np.sqrt((d**2 * cosl).sum())

    rows = []
    t0 = time.time()
    for ic in ICS:
        inp, tgt, frc = build_io(ic, tc, HOR)
        base = roll_z500(apply, sae, inp, tgt, frc, fc.noop_patch(sae), HOR)
        for (i, j) in pairs:
            di = roll_z500(apply, sae, inp, tgt, frc, fc.coef_patch(sae, [i], GAMMA), HOR) - base
            dj = roll_z500(apply, sae, inp, tgt, frc, fc.coef_patch(sae, [j], GAMMA), HOR) - base
            db = roll_z500(apply, sae, inp, tgt, frc, fc.coef_patch(sae, [i, j], GAMMA), HOR) - base
            add = di + dj; resid = db - add
            inter = wnorm(resid) / (wnorm(add) + 1e-9)
            proj = float((db*add*cosl).sum() / ((add**2*cosl).sum() + 1e-9))   # >1 synergy, <1 redundancy
            rows.append(dict(ic=ic, i=i, j=j, inter=inter, proj=proj,
                             ni=wnorm(di), nj=wnorm(dj), nb=wnorm(db)))
        print(f"  IC {ic} done  {(time.time()-t0)/60:.1f}m", flush=True)

    np.save(fc.ROOT / "results/fs_interact.npy", dict(rows=rows, pairs=pairs, gamma=GAMMA), allow_pickle=True)
    # aggregate per pair across ICs
    import collections
    agg = collections.defaultdict(list)
    for r in rows: agg[(r["i"], r["j"])].append((r["inter"], r["proj"]))
    print(f"\n{'pair':>14} {'interaction':>12} {'proj(syn/red)':>14}")
    inters = []
    for (i, j), vals in sorted(agg.items(), key=lambda kv: -np.mean([v[0] for v in kv[1]])):
        m = np.mean([v[0] for v in vals]); p = np.mean([v[1] for v in vals]); inters.append(m)
        tag = "synergy" if p > 1.1 else ("redundant" if p < 0.9 else "additive-ish")
        print(f"  {i:>5}+{j:<5} {m:>12.2f} {p:>10.2f}  {tag}", flush=True)
    inters = np.array(inters)
    print(f"\nmedian interaction = {np.median(inters):.2f}  (0 = additive; >~0.3 = clearly non-additive)")
    print(f"pairs strongly non-additive (>0.3): {(inters>0.3).sum()}/{len(inters)}")
    print("VERDICT:", "features INTERACT — the model's feature computation is non-additive"
          if np.median(inters) > 0.3 else
          "features compose ~ADDITIVELY at this magnitude")
    print("-> results/fs_interact.npy")

if __name__ == "__main__":
    main()
