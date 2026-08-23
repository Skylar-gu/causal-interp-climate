"""Interventional causal graph of GraphCast SAE features — the do-operator, not PCMCI.

For each source feature i: dose it at t (scale x2 at its firing sites), advance ONE 6-h step, and
read how every feature j's activation changed vs the untouched baseline:
    A[i,j] = < codes_j( do(dose i) )  -  codes_j( baseline ) >   at t+6h, averaged over nodes & ICs.
This is a directed causal effect through the model's own dynamics — no lag->speed, no
deseasonalization, no persistence null. The baseline IS the counterfactual. Averaged over a
seasonal spread of initial conditions so the graph is the model's typical wiring, not one day's.

Paper: not in the paper; kept for provenance only
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/fs_cgraph.npy
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.legacy.steer_cgraph
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

GAMMA = 1.0                                                    # dose = double the feature at its sites
ICS = os.environ.get("CG_ICS", "2018-09-10,2020-01-05,2020-04-06,2020-07-06,2020-10-05,2019-06-15").split(",")
OUT = os.environ.get("CG_OUT", "results/fs_cgraph.npy")
TC = 3243

def pick_features(cat, k_per_bin=1):
    """~40 compact, interpretable features spread across the globe (+ the TC feature)."""
    fr, coh, clat, clon = cat["firerate"], cat["coh"], cat["clat"], cat["clon"]
    ok = np.isfinite(coh) & (fr > 0.002) & (fr < 0.06) & (coh < 4500)
    latb = [-90, -55, -30, -10, 10, 30, 55, 90]; lonb = list(range(-180, 181, 45))
    chosen = []
    for a, b in zip(latb, latb[1:]):
        for c, d in zip(lonb, lonb[1:]):
            m = ok & (clat >= a) & (clat < b) & (clon >= c) & (clon < d)
            idx = np.where(m)[0]
            if len(idx): chosen.append(idx[np.argmin(coh[idx])])   # most compact in the cell
    chosen = list(dict.fromkeys(chosen))
    if TC not in chosen: chosen.append(TC)
    return np.array(chosen)

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def build_io(t0, tc):
    blk, times, statics = fc.load_block(np.datetime64(t0), nframes=4)      # 2 in + 2 target steps
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = statics[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", "12h"), **dataclasses.asdict(tc))

def main():
    cat = np.load(fc.ROOT / "candidates/fs_feature_catalog.npy", allow_pickle=True).item()
    feats = pick_features(cat); F = len(feats)
    labels = [f"{fi}:({cat['clat'][fi]:+.0f},{cat['clon'][fi]:+.0f})" for fi in feats]
    print(f"{F} features spanning the globe (+ TC {TC}); ICs={len(ICS)}", flush=True)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    def codes_sum(acts):
        X = jnp.asarray(np.asarray(acts, np.float32).reshape(-1, fc.D_IN))
        return np.asarray(sae.codes(X).sum(0))                            # (4096,) total activation per feature

    def step(inp, tgt1, frc1, patch):
        preds, acts = apply(inp, tgt1, frc1, patch)                      # step; capture layer-8 acts
        return numpyify(preds), acts

    A = np.zeros((F, F)); nIC = 0
    t0 = time.time()
    for ic in ICS:
        inp, tgt, frc = build_io(ic, tc)
        tct = tgt.time.isel(time=slice(0, 1))
        for c in ("datetime",):
            if c in tgt.coords: tgt = tgt.drop_vars(c)
            if c in frc.coords: frc = frc.drop_vars(c)
        tgt0 = tgt.isel(time=slice(0, 1)).assign_coords(time=tct); frc0 = frc.isel(time=slice(0, 1)).assign_coords(time=tct)
        tgt1 = tgt.isel(time=slice(1, 2)).assign_coords(time=tct); frc1 = frc.isel(time=slice(1, 2)).assign_coords(time=tct)

        # baseline: step0 noop -> preds0; step1 noop on preds0 -> feature activations at t+6h
        p0b, _ = step(inp, tgt0, frc0, noop)
        in1b = rollout._get_next_inputs(inp, xr.merge([p0b, frc0])).assign_coords(time=inp.coords["time"])
        _, a1b = step(in1b, tgt1, frc1, noop); base = codes_sum(a1b)

        for r, fi in enumerate(feats):
            p0, _ = step(inp, tgt0, frc0, fc.coef_patch(sae, [int(fi)], GAMMA))
            in1 = rollout._get_next_inputs(inp, xr.merge([p0, frc0])).assign_coords(time=inp.coords["time"])
            _, a1 = step(in1, tgt1, frc1, noop)
            A[r] += codes_sum(a1)[feats] - base[feats]                    # effect on the F target features
        nIC += 1
        print(f"  IC {ic} done ({nIC}/{len(ICS)})  {(time.time()-t0)/60:.1f}m", flush=True)
    A /= nIC
    np.fill_diagonal(A, 0.0)                                              # drop self-loops

    np.save(fc.ROOT / OUT,
            dict(A=A, feats=feats, labels=labels, clat=cat["clat"][feats], clon=cat["clon"][feats],
                 gamma=GAMMA, ics=ICS, tc=TC), allow_pickle=True)

    # quick read: strongest directed causal edges
    print("\nstrongest interventional causal edges (source -> target, effect):")
    flat = [(r, c, A[r, c]) for r in range(F) for c in range(F) if r != c]
    for r, c, w in sorted(flat, key=lambda x: -abs(x[2]))[:15]:
        print(f"  {labels[r]:>16}  ->  {labels[c]:<16}  {w:+.2f}", flush=True)
    # is the graph geographically causal? correlate edge strength with -distance (near-field wins)
    from math import radians
    print("\n-> results/fs_cgraph.npy")

if __name__ == "__main__":
    main()
