"""Genesis knockout — which upstream features must GraphCast keep on to spin up Ida's TC feature.

Intervene at FORMATION (Aug 26 2021, when TC feature 3243 ~ 0), hold each tropical-Atlantic cast
feature persistently dosed (+1, enhance) or ablated (-1, remove) through a 48-h rollout, and track
feature 3243's activation in the Caribbean+Gulf box vs the untouched baseline (which spins Ida up).
  ablate suppresses genesis   -> the feature is causally NECESSARY for the model to form the cyclone
  dose accelerates genesis     -> causally SUFFICIENT to intensify it

Paper: not in the paper; kept for provenance only
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); results/fs_ida_trop.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/fs_ida_genesis.npy
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.legacy.steer_ida_genesis
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

TC = 3243
IC = "2021-08-26"
H = 8                                                          # 48 h
BOX = dict(lat=(10, 33), lon=(-98, -58))                       # Caribbean -> Gulf, Ida's track

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

def main():
    cast = list(np.load(fc.ROOT / "results/fs_ida_trop.npy", allow_pickle=True).item()["cast"])
    cat = np.load(fc.ROOT / "candidates/fs_feature_catalog.npy", allow_pickle=True).item()
    lo = np.where(cat["clon"] > 180, cat["clon"] - 360, cat["clon"])
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]; mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    inbox = (mlat >= BOX["lat"][0]) & (mlat <= BOX["lat"][1]) & (mlon >= BOX["lon"][0]) & (mlon <= BOX["lon"][1])

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    def tc_in_box(acts):
        X = jnp.asarray(np.asarray(acts, np.float32).reshape(-1, fc.D_IN))
        return float(np.asarray(sae.codes(X))[inbox, TC].sum())

    inp, tgt, frc = build_io(IC, tc, H)
    tct = tgt.time.isel(time=slice(0, 1))
    for c in ("datetime",):
        if c in tgt.coords: tgt = tgt.drop_vars(c)
        if c in frc.coords: frc = frc.drop_vars(c)

    def persistent_roll(patch):
        """apply `patch` at EVERY step (a sustained knockout); return TC-in-box trajectory."""
        cur = inp; traj = []
        for h in range(H):
            ct = tgt.isel(time=slice(h, h+1)).assign_coords(time=tct); cf = frc.isel(time=slice(h, h+1)).assign_coords(time=tct)
            preds, acts = apply(cur, ct, cf, patch)
            traj.append(tc_in_box(acts))
            preds = numpyify(preds)
            if h < H-1: cur = rollout._get_next_inputs(cur, xr.merge([preds, cf])).assign_coords(time=cur.coords["time"])
        return np.array(traj)

    t0 = time.time()
    base = persistent_roll(noop)
    print(f"baseline TC-feature genesis (box, +6h..+48h): {np.array2string(base, precision=0)}", flush=True)
    print(f"  Ida spins up: {base[0]:.0f} -> {base[-1]:.0f}\n", flush=True)

    rows = {}
    print(f"{'feature':>14}{'ablate:TC@48h':>15}{'Δ vs base':>11}{'dose:TC@48h':>13}{'Δ':>9}")
    for fi in cast:
        abl = persistent_roll(fc.coef_patch(sae, [int(fi)], -1.0))
        dos = persistent_roll(fc.coef_patch(sae, [int(fi)], +1.0))
        rows[fi] = dict(abl=abl, dos=dos)
        da = abl[-1] - base[-1]; dd = dos[-1] - base[-1]
        nec = "  <- NECESSARY" if da < -0.15*base[-1] else ""
        print(f"  {fi}({cat['clat'][fi]:+.0f},{lo[fi]:+.0f})".rjust(14) +
              f"{abl[-1]:>15.0f}{da:>+11.0f}{dos[-1]:>13.0f}{dd:>+9.0f}{nec}", flush=True)
    np.save(fc.ROOT / "results/fs_ida_genesis.npy",
            dict(base=base, rows=rows, cast=cast, box=BOX, ic=IC), allow_pickle=True)
    print(f"\n{(time.time()-t0)/60:.1f}m  -> results/fs_ida_genesis.npy")

if __name__ == "__main__":
    main()
