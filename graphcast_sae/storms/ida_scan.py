"""Hurricane Ida — confirm the TC feature fires, and identify the local cast of features.

For a few timesteps around Ida (Gulf of Mexico, Aug 2021): run the forward, encode, and report
  - MSLP min in the Ida box (is the storm there?)
  - feature 3243 (TC) activation in the box (does the TC feature fire on Ida?)
  - the top features firing in the box = the interpretable local cast for the causal chain.

Paper: Fig. 5, Ida dial-up (defines the local cast)
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/fs_ida_cast.npy
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.storms.ida_scan
"""
import os, sys
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax.numpy as jnp
import graphcast_sae.common.fs_common as fc

TC = 3243
ICS = ["2021-08-26", "2021-08-27", "2021-08-28", "2021-08-29"]
BOX = dict(lat=(14, 32), lon=(-98, -74))                       # Caribbean + Gulf of Mexico

def main():
    def lon_in(lon):
        lo = np.where(lon > 180, lon - 360, lon)              # normalize any 0-360 grid to -180..180
        return (lo >= BOX["lon"][0]) & (lo <= BOX["lon"][1])

    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat, mlon = geom["lat"], geom["lon"]
    inbox = (mlat >= BOX["lat"][0]) & (mlat <= BOX["lat"][1]) & lon_in(mlon)
    print(f"Ida box: {inbox.sum()} mesh nodes")
    cat = np.load(fc.ROOT / "candidates/fs_feature_catalog.npy", allow_pickle=True).item()

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    box_acc = np.zeros(sae.n_features)
    for ic in ICS:
        blk = fc.load_block(np.datetime64(ic), nframes=fc.INPUT_WINDOW)
        inp, tgt, frc = fc.build_batch_inputs([blk], 0, tc)
        preds, acts = apply(inp, tgt, frc, noop)
        # storm present?
        mslp = np.asarray(preds["mean_sea_level_pressure"].isel(batch=0, time=0).values)
        glat = np.asarray(blk[0]["lat"].values); glon = np.asarray(blk[0]["lon"].values)
        gla = (glat >= BOX["lat"][0]) & (glat <= BOX["lat"][1]); glo = lon_in(glon)
        pmin = mslp[np.ix_(gla, glo)].min() / 100
        # features firing in the box
        X = jnp.asarray(np.asarray(acts, np.float32).reshape(-1, fc.D_IN))
        code = np.asarray(sae.codes(X))                        # (nodes, 4096)
        box_code = code[inbox].sum(0)                          # per-feature activation in the box
        box_acc += box_code
        print(f"{ic}: box MSLP_min={pmin:6.1f} hPa   feat{TC}(TC) box-activation={box_code[TC]:8.1f}", flush=True)

    print(f"\nTop 20 features firing in the Ida box (the causal-chain cast):")
    print(f"  {'feat':>5}{'box-act':>10}{'firerate':>10}{'coh_km':>8}   centroid")
    for fi in np.argsort(-box_acc)[:20]:
        print(f"  {fi:>5}{box_acc[fi]:>10.1f}{cat['firerate'][fi]:>10.4f}{cat['coh'][fi]:>8.0f}"
              f"   ({cat['clat'][fi]:+.0f},{cat['clon'][fi]:+.0f})"
              f"{'   <- TC feature' if fi == TC else ''}", flush=True)
    np.save(fc.ROOT / "results/fs_ida_cast.npy",
            dict(box_acc=box_acc, ics=ICS, box=BOX, tc=TC), allow_pickle=True)
    print("-> results/fs_ida_cast.npy")

if __name__ == "__main__":
    main()
