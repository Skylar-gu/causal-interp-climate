"""Heat-dome broad scan (GPU, bf16) — data-driven: encode ALL 4096 features on the ridge.

Roll the flagship from IC 2021-06-24 through the ridge peak (+144h). At each lead, encode
layer-8 -> SAE codes over the full mesh, and for EVERY feature record:
  - box_sum / box_cnt : firing summed / active-node count in the W-NA ridge box (45-62N,-135..-100)
  - firing centroid (lat,lon) over the NA ridge region and total NA firing -> PROPAGATION track
Plus node-level firing maps (float16) over the ridge region for the top-30 box-firing features
(for the propagation figure), and the model's own z500 ridge-anomaly max (baseline fidelity).

This is the full ridge firing, re-analyzable on CPU (heatdome_scan_analyze.py). No pre-selected
list. Serialize behind other GPU jobs. Crash-safe: results/heatdome/scan.npy.

Paper: not in the paper (2021 heat-dome study; results shipped, demo notebook)
Inputs: GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/heatdome
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.heatdome.heatdome_scan
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout
import graphcast_sae.heatdome.heatdome_config as C
from graphcast_sae.heatdome.heatdome_phase1 import numpyify, build_io, ridge_anom_max

OUT = fc.ROOT / "results/heatdome"; OUT.mkdir(parents=True, exist_ok=True)
NTOP = 30

def main():
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]; mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    la0, la1 = C.BOX["lat"]; blo0, blo1 = C.BOX["lon"]
    inbox = (mlat >= la0) & (mlat <= la1) & (mlon >= blo0) & (mlon <= blo1)
    reg = (mlat >= 25) & (mlat <= 75) & (mlon >= -165) & (mlon <= -85)   # NA ridge region for tracks/maps
    rlat = mlat[reg]; rlon = mlon[reg]
    print(f"box nodes {int(inbox.sum())}; region nodes {int(reg.sum())}", flush=True)

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply_cond(mc, tc, stats, sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = (jnp.zeros(sae.n_features, jnp.float32), jnp.zeros(sae.n_features, jnp.float32),
            jnp.zeros(len(mlat), jnp.float32))

    inp, tgt, frc = build_io(C.IC, C.H, tc)
    tct = tgt.time.isel(time=slice(0, 1))
    for cc in ("datetime",):
        if cc in tgt.coords: tgt = tgt.drop_vars(cc)
        if cc in frc.coords: frc = frc.drop_vars(cc)

    F = sae.n_features
    box_sum = np.zeros((C.H, F), np.float32); box_cnt = np.zeros((C.H, F), np.float32)
    cen_lat = np.zeros((C.H, F), np.float32); cen_lon = np.zeros((C.H, F), np.float32)
    reg_wsum = np.zeros((C.H, F), np.float32); ridge = np.zeros(C.H, np.float32)
    reg_codes = []
    cur = inp; t0 = time.time()
    for h in range(C.H):
        ct = tgt.isel(time=slice(h, h+1)).assign_coords(time=tct)
        cf = frc.isel(time=slice(h, h+1)).assign_coords(time=tct)
        p, a = apply(cur, ct, cf, noop)
        X = jnp.asarray(np.asarray(a, np.float32).reshape(-1, fc.D_IN))
        Cc = np.asarray(sae.codes(X))                       # (n_mesh, F)
        Cb = Cc[inbox]; Cr = Cc[reg]
        box_sum[h] = Cb.sum(0); box_cnt[h] = (Cb > 0).sum(0)
        ws = Cr.sum(0); reg_wsum[h] = ws
        safe = np.maximum(ws, 1e-6)
        cen_lat[h] = (Cr * rlat[:, None]).sum(0) / safe
        cen_lon[h] = (Cr * rlon[:, None]).sum(0) / safe
        reg_codes.append(Cr.astype(np.float16))
        p = numpyify(p); ridge[h] = ridge_anom_max(p, C.BOX)
        if h < C.H - 1:
            cur = rollout._get_next_inputs(cur, xr.merge([p, cf])).assign_coords(time=cur.coords["time"])
        if h % 4 == 0:
            k = int(box_cnt[h].argmax())
            print(f"  +{(h+1)*6:3d}h ridge={ridge[h]:.0f}m  top-firing feat={k} cnt={int(box_cnt[h,k])}  "
                  f"nstrong(cnt>=8)={int((box_cnt[h]>=8).sum())}   ({time.time()-t0:.0f}s)", flush=True)

    peak_cnt = box_cnt.max(0); peak_sum = box_sum.max(0)
    top = np.argsort(peak_cnt)[::-1][:NTOP]
    reg_maps_top = np.stack(reg_codes)[:, :, top]           # (H, n_reg, NTOP) f16
    out = dict(ic=C.IC, box=C.BOX, leads_h=(np.arange(C.H) + 1) * 6, ridge=ridge,
               box_sum=box_sum, box_cnt=box_cnt, cen_lat=cen_lat, cen_lon=cen_lon,
               reg_wsum=reg_wsum, peak_cnt=peak_cnt, peak_sum=peak_sum,
               top_feats=top.astype(int), reg_maps_top=reg_maps_top,
               reg_lat=rlat.astype(np.float32), reg_lon=rlon.astype(np.float32),
               ridge_center=(53.0, -120.75))
    np.save(OUT / "scan.npy", out, allow_pickle=True)
    print(f"\ntop-{NTOP} ridge-firing features by peak box_cnt: {list(top)}", flush=True)
    print(f"  peak_cnt: {[int(peak_cnt[f]) for f in top]}", flush=True)
    print(f"n strong (peak_cnt>=8): {int((peak_cnt>=8).sum())}", flush=True)
    print("-> results/heatdome/scan.npy", flush=True)

if __name__ == "__main__":
    main()
