"""TEST 1 — is f2075's fixed support just climatology?

A near-constant load-bearing spatial field has at least three explanations: a positional
basis, the climatological mean state ("what normally happens here"), or a DC/bias absorber.
Climatology is the cheapest to kill: if the bowties coincide with a real climatological
structure, the geometry is coincidence and there is nothing interesting here.

Sample ERA5 climatological means at the mesh nodes and correlate them with the feature's mean
activation pattern. A climatology feature should light up where its field is extreme.

Paper: Sec. 3, grid-locked-feature ablations (demo notebook part 4)
Inputs: WeatherBench2 ERA5 (GCS, streamed); i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: printed report
Run:   # JAX env, CPU
    python -m graphcast_sae.gridlock.f2075_climatology
"""
import json
import os
import sys

import numpy as np

import graphcast_sae.common.fs_common as fc

NW = 8
FEATS = [int(x) for x in os.environ.get("CL_FEATS", "2075,2954,2109,2474").split(",")]
FIELDS = [("w500", "vertical_velocity", 500), ("q700", "specific_humidity", 700),
          ("t850", "temperature", 850), ("z500", "geopotential", 500),
          ("u250", "u_component_of_wind", 250), ("q925", "specific_humidity", 925),
          ("t2m", "2m_temperature", None), ("msl", "mean_sea_level_pressure", None)]

def main():
    from scipy import stats
    z = np.load(fc.WEIGHTS / "sae_k32_lat4096_lay08.npz")
    Wenc, bpre = z["W_enc"], z["b_pre"]
    META = json.load(open(fc.SCRATCH / "fs_iid_meta.json"))
    L = META["n_mesh"]
    X = np.load(fc.SCRATCH / "fs_iid_dump.npy", mmap_mode="r")
    sel = np.linspace(0, META["n_windows"] - 1, NW).astype(int)
    acc = None
    for j in sel:
        A = np.asarray(X[j * L:(j + 1) * L], np.float32)
        xn = A - A.mean(1, keepdims=True)
        xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
        pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
        idx = np.argpartition(-pre, 32, axis=1)[:, :32]
        C = np.zeros_like(pre); r = np.arange(len(A))[:, None]; C[r, idx] = pre[r, idx]
        acc = C if acc is None else acc + C
    acc /= NW

    g = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = np.asarray(g["lat"], float)
    mlon = np.asarray(g["lon"], float) % 360.0

    ds, _ = fc.open_wb2()
    times = [np.datetime64(str(META["starts"][j])) for j in sel]
    clim = {}
    for nm, var, lev in FIELDS:
        d = ds[var].sel(time=times)
        if lev is not None:
            d = d.sel(level=lev)
        a = np.asarray(d.mean("time").transpose("lat", "lon").values, float)
        la = np.asarray(ds.lat.values, float); lo = np.asarray(ds.lon.values, float)
        i = np.abs(mlat[:, None] - la[None, :]).argmin(1)
        k = np.abs(((mlon[:, None] - lo[None, :] + 180) % 360) - 180).argmin(1)
        clim[nm] = a[i, k]
        print(f"  sampled {nm}", flush=True)

    print(f"\nSpearman of each feature's mean activation against the climatological field,")
    print(f"over all {L:,} mesh nodes. A climatology feature should show a strong |rho|.\n")
    print(f"{'feature':<9}" + "".join(f"{n:>9}" for n, _, _ in FIELDS) + f"{'max|rho|':>10}")
    for f in FEATS:
        v = acc[:, f]
        row = [stats.spearmanr(v, clim[n]).statistic for n, _, _ in FIELDS]
        print(f"f{f:<8}" + "".join(f"{x:>+9.3f}" for x in row) +
              f"{max(abs(x) for x in row):>10.3f}")
    print("\nalso restricted to the feature's OWN support (does the field explain WHERE "
          "inside it the feature is strong?)")
    print(f"{'feature':<9}" + "".join(f"{n:>9}" for n, _, _ in FIELDS))
    for f in FEATS:
        v = acc[:, f]; on = v > 0
        if on.sum() < 50:
            continue
        row = [stats.spearmanr(v[on], clim[n][on]).statistic for n, _, _ in FIELDS]
        print(f"f{f:<8}" + "".join(f"{x:>+9.3f}" for x in row))

if __name__ == "__main__":
    main()
