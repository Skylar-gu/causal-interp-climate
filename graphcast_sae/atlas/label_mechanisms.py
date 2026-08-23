"""Label SAE features by the TC-genesis MECHANISM they track (not by geography).

Classic genesis ingredients, computed from ERA5 at the i.i.d. dump windows:
  vort850  850 hPa relative vorticity   (the pre-existing disturbance / easterly wave)
  q600     600 hPa specific humidity     (mid-level moisture)
  ascent   -omega500                     (mid-level upward motion / convection)
  shear    |V200 - V850|                 (deep-layer wind shear; genesis wants this LOW)
For each feature f and mechanism m: z[f,m] = mean over f's ACTIVE nodes of the per-window
standardized m. Large +z => the feature fires where that mechanism is anomalously strong => it
"is" that mechanism. Runs in the JAX env (needs gcsfs/xarray); numpy SAE encode (no torch).

Paper: Sec. 3 (v1 labels; superseded by label_mechanisms_v2, kept because results/fs_mechanisms.npy ships)
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); results/fs_ida_trop.npy (not shipped, see docs/REPRODUCE.md); WeatherBench2 ERA5 (GCS, streamed); i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/fs_mechanisms.npy
Run:   # JAX env, CPU
    FS_DEVICE=cpu python -m graphcast_sae.atlas.label_mechanisms
"""
import json, os, sys
import numpy as np

import graphcast_sae.common.fs_common as fc

NW_USE = 60                                                    # subsample dump windows for speed
LV = dict(p200=200, p500=500, p600=600, p850=850)
SAE_NPZ = fc.WEIGHTS / "sae_k32_lat4096_lay08.npz"
META = json.load(open(fc.SCRATCH / "fs_iid_meta.json"))
DUMP = fc.SCRATCH / "fs_iid_dump.npy"
L, DIM = META["n_mesh"], META["dim"]

def sae_np():
    z = np.load(SAE_NPZ); return z["W_enc"], z["b_pre"]

def encode(A, Wenc, bpre, k=32):
    xn = A - A.mean(1, keepdims=True); xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
    pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
    idx = np.argpartition(-pre, k, axis=1)[:, :k]
    out = np.zeros_like(pre); r = np.arange(len(A))[:, None]; out[r, idx] = pre[r, idx]
    return out                                                # (nodes, F) dense TopK

def main():
    Wenc, bpre = sae_np(); F = Wenc.shape[0]
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat, mlon = geom["lat"], np.mod(geom["lon"], 360)
    ds, _ = fc.open_wb2()
    glat = np.asarray(ds.lat.values); glon = np.mod(np.asarray(ds.lon.values), 360)
    # nearest grid index for each mesh node (regular grid -> searchsorted)
    iy = np.clip(np.searchsorted(glat, mlat), 0, len(glat) - 1)
    ix = np.clip(np.searchsorted(np.sort(glon), mlon), 0, len(glon) - 1)
    lon_sort = np.argsort(glon); ix = lon_sort[ix]
    dphi = np.gradient(np.radians(glat)); dlam = np.gradient(np.radians(np.sort(glon)))
    R = 6.371e6; coslat = np.cos(np.radians(glat))[:, None]

    starts = np.array(META["starts"])[np.linspace(0, META["n_windows"] - 1, NW_USE).astype(int)]
    X = np.load(DUMP, mmap_mode="r")
    all_starts = list(META["starts"])
    mech_names = ["vort850", "q600", "ascent", "shear"]
    zsum = np.zeros((F, 4)); zcnt = np.zeros(F)
    for wi, s in enumerate(starts):
        t = np.datetime64(str(s)[:19])
        d = ds[["u_component_of_wind", "v_component_of_wind", "specific_humidity", "vertical_velocity"]]\
            .sel(time=t, method="nearest").sel(level=list(LV.values())).load()
        u = d["u_component_of_wind"].values; v = d["v_component_of_wind"].values     # (4lev, lat, lon)
        li = {p: k for k, p in enumerate(LV.values())}
        u850, v850 = u[li[850]], v[li[850]]; u200, v200 = u[li[200]], v[li[200]]
        # 850 relative vorticity = dv/dx - du/dy on the lat/lon grid
        dvdx = np.gradient(v850, axis=1) / (dlam[None, :] * R * coslat + 1e-9)
        dudy = np.gradient(u850, axis=0) / (dphi[:, None] * R + 1e-9)
        vort = dvdx - dudy
        q600 = d["specific_humidity"].values[li[600]]
        ascent = -d["vertical_velocity"].values[li[500]]
        shear = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)
        fields = [vort, q600, ascent, shear]
        # sample to mesh nodes + standardize per-window
        node_m = np.stack([f[iy, ix] for f in fields], 1)                            # (L, 4)
        node_m = (node_m - node_m.mean(0)) / (node_m.std(0) + 1e-9)
        A = np.asarray(X[all_starts.index(str(s)) * L:(all_starts.index(str(s)) + 1) * L], np.float32)
        code = encode(A, Wenc, bpre)
        act = code > 0                                                                # (L, F)
        zsum += act.T.astype(np.float32) @ node_m                                     # (F,4)
        zcnt += act.sum(0)
        if wi % 15 == 0: print(f"  window {wi}/{NW_USE}", flush=True)
    z = zsum / np.maximum(zcnt, 1)[:, None]
    np.save(fc.ROOT / "results/fs_mechanisms.npy",
            dict(z=z, mech=mech_names, active_count=zcnt), allow_pickle=True)

    cat = np.load(fc.ROOT / "candidates/fs_feature_catalog.npy", allow_pickle=True).item()
    lo = np.where(cat["clon"] > 180, cat["clon"] - 360, cat["clon"])
    print(f"\nStrongest feature for each TC-genesis mechanism (z = how anomalous the mechanism is where it fires):")
    for m, name in enumerate(mech_names):
        good = np.where(zcnt > 500)[0]
        top = good[np.argsort(-z[good, m])[:5]]
        print(f"\n {name}:")
        for fi in top:
            print(f"   feat {fi:>4}  z={z[fi,m]:+.2f}  (other mechs: " +
                  ", ".join(f'{mech_names[k]} {z[fi,k]:+.1f}' for k in range(4) if k != m) +
                  f")  home=({cat['clat'][fi]:+.0f},{lo[fi]:+.0f})", flush=True)
    # the Ida tropical cast: what does each track?
    cast = list(np.load(fc.ROOT / "results/fs_ida_trop.npy", allow_pickle=True).item()["cast"])
    print(f"\nIda tropical cast — mechanism each feature tracks:")
    print(f"  {'feat':>5}  " + "".join(f"{n:>9}" for n in mech_names) + "   -> label")
    for fi in cast:
        lab = mech_names[int(np.argmax(z[fi]))] if z[fi].max() > 0.15 else "(none clear)"
        print(f"  {fi:>5}  " + "".join(f"{z[fi,k]:>+9.2f}" for k in range(4)) + f"   -> {lab}", flush=True)
    print("-> results/fs_mechanisms.npy")

if __name__ == "__main__":
    main()
