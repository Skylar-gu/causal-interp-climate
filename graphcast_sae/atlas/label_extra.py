"""Detector battery for the harder encoding categories (extends the atlas).

Adds signals that separate the six hypotheses:
  NUMERICAL machinery:  coast_grad |grad land_sea|, orog_grad |grad orography|, node_density,
                        and STATICNESS = duty cycle (fires the same nodes every window = geometric).
  TELECONNECTION:       blocking (z500 anomaly vs zonal mean), atm_river (q850*|V850| transport),
                        enso (per-window equatorial-Pacific t850 anomaly; interannual).
  PREDICTABILITY/regime: baroclinicity (Eady ~ shear*|sin lat|; storm-track = low predictability).
Merges into results/fs_atlas.npy -> results/fs_atlas_extra.npy.

Paper: Sec. 3, mechanism labels that define the intervention groups
Inputs: WeatherBench2 ERA5 (GCS, streamed); i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/fs_atlas.npy; results/fs_atlas_extra.npy
Run:   # JAX env, CPU
    FS_DEVICE=cpu python -m graphcast_sae.atlas.label_extra
"""
import json, os, sys
import numpy as np

import graphcast_sae.common.fs_common as fc

NW_USE = 72
LEVELS = [200, 500, 850]
SAE_NPZ = fc.WEIGHTS / "sae_k32_lat4096_lay08.npz"
META = json.load(open(fc.SCRATCH / "fs_iid_meta.json"))
L = META["n_mesh"]
DUMP = fc.SCRATCH / "fs_iid_dump.npy"
NODE = ["coast_grad", "orog_grad", "node_density", "blocking", "atm_river", "baroclinicity"]

def encode(A, Wenc, bpre, k=32):
    xn = A - A.mean(1, keepdims=True); xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
    pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
    idx = np.argpartition(-pre, k, axis=1)[:, :k]
    out = np.zeros_like(pre); r = np.arange(len(A))[:, None]; out[r, idx] = pre[r, idx]
    return out

def main():
    z = np.load(SAE_NPZ); Wenc, bpre = z["W_enc"], z["b_pre"]; F = Wenc.shape[0]
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat, mlon, xyz = geom["lat"], np.mod(geom["lon"], 360), geom["xyz"]
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    node_density = np.array([len(tree.query_ball_point(p, 0.05)) for p in xyz], float)  # ~300km ball

    ds, statics = fc.open_wb2()
    glat = np.asarray(ds.lat.values); glon = np.mod(np.asarray(ds.lon.values), 360)
    iy = np.clip(np.searchsorted(glat, mlat), 0, len(glat) - 1)
    order = np.argsort(glon); ix = order[np.clip(np.searchsorted(glon[order], mlon), 0, len(glon) - 1)]
    sm = statics["land_sea_mask"].values; og = statics["geopotential_at_surface"].values
    cg = np.hypot(*np.gradient(sm))[iy, ix]; ogg = np.hypot(*np.gradient(og))[iy, ix]
    sinlat = np.abs(np.sin(np.radians(mlat)))
    nino = (glat >= -5) & (glat <= 5); ninox = (glon >= 190) & (glon <= 240)

    starts_all = list(META["starts"])
    starts = np.array(starts_all)[np.linspace(0, META["n_windows"] - 1, NW_USE).astype(int)]
    X = np.load(DUMP, mmap_mode="r")
    zsum = np.zeros((F, len(NODE))); zcnt = np.zeros(F)
    fire_count = np.zeros((F, L), np.uint16)
    actW = np.zeros((NW_USE, F)); enso_idx = np.zeros(NW_USE)

    for wi, s in enumerate(starts):
        t = np.datetime64(str(s)[:19])
        d = ds[["u_component_of_wind", "v_component_of_wind", "specific_humidity",
                "temperature", "geopotential"]].sel(time=t, method="nearest").sel(level=LEVELS).load()
        li = {p: k for k, p in enumerate(LEVELS)}
        u2, v2 = d["u_component_of_wind"].values[li[200]], d["v_component_of_wind"].values[li[200]]
        u8, v8 = d["u_component_of_wind"].values[li[850]], d["v_component_of_wind"].values[li[850]]
        q8 = d["specific_humidity"].values[li[850]]; z5 = d["geopotential"].values[li[500]] / 9.81
        t8 = d["temperature"].values[li[850]]
        blocking = z5 - z5.mean(1, keepdims=True)                 # anomaly vs zonal mean
        ar = q8 * np.hypot(u8, v8)                                # moisture transport
        shear = np.hypot(u2 - u8, v2 - v8)
        enso_idx[wi] = t8[np.ix_(nino, ninox)].mean() - t8[(glat >= -20) & (glat <= 20)].mean()
        node = np.stack([cg, ogg, node_density,
                         blocking[iy, ix], ar[iy, ix], shear[iy, ix] * sinlat], 1)
        node = (node - node.mean(0)) / (node.std(0) + 1e-9)
        A = np.asarray(X[starts_all.index(str(s)) * L:(starts_all.index(str(s)) + 1) * L], np.float32)
        act = encode(A, Wenc, bpre) > 0
        zsum += act.T.astype(np.float32) @ node; zcnt += act.sum(0)
        fire_count += act.T.astype(np.uint16)
        actW[wi] = act.sum(0)
        if wi % 18 == 0: print(f"  window {wi}/{NW_USE}", flush=True)

    zc = zsum / np.maximum(zcnt, 1)[:, None]
    # STATICNESS: among a feature's ever-active nodes, mean duty cycle (fraction of windows active)
    ever = fire_count > 0
    staticness = np.where(ever.sum(1) > 0, (fire_count / NW_USE).sum(1) / np.maximum(ever.sum(1), 1), 0)
    # ENSO amplitude: |corr| of per-window activation with the enso index
    Aw = actW - actW.mean(0); e = enso_idx - enso_idx.mean()
    enso_amp = np.abs((Aw * e[:, None]).sum(0) / (np.sqrt((Aw ** 2).sum(0)) * np.sqrt((e ** 2).sum()) + 1e-9))

    at = np.load(fc.ROOT / "results/fs_atlas.npy", allow_pickle=True).item()
    at.update(dict(z_extra=zc, node_extra=NODE, staticness=staticness, enso_amp=enso_amp))
    np.save(fc.ROOT / "results/fs_atlas_extra.npy", at, allow_pickle=True)
    print(f"\nextra detectors done: {NODE} + staticness + enso")
    print(f"  staticness range {staticness.min():.2f}-{staticness.max():.2f} (1=fires same nodes every window=geometric)")
    print("-> results/fs_atlas_extra.npy")

if __name__ == "__main__":
    main()
