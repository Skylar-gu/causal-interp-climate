"""Shared engine for the representation atlas: label all 4096 features against everything we know.

Reference battery (from ERA5 at the i.i.d. dump windows):
  PHYSICAL (node-level): vort850, q600(moist), ascent(-w500), shear|V200-V850|, t850,
                         z500(flow), jet|V250|, div250(upper divergence)
  GEOGRAPHIC (static):   |lat|, land_sea, orography
  TEMPORAL (per-window): seasonal phase, diurnal phase
Outputs z[f, ref] (how anomalous each ref is where feature f fires) + temporal amplitude[f,2].
Feeds A (known atlas + census) and B (residual discovery: high-firing features matching nothing).

Paper: Sec. 3, mechanism labels that define the intervention groups
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); WeatherBench2 ERA5 (GCS, streamed); i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/fs_atlas.npy
Run:   # JAX env, CPU
    FS_DEVICE=cpu python -m graphcast_sae.atlas.label_expanded
"""
import json, os, sys
import numpy as np

import graphcast_sae.common.fs_common as fc

NW_USE = 72
LEVELS = [200, 250, 500, 600, 850]
SAE_NPZ = fc.WEIGHTS / "sae_k32_lat4096_lay08.npz"
META = json.load(open(fc.SCRATCH / "fs_iid_meta.json"))
L, DIM = META["n_mesh"], META["dim"]
DUMP = fc.SCRATCH / "fs_iid_dump.npy"
PHYS = ["vort850", "q600", "ascent", "shear", "t850", "z500", "jet250", "div250"]
GEO = ["abslat", "land_sea", "orography"]
NODE_REFS = PHYS + GEO

def encode(A, Wenc, bpre, k=32):
    xn = A - A.mean(1, keepdims=True); xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
    pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
    idx = np.argpartition(-pre, k, axis=1)[:, :k]
    out = np.zeros_like(pre); r = np.arange(len(A))[:, None]; out[r, idx] = pre[r, idx]
    return out

def main():
    z = np.load(SAE_NPZ); Wenc, bpre = z["W_enc"], z["b_pre"]; F = Wenc.shape[0]
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat, mlon = geom["lat"], np.mod(geom["lon"], 360)
    ds, statics = fc.open_wb2()
    glat = np.asarray(ds.lat.values); glon_raw = np.asarray(ds.lon.values); glon = np.mod(glon_raw, 360)
    iy = np.clip(np.searchsorted(glat, mlat), 0, len(glat) - 1)
    order = np.argsort(glon); glon_s = glon[order]
    ix = order[np.clip(np.searchsorted(glon_s, mlon), 0, len(glon) - 1)]
    dphi = np.gradient(np.radians(glat)); dlam = np.gradient(np.radians(glon_s)); R = 6.371e6
    coslat = np.cos(np.radians(glat))[:, None]

    def deriv_x(f): return np.gradient(f[:, order], axis=1)[:, np.argsort(order)] / (dlam[None, :][:, np.argsort(order)] * R * coslat + 1e-9)
    def deriv_y(f): return np.gradient(f, axis=0) / (dphi[:, None] * R + 1e-9)

    # static per-node references (constant across windows)
    sm = statics["land_sea_mask"].values; og = statics["geopotential_at_surface"].values
    stat = {"abslat": np.abs(mlat), "land_sea": sm[iy, ix], "orography": og[iy, ix]}

    starts_all = list(META["starts"])
    starts = np.array(starts_all)[np.linspace(0, META["n_windows"] - 1, NW_USE).astype(int)]
    X = np.load(DUMP, mmap_mode="r")
    zsum = np.zeros((F, len(NODE_REFS))); zcnt = np.zeros(F)
    actW = np.zeros((NW_USE, F)); doy = np.zeros(NW_USE); hr = np.zeros(NW_USE)

    for wi, s in enumerate(starts):
        t = np.datetime64(str(s)[:19])
        d = ds[["u_component_of_wind", "v_component_of_wind", "specific_humidity",
                "vertical_velocity", "temperature", "geopotential"]]\
            .sel(time=t, method="nearest").sel(level=LEVELS).load()
        li = {p: k for k, p in enumerate(LEVELS)}
        u = d["u_component_of_wind"].values; v = d["v_component_of_wind"].values
        u850, v850, u200, v200, u250, v250 = u[li[850]], v[li[850]], u[li[200]], v[li[200]], u[li[250]], v[li[250]]
        vort = deriv_x(v850) - deriv_y(u850)
        div250 = deriv_x(u250) + deriv_y(v250)
        gfields = dict(vort850=vort, q600=d["specific_humidity"].values[li[600]],
                       ascent=-d["vertical_velocity"].values[li[500]],
                       shear=np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2),
                       t850=d["temperature"].values[li[850]],
                       z500=d["geopotential"].values[li[500]] / 9.81,
                       jet250=np.sqrt(u250 ** 2 + v250 ** 2), div250=div250)
        node = np.stack([gfields[p][iy, ix] for p in PHYS] + [stat[g] for g in GEO], 1)  # (L, 11)
        node = (node - node.mean(0)) / (node.std(0) + 1e-9)
        A = np.asarray(X[starts_all.index(str(s)) * L:(starts_all.index(str(s)) + 1) * L], np.float32)
        code = encode(A, Wenc, bpre); act = code > 0
        zsum += act.T.astype(np.float32) @ node; zcnt += act.sum(0)
        actW[wi] = act.sum(0)
        dt = np.datetime64(str(s)[:19]); y0 = np.datetime64(str(s)[:4] + "-01-01")
        doy[wi] = (dt - y0) / np.timedelta64(1, "D"); hr[wi] = int(str(s)[11:13])
        if wi % 18 == 0: print(f"  window {wi}/{NW_USE}", flush=True)

    zc = zsum / np.maximum(zcnt, 1)[:, None]
    # temporal amplitude: corr of per-window activation with season/diurnal phase
    def amp(phase_period, x):
        c = np.cos(2 * np.pi * x / phase_period); s = np.sin(2 * np.pi * x / phase_period)
        A = actW - actW.mean(0)
        rc = (A * (c - c.mean())[:, None]).sum(0) / (np.sqrt((A ** 2).sum(0)) * np.sqrt(((c - c.mean()) ** 2).sum()) + 1e-9)
        rs = (A * (s - s.mean())[:, None]).sum(0) / (np.sqrt((A ** 2).sum(0)) * np.sqrt(((s - s.mean()) ** 2).sum()) + 1e-9)
        return np.hypot(rc, rs)
    season = amp(365.25, doy); diurnal = amp(24.0, hr)

    cat = np.load(fc.ROOT / "candidates/fs_feature_catalog.npy", allow_pickle=True).item()
    np.save(fc.ROOT / "results/fs_atlas.npy",
            dict(z=zc, node_refs=NODE_REFS, phys=PHYS, geo=GEO, season=season, diurnal=diurnal,
                 zcnt=zcnt, firerate=cat["firerate"], clat=cat["clat"], clon=cat["clon"],
                 coh=cat["coh"], nw=NW_USE), allow_pickle=True)
    print(f"\nlabeled {F} features against {len(NODE_REFS)} node refs + season + diurnal")
    print(f"-> results/fs_atlas.npy")

if __name__ == "__main__":
    main()
