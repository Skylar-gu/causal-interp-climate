"""LATITUDE-BANDED relabelling — the repair for a systematic, diagnosed labelling bug.

THE BUG. label_expanded.py standardises each physical reference field GLOBALLY:
    node = (node - node.mean(0)) / (node.std(0) + 1e-9)      # over all 40,962 mesh nodes
Over the poles almost every field is the largest anomaly on Earth: t850 is extremely cold,
z500 extremely low, |V250| far from the global mean. So "where is field m most anomalous"
resolves to "the poles" for nearly every m, and the purity selection then picks the SAME
polar features over and over, relabelling them each time.

MEASURED CONSEQUENCE. 24 of the 40 features in the K=4 concept groups sit poleward of 65 deg:
    t850          centroids -81 -80 -80 -80      z500      -78 -84 -82 -86
    blocking      centroids -85  72 -84 -85      jet250    -69  66 -80 -67
    baroclinicity centroids -84  73 -85  80      vort850    71 -83 -87  75
It also explains every downstream oddity: t850 features score NEGATIVE on t850 (Antarctica is
cold), blocking features score NEGATIVE on the blocking index (they are TROUGHS, the opposite
of a block), all six groups are SILENT inside a hurricane, and all six came out INERT under
the interventional label test.

THE FIX, one line. Standardise each physical field WITHIN LATITUDE BANDS, so "anomalous"
means "anomalous for this latitude" -- which is the physically meaningful notion and the one
the labels always claimed to use. GEO refs (|lat|, land_sea, orography) stay globally
standardised: |lat| is constant within a band and would blow up.

Everything else is identical to label_expanded.py, so the two are directly comparable.
Emits results/fs_atlas_banded.npy with the same keys.

Paper: Sec. 3, mechanism labels that define the intervention groups
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); WeatherBench2 ERA5 (GCS, streamed); i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/fs_atlas_banded.npy
Run:   # JAX env, CPU
    FS_DEVICE=cpu python -m graphcast_sae.atlas.label_banded
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

NBANDS = 18                       # 10-degree bands; ~2,275 mesh nodes each

def band_standardize(node, mlat, n_phys):
    """Standardise the PHYSICAL columns within latitude bands, GEO columns globally.

    This is the whole repair. Global standardisation makes "most anomalous" mean "polar"
    for nearly every field; within-band makes it mean "unusual for this latitude".
    """
    out = node.copy()
    edges = np.linspace(-90, 90, NBANDS + 1)
    bi = np.clip(np.digitize(mlat, edges) - 1, 0, NBANDS - 1)
    for b in range(NBANDS):
        m = bi == b
        if m.sum() < 30:                       # too few nodes to standardise safely
            continue
        blk = out[m][:, :n_phys]
        out[np.ix_(m, np.arange(n_phys))] = (blk - blk.mean(0)) / (blk.std(0) + 1e-9)
    g = out[:, n_phys:]
    out[:, n_phys:] = (g - g.mean(0)) / (g.std(0) + 1e-9)
    return out

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
        node = band_standardize(node, mlat, len(PHYS))
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
    np.save(fc.ROOT / "results/fs_atlas_banded.npy",
            dict(z=zc, node_refs=NODE_REFS, phys=PHYS, geo=GEO, season=season, diurnal=diurnal,
                 zcnt=zcnt, firerate=cat["firerate"], clat=cat["clat"], clon=cat["clon"],
                 coh=cat["coh"], nw=NW_USE), allow_pickle=True)
    print(f"\nlabeled {F} features against {len(NODE_REFS)} node refs + season + diurnal")
    print(f"-> results/fs_atlas_banded.npy")

if __name__ == "__main__":
    main()
