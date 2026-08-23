"""What are f3357, f3319 and f3004? Evidence, not labels.

These three are the endpoints of the ONLY edges that PCMCI+, LPCMCI and J-PCMCI+ all agree
on (f3357 -> f3319 and f3357 -> f3004, both tau=2 == 12 h). Nothing in the repo says what
they are: f3357 sits in the `ASCENT` exclusion set of core_control.py:26-27 (a hand-kept
list, not a measurement) and the other two are unlabelled.

Four independent lines of evidence, none of which reuses an existing label:

  fields   node-level correlation of each feature's firing against ERA5 fields sampled at
           the 40,962 M6 mesh nodes, over 16 IID windows drawn from the same 2016-2020
           dump the atlas used. Fields are standardised WITHIN 18 latitude bands
           (label_banded.py's repair), because global standardisation makes "anomalous"
           mean "polar" for nearly every field -- the diagnosed bug behind the repo's
           contaminated labels. Calibration is the whole 4,096-feature dictionary: a
           feature's correlation is reported as a percentile of the dictionary's own
           distribution for that field, so a bar can fail.
  foot     footprint geometry from results/hybrid_footprint_fires.npz. Masks are the
           COLUMNS, fires[:, j] -- indexing by row has already caused one retraction.
  traj     in-box activation trajectories from results/skill/hyb_series{,_sh,_mega}/,
           correlated against each storm's own MSLP trace at lags -4..+4 (6 h steps).
  edge     the f3357 -> f3319 lead-lag, by eye and by cross-correlation, to say whether the
           estimators' tau=2 is visible in the trajectories at all.

Paper: Appendix app:parity (the f3357 -> f3319 / f3004 edges)
Inputs: results/fs_cgv2_actseries.npy (not shipped, see docs/REPRODUCE.md); results/hybrid_footprint_fires.npz (not shipped, see docs/REPRODUCE.md); results/skill (shipped); WeatherBench2 ERA5 (GCS, streamed); i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: figures/edge_features.png; results/edge_features_evidence.json; results/edge_features_fields.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.appendix.identify_edge_features fields   # ~10 min, network, CPU only
    python -m graphcast_sae.appendix.identify_edge_features rest     # footprints + trajectories + JSON
    python -m graphcast_sae.appendix.identify_edge_features figure
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT, SCRATCH, MESH_GEOM
WEIGHTS = ROOT / "graphcast_sae/weights"
OUT = ROOT / "results/edge_features_evidence.json"
FIELDS_NPY = ROOT / "results/edge_features_fields.npy"

TARGETS = [3357, 3319, 3004]
CONV = [2401, 2067, 3174]          # frozen convection group
TC = 3243                          # TC readout
REF = TARGETS + CONV + [TC]

NW = 16                            # IID windows for the field pass
NBANDS = 18                        # 10-degree latitude bands (label_banded.py)
LEVELS = [200, 250, 400, 500, 600, 700, 850]

# fields built at every mesh node; "phys" are band-standardised, "geo" globally
PHYS = ["ascent500", "ascent700", "q850", "q600", "q400", "t850", "t400", "dt400_850",
        "z500", "vort850", "vort500", "div250", "div850", "shear200_850", "jet250",
        "wspd850", "mslp", "qflux850"]
GEO = ["abslat", "land_sea", "orography"]

# ------------------------------------------------------------------ shared -----
def mesh():
    g = np.load(MESH_GEOM, allow_pickle=True).item()
    return np.asarray(g["lat"], float), np.mod(np.asarray(g["lon"], float), 360.0)

def encode(A, Wenc, bpre, k=32):
    """authors' TopK SAE encoder, numpy. A (n,512) raw -> dense code (n,F)."""
    xn = A - A.mean(1, keepdims=True)
    xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
    pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
    idx = np.argpartition(-pre, k, axis=1)[:, :k]
    out = np.zeros_like(pre)
    r = np.arange(len(A))[:, None]
    out[r, idx] = pre[r, idx]
    return out

def band_standardize(node, mlat, n_phys):
    """label_banded.py's repair: physical columns standardised within latitude bands."""
    out = node.copy()
    edges = np.linspace(-90, 90, NBANDS + 1)
    bi = np.clip(np.digitize(mlat, edges) - 1, 0, NBANDS - 1)
    for b in range(NBANDS):
        m = bi == b
        if m.sum() < 30:
            continue
        blk = out[m][:, :n_phys]
        out[np.ix_(m, np.arange(n_phys))] = (blk - blk.mean(0)) / (blk.std(0) + 1e-9)
    g = out[:, n_phys:]
    out[:, n_phys:] = (g - g.mean(0)) / (g.std(0) + 1e-9)
    return out

# ------------------------------------------------------------------ fields -----
def stage_fields():
    import xarray as xr
    meta = json.load(open(SCRATCH / "fs_iid_meta.json"))
    L, DIM, NWIN = meta["n_mesh"], meta["dim"], meta["n_windows"]
    starts_all = list(meta["starts"])
    X = np.load(SCRATCH / "fs_iid_dump.npy", mmap_mode="r")
    assert X.shape == (NWIN * L, DIM), X.shape                      # data gate
    z = np.load(WEIGHTS / "sae_k32_lat4096_lay08.npz")
    Wenc, bpre = z["W_enc"], z["b_pre"]
    F = Wenc.shape[0]

    mlat, mlon = mesh()
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(meta["source"][5:]), consolidated=True)
    ren = {}
    if "latitude" in ds.coords:
        ren["latitude"] = "lat"
    if "longitude" in ds.coords:
        ren["longitude"] = "lon"
    if ren:
        ds = ds.rename(ren)
    if ds.lat[0] > ds.lat[-1]:
        ds = ds.reindex(lat=ds.lat[::-1])
    glat = np.asarray(ds.lat.values)
    glon = np.mod(np.asarray(ds.lon.values), 360)
    iy = np.clip(np.searchsorted(glat, mlat), 0, len(glat) - 1)
    order = np.argsort(glon)
    glon_s = glon[order]
    ix = order[np.clip(np.searchsorted(glon_s, mlon), 0, len(glon) - 1)]
    inv = np.argsort(order)
    dphi = np.gradient(np.radians(glat))
    dlam = np.gradient(np.radians(glon_s))
    R = 6.371e6
    coslat = np.cos(np.radians(glat))[:, None]

    def dx(f):
        return np.gradient(f[:, order], axis=1)[:, inv] / (dlam[None, :][:, inv] * R * coslat + 1e-9)

    def dy(f):
        return np.gradient(f, axis=0) / (dphi[:, None] * R + 1e-9)

    statics = ds[["geopotential_at_surface", "land_sea_mask"]].load()
    stat = {"abslat": np.abs(mlat),
            "land_sea": statics["land_sea_mask"].values[iy, ix],
            "orography": statics["geopotential_at_surface"].values[iy, ix]}

    sel = np.linspace(0, NWIN - 1, NW).astype(int)
    M = len(PHYS) + len(GEO)
    n_tot = 0
    s_f = np.zeros(M); s_ff = np.zeros(M)
    s_ab = np.zeros(F); s_av = np.zeros(F); s_avv = np.zeros(F)
    s_bf = np.zeros((F, M)); s_vf = np.zeros((F, M))
    used = []

    for wi, w in enumerate(sel):
        t0 = time.time()
        t = np.datetime64(str(starts_all[w])[:19])
        d = ds[["u_component_of_wind", "v_component_of_wind", "specific_humidity",
                "vertical_velocity", "temperature", "geopotential"]] \
            .sel(time=t, method="nearest").sel(level=LEVELS).load()
        p = ds[["mean_sea_level_pressure"]].sel(time=t, method="nearest").load()
        li = {lv: k for k, lv in enumerate(LEVELS)}
        u = d["u_component_of_wind"].values; v = d["v_component_of_wind"].values
        q = d["specific_humidity"].values; T = d["temperature"].values
        w5 = d["vertical_velocity"].values; zg = d["geopotential"].values
        u850, v850 = u[li[850]], v[li[850]]
        u500, v500 = u[li[500]], v[li[500]]
        u250, v250 = u[li[250]], v[li[250]]
        u200, v200 = u[li[200]], v[li[200]]
        g = {
            "ascent500": -w5[li[500]], "ascent700": -w5[li[700]],
            "q850": q[li[850]], "q600": q[li[600]], "q400": q[li[400]],
            "t850": T[li[850]], "t400": T[li[400]], "dt400_850": T[li[400]] - T[li[850]],
            "z500": zg[li[500]] / 9.81,
            "vort850": dx(v850) - dy(u850), "vort500": dx(v500) - dy(u500),
            "div250": dx(u250) + dy(v250), "div850": dx(u850) + dy(v850),
            "shear200_850": np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2),
            "jet250": np.sqrt(u250 ** 2 + v250 ** 2),
            "wspd850": np.sqrt(u850 ** 2 + v850 ** 2),
            "mslp": p["mean_sea_level_pressure"].values / 100.0,
        }
        g["qflux850"] = g["q850"] * g["wspd850"]
        node = np.stack([g[k][iy, ix] for k in PHYS] + [stat[k] for k in GEO], 1)
        assert np.isfinite(node).all(), "non-finite reference field"
        node = band_standardize(node, mlat, len(PHYS))

        A = np.asarray(X[w * L:(w + 1) * L], np.float32)
        assert np.isfinite(A).all() and np.abs(A).sum(1).min() > 0, "bad activation slab"
        code = encode(A, Wenc, bpre)
        act = (code > 0).astype(np.float32)

        n_tot += L
        s_f += node.sum(0); s_ff += (node ** 2).sum(0)
        s_ab += act.sum(0); s_av += code.sum(0); s_avv += (code ** 2).sum(0)
        s_bf += act.T @ node
        s_vf += code.T @ node
        used.append(str(starts_all[w])[:19])
        print(f"  [{wi+1}/{NW}] {used[-1]}  {time.time()-t0:.0f}s", flush=True)

    def corr(s_a, s_aa, s_af):
        ca = s_aa - s_a ** 2 / n_tot
        cf = s_ff - s_f ** 2 / n_tot
        cov = s_af - np.outer(s_a, s_f) / n_tot
        return cov / (np.sqrt(np.maximum(ca, 1e-12))[:, None] * np.sqrt(np.maximum(cf, 1e-12))[None, :])

    r_bin = corr(s_ab, s_ab, s_bf)           # binary firing indicator
    r_val = corr(s_av, s_avv, s_vf)          # activation magnitude
    np.save(FIELDS_NPY, dict(r_bin=r_bin, r_val=r_val, refs=PHYS + GEO, n_phys=len(PHYS),
                             firerate=s_ab / n_tot, n=n_tot, windows=used,
                             note="node-level corr of SAE firing vs band-standardised ERA5"),
            allow_pickle=True)
    print(f"-> {FIELDS_NPY}  r {r_bin.shape} over {n_tot} node-windows")

# -------------------------------------------------------------- footprints -----
def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

def footprint_stats(mask, mlat, mlon):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    idx = np.where(mask)[0]
    out = {"n_nodes": int(len(idx)), "frac": float(len(idx) / len(mask))}
    if len(idx) == 0:
        return out
    xyz = unit(mlat[idx], mlon[idx])
    m = xyz.mean(0)
    nm = np.linalg.norm(m)
    c = m / max(nm, 1e-12)
    clat = float(np.degrees(np.arcsin(np.clip(c[2], -1, 1))))
    clon = float(np.mod(np.degrees(np.arctan2(c[1], c[0])), 360))
    dist = 6371.0 * np.arccos(np.clip(xyz @ c, -1, 1))
    out.update(centroid_lat=clat, centroid_lon=clon,
               resultant=float(nm),                      # 1 = one point, 0 = uniform sphere
               spread_km=float(dist.mean()), med_dist_km=float(np.median(dist)),
               p90_dist_km=float(np.percentile(dist, 90)),
               frac_within_2000km=float((dist < 2000).mean()),
               lat_min=float(mlat[idx].min()), lat_max=float(mlat[idx].max()),
               lat_absmed=float(np.median(np.abs(mlat[idx]))),
               frac_tropical=float((np.abs(mlat[idx]) < 30).mean()),
               frac_polar=float((np.abs(mlat[idx]) > 65).mean()))
    # blob structure: connect nodes within 250 km (M6 mesh spacing ~ 112 km)
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r=2 * np.sin(250.0 / (2 * 6371.0)), output_type="ndarray")
    if len(pairs):
        A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(idx), len(idx)))
        nc, lab = connected_components(A, directed=False)
    else:
        nc, lab = len(idx), np.arange(len(idx))
    sz = np.bincount(lab)
    out.update(n_blobs=int(nc), largest_blob=int(sz.max()),
               largest_blob_frac=float(sz.max() / len(idx)),
               blobs_ge10=int((sz >= 10).sum()))
    return out

# ------------------------------------------------------------ trajectories -----
def load_series():
    runs = {}
    for sub in ("hyb_series", "hyb_series_sh", "hyb_series_mega"):
        dd = ROOT / "results/skill" / sub
        if not dd.exists():
            continue
        for f in sorted(dd.glob("run_*.npy")):
            stem = f.stem
            if stem.endswith(("_m24", "_m48", "_p24")):        # lead-time variants
                continue
            try:
                d = np.load(f, allow_pickle=True).item()
                b = d["res"]["baseline"]
            except Exception:
                continue
            bf = b.get("box_feats")
            if bf is None:
                continue
            runs[f"{sub}:{stem[4:]}"] = dict(
                mslp=np.asarray(b["mslp_min"], float),
                wind=np.asarray(b.get("wind_max", np.zeros(1)), float),
                feats=bf, center=np.asarray(d["center"], float), ic=str(d["ic"]),
                nondev=bool(d.get("nondev", False)))
    return runs

def xcorr(a, b, maxlag=4):
    """corr(a[t], b[t+lag]); lag>0 means a LEADS b. Returns dict lag -> r."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = {}
    for lag in range(-maxlag, maxlag + 1):
        if lag >= 0:
            x, y = a[:len(a) - lag], b[lag:]
        else:
            x, y = a[-lag:], b[:len(b) + lag]
        if len(x) < 6 or x.std() < 1e-12 or y.std() < 1e-12:
            out[lag] = float("nan"); continue
        out[lag] = float(np.corrcoef(x, y)[0, 1])
    return out

def stage_rest():
    mlat, mlon = mesh()
    zf = np.load(ROOT / "results/hybrid_footprint_fires.npz")
    fires = zf["fires"]
    assert fires.shape == (40962, 4096), fires.shape        # columns are features
    ev = {"nw_footprint": int(zf["nw"]), "thresh": int(zf["thresh"])}

    # ---- footprints
    fp = {}
    for j in REF:
        fp[str(j)] = footprint_stats(fires[:, j], mlat, mlon)
    sizes = fires.sum(0)
    fp["_dict_median_nodes"] = float(np.median(sizes))
    fp["_dict_p10_p90_nodes"] = [float(np.percentile(sizes, 10)), float(np.percentile(sizes, 90))]
    for j in REF:
        fp[str(j)]["size_pctile"] = float((sizes < sizes[j]).mean() * 100)
    # pairwise footprint overlap (cosine) among the three
    def cos(a, b):
        return float((a & b).sum() / max(np.sqrt(a.sum() * b.sum()), 1))
    fp["_overlap"] = {f"{i}-{j}": cos(fires[:, i], fires[:, j])
                      for i in REF for j in REF if i < j}
    ev["footprints"] = fp

    # ---- fields
    if FIELDS_NPY.exists():
        d = np.load(FIELDS_NPY, allow_pickle=True).item()
        r = d["r_bin"]; rv = d["r_val"]; refs = list(d["refs"])
        fl = {"refs": refs, "n_node_windows": int(d["n"]), "windows": list(d["windows"])}
        # calibration: the dictionary's own distribution per field
        fl["_dict_absr_p50"] = {m: float(np.median(np.abs(r[:, k]))) for k, m in enumerate(refs)}
        fl["_dict_absr_p99"] = {m: float(np.percentile(np.abs(r[:, k]), 99)) for k, m in enumerate(refs)}
        fl["_dict_r_min_max"] = {m: [float(r[:, k].min()), float(r[:, k].max())] for k, m in enumerate(refs)}
        for j in REF:
            fl[str(j)] = {
                m: dict(r=float(r[j, k]), r_val=float(rv[j, k]),
                        pctile=float((np.abs(r[:, k]) < abs(r[j, k])).mean() * 100))
                for k, m in enumerate(refs)}
            fl[str(j)]["_firerate"] = float(d["firerate"][j])
        ev["fields"] = fl

    # ---- trajectories
    runs = load_series()
    tr = {"storms": sorted(runs)}
    per = {}
    for j in REF:
        rows = {}
        for name, R in runs.items():
            a = np.asarray(R["feats"].get(j, R["feats"].get(str(j))), float)
            if a is None or a.size == 0:
                continue
            gate = dict(finite=bool(np.isfinite(a).all()), allzero=bool(np.all(a == 0)),
                        n=int(a.size))
            rows[name] = dict(series=[float(x) for x in a],
                              mean=float(a.mean()), max=float(a.max()),
                              vs_mslp=xcorr(a, R["mslp"]), gate=gate,
                              d_first_last=float(a[-1] - a[0]),
                              mslp_drop=float(R["mslp"][0] - R["mslp"].min()),
                              nondev=R["nondev"])
        per[str(j)] = rows
    tr["per_feature"] = per
    # f3357 -> f3319 and f3357 -> f3004 lead-lag, per storm
    pairs = {}
    for tgt in (3319, 3004):
        pr = {}
        for name, R in runs.items():
            a = np.asarray(R["feats"].get(3357, R["feats"].get("3357")), float)
            b = np.asarray(R["feats"].get(tgt, R["feats"].get(str(tgt))), float)
            pr[name] = xcorr(a, b)
        pairs[f"3357->{tgt}"] = pr
    tr["pairs"] = pairs
    ev["trajectories"] = tr

    json.dump(ev, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")
    summarize(ev)

def summarize(ev):
    print("\n=== FOOTPRINTS (columns of hybrid_footprint_fires.npz) ===")
    fp = ev["footprints"]
    print(f"{'feat':>6}{'nodes':>8}{'%dict':>7}{'clat':>8}{'clon':>8}{'spread km':>11}"
          f"{'<2000km':>9}{'blobs':>7}{'big%':>7}{'trop%':>7}{'pol%':>7}")
    for j in REF:
        s = fp[str(j)]
        print(f"{j:>6}{s['n_nodes']:>8}{s['size_pctile']:>7.0f}{s['centroid_lat']:>8.1f}"
              f"{s['centroid_lon']:>8.1f}{s['spread_km']:>11.0f}"
              f"{100*s['frac_within_2000km']:>9.0f}{s['n_blobs']:>7}"
              f"{100*s['largest_blob_frac']:>7.0f}{100*s['frac_tropical']:>7.0f}"
              f"{100*s['frac_polar']:>7.0f}")
    print(f"  dictionary median footprint {fp['_dict_median_nodes']:.0f} nodes; "
          f"p10-p90 {fp['_dict_p10_p90_nodes'][0]:.0f}-{fp['_dict_p10_p90_nodes'][1]:.0f}")
    print("  overlaps:", {k: round(v, 3) for k, v in fp["_overlap"].items() if
                          any(str(t) in k for t in TARGETS)})

    if "fields" in ev:
        fl = ev["fields"]
        refs = fl["refs"]
        print("\n=== FIELD CORRELATIONS (band-standardised, r / dictionary percentile) ===")
        hdr = f"{'field':>14}" + "".join(f"{j:>16}" for j in REF)
        print(hdr)
        print(f"{'':>14}" + "".join(f"{'dict|r| p50/p99':>16}" for _ in REF)[:0] + "")
        for m in refs:
            row = f"{m:>14}"
            for j in REF:
                e = fl[str(j)][m]
                row += f"{e['r']:>+9.3f}/{e['pctile']:>4.0f}"
            print(row)
        print("  dictionary |r| p50 / p99 per field (the null, and that it VARIES):")
        for m in refs:
            print(f"    {m:>14}  p50 {fl['_dict_absr_p50'][m]:.3f}  p99 {fl['_dict_absr_p99'][m]:.3f}"
                  f"  range [{fl['_dict_r_min_max'][m][0]:+.3f},{fl['_dict_r_min_max'][m][1]:+.3f}]")

    tr = ev["trajectories"]
    print(f"\n=== IN-BOX TRAJECTORIES ({len(tr['storms'])} storms) ===")
    for j in REF:
        rows = tr["per_feature"][str(j)]
        if not rows:
            print(f"f{j}: no series"); continue
        best = []
        for name, R in rows.items():
            v = {int(k): x for k, x in R["vs_mslp"].items()}
            v = {k: x for k, x in v.items() if np.isfinite(x)}
            if not v:
                continue
            bl = max(v, key=lambda k: abs(v[k]))
            best.append((name, R["mean"], v.get(0, np.nan), bl, v[bl]))
        m0 = np.array([b[2] for b in best], float)
        print(f"\nf{j}  mean in-box act {np.mean([b[1] for b in best]):.2f}   "
              f"corr with MSLP at lag0: median {np.nanmedian(m0):+.2f}  "
              f"n_neg {(m0 < -0.3).sum()}/{len(m0)}  n_pos {(m0 > 0.3).sum()}/{len(m0)}")
        for name, mn, r0, bl, rb in sorted(best):
            print(f"   {name:<32} act {mn:>6.2f}  r(mslp) lag0 {r0:>+6.2f}   "
                  f"best lag {bl:>+2d} ({6*bl:>+4d} h) r {rb:>+6.2f}")
    print("\n=== f3357 -> target lead-lag (lag>0 == f3357 LEADS) ===")
    for key, pr in tr["pairs"].items():
        arr = {}
        for name, v in pr.items():
            v = {int(k): x for k, x in v.items() if np.isfinite(x)}
            if not v:
                continue
            arr[name] = v
        if not arr:
            continue
        lags = sorted(set().union(*[set(v) for v in arr.values()]))
        print(f"\n{key}   (mean r over {len(arr)} storms)")
        print("   lag  " + "".join(f"{l:>7}" for l in lags))
        print("   h    " + "".join(f"{6*l:>7}" for l in lags))
        mean = [np.nanmean([v.get(l, np.nan) for v in arr.values()]) for l in lags]
        print("   r    " + "".join(f"{m:>+7.2f}" for m in mean))
        for name, v in sorted(arr.items()):
            bl = max(v, key=lambda k: abs(v[k]))
            print(f"   {name:<32} peak lag {bl:>+2d} ({6*bl:>+4d} h) r {v[bl]:>+6.2f}  "
                  f"lag2 r {v.get(2, float('nan')):>+6.2f}")

def stage_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = json.load(open(OUT))
    mlat, mlon = mesh()
    fires = np.load(ROOT / "results/hybrid_footprint_fires.npz")["fires"]
    runs = load_series()
    order = [n for n in sorted(runs) if not runs[n]["nondev"]][:4]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 1, 1], hspace=0.42, wspace=0.22)
    lon_p = np.where(mlon > 180, mlon - 360, mlon)
    cols = {3357: "#d1495b", 3319: "#2e86ab", 3004: "#f4a259"}
    for k, j in enumerate(TARGETS):
        ax = fig.add_subplot(gs[0, k])
        m = fires[:, j]
        ax.scatter(lon_p[~m][::7], mlat[~m][::7], s=0.4, c="0.88", lw=0)
        ax.scatter(lon_p[m], mlat[m], s=2.2, c=cols[j], lw=0)
        s = ev["footprints"][str(j)]
        cl = s["centroid_lon"] - 360 if s["centroid_lon"] > 180 else s["centroid_lon"]
        ax.plot([cl], [s["centroid_lat"]], "k+", ms=12, mew=2)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-60, -30, 0, 30, 60])
        ax.set_title(f"f{j}   {s['n_nodes']} nodes ({100*s['frac']:.1f}%)\n"
                     f"centroid {s['centroid_lat']:+.0f}, {s['centroid_lon']:.0f}E   "
                     f"spread {s['spread_km']:.0f} km", fontsize=9)
        ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[0, 3])
    if "fields" in ev:
        fl = ev["fields"]
        refs = [m for m in fl["refs"] if m not in GEO]
        y = np.arange(len(refs))
        for j in TARGETS:
            ax.barh(y + (TARGETS.index(j) - 1) * 0.27, [fl[str(j)][m]["r"] for m in refs],
                    height=0.26, color=cols[j], label=f"f{j}")
        ax.set_yticks(y); ax.set_yticklabels(refs, fontsize=7); ax.invert_yaxis()
        ax.axvline(0, c="k", lw=0.6)
        ax.set_xlabel("corr(fires, band-std field)", fontsize=8)
        ax.legend(fontsize=7); ax.tick_params(labelsize=7)
        ax.set_title("node-level field correlation", fontsize=9)

    for k, name in enumerate(order):
        ax = fig.add_subplot(gs[1, k])
        R = runs[name]
        h = 6 * np.arange(len(R["mslp"]))
        for j in TARGETS:
            a = np.asarray(R["feats"].get(j, R["feats"].get(str(j))), float)
            ax.plot(h, a, color=cols[j], lw=1.6, label=f"f{j}")
        ax.set_title(name.split(":")[1], fontsize=9)
        ax.set_xlabel("forecast hour", fontsize=8); ax.tick_params(labelsize=7)
        if k == 0:
            ax.set_ylabel("in-box activation", fontsize=8); ax.legend(fontsize=7)
        ax2 = ax.twinx()
        ax2.plot(h, R["mslp"], color="0.35", lw=1.1, ls="--")
        ax2.tick_params(labelsize=6, colors="0.35")
        if k == 3:
            ax2.set_ylabel("MSLP min (hPa, dashed)", fontsize=7, color="0.35")

    for k, name in enumerate(order[:3]):
        ax = fig.add_subplot(gs[2, k])
        R = runs[name]
        a = np.asarray(R["feats"].get(3357, R["feats"].get("3357")), float)
        b = np.asarray(R["feats"].get(3319, R["feats"].get("3319")), float)
        za = (a - a.mean()) / (a.std() + 1e-9); zb = (b - b.mean()) / (b.std() + 1e-9)
        h = 6 * np.arange(len(a))
        ax.plot(h, za, color=cols[3357], lw=1.6, label="f3357 (z)")
        ax.plot(h, zb, color=cols[3319], lw=1.6, label="f3319 (z)")
        ax.plot(h + 12, za, color=cols[3357], lw=1.0, ls=":", label="f3357 shifted +12 h")
        ax.set_xlim(0, 6 * (len(a) - 1))
        ax.set_xlabel("forecast hour", fontsize=8); ax.tick_params(labelsize=7)
        v = {int(kk): x for kk, x in ev["trajectories"]["pairs"]["3357->3319"][name].items()}
        ax.set_title(f"{name.split(':')[1]}   r(lag+2)={v.get(2, float('nan')):+.2f}", fontsize=8)
        if k == 0:
            ax.set_ylabel("z-scored", fontsize=8); ax.legend(fontsize=6)
    # the finding: f3319/f3004 fire only at 06Z/18Z, on real ERA5 IID windows
    ax = fig.add_subplot(gs[2, 3])
    ser = np.load(ROOT / "results/fs_cgv2_actseries.npy", allow_pickle=True).item()
    S = ser["series"]; hrs = np.array([int(str(x)[11:13]) for x in ser["starts"]])
    H = [0, 6, 12, 18]
    for i, j in enumerate(TARGETS):
        v = [S[hrs == h, j].mean() for h in H]
        ax.bar(np.arange(4) + (i - 1) * 0.27, v, width=0.26, color=cols[j], label=f"f{j}")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{h:02d}Z" for h in H])
    ax.set_ylabel("global code sum", fontsize=8); ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.set_title("160 real ERA5 windows: f3319/f3004 fire\nONLY at 06Z/18Z (12-h period)", fontsize=9)
    fig.suptitle("f3357, f3319, f3004 — footprints, field correlations, in-box behaviour",
                 fontsize=12)
    p = ROOT / "figures/edge_features.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"-> {p}")

def stage_extras():
    """Guardrail #9 calibration of the field detector + profile similarity + series gates."""
    ev = json.load(open(OUT))
    d = np.load(FIELDS_NPY, allow_pickle=True).item()
    r = d["r_bin"]; refs = list(d["refs"]); n_phys = int(d["n_phys"])
    idx = {m: k for k, m in enumerate(refs)}
    P = r[:, :n_phys]
    nrm = np.linalg.norm(P, axis=1); live = nrm > 1e-6
    rng = np.random.default_rng(0)
    samp = rng.choice(4096, 200, replace=False)
    cal = {
        # (i) the null VARIES: per-field |r| spread already in ev["fields"]
        # (ii) the bar is ATTAINABLE: the convection group reaches p100 on ascent
        "positive_control": {f"f{j}": {m: float(r[j, idx[m]]) for m in
                                       ("ascent700", "ascent500", "div850")} for j in CONV},
        # (iii) a negative control FAILS it: two features the rotation test showed are
        # POSITIONAL (grid-locked), not physical. They must score near the dictionary
        # median on the physics block and at the top on |lat|.
        "negative_control": {f"f{j}": {m: dict(r=float(r[j, idx[m]]),
                                               pctile=float((np.abs(r[:, idx[m]]) < abs(r[j, idx[m]])).mean() * 100))
                                       for m in ("ascent700", "q600", "div850", "abslat")}
                             for j in (2075, 2235)},
        "random_200_absr": {m: dict(p50=float(np.median(np.abs(r[samp, idx[m]]))),
                                    p90=float(np.percentile(np.abs(r[samp, idx[m]]), 90)),
                                    max=float(np.abs(r[samp, idx[m]]).max()))
                            for m in ("ascent700", "q600", "div850")},
    }
    prof = {}
    for a in TARGETS:
        cs = (P[live] @ P[a]) / (nrm[live] * nrm[a])
        prof[f"f{a}"] = dict(
            vs={f"f{b}": float(P[a] @ P[b] / (nrm[a] * nrm[b])) for b in REF},
            dict_p50=float(np.median(cs)), dict_p95=float(np.percentile(cs, 95)),
            dict_p99=float(np.percentile(cs, 99)))
    ev["fields_calibration"] = cal
    ev["field_profile_similarity"] = prof

    # zero-inflation of the in-box series the estimators were fitted on
    zi = {}
    for j in REF + [1033, 3314]:
        rows = ev["trajectories"]["per_feature"].get(str(j))
        if not rows:
            continue
        a = np.array([R["series"] for R in rows.values()], float)
        zi[f"f{j}"] = dict(frac_zero=float((a == 0).mean()),
                           storms_all_zero=int((a.max(1) == 0).sum()),
                           n_storms=int(len(a)), mean=float(a.mean()),
                           lag1_autocorr=float(np.nanmean(
                               [np.corrcoef(x[:-1], x[1:])[0, 1] if x.std() > 0 else np.nan
                                for x in a])))
    ev["series_gate"] = zi
    # SEMIDIURNAL PARITY. In-box parity (rollout ICs are 00Z, so odd 6-h steps == 06Z/18Z)
    # and, independently, the 160 real ERA5 IID windows grouped by UTC hour.
    par = {}
    for j in REF:
        rows = ev["trajectories"]["per_feature"].get(str(j))
        if not rows:
            continue
        a = np.array([R["series"] for R in rows.values()], float)
        par[f"f{j}"] = dict(inbox_even=float(a[:, 0::2].mean()), inbox_odd=float(a[:, 1::2].mean()))
    ser = np.load(ROOT / "results/fs_cgv2_actseries.npy", allow_pickle=True).item()
    S = ser["series"]; hrs = np.array([int(str(x)[11:13]) for x in ser["starts"]])
    for j in REF:
        par.setdefault(f"f{j}", {})["iid_by_utc_hour"] = {
            f"{h:02d}Z": float(S[hrs == h, j].mean()) for h in (0, 6, 12, 18)}
    par["_n_windows_per_hour"] = {f"{h:02d}Z": int((hrs == h).sum()) for h in (0, 6, 12, 18)}
    par["_note"] = ("the banded atlas diurnal statistic uses a 24-h harmonic and is "
                    "structurally blind to a 12-h on/off pattern, which is why this was "
                    "never flagged")
    ev["semidiurnal"] = par
    json.dump(ev, open(OUT, "w"), indent=1)
    print(json.dumps({"fields_calibration": cal, "field_profile_similarity": prof,
                      "series_gate": zi}, indent=1))

if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "rest"
    {"fields": stage_fields, "rest": stage_rest, "figure": stage_figure,
     "extras": stage_extras}[st]()
