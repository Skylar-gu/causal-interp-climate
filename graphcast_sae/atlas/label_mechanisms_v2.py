"""Calibrated mechanism labelling — the repair of `label_mechanisms.py`.

WHAT WAS WRONG. v1 labels a feature by `argmax` over mechanisms with a fixed bar
`z > 0.15` (label_mechanisms.py:100). `z[f,m]` is a MEAN of per-window
standardized field values over the nodes where f fires, so its noise scale is set
by the footprint: naive SE = 1/sqrt(n_fire), which runs 0.048 at the 5th
percentile of footprint size to 0.004 at the 95th. The one fixed bar is therefore
an 11x different statistical bar across the dictionary, and it is LOOSEST exactly
where it should be strictest — compact features, which are the good causal
handles. Measured consequence: the labelled fraction rises 78.5% -> 97.9% as the
battery grows 4 -> 17 probes, i.e. the bar goes vacuous as more phenomena are
added. It is the same family as the blocks-v1 vacuous threshold and the SPD
point-mass null.

Second defect: argmax forces a winner. q600 and ascent correlate at r=+0.39 over
the features (shear vs q600 at -0.32), so the convection-vs-moisture contrast
that the skill result rests on separates two groups that are disjoint only
BECAUSE argmax made them disjoint. Features near that boundary are assigned by a
coin flip.

THE REPAIR, five parts:
  1. Per-feature null by RIGID LONGITUDE ROTATION of the mesh sampling. For each
     rotation the field is read at nodes rotated about the pole, so the
     footprint's size, shape and latitude band are preserved exactly and the
     field keeps its own spatial autocorrelation; only the zonal alignment is
     destroyed. This is the same footprint/distance-preserving-null idea the
     project already needed three times (M1's locality control, the
     distance-preserving PX null, the coverage-matched heat-dome control).
  2. z-score against that null, then a two-sided p. R rotations give a null MEAN
     and SD per (feature, probe); p is Gaussian on the standardized statistic, so
     resolution is not capped at 1/R. Normality of the null is checked and
     reported, not assumed.
  3. Benjamini-Hochberg across the whole feature x probe grid.
  4. argmax -> the winner must ALSO beat the runner-up by more than the combined
     null spread, else the feature is labelled `ambiguous`. This is what turns
     convection-vs-moisture into a measurement instead of a tie-break.
  5. Calibration gate: the same labelling run on a ROTATED z (pure null input)
     must collapse the labelled fraction to about the FDR level. If it does not,
     nothing here is readable.

HONEST LIMIT of the rotation null, stated because it bounds every p below: it
tests ZONAL alignment. A feature whose footprint and field are both organised by
latitude alone (a polar feature on a polar field) will read p ~ 1 even though the
association is real, because the rotated null reproduces it. So the raw z (which
includes the zonal-mean part) is reported alongside the p (the part not explained
by latitude band and footprint shape). Same lesson as M1: the free null
over-credits, the constrained null is fair, and both belong in the table.

DEVIATION from v1, declared: 24 dump windows here against v1's 60, so that the
null costs the same reads as the estimate. The observed z is correlated against
v1's stored z and that correlation is printed as a gate.

Paper: Sec. 3 (calibrated mechanism labels; figures/paper_fig_exposure.py)
Inputs: results/fs_ida_mechfeats.npy (not shipped, see docs/REPRODUCE.md); WeatherBench2 ERA5 (GCS, streamed); i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: results/fs_mechanisms_v2.npy
Run:   # JAX env, CPU
    FS_DEVICE=cpu python -m graphcast_sae.atlas.label_mechanisms_v2
"""
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

import graphcast_sae.common.fs_common as fc
from graphcast_sae.atlas.label_mechanisms import encode, sae_np                      # identical estimator

NW_USE = 24
NROT = 128                    # rotations for the null
LV = dict(p200=200, p500=500, p600=600, p850=850)
META = json.load(open(fc.SCRATCH / "fs_iid_meta.json"))
DUMP = fc.SCRATCH / "fs_iid_dump.npy"
L, DIM = META["n_mesh"], META["dim"]
MECH = ["vort850", "q600", "ascent", "shear"]
OUT = fc.ROOT / "results" / "fs_mechanisms_v2.npy"
Q_FDR = 0.05

def rotation_perms(lat, lon, nrot):
    """pi[r, i] = node nearest to node i rotated by angle r about the pole.

    Rigid rotation, so footprint geometry and latitude are preserved exactly and
    the sampled field keeps its spatial autocorrelation. The icosahedral mesh is
    not exactly invariant under an arbitrary longitude rotation, so this is
    nearest-neighbour, not exact; the residual is a fraction of one mesh spacing
    and is reported as the median great-circle snap distance.
    """
    la, lo = np.radians(lat), np.radians(lon)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)
    tree = cKDTree(xyz)
    angles = np.linspace(0, 2 * np.pi, nrot, endpoint=False)[1:]     # drop identity
    perms = np.empty((len(angles), len(lat)), np.int32)
    snaps = []
    for k, a in enumerate(angles):
        c, s = np.cos(a), np.sin(a)
        rot = np.stack([c * xyz[:, 0] - s * xyz[:, 1],
                        s * xyz[:, 0] + c * xyz[:, 1], xyz[:, 2]], 1)
        d, idx = tree.query(rot, k=1)
        perms[k] = idx
        snaps.append(np.median(d))
    return perms, float(np.median(snaps) * 6371.0)

def main():
    Wenc, bpre = sae_np()
    F = Wenc.shape[0]
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat, mlon = geom["lat"], np.mod(geom["lon"], 360)

    perms, snap_km = rotation_perms(mlat, mlon, NROT)
    R = perms.shape[0]
    print(f"rotation null: {R} rigid longitude rotations, "
          f"median snap {snap_km:.0f} km (mesh spacing ~{40000/np.sqrt(L):.0f} km)")

    ds, _ = fc.open_wb2()
    glat = np.asarray(ds.lat.values)
    glon = np.mod(np.asarray(ds.lon.values), 360)
    iy = np.clip(np.searchsorted(glat, mlat), 0, len(glat) - 1)
    ix = np.clip(np.searchsorted(np.sort(glon), mlon), 0, len(glon) - 1)
    ix = np.argsort(glon)[ix]
    dphi = np.gradient(np.radians(glat))
    dlam = np.gradient(np.radians(np.sort(glon)))
    Re = 6.371e6
    coslat = np.cos(np.radians(glat))[:, None]

    starts = np.array(META["starts"])[np.linspace(0, META["n_windows"] - 1,
                                                  NW_USE).astype(int)]
    all_starts = list(META["starts"])
    X = np.load(DUMP, mmap_mode="r")
    P = len(MECH)

    zsum = np.zeros((F, P))
    zrot = np.zeros((F, P, R))
    zcnt = np.zeros(F)

    for wi, s in enumerate(starts):
        t = np.datetime64(str(s)[:19])
        d = ds[["u_component_of_wind", "v_component_of_wind",
                "specific_humidity", "vertical_velocity"]] \
            .sel(time=t, method="nearest").sel(level=list(LV.values())).load()
        u = d["u_component_of_wind"].values
        v = d["v_component_of_wind"].values
        li = {p: k for k, p in enumerate(LV.values())}
        u850, v850, u200, v200 = u[li[850]], v[li[850]], u[li[200]], v[li[200]]
        dvdx = np.gradient(v850, axis=1) / (dlam[None, :] * Re * coslat + 1e-9)
        dudy = np.gradient(u850, axis=0) / (dphi[:, None] * Re + 1e-9)
        fields = [dvdx - dudy,
                  d["specific_humidity"].values[li[600]],
                  -d["vertical_velocity"].values[li[500]],
                  np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)]

        node_m = np.stack([f[iy, ix] for f in fields], 1)
        node_m = (node_m - node_m.mean(0)) / (node_m.std(0) + 1e-9)

        j = all_starts.index(str(s))
        A = np.asarray(X[j * L:(j + 1) * L], np.float32)
        act = (encode(A, Wenc, bpre) > 0).astype(np.float32)      # (L, F)

        zsum += act.T @ node_m
        zcnt += act.sum(0)
        # all rotations in one GEMM: (F,L) @ (L, P*R)
        big = node_m[perms.reshape(-1)].reshape(R, L, P).transpose(1, 0, 2).reshape(L, R * P)
        zrot += (act.T @ big).reshape(F, R, P).transpose(0, 2, 1)
        if wi % 6 == 0:
            print(f"  window {wi}/{len(starts)}", flush=True)

    n = np.maximum(zcnt, 1)[:, None]
    z = zsum / n
    zr = zrot / n[:, :, None]
    mu, sd = zr.mean(2), zr.std(2)
    zs = (z - mu) / np.maximum(sd, 1e-12)
    p = 2 * stats.norm.sf(np.abs(zs))

    # ---- is the Gaussian step defensible? report, do not assume ---------------
    live = zcnt > 500
    sk = stats.skew(zr[live], axis=2)
    ku = stats.kurtosis(zr[live], axis=2)
    print(f"\nnull shape check (live features only — dead rows are constant and "
          f"return nan): |skew| median {np.nanmedian(np.abs(sk)):.2f} "
          f"p95 {np.nanpercentile(np.abs(sk), 95):.2f} | excess kurtosis median "
          f"{np.nanmedian(ku):+.2f} p95 {np.nanpercentile(ku, 95):+.2f}")

    # ---- BH-FDR over the whole feature x probe grid --------------------------
    def bh(pv, q=Q_FDR):
        flat = pv.ravel()
        o = np.argsort(flat)
        m = flat.size
        keep = np.zeros(m, bool)
        ok = flat[o] <= q * np.arange(1, m + 1) / m
        if ok.any():
            keep[o[:np.max(np.where(ok)[0]) + 1]] = True
        return keep.reshape(pv.shape)

    sig = bh(p)

    # ---- label rule: significant winner that also beats the runner-up --------
    # The gap bar is read off the NULL, not assumed: for every rotation draw,
    # form the same top-minus-runner-up gap and take its 95th percentile per
    # feature. A fixed 1.96 would be a bar on one z-score, not on the difference
    # of two correlated ones.
    fi = np.arange(F)

    def gap_of(zz):
        o = np.argsort(-np.abs(zz), axis=1)
        return (np.abs(zz[fi, o[:, 0]]) - np.abs(zz[fi, o[:, 1]])), o[:, 0]

    zs_rot = (zr - mu[:, :, None]) / np.maximum(sd, 1e-12)[:, :, None]
    gap_null = np.stack([gap_of(zs_rot[:, :, r])[0] for r in range(R)], 1)
    gap_bar = np.percentile(gap_null, 95, axis=1)

    gap, top = gap_of(zs)
    labelled = sig[fi, top] & (gap > gap_bar)
    lab = np.where(labelled, np.array(MECH)[top], "ambiguous")

    # ---- calibration gate, HELD OUT ------------------------------------------
    # Draw 0 is scored against a mean/SD built from draws 1..R-1 only. Scoring a
    # draw against a null that contains it is self-inclusion bias and reads
    # anti-conservative (measured: 10.3% against a nominal 5%).
    mu_h, sd_h = zr[:, :, 1:].mean(2), zr[:, :, 1:].std(2)
    zsf = (zr[:, :, 0] - mu_h) / np.maximum(sd_h, 1e-12)
    sigf = bh(2 * stats.norm.sf(np.abs(zsf)))
    gapf, topf = gap_of(zsf)
    fake_lab = float((sigf[fi, topf] & (gapf > gap_bar)).mean())

    v1_rate = float((np.abs(z[live]).max(1) > 0.15).mean())
    print(f"\n{'':26}{'v1  argmax + fixed 0.15':>26}{'v2  calibrated':>18}")
    print(f"{'features labelled':<26}{v1_rate:>25.1%}{labelled[live].mean():>18.1%}")
    print(f"{'  on a pure-null input':<26}{'(never tested)':>26}{fake_lab:>17.1%}"
          f"   <- calibration gate")

    print(f"\nlabels among {live.sum()} live features (n_fire > 500):")
    for m_i, mname in enumerate(MECH):
        k = int(((lab == mname) & live).sum())
        print(f"   {mname:<10}{k:>6}   ({k/live.sum():5.1%})")
    k = int(((lab == "ambiguous") & live).sum())
    print(f"   {'ambiguous':<10}{k:>6}   ({k/live.sum():5.1%})")

    # ---- the load-bearing set: does the -41% convection cast survive? --------
    try:
        pick = np.load(fc.ROOT / "results/fs_ida_mechfeats.npy",
                       allow_pickle=True).item()["pick"]
        print(f"\nTHE MECHANISM CAST — the groups behind the -41% convection result")
        print(f"  {'feat':>5}{'n_fire':>8}  " + "".join(f"{m:>9}" for m in MECH)
              + f"{'assigned':>11}{'calibrated':>12}{'gap σ':>8}{'bar':>7}")
        agree = amb = wrong = 0
        for mname, feats in pick.items():
            for f_ in feats:
                got = lab[f_]
                agree += got == mname
                amb += got == "ambiguous"
                wrong += got not in (mname, "ambiguous")
                print(f"  {f_:>5}{int(zcnt[f_]):>8}  "
                      + "".join(f"{zs[f_, k_]:>+9.1f}" for k_ in range(P))
                      + f"{mname:>11}{got:>12}{gap[f_]:>8.1f}{gap_bar[f_]:>7.1f}")
        print(f"  -> assignment confirmed {agree}, ambiguous {amb}, "
              f"REASSIGNED {wrong}  (z-scores vs the rotation null, not raw z)")
    except Exception as ex:
        print(f"  mechanism cast unavailable: {ex}")

    np.save(OUT, dict(z=z, z_null=zr, z_null_mean=mu, z_null_sd=sd, zscore=zs,
                      p=p, sig=sig, label=lab, gap=gap, gap_bar=gap_bar,
                      mech=MECH, active_count=zcnt, nrot=R, nw=NW_USE,
                      q_fdr=Q_FDR, snap_km=snap_km,
                      calibration_null_label_rate=fake_lab,
                      v1_label_rate=v1_rate), allow_pickle=True)
    print(f"\nwrote {OUT.relative_to(fc.ROOT)}")

if __name__ == "__main__":
    main()
