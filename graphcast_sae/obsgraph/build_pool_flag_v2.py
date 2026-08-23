"""Flagship pool v2 — the SAME footprints as pool v1 plus `sae_sel_flag`, and a channel
readout `q_c` refit with the Nyquist term removed first.

WHY (two defects in `graphcast_sae/obsgraph/build_pool.py`, both diagnosed on the mini model):

  D1  q_c = top-variance direction of the RAW pooled activations (build_pool.py:168).
      In layer-8 activations the top-variance direction IS the annual cycle, so the
      pipeline picks the most seasonal readout and then regresses the season out —
      it selects for exactly what it deletes. Measured on the flagship pool v1:
      within-member median |cos| 0.256, varfrac mean 0.336.

  D2  Refitting q on deseasonalized data with the PIPELINE design (trend + annual K=3
      + diurnal K=1) is WORSE, not better: at 6-hourly sampling the diurnal K=2
      harmonic is the Nyquist frequency (sine identically zero, cosine a pure (+1,-1)
      alternation separating {00Z,12Z} from {06Z,18Z}) and is NOT in that design. In
      the tropics the aliased diurnal signal is the largest surviving coherent
      variance, so a variance-maximising readout points straight at it. On mini this
      fused five tropical modes at |corr| 0.94-0.99 and took the series condition
      number 4.8 -> 757; edge count tracks log(cond) at Spearman 0.92, i.e. PCMCI+
      MANUFACTURES edges when variables are collinear.

THE REPAIR is `candidates/refit_channel_dirs_v2.py`'s `harmonic_design_v2`: add the
diurnal K=2 column so the alias is removed BEFORE the top-variance direction is chosen.
On mini: cond 757 -> 12.7, zero modes above 50% Nyquist variance, frac_eastward
0.471 -> 0.692 with both anchors at chance.

NEW MEMBER `sae_sel_flag` — 39 INDIVIDUAL SAE features at native footprint, selection
rule frozen in docs/prereg/prereg_flagship_gint.md §3 before this script existed. `sae_flag`
KMeans-pools 4,079 features into 39 clusters to match leiden's N-hat (the data-gate rule), and
that pooling costs it the resolution it is being tested on: 5,665 effective nodes vs
leiden's 710, against a synoptic storm of 560-2,256 nodes at 0.25 deg. `sae_sel_flag`
asks whether the SAE's causal structure was destroyed by that pooling. It does NOT tile
the sphere: edge COUNTS are not comparable to a partition and PX is undefined for it.

ANCHORS. `qperm_flag` (channel-permuted q on vmax footprints) is kept only if it still
decorrelates from vmax on the refit basis — measured here, not assumed. `qrand_flag`
(vmax footprints x random unit q) is added unconditionally; it costs no GPU because it
shares vmax's footprints and therefore vmax's pooled tensor. qrand numbers may NEVER be
compared to published qperm numbers.

Paper: Sec. 4 'The observational graph, audited' (Fig. fig:graphmap)
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); candidates/pool_flag_candidates.npy (not shipped, see docs/REPRODUCE.md); candidates/pool_flag_channel_dirs.npy (not shipped, see docs/REPRODUCE.md); results/fs_atlas_class.npy (not shipped, see docs/REPRODUCE.md); i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: candidates/pool_flag_v2_candidates.npy; candidates/pool_flag_v2_chandirs.npy; results/flag_gint_preflight.json
Run:   # JAX env, CPU
    OMP_NUM_THREADS=8 python -m graphcast_sae.obsgraph.build_pool_flag_v2
"""
import json
from pathlib import Path

import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT, SCRATCH
DUMP = SCRATCH / "fs_iid_dump.npy"
META = json.load(open(SCRATCH / "fs_iid_meta.json"))
NW, N_MESH, DIM = META["n_windows"], META["n_mesh"], META["dim"]

POOL_V1 = ROOT / "candidates/pool_flag_candidates.npy"
CATALOG = ROOT / "candidates/fs_feature_catalog.npy"
ATLAS_CLASS = ROOT / "results/fs_atlas_class.npy"
OUT_C = ROOT / "candidates/pool_flag_v2_candidates.npy"
OUT_CD = ROOT / "candidates/pool_flag_v2_chandirs.npy"
OUT_J = ROOT / "results/flag_gint_preflight.json"

SEED = 0
# ── frozen sae_sel_flag rule (prereg §3) ────────────────────────────────────
SEL_DROP_CATS = ("numerical/geometry", "climatology/clock")
SEL_MIN_FIRERATE = 1e-3
SEL_COS0 = 0.50          # greedy decorrelation start; relaxed +0.05 until N found
SEL_N = None             # set to leiden's N

# ── deseasonalization designs ───────────────────────────────────────────────
def _t(times):
    return times.astype("datetime64[s]").astype(np.float64)

def harmonic_design(times, nyquist=False):
    """Pipeline design (trend + annual K=3 + diurnal K=1); +Nyquist column if asked."""
    t = _t(times)
    t_lin = (t - t.mean()) / (t.std() + 1e-9)
    yr, day = 365.2422 * 86400.0, 86400.0
    cols = [np.ones_like(t), t_lin]
    for k in range(1, 4):
        cols += [np.sin(2 * np.pi * k * t / yr), np.cos(2 * np.pi * k * t / yr)]
    cols += [np.sin(2 * np.pi * t / day), np.cos(2 * np.pi * t / day)]
    if nyquist:
        s2 = np.sin(2 * np.pi * 2 * t / day)
        c2 = np.cos(2 * np.pi * 2 * t / day)
        assert np.abs(s2).max() < 1e-6, "K=2 sine not degenerate; sampling is not 6-hourly"
        assert len(np.unique(np.round(c2, 6))) == 2, "K=2 cosine is not a 2-level alternation"
        cols += [c2]
    return np.stack(cols, 1)

def nyq_col(times):
    return np.cos(2 * np.pi * 2 * _t(times) / 86400.0)

def resid(D, M):
    b, *_ = np.linalg.lstsq(D, M, rcond=None)
    return M - D @ b

def loading_to_footprint(load):
    """Identical to build_pool.py — sign-fix, clip, 5%-of-max floor, sum-normalise."""
    load = np.asarray(load, np.float64).copy()
    if load[np.argmax(np.abs(load))] < 0:
        load = -load
    fp = np.clip(load, 0, None)
    m = fp.max()
    if m > 0:
        fp[fp < 0.05 * m] = 0.0
    s = fp.sum()
    return (fp / s if s > 0 else fp).astype(np.float32)

def eff_nodes(W):
    """Participation ratio of a footprint = effective number of mesh nodes."""
    return np.array([float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0
                     for w in W])

# ── the frozen sae_sel_flag selection ───────────────────────────────────────
def build_sae_sel(N):
    cat = np.load(ATLAS_CLASS, allow_pickle=True).item()["cat"]
    cat = np.asarray(cat)
    cf = np.load(CATALOG, allow_pickle=True).item()
    fire, firerate, coh = cf["fire"], cf["firerate"], cf["coh"]
    nm = cf["node_map"]                                    # (4096, 40962) mean activation
    F = nm.shape[0]
    keep = (fire > 0) & np.isfinite(coh) & (coh > 0) & (firerate >= SEL_MIN_FIRERATE)
    keep &= ~np.isin(cat, SEL_DROP_CATS)
    univ = np.flatnonzero(keep)
    score = firerate / (coh + 1e-6)
    order = univ[np.argsort(-score[univ])]
    print(f"[sae_sel] universe {len(univ)}/{F} features "
          f"(dropped cats {SEL_DROP_CATS}, firerate>={SEL_MIN_FIRERATE})", flush=True)

    fps = {}
    def fp_of(f):
        if f not in fps:
            v = loading_to_footprint(nm[f])
            n = np.linalg.norm(v)
            fps[f] = (v, v / (n + 1e-12))
        return fps[f]

    thr = SEL_COS0
    while thr < 1.0:
        sel, unit = [], []
        for f in order:
            v, u = fp_of(int(f))
            if unit and max(float(u @ x) for x in unit) >= thr:
                continue
            sel.append(int(f)); unit.append(u)
            if len(sel) == N:
                break
        if len(sel) == N:
            break
        print(f"[sae_sel] only {len(sel)} at cos<{thr:.2f}; relaxing", flush=True)
        thr += 0.05
    assert len(sel) == N, f"could not select {N} features"
    W = np.stack([fp_of(f)[0] for f in sel])
    mx = float(np.max([np.abs(np.asarray(unit) @ np.asarray(unit).T
                              - np.eye(N)).max()]))
    print(f"[sae_sel] N={N} feats={sel}\n[sae_sel] final cos threshold {thr:.2f}, "
          f"max pairwise footprint cos {mx:.3f}", flush=True)
    return W, sel, float(thr), mx

def main():
    global SEL_N
    v1 = np.load(POOL_V1, allow_pickle=True).item()
    cands = {k: v.astype(np.float32) for k, v in v1["cands"].items()}
    lat, lon, xyz = v1["lat"], v1["lon"], v1["xyz"]
    N = cands["leiden_flag"].shape[0]; SEL_N = N
    print(f"pool v1 members={list(cands)}  N={N}", flush=True)

    W_sel, sel_feats, sel_thr, sel_maxcos = build_sae_sel(N)
    cands["sae_sel_flag"] = W_sel
    cands["qrand_flag"] = cands["vmax_flag"].copy()

    # ── pooled i.i.d. tensor per member (q-agnostic) ────────────────────────
    times = np.array(META["starts"], dtype="datetime64[ns]")
    X = np.load(DUMP, mmap_mode="r")
    pooled = {n: np.empty((W.shape[0], NW, DIM), np.float32) for n, W in cands.items()}
    for w in range(NW):
        A = np.asarray(X[w * N_MESH:(w + 1) * N_MESH], np.float32)
        for n, W in cands.items():
            pooled[n][:, w, :] = W @ A
        if w % 40 == 0:
            print(f"  pooled window {w}/{NW}", flush=True)

    D2 = harmonic_design(times, nyquist=True)      # trend+annual K3+diurnal K1+K2
    D1 = harmonic_design(times, nyquist=False)     # what the graph pipeline removes
    nyq = nyq_col(times); nyq = (nyq - nyq.mean()) / nyq.std()
    print(f"design v2 cols={D2.shape[1]}  NW={NW}", flush=True)

    cd, rep = {}, {}
    for n, P in pooled.items():
        Nn = P.shape[0]
        q = np.empty((Nn, DIM), np.float32)
        mbar = P.mean(1).astype(np.float32)
        vf = np.empty(Nn, np.float32)
        for c in range(Nn):
            M = resid(D2, P[c] - P[c].mean(0))
            _, s, vt = np.linalg.svd(M, full_matrices=False)
            q[c] = vt[0]; vf[c] = float(s[0] ** 2 / (s ** 2).sum())
        cd[n] = dict(q=q, mbar=mbar, varfrac=vf)

    # anchors: q constructed FROM the refit vmax q, exactly as build_pool.py v1 did
    rng = np.random.default_rng(SEED)
    qv = cd["vmax_flag"]["q"]
    cd["qperm_flag"]["q"] = np.stack([qv[c][rng.permutation(DIM)]
                                      for c in range(qv.shape[0])]).astype(np.float32)
    Qr = rng.standard_normal((qv.shape[0], DIM)).astype(np.float32)
    Qr /= np.linalg.norm(Qr, axis=1, keepdims=True)
    cd["qrand_flag"]["q"] = Qr

    # ── PRE-FLIGHT diagnostics on the i.i.d. dump ───────────────────────────
    v1cd = np.load(ROOT / "candidates/pool_flag_channel_dirs.npy", allow_pickle=True).item()
    print("\n" + "=" * 78)
    print("PRE-FLIGHT — refit-v2 basis, measured on the i.i.d. dump (160 windows)")
    print("=" * 78)
    hdr = (f"{'member':<15}{'medcos':>8}{'maxcos':>8}{'varfr':>7}{'cond':>9}"
           f"{'mineig':>9}{'max|r|':>8}{'nyqmax':>8}{'>50%':>6}{'effnodes':>10}")
    print(hdr)
    for n in cands:
        q = cd[n]["q"]
        C = np.abs(q @ q.T); np.fill_diagonal(C, np.nan)
        medcos, maxcos = float(np.nanmedian(C)), float(np.nanmax(C))
        # series on the i.i.d. dump under this q, then the PIPELINE deseasonalization
        S = np.einsum("cwd,cd->wc", pooled[n], q) - (cd[n]["mbar"] * q).sum(1)
        R = resid(D1, S)
        Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
        nyqfrac = np.array([float(((nyq @ Z[:, c]) / len(nyq)) ** 2) for c in range(Z.shape[1])])
        CM = np.corrcoef(Z.T)
        ev = np.linalg.eigvalsh(CM)
        offd = np.abs(CM - np.eye(CM.shape[0]))
        cond = float(ev.max() / max(ev.min(), 1e-12))
        en = eff_nodes(cands[n])
        rep[n] = dict(med_cos_q=medcos, max_cos_q=maxcos,
                      varfrac_mean=float(cd[n]["varfrac"].mean()),
                      cond=cond, min_eig=float(ev.min()), max_abs_corr=float(offd.max()),
                      nyq_max=float(nyqfrac.max()), n_nyq_over_50=int((nyqfrac > 0.5).sum()),
                      nyq_mean=float(nyqfrac.mean()),
                      eff_nodes_median=float(np.median(en)), N=int(cands[n].shape[0]))
        print(f"{n:<15}{medcos:>8.3f}{maxcos:>8.3f}{cd[n]['varfrac'].mean():>7.3f}"
              f"{cond:>9.1f}{float(ev.min()):>9.4f}{float(offd.max()):>8.3f}"
              f"{float(nyqfrac.max()):>8.3f}{int((nyqfrac>0.5).sum()):>6d}"
              f"{float(np.median(en)):>10.0f}")

    print("\nv1 (published, RAW top-variance q) for the same members, same statistic:")
    for n in v1cd:
        q = v1cd[n]["q"]
        C = np.abs(q @ q.T); np.fill_diagonal(C, np.nan)
        S = np.einsum("cwd,cd->wc", pooled[n], q) - (v1cd[n]["mbar"] * q).sum(1)
        R = resid(D1, S); Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
        nf = np.array([float(((nyq @ Z[:, c]) / len(nyq)) ** 2) for c in range(Z.shape[1])])
        CM = np.corrcoef(Z.T); ev = np.linalg.eigvalsh(CM)
        offd = np.abs(CM - np.eye(CM.shape[0]))
        print(f"{n:<15}{float(np.nanmedian(C)):>8.3f}{float(np.nanmax(C)):>8.3f}"
              f"{v1cd[n]['varfrac'].mean():>7.3f}{float(ev.max()/max(ev.min(),1e-12)):>9.1f}"
              f"{float(ev.min()):>9.4f}{float(offd.max()):>8.3f}{float(nf.max()):>8.3f}"
              f"{int((nf>0.5).sum()):>6d}")

    # anchor portability, measured
    cos_perm = float(np.median(np.abs((cd["qperm_flag"]["q"] * qv).sum(1))))
    cos_rand = float(np.median(np.abs((cd["qrand_flag"]["q"] * qv).sum(1))))
    print(f"\nANCHOR PORTABILITY on the refit basis (bar: median |cos| vs vmax q < 0.15)")
    print(f"  qperm_flag  median|cos(q,q_vmax)| = {cos_perm:.3f}  "
          f"{'PORTS' if cos_perm < 0.15 else '** DOES NOT PORT **'}")
    print(f"  qrand_flag  median|cos(q,q_vmax)| = {cos_rand:.3f}  "
          f"{'OK' if cos_rand < 0.15 else '** FAILED **'}")

    pf = dict(rep=rep, qperm_cos=cos_perm, qrand_cos=cos_rand,
              qperm_ports=bool(cos_perm < 0.15),
              sae_sel=dict(features=sel_feats, cos_threshold=sel_thr,
                           max_pairwise_footprint_cos=sel_maxcos,
                           drop_cats=list(SEL_DROP_CATS), min_firerate=SEL_MIN_FIRERATE),
              design="trend + annual K=3 + diurnal K=1 + diurnal K=2 (Nyquist)",
              n_iid=NW, iid_span=[str(times.min()), str(times.max())])
    OUT_J.parent.mkdir(exist_ok=True)
    json.dump(pf, open(OUT_J, "w"), indent=1, default=float)

    np.save(OUT_C, dict(cands=cands, lat=lat, lon=lon, xyz=xyz,
            provenance=dict(pool="flagship_v2", N=int(N), mesh="2to6", n_iid=NW,
                            members=list(cands), base=str(POOL_V1),
                            sae_sel_features=sel_feats,
                            q_design="harmonic_design_v2 (+diurnal K=2)")),
            allow_pickle=True)
    np.save(OUT_CD, cd, allow_pickle=True)
    print(f"\nwrote {OUT_C.name}, {OUT_CD.name}, {OUT_J.name}")

if __name__ == "__main__":
    main()
