"""Flagship retry #1 — Ising/Leiden grouping of published-SAE features.

Port of `sae/retry1_ising_leiden.py`. Pipeline unchanged: harmonic deseasonalization
removed in closed form from the coupling, exact deseasonalized Pearson correlation
from the Gram, |corr| graph thresholded at the median positive edge, Leiden
(RBConfiguration, seed 0), identity-R² per community.

Declared deviation (prereg §2): 24 windows instead of 480, so the harmonic design
drops to trend + annual K=1 + diurnal K=1 (5 columns, ~5:1 points-per-parameter).

BAR: PASS iff #non-trivial communities >= 2 AND median per-community identity-R²
> 0.03. Also writes the community mode basis (footprints + channel directions) that
retry #2's steering readout uses.

Paper: not in the paper; kept for provenance only
Inputs: none beyond the arguments above
Outputs: results/graphcast_sae_communities.arrays.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.legacy.fs_retry1_communities
"""
import argparse
import json

import numpy as np

import graphcast_sae.common.fs_common as fc
from graphcast_sae.paths import FS_CATALOG, FS_MODES

MIN_ACT = 0.005
K_ANNUAL, K_DIURNAL = 1, 1

def harmonic_design(times):
    t = times.astype("datetime64[s]").astype(np.float64)
    t_lin = (t - t.mean()) / (t.std() + 1e-9)
    yr, day = 365.2422 * 86400.0, 86400.0
    cols = [np.ones_like(t), t_lin]
    for k in range(1, K_ANNUAL + 1):
        cols += [np.sin(2 * np.pi * k * t / yr), np.cos(2 * np.pi * k * t / yr)]
    for k in range(1, K_DIURNAL + 1):
        cols += [np.sin(2 * np.pi * k * t / day), np.cos(2 * np.pi * k * t / day)]
    return np.stack(cols, 1)

def deseason(P, keep, starts, n_mesh):
    """Seasonal fit s_f(w) of the node-mean series, per kept feature."""
    D = harmonic_design(starts)
    m = P[keep] / n_mesh
    beta, *_ = np.linalg.lstsq(D, m.T, rcond=None)
    return (D @ beta).T, m

def deseason_corr(P, G, keep, starts, n_mesh):
    """Exact deseasonalized Pearson corr among `keep` features (no dense sample matrix)."""
    NW = P.shape[1]
    Ntot = NW * n_mesh
    s, _ = deseason(P, keep, starts, n_mesh)
    Pk, Gk = P[keep], G[np.ix_(keep, keep)]
    cross = Pk @ s.T
    S_fg = Gk - cross - cross.T + n_mesh * (s @ s.T)
    tot = Pk.sum(1) - n_mesh * s.sum(1)
    cov = S_fg / Ntot - np.outer(tot, tot) / Ntot ** 2
    d = np.clip(np.diag(cov), 1e-12, None)
    corr = cov / np.sqrt(np.outer(d, d))
    np.fill_diagonal(corr, 0.0)
    return np.clip(corr, -1, 1)

def leiden_communities(corr):
    import igraph as ig
    import leidenalg as la
    A = np.abs(corr)
    pos = A[A > 0]
    thr = np.percentile(pos, 50) if pos.size else 0.0
    iu = np.triu_indices_from(A, 1)
    mask = A[iu] > thr
    g = ig.Graph(n=A.shape[0],
                 edges=list(zip(iu[0][mask].tolist(), iu[1][mask].tolist())),
                 edge_attrs={"weight": A[iu][mask].tolist()})
    part = la.find_partition(g, la.RBConfigurationVertexPartition,
                             weights="weight", seed=0)
    return np.array(part.membership), float(thr), int(mask.sum())

def identity_r2(sig):
    if sig.shape[0] < 2:
        return 1.0
    R = sig / (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-12)
    s = np.linalg.svd(R, compute_uv=False)
    return float(s[0] ** 2 / (s ** 2).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default=str(FS_CATALOG))
    ap.add_argument("--out", default="results/graphcast_sae_communities.json")
    args = ap.parse_args()

    z = np.load(args.cat, allow_pickle=True)
    featmap = z["featmap"].astype(np.float64)
    P, G, fire = z["P"], z["G"], z["fire"]
    N, F = int(z["n_mesh"]), int(z["F"])
    NW = P.shape[1]
    starts = np.array([np.datetime64(s) for s in z["starts"]], dtype="datetime64[ns]")
    rate = fire / (NW * N)
    keep = np.where(rate > MIN_ACT)[0]
    print(f"alive {int((fire>0).sum())}/{F}; active(rate>{MIN_ACT:.1%}) {keep.size}",
          flush=True)

    corr = deseason_corr(P, G, keep, starts, N)
    print(f"deseasonalized |corr|: median {np.median(np.abs(corr)):.3f} "
          f"max {np.abs(corr).max():.3f}", flush=True)

    membership, thr, nedge = leiden_communities(corr)
    sig = featmap[keep]
    ncomm = int(membership.max()) + 1
    comm = []
    for c in range(ncomm):
        idx = np.where(membership == c)[0]
        if idx.size:
            comm.append(dict(c=c, size=int(idx.size), id_r2=identity_r2(sig[idx]),
                             idx=idx))
    comm.sort(key=lambda d: -d["size"])
    minf = max(5, int(0.01 * keep.size))
    nontriv = [d for d in comm if d["size"] >= minf]
    med_id = float(np.median([d["id_r2"] for d in nontriv])) if nontriv else float("nan")

    gm = sig.mean(0)
    ss_tot = ((sig - gm) ** 2).sum()
    ss_within = sum(((sig[d["idx"]] - sig[d["idx"]].mean(0)) ** 2).sum() for d in comm)
    clust_r2 = float(1 - ss_within / (ss_tot + 1e-12))

    print(f"\nLeiden: {ncomm} communities (thr {thr:.3f}, {nedge} edges); "
          f"{len(nontriv)} non-trivial (>={minf} feats)")
    for d in nontriv[:12]:
        print(f"  comm {d['c']:2d}: {d['size']:4d} feats  identity-R2={d['id_r2']:.3f}")
    print(f"\nmedian non-trivial identity-R2 = {med_id:.3f}")
    print(f"between-community clustering R2 = {clust_r2:.3f}")
    passed = len(nontriv) >= 2 and med_id > 0.03
    print(f"\nBAR(#1): >=2 non-trivial communities AND median identity-R2 > 0.03 -> "
          f"{'PASS' if passed else 'MISS'}")

    # ---- mode basis for retry #2 (prereg §4): footprint + channel direction ----
    sae = fc.sae_numpy()
    Wc, Qc, members = [], [], []
    s_fit, m_series = deseason(P, keep, starts, N)
    series = []
    for d in nontriv:
        f_idx = keep[d["idx"]]
        fp = featmap[f_idx].mean(0)
        fp = fp / (np.abs(fp).sum() + 1e-12)
        q = sae["W_dec"][:, f_idx].mean(1)
        q = q / (np.linalg.norm(q) + 1e-12)
        Wc.append(fp); Qc.append(q); members.append(f_idx)
        series.append((m_series[d["idx"]] - s_fit[d["idx"]]).mean(0))
    if nontriv:
        np.savez(str(FS_MODES),
                 W=np.array(Wc), Q=np.array(Qc), sigma=np.std(series, axis=1),
                 series=np.array(series), comm_id=np.array([d["c"] for d in nontriv]),
                 members=np.array(members, dtype=object), keep=keep,
                 rate=rate, starts=z["starts"])

    json.dump(dict(
        program="flagship-SAE suite -- NOT cross-comparable to small-model G1",
        prereg="docs/prereg/prereg_flagship_g2_suite.md §2",
        n_windows=NW, n_active=int(keep.size), n_alive=int((fire > 0).sum()),
        min_act=MIN_ACT, harmonic="trend + annual K=1 + diurnal K=1 (24 windows)",
        n_communities=ncomm, n_nontrivial=len(nontriv), min_feat=minf,
        median_identity_r2=med_id, clustering_r2=clust_r2,
        edge_threshold=thr, n_edges=nedge, passed=bool(passed),
        communities=[dict(c=d["c"], size=d["size"], id_r2=d["id_r2"]) for d in comm[:40]],
    ), open(fc.ROOT / args.out, "w"), indent=1)
    np.save(fc.ROOT / "results/graphcast_sae_communities.arrays.npy",
            dict(membership=membership, keep=keep, corr_abs_median=float(np.median(np.abs(corr)))),
            allow_pickle=True)
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()
