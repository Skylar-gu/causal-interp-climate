"""CG-v2 node definition: the FROZEN purity+decorrelation selection rule.

Rule (frozen, docs/prereg/prereg_concept_graph_v2.md):
  1. ALIVE        zcnt >= 300
  2. LABELLED     z_top >= 1.0        (z_top = max_c |z[:,c]| over the 10 concepts)
  3. PURE         z_top - z_second >= 0.5      <-- the new gate; v1 had none
  4. DECORRELATED greedy over the pure candidates in descending |z_c|: add a feature only
                  if |Pearson| of its ACTIVATION SERIES with EVERY already-chosen feature
                  is < 0.5. Series = results/fs_cgv2_actseries.npy (160 IID windows,
                  2016-01..2020-12, per-window SAE code sum over 40,962 mesh nodes).
  5. K = 4, fixed across all concepts.
  6. Any concept that cannot fill K is STRUCK and reported as a coverage gap.

Readout combiner (also frozen here): concept j's scalar = the FIRST PRINCIPAL COMPONENT of
its K standardized activation series. Stored as (mu, sd, w) with w unit-norm and sign fixed
so sum(w) > 0. A dose's effect on concept j is then  sum_k w[k] * d[f_k] / sd[k]  -- scale
free, and independent of group size by construction.

Produces NO causal number. system python3.

Paper: Fig. fig:contrast (a)/(c): concept response operators on generic initial states
Inputs: results/fs_atlas_extra.npy (not shipped, see docs/REPRODUCE.md); results/fs_cgv2_actseries.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/fs_cgv2_groups.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.concepts.cgv2_select
"""
from pathlib import Path
import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT
OUT = ROOT / "results/fs_cgv2_groups.npy"

CONCEPTS = ["vort850", "q600", "ascent", "shear", "t850", "z500", "jet250",
            "blocking", "atm_river", "baroclinicity"]
ZCNT_MIN = 300.0
ZMIN = 1.0
MARGIN_MIN = 0.5
CORR_MAX = 0.5
K = 4
SEED = 0

def load_z():
    a = np.load(ROOT / "results/fs_atlas_extra.npy", allow_pickle=True).item()
    z, refs, ze, ne = a["z"], a["node_refs"], a["z_extra"], a["node_extra"]
    zc = lambda n: z[:, refs.index(n)] if n in refs else ze[:, ne.index(n)]
    Z = np.stack([zc(n) for n in CONCEPTS], 1)
    return Z, a["zcnt"]

def greedy_decorr(cand, S, kmax=K, corr_max=CORR_MAX):
    """cand ordered by preference; S (nwin, F) series. Returns chosen list + trace."""
    chosen, trace = [], []
    for f in cand:
        if len(chosen) >= kmax:
            break
        if not chosen:
            chosen.append(f); trace.append((int(f), 0.0, True)); continue
        r = np.array([abs(pearson(S[:, f], S[:, g])) for g in chosen])
        ok = bool((r < corr_max).all())
        trace.append((int(f), float(r.max()), ok))
        if ok:
            chosen.append(f)
    return chosen, trace

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0

def pc1(S, feats):
    """First PC of the K standardized series. Returns mu, sd, w (unit, sum(w)>0), evr."""
    X = np.asarray(S[:, feats], np.float64)
    mu = X.mean(0); sd = X.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xs = (X - mu) / sd
    U, s, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    w = Vt[0]
    if w.sum() < 0:
        w = -w
    evr = float(s[0] ** 2 / max((s ** 2).sum(), 1e-12))
    return mu, sd, w, evr

def main():
    Z, zcnt = load_z()
    AZ = np.abs(Z)
    srt = np.sort(AZ, 1)
    ztop, zsec = srt[:, -1], srt[:, -2]
    margin = ztop - zsec
    lab = np.argmax(AZ, 1)
    alive = zcnt >= ZCNT_MIN
    S = np.load(ROOT / "results/fs_cgv2_actseries.npy", allow_pickle=True).item()["series"]

    print("CG-v2 SELECTION — purity gate + activation-series decorrelation, K = 4\n")
    print(f"  {'concept':<15}{'alive':>7}{'+lab':>7}{'+pure':>7}{'chosen':>8}{'evr(PC1)':>10}"
          f"{'max|r|':>9}   verdict")
    groups, pcs, diag = {}, {}, {}
    for k, n in enumerate(CONCEPTS):
        m_alive = alive & (lab == k)
        m_lab = m_alive & (ztop >= ZMIN)
        m_pure = m_lab & (margin >= MARGIN_MIN)
        cand = np.where(m_pure)[0]
        cand = cand[np.argsort(-AZ[cand, k])]
        chosen, trace = greedy_decorr(cand, S)
        struck = len(chosen) < K
        if not struck:
            mu, sd, w, evr = pc1(S, chosen)
            pcs[n] = dict(mu=mu, sd=sd, w=w, evr=evr)
            R = np.array([[abs(pearson(S[:, a], S[:, b])) for b in chosen] for a in chosen])
            mx = float(R[~np.eye(len(chosen), dtype=bool)].max())
        else:
            evr, mx = float("nan"), float("nan")
        groups[n] = [int(f) for f in chosen]
        diag[n] = dict(n_alive=int(m_alive.sum()), n_lab=int(m_lab.sum()),
                       n_pure=int(m_pure.sum()), trace=trace,
                       margins=[float(margin[f]) for f in chosen],
                       ztop=[float(AZ[f, k]) for f in chosen])
        print(f"  {n:<15}{m_alive.sum():>7}{m_lab.sum():>7}{m_pure.sum():>7}"
              f"{len(chosen):>8}{evr:>10.3f}{mx:>9.3f}   "
              f"{'** STRUCK (coverage gap) **' if struck else 'ok'}")

    kept = [n for n in CONCEPTS if len(groups[n]) == K]
    struck = [n for n in CONCEPTS if len(groups[n]) < K]
    print(f"\n  kept {len(kept)}/{len(CONCEPTS)}: {kept}")
    print(f"  STRUCK: {struck if struck else 'none'}")

    print("\n  chosen features (idx, |z|, margin):")
    for n in kept:
        d = diag[n]
        print(f"    {n:<15}" + "  ".join(f"{f}({zt:.2f}/{mg:.2f})" for f, zt, mg
                                         in zip(groups[n], d["ztop"], d["margins"])))
        print(f"      PC1 loadings {np.round(pcs[n]['w'], 3).tolist()}  evr {pcs[n]['evr']:.3f}")

    # CG-2 NEG control, defined here so it is frozen with the groups: the SAME K*len(kept)
    # features re-partitioned at random into len(kept) groups of K (seed 0).
    rng = np.random.default_rng(SEED)
    pool = np.concatenate([groups[n] for n in kept])
    perm = rng.permutation(pool)
    perm_groups = [[int(x) for x in perm[i * K:(i + 1) * K]] for i in range(len(kept))]
    perm_pcs = []
    for g in perm_groups:
        mu, sd, w, evr = pc1(S, g)
        perm_pcs.append(dict(mu=mu, sd=sd, w=w, evr=evr))
    print(f"\n  CG-2 NEG concept_perm: {len(pool)} features -> {len(kept)} random groups of {K}")
    print(f"    perm PC1 evr: {[round(p['evr'], 3) for p in perm_pcs]}")
    print(f"    real PC1 evr: {[round(pcs[n]['evr'], 3) for n in kept]}")

    np.save(OUT, dict(concepts=kept, struck=struck, groups={n: groups[n] for n in kept},
                      pcs={n: pcs[n] for n in kept}, perm_groups=perm_groups,
                      perm_pcs=perm_pcs, diag=diag, K=K, seed=SEED,
                      rule=dict(zcnt_min=ZCNT_MIN, zmin=ZMIN, margin_min=MARGIN_MIN,
                                corr_max=CORR_MAX, K=K),
                      series_src="results/fs_cgv2_actseries.npy"),
            allow_pickle=True)
    print(f"\n-> {OUT}")

if __name__ == "__main__":
    main()
