"""Score the concept-group global-RMSE run, with the two confound checks it needs.

THE RUN. 8 screened concept groups (6 features each) plus 8 firing-rate-matched controls
plus baseline, ablated at coef -1 at EVERY step, 8 ICs x 120 h, cos-lat-weighted RMSE against
ERA5 on the WeatherBench2 headline fields.

WHY THE PER-FIELD TABLE IS NOT THE ANSWER. 8 groups x 7 fields = 56 paired t-tests at n = 8.
At alpha 0.05 that yields 2.8 false positives in expectation; the run produced 2, and the
smallest Benjamini-Hochberg q is 0.54. Read one cell at a time, this is a flat negative.

WHAT ACTUALLY HAS POWER. The question is not "does group g move field f" but "do
concept-labelled features carry MORE forecast information than firing-rate-matched
controls". That pools all 8 groups. Two confounds have to die first:

  C1 NESTING. The 64 (group, IC) cells are not independent -- 8 groups share 8 ICs. Both
     clusterings are reported. Clustering by IC is the honest one because the ICs are
     independent draws while the GROUPS ARE NOT: 8 groups x 6 slots contain only 39 distinct
     features, and ascent/div250, q600/div250, shear/jet250 each share 3 of 6.

  C2 ABLATION MAGNITUDE. The controls match on FIRING RATE only. If concept features also
     fire harder, ablating them deletes a bigger vector and any gap is mechanical. The test
     is the deleted norm ||W_dec[:,f]|| * E[a|a>0] * firerate summed over the group, and
     then whether that ratio PREDICTS the gap across groups.

Paper: supporting: concept-group global RMSE (not a paper figure)
Inputs: results/fs_concept_rmse.npy (not shipped, see docs/REPRODUCE.md); i.i.d. dump (GC_SCRATCH/fs_iid_dump.npy, extraction/extract_iid_dump.py)
Outputs: printed report
Run:   # JAX env, CPU
    python -m graphcast_sae.concepts.concept_rmse_analyze
"""
import json
import os
import pathlib
import sys

import numpy as np
from scipy import stats

from graphcast_sae.paths import REPO_ROOT as ROOT, WEIGHTS, SCRATCH
RES = ROOT / os.environ.get("CR_RES", "results/fs_concept_rmse.npy")
GRP = os.environ.get("CR_GROUPS", "/tmp/screened_groups.json")

def deleted_norm(G, nwin=8):
    """Expected norm of what each group's patch removes from the residual stream."""
    z = np.load(WEIGHTS / "sae_k32_lat4096_lay08.npz")
    Wenc, bpre, Wdec = z["W_enc"], z["b_pre"], z["W_dec"]
    dn = np.linalg.norm(Wdec, axis=0 if Wdec.shape[0] == 512 else 1)
    META = json.load(open(SCRATCH / "fs_iid_meta.json"))
    L = META["n_mesh"]
    X = np.load(SCRATCH / "fs_iid_dump.npy", mmap_mode="r")
    tot = np.zeros(len(dn)); cnt = np.zeros(len(dn)); nrow = 0
    for j in np.linspace(0, META["n_windows"] - 1, nwin).astype(int):
        A = np.asarray(X[j * L:(j + 1) * L], np.float32)
        xn = A - A.mean(1, keepdims=True)
        xn /= (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-6)
        pre = np.maximum((xn - bpre) @ Wenc.T, 0.0)
        idx = np.argpartition(-pre, 32, axis=1)[:, :32]
        a = np.zeros_like(pre); r = np.arange(len(A))[:, None]; a[r, idx] = pre[r, idx]
        tot += a.sum(0); cnt += (a > 0).sum(0); nrow += L
    mag, fr = tot / np.maximum(cnt, 1), cnt / nrow
    return {k: float((dn[np.array(v)] * mag[np.array(v)] * fr[np.array(v)]).sum())
            for k, v in G.items()}, dn

def main():
    d = np.load(RES, allow_pickle=True).item()
    acc, F, arms, n = d["acc"], d["fields"], d["arms"], d["n"]
    G = json.load(open(GRP))
    ks = [a for a in arms if a != "baseline" and not a.startswith("ctrl_")
          and "ctrl_" + a in acc]
    B = acc["baseline"][:, -1, :]

    # --- 0. group independence, stated before any p-value is read ------------
    allf = sorted({f for k in ks for f in G[k]})
    print(f"{len(ks)} groups x {len(G[ks[0]])} slots contain {len(allf)} DISTINCT features")
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            sh = len(set(G[a]) & set(G[b]))
            if sh >= 3:
                print(f"  {a} and {b} share {sh} of {len(G[a])} -- not independent groups")

    # --- 1. the flat per-cell result, with FDR -------------------------------
    P = np.array([[stats.ttest_rel(acc[k][:, -1, i], acc["ctrl_" + k][:, -1, i],
                                   nan_policy="omit").pvalue for i in range(len(F))]
                  for k in ks]).ravel()
    m = len(P); o = np.argsort(P)
    q = np.minimum.accumulate((P[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    print(f"\nPER-CELL: {m} paired tests, {(P < 0.05).sum()} at p<0.05 "
          f"(expected {0.05*m:.1f} by chance), smallest BH q = {q.min():.3f}")

    # --- 2. pooled, and both clusterings -------------------------------------
    print(f"\n{'field':>6}{'ratio |arm|/|ctl|':>19}{'signed mean':>14}"
          f"{'p pooled':>10}{'p by group':>12}{'p by IC':>10}{'ICs worse':>11}")
    out = {}
    for i, f in enumerate(F):
        M = np.stack([acc[k][:, -1, i] - acc["ctrl_" + k][:, -1, i] for k in ks])
        A = np.concatenate([np.abs(acc[k][:, -1, i] - B[:, i]) for k in ks])
        C = np.concatenate([np.abs(acc["ctrl_" + k][:, -1, i] - B[:, i]) for k in ks])
        pg = stats.ttest_1samp(M.mean(1), 0).pvalue
        pi = stats.ttest_1samp(M.mean(0), 0).pvalue
        pp = stats.ttest_1samp(M.ravel(), 0).pvalue
        out[f] = (M.mean(), pg, pi)
        print(f"{f:>6}{A.mean()/max(C.mean(),1e-30):>19.2f}{M.mean():>+14.5f}"
              f"{pp:>10.3f}{pg:>12.3f}{pi:>10.3f}{int((M.mean(0)>0).sum()):>9}/{n}")

    # --- 3. the magnitude confound -------------------------------------------
    dnorm, dn = deleted_norm(G)
    print(f"\ndecoder norms: min {dn.min():.4f} max {dn.max():.4f} "
          f"(unit-normalised dictionary => no decoder-norm confound)")
    r = np.array([dnorm[k] / dnorm["ctrl_" + k] for k in ks])
    print(f"deleted-norm ratio concept/control: median {np.median(r):.2f} "
          f"range {r.min():.2f}-{r.max():.2f}")
    print("does it PREDICT the gap? (positive rho => mechanical; negative => not)")
    for i, f in enumerate(F):
        g = np.array([np.nanmean(acc[k][:, -1, i] - acc["ctrl_" + k][:, -1, i]) for k in ks])
        rho, p = stats.spearmanr(r, g)
        print(f"  {f:>5}  spearman {rho:>+6.3f}  p {p:.3f}")

if __name__ == "__main__":
    main()
