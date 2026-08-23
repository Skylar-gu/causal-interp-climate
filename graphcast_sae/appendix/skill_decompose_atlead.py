"""At-lead (contemporaneous) decompose: does day-L internal state predict the day-L
skill advantage where hour-0 state could not?

For each lead L in {72,120,168}h: features = layer-8 SAE region-mean firing captured
AT lead L during the full rollout (results/skill/atlead/case_*.npy, snapshot str(L)),
target = per-case Z500 NHext advantage AT lead L (sanity_gate.npy, single lead).
Same regularized month-grouped nested-CV regression + permuted-label and random-feature
nulls as the hour-0 decompose. Reports per-lead CV R2, p_perm, p_rand, and top
skill-features (classified via fs_atlas_class.npy). Compares to the hour-0 null.

Paper: Appendix app:taxonomy (skill decomposition)
Inputs: results/fs_atlas_class.npy (not shipped, see docs/REPRODUCE.md); results/skill/atlead (not shipped, see docs/REPRODUCE.md); results/skill/decompose.npy (not shipped, see docs/REPRODUCE.md); results/skill/sanity_gate.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/skill/decompose_atlead.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.appendix.skill_decompose_atlead
"""
import os, sys, collections

import numpy as np
from sklearn.preprocessing import StandardScaler
from graphcast_sae.appendix.skill_decompose import abscorr, nested_cv, month_folds, TOPK, REGS, NF

from graphcast_sae.paths import REPO_ROOT
ROOT = str(REPO_ROOT)
ATLEAD = f"{ROOT}/results/skill/atlead"
LEADS = [72, 120, 168]
N_NULL = 50
RNG = np.random.default_rng(0)

def adv_per_lead():
    sg = np.load(f"{ROOT}/results/skill/sanity_gate.npy", allow_pickle=True).item()
    adv = {}   # (ci,lead) -> hres-gc
    for case, lead_h, rh, rg in sg["rows"]:
        adv[(int(case), int(lead_h))] = rh - rg
    return adv

def load_lead(L, adv):
    """Feature matrix at lead L and target adv_L, matched by ci."""
    snap = str(L)
    files = sorted(f for f in os.listdir(ATLEAD) if f.startswith("case_") and f.endswith(".npy"))
    X, y, months, cis = [], [], [], []
    for fn in files:
        d = np.load(f"{ATLEAD}/{fn}", allow_pickle=True).item()
        ci = int(d["ci"])
        if (ci, L) not in adv or snap not in d["featvec"]:
            continue
        X.append(np.concatenate([d["featvec"][snap][r] for r in REGS]))
        y.append(adv[(ci, L)])
        months.append(np.datetime64(d["init"], "M").astype(int) % 12 + 1)
        cis.append(ci)
    colmap = np.array([(snap, r, f) for r in REGS for f in range(NF)], dtype=object)
    return (np.array(X, np.float64), np.array(y, np.float64),
            np.array(months), np.array(cis), colmap)

def classify_top(colk, rfull, cat):
    feat_score = np.zeros(NF); feat_best = {}
    for j in np.argsort(-rfull):
        f = int(colk[j][2])
        if rfull[j] > feat_score[f]:
            feat_score[f] = rfull[j]; feat_best[f] = colk[j][1]
    top = np.argsort(-feat_score)[:TOPK]
    def bucket(c):
        if c in ("physics(single)", "joint-coupling"): return "known-physics"
        if c == "residual": return "residual/novel"
        if c == "climatology/clock": return "geography/clock"
        if c == "teleconnection/mode": return "teleconnection"
        if c == "numerical/geometry": return "numerical/geometry"
        return "regime/other"
    bc = collections.Counter(bucket(cat[int(f)]) for f in top)
    return top, feat_score, bc

def main():
    adv = adv_per_lead()
    cat = np.asarray(np.load(f"{ROOT}/results/fs_atlas_class.npy", allow_pickle=True).item()["cat"])
    out = {}
    print("===== At-lead (contemporaneous) decompose =====")
    for L in LEADS:
        X, y, months, cis, colmap = load_lead(L, adv)
        if len(y) < 20:
            print(f"lead {L}h: only {len(y)} cases available -- skip"); continue
        vkeep = X.std(0) > 1e-9
        Xk, colk = X[:, vkeep], colmap[vkeep]
        folds = month_folds(months, k=6)
        r2_skill, _ = nested_cv(Xk, y, folds, select="corr", topk=TOPK)
        r2_all, _ = nested_cv(Xk, y, folds, select="all", alpha=100.0)
        r2_perm = np.array([nested_cv(Xk, RNG.permutation(y), folds, select="corr", topk=TOPK)[0]
                            for _ in range(N_NULL)])
        r2_rand = np.array([nested_cv(Xk, y, folds, select="random", topk=TOPK)[0]
                            for _ in range(N_NULL)])
        p_perm = float(np.mean(r2_perm >= r2_skill))
        p_rand = float(np.mean(r2_rand >= r2_skill))
        sc = StandardScaler().fit(Xk); rfull = abscorr(sc.transform(Xk), y)
        top, feat_score, bc = classify_top(colk, rfull, cat)
        print(f"\n--- lead {L}h  (n={len(y)}, adv mean={y.mean():+.2f} gpm, {Xk.shape[1]} predictors) ---")
        print(f"  skill-features (top{TOPK}) CV R2 = {r2_skill:+.3f}")
        print(f"  all-features ridge     CV R2 = {r2_all:+.3f}")
        print(f"  permuted-y null  CV R2 = {r2_perm.mean():+.3f} +- {r2_perm.std():.3f}  (p={p_perm:.3f})")
        print(f"  random-feature   CV R2 = {r2_rand.mean():+.3f} +- {r2_rand.std():.3f}  (p={p_rand:.3f})")
        print(f"  top-feature |corr| max = {rfull.max():.2f}")
        print(f"  top-{TOPK} skill-feature classes: " +
              ", ".join(f"{k}:{v}" for k, v in bc.most_common()))
        out[L] = dict(r2_skill=r2_skill, r2_all=r2_all, r2_perm=r2_perm, r2_rand=r2_rand,
                      p_perm=p_perm, p_rand=p_rand, top=top.astype(int),
                      feat_score=feat_score, n=len(y), bc=dict(bc), rmax=float(rfull.max()))
    # hour-0 reference
    try:
        init = np.load(f"{ROOT}/results/skill/decompose.npy", allow_pickle=True).item()
        print(f"\n[hour-0 reference] skill CV R2={float(init['r2_skill']):+.3f} "
              f"p_perm={float(init['p_perm']):.3f} p_rand={float(init['p_rand']):.3f}")
        out["init"] = dict(r2_skill=float(init["r2_skill"]), p_perm=float(init["p_perm"]),
                           p_rand=float(init["p_rand"]))
    except Exception as e:
        print("hour-0 reference unavailable:", e)
    np.save(f"{ROOT}/results/skill/decompose_atlead.npy", out, allow_pickle=True)
    print("\n-> results/skill/decompose_atlead.npy")

if __name__ == "__main__":
    main()
