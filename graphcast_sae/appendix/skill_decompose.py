"""Phase 4 (CPU): decompose GC's skill advantage onto layer-8 SAE features.

4a Predictive: nested-CV (grouped by month) ridge on adv ~ features, with leakage-free
   feature selection INSIDE each fold. Compare CV R2 to permuted-y null and random-feature
   subsets (identical pipeline). Rank stable skill-features on full data (for Phase 5 / 4c).
4b Regime: correlate per-case adv with atlas regime detectors (blocking/atm_river/baroclinicity).
4c Characterize: classify top skill-features via results/fs_atlas_class.npy.

Paper: Appendix app:taxonomy (skill decomposition)
Inputs: results/fs_atlas_class.npy (not shipped, see docs/REPRODUCE.md); results/fs_atlas_extra.npy (not shipped, see docs/REPRODUCE.md); results/skill (shipped); results/skill/sanity_gate.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/skill/decompose.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.appendix.skill_decompose     (pure numpy/sklearn, no GCS needed)
"""
import os, sys, json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats

from graphcast_sae.paths import REPO_ROOT
ROOT = str(REPO_ROOT)
REGS = ["global", "nhext", "tropics"]
NF = 4096
TARGET_LEADS = [120, 168]   # headline medium-range
TOPK = 20
RNG = np.random.default_rng(0)

def _adv_from_sanity():
    """Per-case medium-range Z500 NH-ext advantage (rmse_hres - rmse_gc), mean over
    TARGET_LEADS. Source: results/skill/sanity_gate.npy (full-rollout scoring, all 120)."""
    sg = np.load(f"{ROOT}/results/skill/sanity_gate.npy", allow_pickle=True).item()
    rows = sg["rows"]  # cols: case, lead_h, rmse_hres, rmse_gc
    adv = {}
    for case, lead_h, rh, rg in rows:
        if int(lead_h) in TARGET_LEADS:
            adv.setdefault(int(case), []).append(rh - rg)
    return {c: float(np.mean(v)) for c, v in adv.items()}

def load_cases():
    """Prefer init_*.npy (fast INIT-only cut); fall back to case_*.npy (full-rollout).
    Features from the per-case featvec; adv target from sanity_gate.npy (matched by ci)."""
    allf = os.listdir(f"{ROOT}/results/skill")
    init_files = sorted(f for f in allf if f.startswith("init_"))
    if init_files:
        files, SNAPS = init_files, ["init"]
    else:
        files, SNAPS = sorted(f for f in allf if f.startswith("case_")), ["init", "72", "120", "168"]
    adv_map = _adv_from_sanity()
    colmap = np.array([(s, r, f) for s in SNAPS for r in REGS for f in range(NF)], dtype=object)
    X, y_all, months, cis = [], [], [], []
    for fn in files:
        d = np.load(f"{ROOT}/results/skill/{fn}", allow_pickle=True).item()
        ci = int(d["ci"])
        if ci not in adv_map:
            continue
        X.append(np.concatenate([d["featvec"][s][r] for s in SNAPS for r in REGS]))
        y_all.append(adv_map[ci])
        months.append(np.datetime64(d["init"], "M").astype(int) % 12 + 1)
        cis.append(ci)
    return (np.array(X, np.float64), np.array(y_all, np.float64),
            np.array(months), np.array(cis), colmap, SNAPS)

def month_folds(months, k=6):
    """Group cases into k folds by month blocks (contiguous months) to avoid seasonal leakage."""
    order = np.argsort(months)
    folds = np.zeros(len(months), int)
    # assign each month to a fold cyclically
    umon = np.unique(months)
    m2f = {m: i % k for i, m in enumerate(umon)}
    return np.array([m2f[m] for m in months])

def abscorr(Xc, y):
    """Vectorized |Pearson r| of each column of Xc with y (identical to looping corrcoef)."""
    xc = Xc - Xc.mean(0); sx = xc.std(0)
    yc = y - y.mean(); sy = y.std()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (xc * yc[:, None]).mean(0) / (sx * sy)
    return np.abs(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0))

def nested_cv(X, y, folds, select="corr", topk=TOPK, alpha=10.0):
    """Leakage-free out-of-fold predictions. select: 'corr' top-k, 'random' k, 'all'."""
    oof = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ytr = y[tr]
        if select == "all":
            idx = np.arange(X.shape[1])
        elif select == "random":
            idx = RNG.choice(X.shape[1], topk, replace=False)
        else:
            r = abscorr(Xtr, ytr)
            idx = np.argsort(-r)[:topk]
        m = Ridge(alpha=alpha).fit(Xtr[:, idx], ytr)
        oof[te] = m.predict(Xte[:, idx])
    ss_res = np.sum((y - oof) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot, oof

def main():
    X, y, months, cis, colmap, SNAPS = load_cases()
    n = len(y)
    print(f"snapshots used: {SNAPS}")
    print(f"loaded {n} cases; target = z500 nhext adv mean over {TARGET_LEADS}h")
    print(f"  adv: mean={y.mean():+.2f} std={y.std():.2f} gpm  frac>0={np.mean(y>0):.2f}")
    # prune zero-variance columns
    vkeep = X.std(0) > 1e-9
    Xk = X[:, vkeep]; colk = colmap[vkeep]
    print(f"  predictors: {Xk.shape[1]} nonconstant of {X.shape[1]}")
    folds = month_folds(months, k=6)

    # --- 4a nested CV ---
    r2_skill, oof = nested_cv(Xk, y, folds, select="corr", topk=TOPK)
    r2_all, _ = nested_cv(Xk, y, folds, select="all", alpha=100.0)
    # permuted null (same corr-select pipeline)
    r2_perm = []
    for s in range(50):
        yp = RNG.permutation(y)
        r2p, _ = nested_cv(Xk, yp, folds, select="corr", topk=TOPK)
        r2_perm.append(r2p)
    r2_perm = np.array(r2_perm)
    # random-feature control
    r2_rand = []
    for s in range(50):
        r2r, _ = nested_cv(Xk, y, folds, select="random", topk=TOPK)
        r2_rand.append(r2r)
    r2_rand = np.array(r2_rand)

    p_perm = np.mean(r2_perm >= r2_skill)
    p_rand = np.mean(r2_rand >= r2_skill)
    print("\n===== Phase 4a: predictive decomposition (nested CV, grouped by month) =====")
    print(f"  skill-features (top{TOPK}, corr-select in-fold) CV R2 = {r2_skill:+.3f}")
    print(f"  all-features ridge CV R2                        = {r2_all:+.3f}")
    print(f"  permuted-y null  CV R2 = {r2_perm.mean():+.3f} +- {r2_perm.std():.3f}  (p={p_perm:.3f})")
    print(f"  random-feature   CV R2 = {r2_rand.mean():+.3f} +- {r2_rand.std():.3f}  (p={p_rand:.3f})")

    # --- rank stable skill-features on full data (for Phase 5 / 4c) ---
    sc = StandardScaler().fit(Xk); Xs = sc.transform(Xk)
    rfull = abscorr(Xs, y)
    order = np.argsort(-rfull)
    # aggregate to SAE-feature level: best |corr| across snapshot/region
    feat_score = np.zeros(NF)
    feat_best = {}
    for j in order:
        s, r, f = colk[j]
        if rfull[j] > feat_score[f]:
            feat_score[f] = rfull[j]; feat_best[f] = (s, r)
    top_feats = np.argsort(-feat_score)[:TOPK]
    print(f"\n  top-{TOPK} skill SAE features (idx, |corr|, best snap/reg):")
    for f in top_feats:
        s, r = feat_best[int(f)]
        print(f"    feat {int(f):4d}  |corr|={feat_score[f]:.2f}  ({s}/{r})")

    # --- 4c classify ---
    cls = np.load(f"{ROOT}/results/fs_atlas_class.npy", allow_pickle=True).item()
    cat = np.asarray(cls["cat"])
    KNOWN = {"physics(single)", "joint-coupling"}
    NOVEL = {"residual"}
    def bucket(c):
        if c in KNOWN: return "known-physics"
        if c in NOVEL: return "residual/novel"
        if c == "climatology/clock": return "geography/clock"
        if c == "teleconnection/mode": return "teleconnection"
        if c == "numerical/geometry": return "numerical/geometry"
        return "regime/other"
    import collections
    bc = collections.Counter(bucket(cat[int(f)]) for f in top_feats)
    print("\n===== Phase 4c: classification of top skill-features =====")
    for k, v in sorted(bc.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v}/{TOPK}  ({v/TOPK:.0%})")
    nphys = sum(bucket(cat[int(f)]) == "known-physics" for f in top_feats)
    nnov = sum(bucket(cat[int(f)]) == "residual/novel" for f in top_feats)
    print(f"  -> known-mechanism (physics/joint): {nphys}/{TOPK};  residual/novel: {nnov}/{TOPK}")

    # --- 4b regime: correlate adv with regime-detector loadings of active features ---
    ex = np.load(f"{ROOT}/results/fs_atlas_extra.npy", allow_pickle=True).item()
    zx = np.asarray(ex["z_extra"]); ncols = list(ex["node_extra"])
    print("\n===== Phase 4b: regime association =====")
    # per-case regime index = sum over features of (nhext activation) * detector-loading
    regime_snap = "120" if "120" in SNAPS else "init"
    prefix = "init_" if SNAPS == ["init"] else "case_"
    files_b = sorted(f for f in os.listdir(f"{ROOT}/results/skill") if f.startswith(prefix))
    _adv = _adv_from_sanity()
    act, yb = [], []
    for fn in files_b:
        d = np.load(f"{ROOT}/results/skill/{fn}", allow_pickle=True).item()
        if int(d["ci"]) not in _adv: continue
        act.append(d["featvec"][regime_snap]["nhext"]); yb.append(_adv[int(d["ci"])])
    act = np.stack(act); yb = np.array(yb)   # (n, 4096)
    for det in ("blocking", "atm_river", "baroclinicity"):
        w = zx[:, ncols.index(det)]
        w = np.clip(w, 0, None)
        idx = (act @ w)                             # per-case regime activity
        if idx.std() > 0:
            rr = np.corrcoef(idx, yb)[0, 1]
            print(f"  adv vs {det:14s} activity: r={rr:+.2f}")

    np.save(f"{ROOT}/results/skill/decompose.npy", dict(
        r2_skill=r2_skill, r2_all=r2_all, r2_perm=r2_perm, r2_rand=r2_rand,
        p_perm=p_perm, p_rand=p_rand, top_feats=top_feats.astype(int),
        feat_score=feat_score, y=y, oof=oof, cis=cis,
        classes={int(f): str(cat[int(f)]) for f in top_feats}), allow_pickle=True)
    print("\n-> results/skill/decompose.npy")

if __name__ == "__main__":
    main()
