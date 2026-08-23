"""Score the CG-v2 concept causal graph against docs/prereg/prereg_concept_graph_v2.md.

Reads results/fs_concept_cgraph_v2.npy (full stored feature deltas) and projects it through
the three pre-registered readouts and the three pre-registered matrix transforms. No new
forwards. system python3.

Every bar below is quoted from the prereg, which was frozen before the run.

Paper: supporting: interventional concept graph (not a paper figure)
Inputs: results/fs_cgv2_groups.npy (not shipped, see docs/REPRODUCE.md); results/fs_concept_cgraph_v2.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/concept_cgraph_v2_score.txt
Run:   # JAX env, CPU
    python -m graphcast_sae.concepts.concept_cgraph_v2_score
"""
from pathlib import Path
from math import comb
import numpy as np

from graphcast_sae.paths import REPO_ROOT as ROOT
R = np.load(ROOT / "results/fs_concept_cgraph_v2.npy", allow_pickle=True).item()
G = np.load(ROOT / "results/fs_cgv2_groups.npy", allow_pickle=True).item()

D, names, gammas = R["D"], list(R["names"]), list(R["gammas"])
ICS, SET_A, SET_B = list(R["ics"]), list(R["set_a"]), list(R["set_b"])
ia = [ICS.index(x) for x in SET_A]; ib = [ICS.index(x) for x in SET_B]
C = len(names); ix = {n: i for i, n in enumerate(names)}
members = [list(m) for m in R["members"]]; perm = [list(m) for m in R["perm"]]
pcs_real = [G["pcs"][n] for n in names]; pcs_perm = list(G["perm_pcs"])

BATTERY = [("R2", "t850", "z500", +1, "hydrostatic thickness"),
           ("R3", "jet250", "baroclinicity", +1, "thermal wind"),
           ("R4", "baroclinicity", "vort850", +1, "baroclinic growth"),
           ("R5", "q600", "ascent", +1, "latent heating"),
           ("R6", "ascent", "q600", +1, "moisture convergence"),
           ("R7", "shear", "ascent", -1, "shear suppresses organized convection"),
           ("R8", "blocking", "jet250", -1, "blocking diverts the jet"),
           ("R9", "q600", "atm_river", +1, "moisture transport"),
           ("R10", "atm_river", "ascent", +1, "AR-forced ascent"),
           ("R11", "baroclinicity", "z500", +1, "baroclinic height response")]

OUTLINES = []
def P(s=""):
    print(s, flush=True); OUTLINES.append(s)

# ---------------------------------------------------------------- readouts --
def project(d, groups, pcs, ro):
    """d (..., F) -> (..., C) via the pre-registered readout `ro`."""
    out = []
    for g, pc in zip(groups, pcs):
        v = d[..., g]                                    # (..., K)
        if ro == "sum":
            out.append(v.sum(-1))
        else:
            zs = v / pc["sd"][None, :] if v.ndim > 1 else v / pc["sd"]
            out.append(zs.mean(-1) if ro == "zmean" else (zs * pc["w"]).sum(-1))
    return np.stack(out, -1)

def matrices(gi, ro):
    """-> A_real, A_perm each (n_ic, C, C) under readout `ro` at gamma index gi."""
    ar = project(D[gi, 0], members, pcs_real, ro)         # (nic, C_src, C_tgt)
    ap = project(D[gi, 1], perm, pcs_perm, ro)
    return ar, ap

# -------------------------------------------------------------- transforms --
def dcenter(M):
    """Double-centre the off-diagonal cells; diagonal returned as-is (unused)."""
    X = M.astype(float).copy(); np.fill_diagonal(X, np.nan)
    rm = np.nanmean(X, 1, keepdims=True); cm = np.nanmean(X, 0, keepdims=True)
    gm = np.nanmean(X)
    Y = X - rm - cm + gm
    Y[np.isnan(Y)] = 0.0
    return Y

def offdiag(M):
    return M[~np.eye(C, dtype=bool)]

def repro(a0, b0, label, k=10):
    x, y = offdiag(a0), offdiag(b0)
    rho = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    sign = float(np.mean(np.sign(x) == np.sign(y)))
    ta = set(map(tuple, np.argwhere(np.abs(a0) >= np.sort(np.abs(x))[-k])))
    tb = set(map(tuple, np.argwhere(np.abs(b0) >= np.sort(np.abs(y))[-k])))
    shared = len(ta & tb)
    P(f"    {label:<26} rho {rho:+.3f}   sign {sign:.3f}   top-10 shared {shared}/10")
    return rho, sign, shared

def split(stack, tf, other=None):
    """mean over each IC set, then apply transform. Returns (A_setA, A_setB)."""
    out = []
    for idxs in (ia, ib):
        M = stack[idxs].mean(0).copy()
        if tf == "diff":
            M = M - other[idxs].mean(0)
        Z = M.copy(); np.fill_diagonal(Z, 0.0)
        out.append(dcenter(M) if tf == "dc" else Z)
    return out

def cg1_verdict(rho, sign, shared):
    if rho >= 0.50 and sign >= 0.70 and shared >= 6:
        return "REPRODUCIBLE"
    return "NOT REPRODUCIBLE" if rho < 0.30 else "PARTIAL"

# ------------------------------------------------------------------- score --
P("=" * 86)
P("CONCEPT CAUSAL GRAPH v2 — PURITY GATE, scored against the frozen prereg")
P("=" * 86)
P(f"concepts ({C}): {names}")
P(f"struck: {list(R['struck']) or 'none'}   (div250 remains struck from v1: 0 features)")
P(f"K = {R['K']}   gammas {gammas}   ICs {len(ICS)} (Set A {len(SET_A)} / Set B {len(SET_B)})")
P("")
P("v1 reference, cannot move:  concept-level rho +0.976 sign 0.711 top-10 7/10")
P("                            NEG perm      rho +0.893 sign 0.911 top-10 9/10  -> VOID")
P("                            feature level rho +0.181 sign 0.513 top-20 6/20")
P("")

gi_primary = gammas.index(1.0)
summary = {}

for ro, rolabel in (("pc1", "RO-A pc1 (as originally specified)"),
                    ("zmean", "RO-B zmean (sign-coherent standardized mean)"),
                    ("sum", "RO-C sum (identical to v1's readout)")):
    P("-" * 86)
    P(f"READOUT {rolabel}    [gamma = 1.0, the primary dose]")
    P("-" * 86)
    ar, ap = matrices(gi_primary, ro)

    # CG-2 POS: self-effect dominates its own row (real arm, all ICs)
    A_all = ar.mean(0)
    diag_wins = sum(int(np.argmax(np.abs(A_all[r])) == r) for r in range(C))
    pos_ok = diag_wins >= C - 1
    P(f"  CG-2 POS  self-effect largest in its own row: {diag_wins}/{C}  (bar >= {C-1})"
      f"  -> {'PASS' if pos_ok else '** FAIL: readout broken, no verdict **'}")
    if not pos_ok:
        bad = [names[r] for r in range(C) if np.argmax(np.abs(A_all[r])) != r]
        P(f"            rows whose diagonal is not largest: {bad}")

    P("  CG-2 NEG / CG-1 across the three declared transforms:")
    res = {}
    for tf, tflab in (("raw", "T-raw"), ("diff", "T-diff (A_real-A_perm)"),
                      ("dc", "T-dc (double-centred)")):
        # T-diff has no matched null (there is no second perm partition to difference
        # against), so its NEG row is the RAW perm and T-diff stays descriptive -- as the
        # prereg declared. T-raw and T-dc both have a properly matched perm null.
        pa, pb = split(ap, tf if tf != "diff" else "raw")
        na = repro(pa, pb, f"{tflab}  NEG perm" + (" [raw, unmatched]" if tf == "diff" else ""))
        if tf == "diff":
            ra, rb = split(ar, "diff", ap)
        else:
            ra, rb = split(ar, tf)
        rr = repro(ra, rb, f"{tflab}  concept-level")
        # CG-4 matrix: mean over ALL ICs (v1 used A_ic.mean(0)), transform applied after
        M = ar.mean(0) - (ap.mean(0) if tf == "diff" else 0.0)
        res[tf] = dict(neg=na, real=rr, A=dcenter(M) if tf == "dc" else M)
        neg_ok = na[0] < 0.30
        P(f"      -> NEG {na[0]:+.3f} < 0.30 : {'PASS' if neg_ok else '** FAIL **'}"
          f"    CG-1: {cg1_verdict(*rr)}"
          f"{'' if neg_ok else '   (VOID: calibration failed)'}")

    # headline transform is T-raw (like-for-like with v1)
    neg_raw = res["raw"]["neg"][0]
    v_raw = cg1_verdict(*res["raw"]["real"])
    P(f"  HEADLINE (T-raw, like-for-like with v1): CG-1 {v_raw}"
      f"{'' if neg_raw < 0.30 else '  -> **VOID** (NEG perm rho >= 0.30)'}")
    P(f"  perm rho moved from v1's +0.893 to {neg_raw:+.3f}  "
      f"(delta {neg_raw - 0.893:+.3f})")

    # CG-3
    A = ar.mean(0)
    mass = np.abs(A).sum(1)
    sp = float(np.corrcoef(np.argsort(np.argsort(np.repeat(mass, C))),
                           np.argsort(np.argsort(np.abs(A).ravel())))[0, 1])
    P(f"  CG-3 magnitude confound: Spearman(|A[i,j]|, source total effect) = {sp:+.3f}"
      f"   (v1 +0.110)")

    # CG-4 on T-raw and T-diff
    P("  CG-4 ANSWER KEY (R1 struck: div250 has no features)")
    P(f"      {'rel':<5}{'relation':<30}{'pred':>5}{'A_raw':>12}{'ok':>4}"
      f"{'A_diff':>12}{'ok':>4}   asym")
    hits = {"raw": 0, "diff": 0}
    Araw = res["raw"]["A"]
    Adif = res["diff"]["A"]
    for rid, s, t, pred, why in BATTERY:
        vr, vd = Araw[ix[s], ix[t]], Adif[ix[s], ix[t]]
        rv = Araw[ix[t], ix[s]]
        okr, okd = np.sign(vr) == pred, np.sign(vd) == pred
        hits["raw"] += int(okr); hits["diff"] += int(okd)
        asym = abs(vr - rv) / max(abs(vr) + abs(rv), 1e-12)
        P(f"      {rid:<5}{s+' -> '+t:<30}{'+' if pred > 0 else '-':>5}{vr:>12.4g}"
          f"{'Y' if okr else 'n':>4}{vd:>12.4g}{'Y' if okd else 'n':>4}   {asym:.2f}")
    n = len(BATTERY)
    for tf in ("raw", "diff"):
        h = hits[tf]
        p = sum(comb(n, k) for k in range(h, n + 1)) / 2 ** n
        vv = "CONFIRMED" if h >= 8 else ("NULL" if h <= 6 else "INCONCLUSIVE")
        P(f"      signed-edge accuracy [{tf:<4}] {h}/{n}  binomial p = {p:.4f}  -> {vv}"
          f"{'' if neg_raw < 0.30 else '   (descriptive only: CG-2 NEG failed)'}")
    r7r, r8r = Araw[ix['shear'], ix['ascent']], Araw[ix['blocking'], ix['jet250']]
    r7d, r8d = Adif[ix['shear'], ix['ascent']], Adif[ix['blocking'], ix['jet250']]
    P(f"      R7/R8 — the two NEGATIVE relations, which co-occurrence CANNOT fake:")
    P(f"        R7 shear->ascent    raw {r7r:+.4g} ({'NEG ok' if r7r < 0 else 'positive'})"
      f"   diff {r7d:+.4g} ({'NEG ok' if r7d < 0 else 'positive'})   [v1: +3.415, positive]")
    P(f"        R8 blocking->jet250 raw {r8r:+.4g} ({'NEG ok' if r8r < 0 else 'positive'})"
      f"   diff {r8d:+.4g} ({'NEG ok' if r8d < 0 else 'positive'})   [v1: +14.28, positive]")
    if hits["raw"] >= 8 and r7r > 0 and r8r > 0:
        P("        -> pre-declared DOWNGRADE: 'co-occurrence recovered, causation not"
          " demonstrated'")
    summary[ro] = dict(pos=pos_ok, diag_wins=diag_wins, neg_raw=neg_raw,
                       cg1=v_raw, cg4_raw=hits["raw"], cg4_diff=hits["diff"],
                       r7=float(r7r), r8=float(r8r), r7d=float(r7d), r8d=float(r8d),
                       cg3=sp, res={k: dict(neg=v["neg"], real=v["real"])
                                    for k, v in res.items()})
    P("")

# ---------------------------------------------------------------- dose sweep -
P("-" * 86)
P("DOSE SWEEP — the pre-registered open question: does the perm-control rho depend on dose?")
P("-" * 86)
P(f"  {'readout':<8}{'transform':<10}" + "".join(f"{'g=' + str(g):>12}" for g in gammas))
sweep = {}
for ro in ("pc1", "zmean", "sum"):
    for tf in ("raw", "dc"):
        row = []
        for gi in range(len(gammas)):
            ar, ap = matrices(gi, ro)
            pa, pb = split(ap, tf)
            x, y = offdiag(pa), offdiag(pb)
            row.append(float(np.corrcoef(x, y)[0, 1]))
        sweep[(ro, tf, "perm")] = row
        P(f"  {ro:<8}{tf + ' perm':<10}" + "".join(f"{v:>+12.3f}" for v in row))
    for tf in ("raw",):
        row = []
        for gi in range(len(gammas)):
            ar, ap = matrices(gi, ro)
            ra, rb = split(ar, tf)
            row.append(float(np.corrcoef(offdiag(ra), offdiag(rb))[0, 1]))
        sweep[(ro, tf, "real")] = row
        P(f"  {ro:<8}{'raw real':<10}" + "".join(f"{v:>+12.3f}" for v in row))
spread = {k: max(v) - min(v) for k, v in sweep.items()}
P(f"  max spread of perm rho across gamma (any readout/transform): "
  f"{max(s for k, s in spread.items() if k[2] == 'perm'):.3f}")
P("  -> perm rho is DOSE-DEPENDENT if that spread is large; a flat row is the answer that")
P("     the label-blind structure is scale-free, matching the flagship 0.10<=gamma<=1.0 result.")
P("")

# ------------------------------------------------------------------- CG-5 ---
P("CG-5 co-occurrence: feature-set overlap (Jaccard) for scored pairs")
mem = [set(m) for m in members]
worst = max(len(mem[ix[s]] & mem[ix[t]]) / len(mem[ix[s]] | mem[ix[t]])
            for _, s, t, _, _ in BATTERY)
P(f"  max Jaccard over scored pairs = {worst:.3f} "
  f"({'as expected, argmax groups are disjoint' if worst <= 0.1 else '** FLAG **'})")
P("")

# ---------------------------------------------------------------- headline ---
P("=" * 86)
head = "pc1" if summary["pc1"]["pos"] else "zmean"
s = summary[head]
P(f"HEADLINE READOUT (pre-declared rule): RO-{'A pc1' if head == 'pc1' else 'B zmean'}"
  f"{'' if head == 'pc1' else '  (pc1 failed CG-2 POS)'}")
P(f"  CG-2 POS {s['diag_wins']}/{C}   CG-2 NEG perm rho {s['neg_raw']:+.3f} (v1 +0.893)")
P(f"  CG-1 {s['cg1']}"
  f"{'' if s['neg_raw'] < 0.30 else '  -> VOID (NEG perm >= 0.30)'}")
P(f"  CG-4 {s['cg4_raw']}/10 raw, {s['cg4_diff']}/10 differenced;  "
  f"R7 {s['r7']:+.4g}  R8 {s['r8']:+.4g}")
P("=" * 86)

(ROOT / "results/concept_cgraph_v2_score.txt").write_text("\n".join(OUTLINES) + "\n")
print(f"\n-> results/concept_cgraph_v2_score.txt")
