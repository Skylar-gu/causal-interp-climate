"""Score the concept causal graph against docs/prereg/prereg_concept_graph.md (CG-1..CG-5).

Reads results/fs_concept_cgraph.npy. No new forwards. system python3.
Every bar below is quoted from the prereg, which was frozen before the run.

Paper: not in the paper; kept for provenance only
Inputs: results/fs_concept_cgraph.npy (not shipped, see docs/REPRODUCE.md)
Outputs: printed report
Run:   # JAX env, CPU
    python -m graphcast_sae.legacy.concept_cgraph_score
"""
from pathlib import Path
import numpy as np
from math import comb

from graphcast_sae.paths import REPO_ROOT as ROOT
D = np.load(ROOT / "results/fs_concept_cgraph.npy", allow_pickle=True).item()
A_ic, P_ic, names = D["A_ic"], D["P_ic"], list(D["names"])
ICS, SET_A, SET_B = list(D["ics"]), list(D["set_a"]), list(D["set_b"])
ia = [ICS.index(x) for x in SET_A]; ib = [ICS.index(x) for x in SET_B]
C = len(names); ix = {n: i for i, n in enumerate(names)}

# CG-4 battery, frozen. R1 (ascent->div250) STRUCK: div250 has no features.
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

def offdiag(M):
    return M[~np.eye(C, dtype=bool)]

def repro(stack, label):
    a, b = stack[ia].mean(0), stack[ib].mean(0)
    a0, b0 = a.copy(), b.copy(); np.fill_diagonal(a0, 0); np.fill_diagonal(b0, 0)
    x, y = offdiag(a0), offdiag(b0)
    rho = float(np.corrcoef(x, y)[0, 1])
    sign = float(np.mean(np.sign(x) == np.sign(y)))
    k = 10
    ta = set(map(tuple, np.argwhere(np.abs(a0) >= np.sort(np.abs(x))[-k])))
    tb = set(map(tuple, np.argwhere(np.abs(b0) >= np.sort(np.abs(y))[-k])))
    shared = len(ta & tb)
    print(f"  {label:<14} rho {rho:+.3f}   sign agreement {sign:.3f}   top-10 shared {shared}/10")
    return rho, sign, shared, a0, b0

print("CONCEPT CAUSAL GRAPH — scored against the frozen prereg")
print(f"concepts: {names}\n")

print("CG-2 CALIBRATION (both sides, read before CG-1)")
A_all = A_ic.mean(0)
diag_wins = sum(int(np.argmax(np.abs(A_all[r])) == r) for r in range(C))
print(f"  POS self-effect dominates its own row: {diag_wins}/{C} "
      f"-> {'PASS' if diag_wins >= C - 1 else '** FAIL: readout broken, no verdict **'}")
rho_p, sign_p, shared_p, _, _ = repro(P_ic, "NEG perm")
neg_ok = rho_p < 0.30
print(f"  NEG concept_perm rho {rho_p:+.3f} < 0.30 -> {'PASS' if neg_ok else '** FAIL: CG-1 VOID **'}")

print("\nCG-1 PRIMARY — reproducibility across disjoint IC sets")
print(f"  {'published feature-level':<14} rho +0.181   sign agreement 0.513   top-20 shared 6/20")
rho, sign, shared, a0, b0 = repro(A_ic, "concept-level")
if rho >= 0.50 and sign >= 0.70 and shared >= 6:
    verdict = "REPRODUCIBLE"
elif rho < 0.30:
    verdict = "NOT REPRODUCIBLE"
else:
    verdict = "PARTIAL"
print(f"  -> CG-1 VERDICT: {verdict}")
if not neg_ok or diag_wins < C - 1:
    print("     (calibration failed -> this verdict is VOID)")

print("\nCG-3 magnitude confound")
mass = np.abs(A_all).sum(1)
sp = np.corrcoef(np.argsort(np.argsort(np.repeat(mass, C))),
                 np.argsort(np.argsort(np.abs(A_all).ravel())))[0, 1]
print(f"  Spearman(|A[i,j]|, source total effect) = {sp:+.3f}")

print("\nCG-4 ANSWER KEY — 10 testable relations (R1 struck: div250 has no features)")
A = A_ic.mean(0)
hit = 0
print(f"  {'rel':<5}{'relation':<32}{'pred':>5}{'A[i,j]':>12}{'ok':>4}   asymmetry")
for rid, s, t, pred, why in BATTERY:
    v = A[ix[s], ix[t]]; rv = A[ix[t], ix[s]]
    ok = np.sign(v) == pred
    hit += int(ok)
    asym = abs(v - rv) / max(abs(v) + abs(rv), 1e-12)
    print(f"  {rid:<5}{s+' -> '+t:<32}{'+' if pred>0 else '-':>5}{v:>12.4g}{'Y' if ok else 'n':>4}   {asym:.2f}   {why}")
n = len(BATTERY)
p = sum(comb(n, k) for k in range(hit, n + 1)) / 2 ** n
print(f"\n  signed-edge accuracy {hit}/{n}   one-sided binomial p = {p:.4f}")
score = "CONFIRMED" if hit >= 8 else ("NULL" if hit <= 6 else "INCONCLUSIVE")
print(f"  -> CG-4 VERDICT: {score}")

r7 = np.sign(A[ix['shear'], ix['ascent']]) < 0
r8 = np.sign(A[ix['blocking'], ix['jet250']]) < 0
print(f"\n  CG-4 secondary (pre-declared downgrade): the two NEGATIVE relations")
print(f"    R7 shear->ascent negative: {r7}    R8 blocking->jet250 negative: {r8}")
if score == "CONFIRMED" and not r7 and not r8:
    print("    -> DOWNGRADED to 'co-occurrence recovered, causation not demonstrated' (prereg CG-4)")

print("\nCG-5 co-occurrence: feature-set overlap (Jaccard) for scored pairs")
mem = [set(m) for m in D["members"]]
worst = max(len(mem[ix[s]] & mem[ix[t]]) / len(mem[ix[s]] | mem[ix[t]]) for _, s, t, _, _ in BATTERY)
print(f"  max Jaccard over scored pairs = {worst:.3f} "
      f"({'as expected, argmax groups are disjoint' if worst <= 0.1 else '** FLAG **'})")
