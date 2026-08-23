"""Pick mediator candidates j for each source i, from MEASURED response — not from a graph.

Reads ONLY what is already on disk:
    results/skill/hyb_abl_f<i>/run_<storm>.npy   arms 'baseline' and 'conv-normal', box_feats
                                                 for all 4096 features (those runs used
                                                 MECH_TRACK=all), 17 source features x 2 storms.

MOVEMENT.  For source i, storm s, the movement of feature j is

    mov_i(j) = max_h | S_j^{do(i)}(h) - S_j^{base}(h) |          in-box summed code units
    amp(j)   = max_h  S_j^{base}(h)                              j's own baseline amplitude

SPECIFICITY, and why raw movement is not enough.  The P0 probe found that ablating a strong
feature and a weak one move the same NUMBER of other features (302 vs 298): a single-feature
ablation perturbs the residual stream and the encoder re-reads it, so hundreds of features
twitch through dictionary overlap alone.  Ranking on raw `mov` therefore mostly ranks
features by their own amplitude.  Because 17 independent single-feature ablations of the SAME
storm are already on disk, we can subtract that generic response:

    spec_i(j) = ( mov_i(j) - median_s mov_s(j) ) / ( 1.4826 * MAD_s mov_s(j) + eps )

i.e. how far j's response to do(i) sticks out above its response to the other 16 ablations.
A generic responder has spec ~ 0 no matter how large mov is.  Candidates must clear BOTH a
raw-movement floor (it has to move at all) and a specificity bar.

CONTROL (the control-must-be-able-to-fail rule).  For each chosen mediator j, the control j' is drawn from features
whose baseline in-box amplitude is within AMP_TOL of amp(j) — an amplitude-MATCHED draw —
taking the one with the smallest mov_i.  If no feature within the band has small movement,
that is reported as a control failure rather than papered over: it would mean amplitude and
response are confounded and the control cannot isolate "freezing anything".

Excluded from candidacy: i itself and every other source (a mediator that is also a source
confounds the do(j) arm with a second do(i) arm).  TC = 3243 is NOT excluded — Y is min MSLP,
a physical readout, not 3243 — but it is flagged wherever it appears, because freezing the
model's own cyclone feature is a much stronger intervention than freezing a covariate.

Paper: Appendix app:topk
Inputs: none beyond the arguments above
Outputs: results/mediation_candidates.json
Run:   # JAX env, CPU
    OMP_NUM_THREADS=4 python -m graphcast_sae.appendix.mediation_select
"""
import json
import os

import numpy as np

from graphcast_sae.paths import REPO_ROOT
ROOT = str(REPO_ROOT)
SKILL = os.path.join(ROOT, "results", "skill")
# every single-feature ablation already on disk; the 5 SOURCES are a subset
ALL_ABL = [165, 292, 339, 617, 649, 1311, 1630, 1732, 1948, 2521,
           2681, 2876, 2990, 3075, 3465, 3605, 3899]
SOURCES = [2681, 1732, 165, 292, 3465]
STORMS = ["ida2021", "haishen2020"]
TC = 3243
EPS = 1e-9
NTOP = 3
AMP_FLOOR = 5.0        # a feature with no in-box amplitude cannot carry a pathway
AMP_TOL = 0.25         # control must sit within +/-25% of the mediator's amplitude
SPEC_BAR = 2.0         # mediator must stick out >=2 robust sigma above the generic response
OUT = os.path.join(ROOT, "results", "mediation_candidates.json")

def load(feat, storm):
    p = os.path.join(SKILL, "hyb_abl_f%d" % feat, "run_%s.npy" % storm)
    if not os.path.exists(p):
        return None, "missing"
    d = np.load(p, allow_pickle=True).item()
    res = d["res"]
    for a in ("baseline", "conv-normal"):
        if a not in res:
            return None, "no arm %s" % a
    bf, af = res["baseline"]["box_feats"], res["conv-normal"]["box_feats"]
    if len(bf) != 4096 or len(af) != 4096:
        return None, "tracked %d features, need 4096" % len(bf)
    B = np.stack([np.asarray(bf[f], np.float64) for f in range(4096)], 1)
    A = np.stack([np.asarray(af[f], np.float64) for f in range(4096)], 1)
    if not (np.isfinite(B).all() and np.isfinite(A).all()):
        return None, "non-finite"
    if B.shape[0] < 8 or np.allclose(B, 0):
        return None, "degenerate (H=%d, allzero=%s)" % (B.shape[0], np.allclose(B, 0))
    return (B, A), "ok"

def main():
    report, bank = {}, {}
    for storm in STORMS:
        movs, base_amp, ok = {}, None, []
        for f in ALL_ABL:
            got, why = load(f, storm)
            if got is None:
                print("  gate: f%d %s -> %s" % (f, storm, why))
                continue
            B, A = got
            movs[f] = np.abs(A - B).max(0)
            base_amp = B.max(0) if base_amp is None else np.maximum(base_amp, B.max(0))
            ok.append(f)
        M = np.stack([movs[f] for f in ok])                      # (n_abl, 4096)
        med = np.median(M, 0)
        mad = 1.4826 * np.median(np.abs(M - med), 0)
        bank[storm] = dict(M=M, ok=ok, med=med, mad=mad, amp=base_amp, movs=movs)
        print("[%s] %d ablations gated in: %s" % (storm, len(ok), ok))

    for storm in STORMS:
        b = bank[storm]
        amp, med, mad = b["amp"], b["med"], b["mad"]
        active = np.array([f for f in range(4096) if amp[f] >= AMP_FLOOR])
        print("\n=== %s : %d features with in-box amplitude >= %.1f ===" % (storm, len(active), AMP_FLOOR))
        for src in SOURCES:
            if src not in b["movs"]:
                continue
            mov = b["movs"][src]
            spec = (mov - med) / (mad + EPS)
            cand = np.array([f for f in active if f not in set(SOURCES)])
            sel = cand[(spec[cand] >= SPEC_BAR)]
            sel = sel[np.argsort(-mov[sel])][:NTOP]
            entry = dict(self_mov=float(mov[src]), self_amp=float(amp[src]),
                         self_frac=float(mov[src] / max(amp[src], EPS)),
                         n_active=int(len(cand)), n_specific=int((spec[cand] >= SPEC_BAR).sum()),
                         movers=[], )
            print(" f%-5d self-move %.1f / amp %.1f (%.0f%%);  %d/%d candidates clear spec>=%.1f"
                  % (src, mov[src], amp[src], 100 * mov[src] / max(amp[src], EPS),
                     entry["n_specific"], len(cand), SPEC_BAR))
            for j in sel:
                lo, hi = amp[j] * (1 - AMP_TOL), amp[j] * (1 + AMP_TOL)
                band = cand[(amp[cand] >= lo) & (amp[cand] <= hi) & (cand != j)]
                if len(band) == 0:
                    ctl = dict(feat=-1, why="no feature within +/-%.0f%% amplitude" % (100 * AMP_TOL))
                else:
                    k = int(band[np.argmin(spec[band])])
                    ctl = dict(feat=k, amp=float(amp[k]), mov=float(mov[k]), spec=float(spec[k]),
                               amp_ratio=float(amp[k] / max(amp[j], EPS)),
                               mov_ratio=float(mov[k] / max(mov[j], EPS)),
                               n_in_band=int(len(band)))
                entry["movers"].append(dict(feat=int(j), mov=float(mov[j]), amp=float(amp[j]),
                                            rel=float(mov[j] / max(amp[j], EPS)),
                                            spec=float(spec[j]), generic=float(med[j]),
                                            is_tc=bool(j == TC), control=ctl))
                print("    j=%-5d mov %7.1f amp %7.1f rel %4.2f spec %5.1f (generic %6.1f)%s"
                      % (j, mov[j], amp[j], mov[j] / max(amp[j], EPS), spec[j], med[j],
                         "   <-- TC READOUT" if j == TC else ""))
                if ctl["feat"] < 0:
                    print("        CONTROL FAIL: %s" % ctl["why"])
                else:
                    print("        ctl j'=%-5d amp %7.1f (%.2fx, %d in band) mov %7.1f (%.2fx) spec %5.1f"
                          % (ctl["feat"], ctl["amp"], ctl["amp_ratio"], ctl["n_in_band"],
                             ctl["mov"], ctl["mov_ratio"], ctl["spec"]))
            report.setdefault(str(src), {})[storm] = entry

    print("\n=== cross-storm consensus (mediators specific in BOTH storms) ===")
    cons = {}
    for src in SOURCES:
        sets = [set(r["feat"] for r in report[str(src)][s]["movers"])
                for s in STORMS if s in report.get(str(src), {})]
        cons[str(src)] = sorted(set.intersection(*sets)) if len(sets) == 2 else []
        print(" f%-5d %s" % (src, cons[str(src)] or "(none in top-%d of both)" % NTOP))
    json.dump(dict(report=report, consensus=cons, sources=SOURCES, storms=STORMS,
                   amp_floor=AMP_FLOOR, amp_tol=AMP_TOL, spec_bar=SPEC_BAR,
                   ablations_used=ALL_ABL), open(OUT, "w"), indent=1)
    print("\n-> %s" % OUT)

if __name__ == "__main__":
    main()
