"""Redraw the CORE-matched control, fixing the zero-target gate in core_control.py.

WHY THIS EXISTS. core_control.py wrote a control for only 3 of 7 developing storms, and the
conv_corectl2 battery silently ran the other 5 against the global-firing-rate fallback
(skill_conv_run.py:195), then reported ONE median over both. The reason only 3 matched is NOT
that the dictionary lacks a core-localised alternative -- there are 99-149 free features
firing in every storm's core, and both non-zero convection targets matched to <1% at a
tolerance of 0.10. The reason is a gate bug:

    core_control.py:41-42   if t <= 0: pick.append(None); continue
    core_control.py:60      if len(got) == len(S.CONV):

Convection feature 2067 has EXACTLY ZERO activation in the 300 km core for ida2021,
michael2018, patricia2015, wilma2005 (and nondev2013). A zero target is unmatchable by
construction, so `got` has 2 entries, so the 3-of-3 gate rejects the storm -- even though the
match that mattered was perfect. The tolerance is irrelevant: the matched count is 3 at every
CC_TOL from 0.05 to 2.00.

THE FIX. Require a match for every convection feature that ACTUALLY FIRES in the core, and
emit a control of that size. A convection feature that is zero in the core contributes
nothing to a core-matched comparison, so it needs no counterpart; demanding one is demanding
a match to nothing.

nondev2013 still gets no control and should not: all three convection features are zero in
its core AND zero in its box, so there is nothing to ablate and nothing to match. Coverage
tops out at 7 of 8, not 8 of 8.

Paper: Table tab:mechanism-interventions (core-matched control)
Inputs: results/fs_core_scan.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/core_control_all.json
Run:   # JAX env, CPU
    CC_OUT=results/core_control_all.json python -m graphcast_sae.storms.core_control_all
"""
import json
import os
import sys

import numpy as np

import graphcast_sae.common.fs_common as fc
import graphcast_sae.common.skill_conv_storms as S

TOL = float(os.environ.get("CC_TOL", "0.35"))
OUT = fc.ROOT / os.environ.get("CC_OUT", "results/core_control_all.json")
MIN_MATCH = int(os.environ.get("CC_MIN", "2"))     # refuse a 1-feature control
ASCENT = set(S.CONV) | {553, 866, 1981, 2401, 1426, 925, 2067, 3174, 141, 1538, 591,
                        3357, 1033, 3314, 1957, 3952, 1471, 2}

def main():
    d = np.load(fc.ROOT / "results/fs_core_scan.npy", allow_pickle=True).item()
    core = d["core"]
    out, report = {}, []
    for name, rec in core.items():
        if S.STORMS.get(name, {}).get("nondev"):
            continue
        c = np.asarray(rec["core"], float)
        tgt = c[np.array(S.CONV)]
        need = int((tgt > 0).sum())
        banned = set(S.CONV) | {S.TC} | ASCENT
        pick = []
        for t in tgt:
            if t <= 0:
                pick.append(None); continue
            cand = np.where((np.abs(c - t) <= TOL * t) &
                            ~np.isin(np.arange(len(c)), list(banned | set(p for p in pick if p)))
                            )[0]
            if len(cand) == 0:
                pick.append(None)
            else:
                j = int(cand[np.argmin(np.abs(c[cand] - t))])
                pick.append(j); banned.add(j)
        got = [p for p in pick if p is not None]
        zero = [f for f, t in zip(S.CONV, tgt) if t <= 0]
        report.append((name, tgt, got, [float(c[p]) for p in got], need, zero))
        if len(got) == need and need >= MIN_MATCH:
            out[name] = dict(rand=got,
                             conv_inbox=[float(x) for x in tgt if x > 0],
                             new_inbox=[float(c[p]) for p in got],
                             old_inbox=[float(c[p]) for p in S.RANDOM_CTRL],
                             conv_core_all=[float(x) for x in tgt],
                             zero_core_conv=zero, tol=TOL)
    print(f"CORE-MATCHED CONTROL, zero-target gate fixed (tolerance +-{100*TOL:.0f}% per "
          f"NON-ZERO feature; ascent-family and TC excluded)\n")
    print(f"{'storm':<14}{'conv core':>11}{'need':>6}{'got':>5}   control / core act. / "
          f"conv features that are ZERO in the core")
    for name, tgt, got, gotc, need, zero in report:
        print(f"{name:<14}{tgt.sum():>11.1f}{need:>6}{len(got):>5}   {got} "
              f"{[round(x, 2) for x in gotc]}  zero:{zero}")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n-> {OUT}   ({len(out)} storms matched; nondev2013 excluded by design -- its "
          "convection group is zero in the core AND in the box)")

if __name__ == "__main__":
    main()
