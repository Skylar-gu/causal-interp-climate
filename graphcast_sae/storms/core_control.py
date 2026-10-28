"""Draw a control matched on CORE activation, and say plainly if none exists.

Reads results/fs_core_scan.npy (per-feature activation within 300 km of each storm centre at
+48 h). For each storm, find features that fire in the core at the same rate as the convection
group but are NOT part of it, not the TC readout, and not ascent-family. Writes a per-storm
control in the same format MECH_INBOX_CTL expects.

A control is only written if it can actually be matched. If the dictionary contains no
core-localised alternative, this writes nothing and says so -- which is a real finding, not a
failure, and is far better than shipping a control that could not have failed.

Superseded by core_control_all.py (this script's zero-target gate matched only 3 of 7 storms);
kept because conv_corectl2 was drawn with it.
Paper: Sec. 3.3 / App. B (core-matched control)
Inputs: results/fs_core_scan.npy
Outputs: results/core_control.json
Run:   # JAX env, CPU
    python -m graphcast_sae.storms.core_control
"""
import json
import os
import sys

import numpy as np

import graphcast_sae.common.fs_common as fc
import graphcast_sae.common.skill_conv_storms as S

TOL = float(os.environ.get("CC_TOL", "0.35"))       # per-feature match tolerance
OUT = fc.ROOT / "results/core_control.json"
# ascent-family groups already measured; excluded so the control is a DIFFERENT concept
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
        banned = set(S.CONV) | {S.TC} | ASCENT
        pick, ok_all = [], True
        for t in tgt:
            if t <= 0:
                pick.append(None); continue
            cand = np.where((np.abs(c - t) <= TOL * t) &
                            ~np.isin(np.arange(len(c)), list(banned | set(p for p in pick if p)))
                            )[0]
            if len(cand) == 0:
                ok_all = False
                pick.append(None)
            else:
                # CLOSEST to the target, not the strongest in the band. Taking argmax inside a
                # symmetric +-35% window lands near +35% every time: the first version of this
                # picked a control firing MORE in the core than the treatment on 9 of 9
                # feature-level draws, 1.24-1.32x at group level, which hands the control a
                # head start in exactly the comparison it is supposed to referee.
                j = int(cand[np.argmin(np.abs(c[cand] - t))])
                pick.append(j); banned.add(j)
        got = [p for p in pick if p is not None]
        report.append((name, float(tgt.sum()), [float(x) for x in tgt],
                       got, [float(c[p]) for p in got], ok_all))
        if len(got) == len(S.CONV):
            out[name] = dict(rand=got,
                             conv_inbox=[float(x) for x in tgt],
                             new_inbox=[float(c[p]) for p in got],
                             old_inbox=[float(c[p]) for p in S.RANDOM_CTRL])
    print(f"CORE-MATCHED CONTROL  (tolerance +-{100*TOL:.0f}% per feature, "
          f"ascent-family and TC excluded)\n")
    print(f"{'storm':<14}{'conv core':>11}{'matched':>9}   control features / their core act.")
    for name, tot, tgt, got, gotc, ok in report:
        print(f"{name:<14}{tot:>11.1f}{len(got):>6}/{len(tgt)}   "
              f"{got} {[round(x,1) for x in gotc]}")
    if out:
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"\n-> {OUT}   ({len(out)} storms fully matched)")
    else:
        print("\nNO storm could be fully matched. The dictionary contains no non-ascent "
              "feature that fires in the storm core at the convection group's rate.")
        print("That is the result: the specificity question is not answerable with this SAE, "
              "and must be reported as unanswerable rather than as a passing control.")


if __name__ == "__main__":
    main()
