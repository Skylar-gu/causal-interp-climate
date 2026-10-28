"""Storm-matched controls for a SINGLE intervened feature (Fig. 2b: 2401, Fig. 2c: 3316).

For each storm, pick the CC_K features whose activation within 300 km of the storm centre at
+48 h (results/fs_core_scan.npy, unpatched baseline) are closest to the target feature's, within
+/-CC_TOL (widened to 2 x CC_TOL only if fewer than CC_K qualify), among features that are a different
concept: not the target, not the convection or
low-level-spin groups, not the TC readout 3243, not the ascent family, and not significant for
vorticity or ascent under the calibrated labels (results/fs_mechanisms_v2.npy). Each control is
run as its own arm at every gain, so it receives the identical operator, disk and alpha as the
target; the control curve is the mean over the CC_K controls at each gain.

Control k is written to its own file, in the per-storm format skill_conv_run.py reads through MECH_INBOX_CTL
(`rand`, `conv_inbox` = target, `new_inbox` = control, `old_inbox` = global-rate control), with
the matched quantity being core rather than climatological in-box activation.

Paper: Fig. 2b, 2c (controls for the single-feature gain sweeps)
Inputs: results/fs_core_scan.npy, results/fs_mechanisms_v2.npy (shipped)
Outputs: results/core_control_<feature>_c<k>.json, k = 0..CC_K-1
Run:   # any env with numpy, CPU
    CC_FEAT=2401 python -m graphcast_sae.storms.core_control_single
"""
import json
import os

import numpy as np

from graphcast_sae.paths import RESULTS

FEAT = int(os.environ.get("CC_FEAT", "2401"))
TOL = float(os.environ.get("CC_TOL", "0.35"))
K = int(os.environ.get("CC_K", "3"))
CONV, SPIN, TC = [2401, 2067, 3174], [2089, 2514, 3316, 3861], 3243
RANDOM_CTRL = [3667, 2875, 2850]
ASCENT = {553, 866, 1981, 1426, 925, 141, 1538, 591, 3357, 1033, 3314, 1957, 3952, 1471, 2}
STORMS = ["ida2021", "michael2018", "haishen2020", "goni2020", "haiyan2013", "patricia2015", "wilma2005"]

scan = np.load(RESULTS / "fs_core_scan.npy", allow_pickle=True).item()["core"]
lab = np.load(RESULTS / "fs_mechanisms_v2.npy", allow_pickle=True).item()
mech = lab["mech"]
sig_phys = lab["sig"][:, mech.index("vort850")] | lab["sig"][:, mech.index("ascent")]
banned = set(CONV) | set(SPIN) | {TC} | ASCENT | set(np.where(sig_phys)[0].tolist())

out = [{} for _ in range(K)]
for s in STORMS:
    core = np.asarray(scan[s]["core"], np.float64)
    t = core[FEAT]
    assert t > 0, f"{s}: feature {FEAT} does not fire in the core; nothing to match"
    for tol in (TOL, 2 * TOL):
        ok = [f for f in np.where(np.abs(core - t) <= tol * t)[0] if f not in banned]
        if len(ok) >= K:
            break
    assert len(ok) >= K, f"{s}: only {len(ok)} features within {tol:.0%} of {t:.2f}"
    picks = sorted(ok, key=lambda f: (abs(core[f] - t), f))[:K]
    for k, c in enumerate(picks):
        out[k][s] = dict(rand=[int(c)], conv_inbox=[float(t)], new_inbox=[float(core[c])],
                         old_inbox=[float(core[RANDOM_CTRL].sum())], match="core 300 km, +48 h", tol=tol)
    print(f"{s}: f{FEAT} core {t:6.2f} -> controls " +
          ", ".join(f"f{c} {100 * core[c] / t:.0f}%" for c in picks) + f"  (tol {tol:.0%}, {len(ok)} candidates)")
for k in range(K):
    p = RESULTS / f"core_control_{FEAT}_c{k}.json"
    json.dump(out[k], open(p, "w"), indent=1)
    print("->", p)
