"""Coverage- and connectivity-matched controls for the four grid-locked features that have no
single-feature ablation yet (f407, f2585, f2989, f3535). Same procedure, tolerances and seed as
matched_control_draw.py, but reads the per-feature coverage/connectivity table that script
saved (results/fs_connectivity.npy) instead of recomputing it from the 6.7 GB IID dump, and
bans the original 17 candidates, their 17 controls, and the rotation-defined grid-locked set
from the control pool.

    python -m graphcast_sae.gridlock.matched_control_draw_six   -> results/matched_controls_six.json

Paper: App. C, Fig. 4 (matched controls for the six grid-locked features)
Inputs: results/fs_connectivity.npy, results/matched_controls.json (shipped);
        results/hybrid_footprint_fires.npz (not shipped, see docs/REPRODUCE.md)
Outputs: results/matched_controls_six.json
"""
import json
import numpy as np
from pathlib import Path

from graphcast_sae.paths import REPO_ROOT as ROOT
NEW = [407, 2585, 2989, 3535]
SEED = 31
C = np.load(ROOT / "results/fs_connectivity.npy", allow_pickle=True).item()
cov, ncomp, fl, dg = C["cov"], C["ncomp"], C["frac_largest"], C["degree"]
z = np.load(ROOT / "results/hybrid_footprint_fires.npz")
fires = z["fires"]
print("fires.npz nw/thresh:", int(z["nw"]), float(z["thresh"]))
old = json.load(open(ROOT / "results/matched_controls.json"))
banned = {f for v in old.values() for f in v} | set(NEW) | {2954, 2075, 1331, 656}
alive = (cov > 0) & (fires.sum(0) >= 10)
rng = np.random.default_rng(SEED)
out, report = {}, []
for f in NEW:
    for tol in (0.30, 0.50, 0.80, 2.0):
        ok = np.where(alive & (np.abs(cov - cov[f]) <= tol * max(cov[f], 1e-9))
                      & (np.abs(fl - fl[f]) <= 0.20)
                      & ~np.isin(np.arange(4096), list(banned)))[0]
        if len(ok) >= 3:
            break
    if len(ok) == 0:
        report.append((f, None, tol)); continue
    score = (np.abs(np.log(np.maximum(ncomp[ok], 1)) - np.log(max(ncomp[f], 1)))
             + np.abs(dg[ok] - dg[f]) / max(dg[f], 1e-9))
    c = int(ok[np.argsort(score)[:3][rng.integers(0, min(3, len(ok)))]])
    banned.add(c)
    out[f"f{f}"] = [f]; out[f"ctrl_f{f}"] = [c]
    report.append((f, c, tol))
print(f"\n{'feat':>6}{'cov':>8}{'ncomp':>7}{'fracL':>7}{'deg':>6}   ->{'ctrl':>6}{'cov':>8}{'ncomp':>7}{'fracL':>7}{'deg':>6}  tol")
for f, c, tol in report:
    if c is None:
        print(f"{f:>6}   NO MATCH"); continue
    print(f"{f:>6}{100*cov[f]:>7.2f}%{int(ncomp[f]):>7}{fl[f]:>7.2f}{dg[f]:>6.1f}"
          f"   ->{c:>6}{100*cov[c]:>7.2f}%{int(ncomp[c]):>7}{fl[c]:>7.2f}{dg[c]:>6.1f}  {tol}")
out["floor"] = []
json.dump(out, open(ROOT / "results/matched_controls_six.json", "w"), indent=1)
print("ARMS=" + ",".join(out))
