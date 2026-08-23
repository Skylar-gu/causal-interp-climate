"""Score the footprint-local run: is there anywhere the ablation HELPS?

Global RMSE averages over 40,962 nodes, so a feature could damage its own neighbourhood and
improve everything else, or vice versa, and the global number would hide both. Every arm was
scored inside each mask, inside each mask's COMPLEMENT, and globally in one pass. The
signature of a genuine artifact is help-INSIDE plus hurt-OUTSIDE, or help inside with nothing
outside; a load-bearing feature hurts in both.

Paper: Sec. 3, grid-locked-feature ablations (demo notebook part 4)
Inputs: results/fs_footprint_masks.npz (not shipped, see docs/REPRODUCE.md); results/fs_footprint_rmse.npy (shipped)
Outputs: printed report
Run:   # JAX env, CPU
    python -m graphcast_sae.gridlock.footprint_rmse_analyze
"""
import os
import sys

import numpy as np
from scipy import stats

from graphcast_sae.paths import REPO_ROOT
ROOT = str(REPO_ROOT)
FIELD = os.environ.get("FR_FIELD", "z500")

def main():
    d = np.load(f"{ROOT}/results/fs_footprint_rmse.npy", allow_pickle=True).item()
    acc, F, arms, S = d["acc"], d["fields"], d["arms"], d["S"]
    MASKS = np.load(f"{ROOT}/results/fs_footprint_masks.npz")
    MK = [k for k in MASKS.files if k not in ("lat", "lon")]
    REG = ["global"] + MK + [f"~{k}" for k in MK]
    fi = F.index(FIELD)
    B = acc["baseline"]
    n = B.shape[0]
    print(f"{FIELD} at +{6*S} h, {n} paired ICs. Change vs baseline, in metres.")
    print("'own' = inside that arm's own footprint; '~own' = everywhere else.\n")
    print(f"{'arm':<14}{'global':>10}{'own fp':>10}{'~own fp':>10}"
          f"{'own p':>9}{'ICs better own':>16}{'own area':>10}")
    lat = MASKS["lat"]
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(MASKS["lon"])))
    for a in arms:
        if a == "baseline":
            continue
        own = a[5:] if a.startswith("ctrl_") else a
        own = {"mesh_locked": "mesh_locked", "ctrl_mesh": "ctrl_mesh",
               "f2075": "f2075", "ctrl_f2075": "f586",
               "f656": "f656", "ctrl_f656": "f683",
               "f2235": "f2235", "ctrl_f2235": "f1850"}.get(a)
        if own is None or own not in MK:
            continue
        ri, ci = REG.index(own), REG.index(f"~{own}")
        g = acc[a][:, -1, fi, 0] - B[:, -1, fi, 0]
        i_ = acc[a][:, -1, fi, ri] - B[:, -1, fi, ri]
        o_ = acc[a][:, -1, fi, ci] - B[:, -1, fi, ci]
        p = stats.ttest_rel(acc[a][:, -1, fi, ri], B[:, -1, fi, ri]).pvalue
        area = 100 * (w * np.asarray(MASKS[own], bool)).sum() / w.sum()
        print(f"{a:<14}{np.nanmean(g):>+10.3f}{np.nanmean(i_):>+10.3f}"
              f"{np.nanmean(o_):>+10.3f}{p:>9.4f}{int((i_ < 0).sum()):>13}/{n}"
              f"{area:>9.1f}%")
    print("\nCONCENTRATION: damage inside the footprint relative to outside it.")
    print("A feature whose harm is confined to its own footprint has a ratio >> 1;")
    print("one that harms the whole forecast equally sits at ~1.\n")
    print(f"{'arm':<14}{'inside/outside':>16}")
    for a in arms:
        own = {"mesh_locked": "mesh_locked", "ctrl_mesh": "ctrl_mesh",
               "f2075": "f2075", "ctrl_f2075": "f586", "f656": "f656",
               "ctrl_f656": "f683", "f2235": "f2235", "ctrl_f2235": "f1850"}.get(a)
        if own is None or own not in MK:
            continue
        ri, ci = REG.index(own), REG.index(f"~{own}")
        i_ = np.nanmean(acc[a][:, -1, fi, ri] - B[:, -1, fi, ri])
        o_ = np.nanmean(acc[a][:, -1, fi, ci] - B[:, -1, fi, ci])
        print(f"{a:<14}{(i_/o_ if abs(o_) > 1e-9 else np.nan):>16.2f}")

if __name__ == "__main__":
    main()
