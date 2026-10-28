"""Cross-angle check of the grid-locked definition.

For every rotation angle with a saved results/fs_rotation_all*.npy (180, 90, 45, 135, 270 deg),
report stay / follow / positional for the features the paper calls grid-locked (f2954, f2075,
f656, f3680, f560) and for the content controls (f3404 land-shaped, f2474 land mask, f2109
convection, f2789 Maritime-Continent convection); then the set of features with positional
score >= CUT at EVERY angle (among features powered at every angle), and the Spearman rank
agreement between angles over the union of powered features.

    python -m graphcast_sae.gridlock.rotation_angles_summary

Paper: App. C (grid-lock criterion: s_f >= 0.5 at all five angles, six features)
Inputs: results/fs_rotation_all*.npy (shipped)
Outputs: stdout
"""
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

from graphcast_sae.paths import RESULTS as R
ANGLES = [(180, "fs_rotation_all.npy"), (90, "fs_rotation_all_90.npy"),
          (45, "fs_rotation_all_45.npy"), (135, "fs_rotation_all_135.npy"),
          (270, "fs_rotation_all_270.npy")]
GRID = [2954, 2075, 656, 3680, 560]
CTRL = [3404, 2474, 2109, 2789]
CUT = 0.5

D = {a: np.load(R / f, allow_pickle=True).item() for a, f in ANGLES if (R / f).exists()}
angles = list(D)
print("angles available:", angles)

print("\nstay / follow / positional  (positional = stay - follow; '.' = no power at that angle)")
hdr = f"{'feature':<8}" + "".join(f"{a:>22}°" for a in angles)
print(hdr)
for f in GRID + [None] + CTRL:
    if f is None:
        print("-" * len(hdr)); continue
    row = f"f{f:<7}"
    for a in angles:
        d = D[a]
        cell = f"{d['stays'][f]:+.2f}/{d['follows'][f]:+.2f}/{d['positional'][f]:+.2f}"
        row += f"{cell:>23}" + ("" if d["powered"][f] else "·")
    print(row)

# features grid-locked at every angle
allpow = np.logical_and.reduce([D[a]["powered"].astype(bool) for a in angles])
allpos = np.logical_and.reduce([D[a]["positional"] >= CUT for a in angles])
anypos = {a: set(np.where(D[a]["powered"].astype(bool) & (D[a]["positional"] >= CUT))[0]) for a in angles}
print(f"\nfeatures with positional >= {CUT} per angle:",
      {a: len(v) for a, v in anypos.items()})
inter = set(np.where(allpow & allpos)[0])
print(f"grid-locked at EVERY angle (powered everywhere, positional >= {CUT}): {len(inter)} ->",
      sorted(inter))
minpos = np.min(np.stack([D[a]["positional"] for a in angles]), 0)
print("\nminimum positional score across angles, paper's grid-locked set:")
for f in GRID:
    print(f"  f{f}: min {minpos[f]:+.2f}   " +
          ("GRID-LOCKED at all angles" if allpow[f] and minpos[f] >= CUT else
           f"below {CUT} at " + ", ".join(f"{a}°" for a in angles if D[a]['positional'][f] < CUT)))
print("\nrank agreement (Spearman over features powered at both angles):")
for i, a in enumerate(angles):
    for b in angles[i + 1:]:
        m = D[a]["powered"].astype(bool) & D[b]["powered"].astype(bool)
        r = spearmanr(D[a]["positional"][m], D[b]["positional"][m]).correlation
        top = m & ((D[a]["positional"] >= 0.3) | (D[b]["positional"] >= 0.3))
        rt = spearmanr(D[a]["positional"][top], D[b]["positional"][top]).correlation if top.sum() > 3 else np.nan
        print(f"  {a:>3}° vs {b:>3}°: all {r:+.2f}   among features >= 0.3 at either angle (n={int(top.sum())}) {rt:+.2f}")
