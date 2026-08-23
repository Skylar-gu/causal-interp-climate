"""Steering verdict — is the SAE feature's learned DIRECTION what carries the physics?

Compares three arms per feature:
  real   : dose the real SAE feature (real decoder direction)
  fair   : dose the same feature with a RANDOM decoder direction (same firing locations+magnitude)
  unif   : a spatially-uniform random kick of matched per-node magnitude
Diagnostics per arm: localization(+6h), coherence (smoothed-energy fraction, mean over leads),
downstream drift (total eastward deg of |Δ| centroid), amplitude growth.
The claim survives iff real >> fair on coherence AND downstream drift (else "any localized kick
propagates" and the specific SAE direction is not special).

Paper: not in the paper; kept for provenance only
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); results/fs_steer_randdec.npy (not shipped, see docs/REPRODUCE.md); results/fs_steer_real.npy (not shipped, see docs/REPRODUCE.md)
Outputs: figures/steer_real_vs_fair.png
Run:   # JAX env, CPU
    python -m graphcast_sae.legacy.steer_summary
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from graphcast_sae.paths import REPO_ROOT as ROOT
Rr = np.load(ROOT / "results/fs_steer_real.npy", allow_pickle=True).item()
Rf = np.load(ROOT / "results/fs_steer_randdec.npy", allow_pickle=True).item()
feats = Rr["feats"]; leads = Rr["leads_h"]; d = leads / 24.0
cat = np.load(ROOT / "candidates/fs_feature_catalog.npy", allow_pickle=True).item()

def drift(diag):    # net eastward movement of |Δ| centroid, deg (unwrapped)
    e = diag["east"]; return float(((e - e[0] + 180) % 360 - 180)[-3:].mean())

print(f"{'feat':>5} {'centroid':>12} {'arm':>6}  {'loc@6h':>7} {'coh(mean)':>10} {'drift°':>8} {'amp_end':>8}")
verdict = {}
for fi in feats:
    clat, clon = cat["clat"][fi], cat["clon"][fi]
    arms = {"real": Rr["diag"][(fi, "dose")], "fair": Rf["diag"][(fi, "dose")],
            "unif": Rr["diag"][(fi, "rand")]}
    for a, dg in arms.items():
        print(f"{fi:>5} {f'({clat:+.0f},{clon:+.0f})':>12} {a:>6}  {dg['loc'][0]:>7.2f} "
              f"{dg['coh'].mean():>10.3f} {drift(dg):>8.1f} {dg['amp'][-1]:>8.2f}")
    re, fa = arms["real"], arms["fair"]
    midlat = abs(clat) > 25
    win = (re["coh"].mean() > 1.3 * fa["coh"].mean()) and (abs(drift(re)) > abs(drift(fa)) + 5) if midlat else None
    verdict[fi] = (midlat, win)
    print()

nm = [f for f in feats if verdict[f][0]]
wins = [f for f in nm if verdict[f][1]]
print("=== VERDICT (mid-latitude features) ===")
print(f"real direction beats fair random-decoder control (more coherent AND more downstream drift): "
      f"{len(wins)}/{len(nm)} features: {wins}")
if len(wins) >= max(1, len(nm)//2):
    print("-> The SAE's LEARNED DIRECTION carries the propagating dynamics — not just 'kick that spot'.")
else:
    print("-> Localized kicks propagate generically; the SAE mainly supplies the localized handle,")
    print("   the specific direction is not special. Report honestly.")

# comparison figure: real vs fair day-5 z500 response for mid-lat features
mid = [f for f in feats if abs(cat["clat"][f]) > 25]
fig, axes = plt.subplots(len(mid), 2, figsize=(11, 2.6 * len(mid)), squeeze=False)
lon, lat = Rr["lon"], Rr["lat"]
for i, fi in enumerate(mid):
    vmax = np.percentile(np.abs(Rr["fields"][(fi, "dose")][-1]), 99.5)
    for j, (R, ttl) in enumerate([(Rr, "real feature"), (Rf, "fair control (random decoder)")]):
        ax = axes[i][j]
        ax.pcolormesh(lon, lat, R["fields"][(fi, "dose")][-1], cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        cl = cat["clon"][fi]; ax.plot(cl % 360 if cl < 0 else cl, cat["clat"][fi], "k*", ms=10, mfc="yellow")
        ax.set_title(f"feat {fi}  {ttl}  +{leads[-1]}h", fontsize=8); ax.tick_params(labelsize=6)
fig.suptitle("Same feature, same firing sites & magnitude — real vs random decoder direction (day 5 Δz500)", fontsize=10)
fig.tight_layout(); out = ROOT / "figures/steer_real_vs_fair.png"; fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nwrote {out}")
