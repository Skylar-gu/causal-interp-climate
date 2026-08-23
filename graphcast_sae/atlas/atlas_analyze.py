"""Census the representation atlas (A) and surface the residual/novel features (B).

From fs_atlas.npy (z[feature, reference] + temporal amplitudes):
  A: what fraction of the dictionary is physics / geography / calendar, and the top feature per
     physical mechanism.
  B: the DISCOVERY — high-firing features whose firing matches NOTHING we know (low z on every
     physical field, geography, and the calendar). Characterize them.

Paper: Sec. 3, mechanism labels that define the intervention groups
Inputs: results/fs_atlas.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/fs_atlas_novel.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.atlas.atlas_analyze
"""
from pathlib import Path
import numpy as np
from graphcast_sae.paths import REPO_ROOT as ROOT
a = np.load(ROOT / "results/fs_atlas.npy", allow_pickle=True).item()
z, phys, geo, refs = a["z"], a["phys"], a["geo"], a["node_refs"]
season, diurnal, fr, coh = a["season"], a["diurnal"], a["firerate"], a["coh"]
clat, clon = a["clat"], np.where(a["clon"] > 180, a["clon"] - 360, a["clon"])
F = z.shape[0]
pj = [refs.index(p) for p in phys]; gj = [refs.index(g) for g in geo]
phys_max = np.abs(z[:, pj]).max(1); phys_arg = np.array(phys)[np.abs(z[:, pj]).argmax(1)]
geo_max = np.abs(z[:, gj]).max(1); geo_arg = np.array(geo)[np.abs(z[:, gj]).argmax(1)]
temp = np.maximum(season, diurnal)
alive = a["zcnt"] > 300

# ---- A: census ----
cat = np.full(F, "novel", dtype=object)
cat[temp > 0.30] = "calendar"
cat[geo_max > 1.1] = "geography"
cat[phys_max > 1.0] = "physics"
print("=== A. ATLAS CENSUS (alive features) ===")
for c in ["physics", "geography", "calendar", "novel"]:
    m = alive & (cat == c)
    print(f"  {c:>10}: {m.sum():>4}  ({100*m.sum()/alive.sum():.0f}%)")
print("\ntop feature per physical mechanism (z = how anomalous the field is where it fires):")
for p, j in zip(phys, pj):
    good = np.where(alive)[0]; top = good[np.argsort(-z[good, j])[:3]]
    s = "  ".join(f"{fi}(z{z[fi,j]:+.1f},{clat[fi]:+.0f}/{clon[fi]:+.0f})" for fi in top)
    print(f"  {p:>9}: {s}")

# ---- B: residual discovery ----
print("\n=== B. RESIDUAL DISCOVERY — high-firing features that match NOTHING known ===")
novel = alive & (phys_max < 0.6) & (geo_max < 0.9) & (temp < 0.18)
cand = np.where(novel)[0]
cand = cand[np.argsort(-fr[cand])][:20]
print(f"{novel.sum()} novel features; top 20 by firing rate:")
print(f"  {'feat':>5}{'fire':>8}{'coh_km':>8}{'phys_max':>9}{'geo_max':>8}{'season':>7}{'diur':>6}   centroid   nearest-known")
for fi in cand:
    print(f"  {fi:>5}{fr[fi]:>8.3f}{coh[fi]:>8.0f}{phys_max[fi]:>9.2f}{geo_max[fi]:>8.2f}"
          f"{season[fi]:>7.2f}{diurnal[fi]:>6.2f}   ({clat[fi]:+.0f},{clon[fi]:+.0f})"
          f"   {phys_arg[fi]}~{z[fi,pj[list(phys).index(phys_arg[fi])]]:+.1f}")
# how spatially organized are the novel ones? (compact vs global vs banded)
comp = coh[cand]
print(f"\nnovel-feature footprint: {(comp<3500).sum()} compact(<3500km), {((comp>=3500)&(comp<7000)).sum()} regional, {(comp>=7000).sum()} broad")
np.save(ROOT / "results/fs_atlas_novel.npy", dict(novel_feats=cand.astype(int), cat=cat), allow_pickle=True)
print("-> results/fs_atlas_novel.npy")
