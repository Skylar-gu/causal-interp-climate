"""Classify every SAE feature into the full taxonomy of what a weather model can encode.

Uses the atlas + extra detectors to sort each feature into:
  physics (single mechanism) | joint-coupling (multi-mechanism) | numerical machinery
  | climatology/clock | teleconnection/mode | predictability/regime | residual (novel or self-correction)
Prints the census + exemplars per category so each hypothesis can be validated physically.

Paper: Sec. 3, mechanism labels that define the intervention groups
Inputs: results/fs_atlas_extra.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/fs_atlas_class.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.atlas.atlas_classify
"""
from pathlib import Path
import numpy as np
from graphcast_sae.paths import REPO_ROOT as ROOT
a = np.load(ROOT / "results/fs_atlas_extra.npy", allow_pickle=True).item()
z, refs, phys, geo = a["z"], a["node_refs"], a["phys"], a["geo"]
ze, ne = a["z_extra"], a["node_extra"]
season, diurnal, static, enso = a["season"], a["diurnal"], a["staticness"], a["enso_amp"]
fr, coh = a["firerate"], a["coh"]
clat, clon = a["clat"], np.where(a["clon"] > 180, a["clon"] - 360, a["clon"])
F = z.shape[0]; alive = a["zcnt"] > 300

def zc(name):
    if name in refs: return z[:, refs.index(name)]
    return ze[:, ne.index(name)]

# --- category signals ---
phys_abs = np.abs(np.stack([zc(p) for p in phys], 1))          # (F, 8)
phys_max = phys_abs.max(1); phys_n = (phys_abs > 1.0).sum(1)
geo_val = np.abs(np.stack([zc(g) for g in geo], 1)).max(1)     # lat/land_sea/orog VALUE
clock = np.maximum(season, diurnal)
geom = np.maximum.reduce([np.abs(zc("coast_grad")), np.abs(zc("orog_grad")), np.abs(zc("node_density"))])
tele = np.maximum.reduce([np.abs(zc("blocking")), np.abs(zc("atm_river")), enso])
regime = np.abs(zc("baroclinicity"))

# --- primary classification (transparent decision rule) ---
cat = np.full(F, "residual", dtype=object)
cat[np.where(regime > 1.0)] = "regime/predictability"
cat[np.where(tele > 1.2)] = "teleconnection/mode"
cat[np.where(clock > 0.30)] = "climatology/clock"
cat[np.where(geo_val > 1.2)] = "climatology/clock"
cat[np.where(geom > 1.3)] = "numerical/geometry"
cat[np.where(static > 0.55)] = "numerical/geometry"          # fires same nodes every window
cat[np.where((phys_max > 1.0) & (phys_n >= 2))] = "joint-coupling"
cat[np.where((phys_max > 1.0) & (phys_n == 1))] = "physics(single)"
CATS = ["physics(single)", "joint-coupling", "numerical/geometry", "climatology/clock",
        "teleconnection/mode", "regime/predictability", "residual"]

print("=== ATLAS CENSUS (alive features) ===")
for c in CATS:
    m = alive & (cat == c); print(f"  {c:>22}: {m.sum():>4}  ({100*m.sum()/alive.sum():.0f}%)")

print("\n=== YOUR SIX CATEGORIES — exemplars (does each detector catch the right thing?) ===")
def show(title, mask, key, extra=lambda fi: ""):
    idx = np.where(alive & mask)[0]
    idx = idx[np.argsort(-key[idx])][:5]
    print(f"\n {title}:")
    for fi in idx:
        print(f"   feat {fi:>4} fire={fr[fi]:.3f} coh={coh[fi]:.0f} home=({clat[fi]:+.0f},{clon[fi]:+.0f})  {extra(fi)}")

show("1. NUMERICAL machinery (static/coastline/orography/density)", cat == "numerical/geometry", geom + static,
    lambda fi: f"static={static[fi]:.2f} coast_z={zc('coast_grad')[fi]:+.1f} orog_z={zc('orog_grad')[fi]:+.1f} dens_z={zc('node_density')[fi]:+.1f}")
show("2. CLIMATOLOGY/clock (lat/land-sea/orog value, season, diurnal)", cat == "climatology/clock", np.maximum(clock, geo_val),
    lambda fi: f"season={season[fi]:.2f} diurnal={diurnal[fi]:.2f} geo_z={geo_val[fi]:+.1f}")
show("3. TELECONNECTION/mode (blocking/atm-river/ENSO)", cat == "teleconnection/mode", tele,
    lambda fi: f"block_z={zc('blocking')[fi]:+.1f} AR_z={zc('atm_river')[fi]:+.1f} enso={enso[fi]:.2f}")
show("4. JOINT couplings (>=2 physical mechanisms at once)", cat == "joint-coupling", phys_n.astype(float),
    lambda fi: "+".join(f"{phys[k]}({phys_abs[fi,k]:.1f})" for k in np.argsort(-phys_abs[fi])[:3]))
show("6. REGIME/predictability (baroclinicity, storm-track)", cat == "regime/predictability", regime,
    lambda fi: f"baroc_z={zc('baroclinicity')[fi]:+.1f} shear_z={zc('shear')[fi]:+.1f}")

# 5 self-correction + novel = residual; split by spatial structure
res = np.where(alive & (cat == "residual") & (fr > np.median(fr[alive])))[0]
res = res[np.argsort(-fr[res])]
struct = res[coh[res] < 5000]; diffuse = res[coh[res] >= 7000]
print(f"\n 5+novel. RESIDUAL — matches NOTHING known ({len(res)} high-firing):")
print(f"   structured (candidate NOVEL mechanism, compact): {len(struct)}")
for fi in struct[:6]:
    print(f"     feat {fi:>4} fire={fr[fi]:.3f} coh={coh[fi]:.0f} home=({clat[fi]:+.0f},{clon[fi]:+.0f}) static={static[fi]:.2f}")
print(f"   diffuse (candidate self-correction / superposition, no spatial structure): {len(diffuse)}")
for fi in diffuse[:4]:
    print(f"     feat {fi:>4} fire={fr[fi]:.3f} coh={coh[fi]:.0f} static={static[fi]:.2f}")

np.save(ROOT / "results/fs_atlas_class.npy",
        dict(cat=cat, novel_struct=struct.astype(int), novel_diffuse=diffuse.astype(int)), allow_pickle=True)
print("\n-> results/fs_atlas_class.npy")
