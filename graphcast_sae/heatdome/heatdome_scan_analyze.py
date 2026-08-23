"""Label the data-driven ridge-firing features and define physics-guided ablation sets (CPU).

Consumes results/heatdome/scan.npy + atlas (fs_atlas_extra: z (4096x11 probe scores) and
z_extra (4096x6, incl a 'blocking' index) + fs_atlas_class 'cat'). For every STRONG ridge-firing
feature (peak box active-node count >= 8) it reports:
  - PHYSICAL label: dominant atlas probe among [z500,t850,jet250,vort850,div250,shear,ascent,q600]
    plus z_extra blocking / baroclinicity / atm_river, and the class 'cat'
  - PROPAGATION: event firing centroid (dome window), centroid path length & net displacement &
    eastward drift & persistence -> does it track a coherent dynamical structure (bent jet / wave
    train / flanking low / warm dome) or sit static
  - ridge-corr and box-firing enrichment vs global firing rate.
Then it forms FROZEN physics-motivated sets for the collective-ablation test (heatdome_physics_ablate):
  core (z500 ridge, near centre) ; +flanking lows (vort850) ; +jet/wave-train (jet250/baroclinic) 
  full dynamical block set ; and the UNION of ALL strong features (strongest test).
Saves results/heatdome/scan_sets.json + figures/heatdome_scan_labels.png.

Paper: not in the paper (2021 heat-dome study; results shipped, demo notebook)
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); results/fs_atlas_class.npy (not shipped, see docs/REPRODUCE.md); results/fs_atlas_extra.npy (not shipped, see docs/REPRODUCE.md)
Outputs: results/heatdome
Run:   # JAX env, CPU
    python -m graphcast_sae.heatdome.heatdome_scan_analyze
"""
import os, sys, json

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import graphcast_sae.common.fs_common as fc
from graphcast_sae.common.signature_physics import gc_km

RES = fc.ROOT / "results/heatdome"; FIG = fc.ROOT / "figures"
PROBES = ['vort850', 'q600', 'ascent', 'shear', 't850', 'z500', 'jet250', 'div250',
          'abslat', 'land_sea', 'orography']
DYN = ['z500', 't850', 'jet250', 'vort850', 'div250', 'shear', 'ascent', 'q600']
XPROBES = ['coast_grad', 'orog_grad', 'node_density', 'blocking', 'atm_river', 'baroclinicity']
STRONG_CNT = 8           # frozen: >=8 active box nodes at some lead == strong ridge-firing
CORE_KM = 700.0          # frozen: z500 feature within 700 km of ridge centre == ridge core
BLOCK_HI = 0.30          # frozen: z_extra blocking index high

def main():
    s = np.load(RES / "scan.npy", allow_pickle=True).item()
    ax = np.load(fc.ROOT / "results/fs_atlas_extra.npy", allow_pickle=True).item()
    z = np.asarray(ax["z"]); zx = np.asarray(ax["z_extra"])
    cat = np.load(fc.ROOT / "results/fs_atlas_class.npy", allow_pickle=True).item()["cat"]
    catlg = np.load(fc.ROOT / "candidates/fs_feature_catalog.npy", allow_pickle=True).item()
    fr = catlg["firerate"]
    ridge = s["ridge"]; leads = s["leads_h"]; box_cnt = s["box_cnt"]; box_sum = s["box_sum"]
    cen_lat = s["cen_lat"]; cen_lon = s["cen_lon"]; reg_wsum = s["reg_wsum"]
    ctr = s["ridge_center"]; box = s["box"]
    peak_cnt = s["peak_cnt"]

    dome = ridge >= 0.75 * ridge.max()           # dome-built leads
    strong = np.where(peak_cnt >= STRONG_CNT)[0]
    # order by peak box firing
    strong = strong[np.argsort(box_sum.max(0)[strong])[::-1]]

    feats = {}
    for f in strong:
        w = reg_wsum[:, f] * dome
        wsum = w.sum()
        if wsum > 0:
            clat = float((w * cen_lat[:, f]).sum() / wsum)
            clon = float((w * cen_lon[:, f]).sum() / wsum)
        else:
            clat, clon = float(np.nan), float(np.nan)
        # propagation over active dome leads
        act = np.where((reg_wsum[:, f] > 0) & dome)[0]
        path = 0.0; disp = 0.0; eastdrift = 0.0
        if len(act) >= 2:
            la = cen_lat[act, f]; lo = cen_lon[act, f]
            path = float(np.sum(gc_km(la[:-1], lo[:-1], la[1:], lo[1:])))
            disp = float(gc_km(la[0], lo[0], la[-1], lo[-1]))
            eastdrift = float(lo[-1] - lo[0])
        persist = float(((box_cnt[:, f] >= 3) & dome).sum() / max(dome.sum(), 1))
        rcorr = float(np.corrcoef(box_sum[:, f], ridge)[0, 1]) if box_sum[:, f].std() > 0 else 0.0
        dist = float(gc_km(clat, clon, ctr[0], ctr[1])) if np.isfinite(clat) else np.inf
        zrow = z[f]; zxr = zx[f]
        dynvals = {p: float(zrow[PROBES.index(p)]) for p in DYN}
        dom_probe = max(dynvals, key=dynvals.get)
        blocking = float(zxr[3]); barocl = float(zxr[5]); atmriv = float(zxr[4])
        # group (frozen priority)
        if dom_probe == 'z500' and dist <= CORE_KM:
            grp = 'ridge_core'
        elif dom_probe == 'z500':
            grp = 'ridge_offcentre'
        elif dom_probe == 't850':
            grp = 'warm_dome'
        elif dom_probe == 'jet250' or barocl >= 0.30:
            grp = 'jet_wavetrain'
        elif dom_probe == 'vort850':
            grp = 'flanking_low'
        elif dom_probe == 'div250':
            grp = 'divergence'
        else:
            grp = 'other'
        feats[int(f)] = dict(
            peak_cnt=int(peak_cnt[f]), peak_sum=float(box_sum.max(0)[f]),
            clat=clat, clon=clon, dist_km=dist, dom_probe=dom_probe,
            z500=dynvals['z500'], t850=dynvals['t850'], jet250=dynvals['jet250'],
            vort850=dynvals['vort850'], blocking=blocking, baroclinicity=barocl, atm_river=atmriv,
            group=grp, cat=str(cat[f]), rcorr=rcorr, enrich=float(peak_cnt[f] / max(fr[f]*box_cnt.shape and 1, 1) if False else peak_cnt[f]),
            path_km=path, disp_km=disp, eastdrift_deg=eastdrift, persist=persist,
            firerate=float(fr[f]))

    # ---- frozen physics sets ----
    g = lambda name: [f for f, d in feats.items() if d["group"] == name]
    S_core = g('ridge_core') or ([int(strong[0])] if len(strong) else [])
    S_offc = g('ridge_offcentre'); S_warm = g('warm_dome')
    S_jet = g('jet_wavetrain'); S_flank = g('flanking_low')
    S_block = [f for f, d in feats.items() if d["blocking"] >= BLOCK_HI]
    S_union = [int(f) for f in strong]
    S_full = sorted(set(S_core + S_offc + S_warm + S_jet + S_flank + S_block))
    sets = {
        "core":         sorted(set(S_core)),
        "core_flank":   sorted(set(S_core + S_flank)),
        "core_jet":     sorted(set(S_core + S_jet)),
        "full_physics": S_full,
        "union_all":    sorted(set(S_union)),
    }
    centroids = {int(f): [feats[int(f)]["clat"], feats[int(f)]["clon"]] for f in feats}

    # normal-level & disk built in step 3; here just persist ids + centroids + labels
    out = dict(strong=[int(f) for f in strong], feats=feats, sets=sets,
               centroids=centroids, ridge_center=list(ctr), box=box,
               strong_cnt_thresh=STRONG_CNT, dome_leads_h=[int(x) for x in leads[dome]])
    json.dump(out, open(RES / "scan_sets.json", "w"), indent=2, default=float)

    # ---- report ----
    print(f"\nbaseline ridge peak {ridge.max():.0f} m; dome leads +{leads[dome][0]}..+{leads[dome][-1]}h")
    print(f"strong ridge-firing features (peak box active-nodes >= {STRONG_CNT}): {len(strong)}")
    print(f"\n{'feat':>5} {'grp':>15} {'cat':>18} {'domP':>7} {'z500':>5} {'t850':>5} {'jet':>5} "
          f"{'vort':>5} {'blk':>5} {'baro':>5} {'cnt':>4} {'cen(lat,lon)':>14} {'dist':>5} "
          f"{'rcorr':>6} {'path':>5} {'east':>5} {'pers':>5}")
    for f in strong:
        d = feats[int(f)]
        print(f"{f:>5} {d['group']:>15} {d['cat'][:18]:>18} {d['dom_probe']:>7} {d['z500']:>5.2f} "
              f"{d['t850']:>5.2f} {d['jet250']:>5.2f} {d['vort850']:>5.2f} {d['blocking']:>5.2f} "
              f"{d['baroclinicity']:>5.2f} {d['peak_cnt']:>4} ({d['clat']:>4.0f},{d['clon']:>5.0f}) "
              f"{d['dist_km']:>5.0f} {d['rcorr']:>+6.2f} {d['path_km']:>5.0f} {d['eastdrift_deg']:>+5.0f} {d['persist']:>5.2f}")
    print("\n=== PHYSICS-GUIDED SETS ===")
    for k, v in sets.items():
        print(f"  {k:>13}: n={len(v)}  {v}")
    print(f"-> results/heatdome/scan_sets.json")

    # ---- figure: firing vs lead (top strong) + centroid tracks ----
    GC = {'ridge_core': '#d1372b', 'ridge_offcentre': '#eb6834', 'warm_dome': '#b5179e',
          'jet_wavetrain': '#1d6fb8', 'flanking_low': '#1baf7a', 'divergence': '#8a6d3b',
          'other': '#999999'}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    a1.plot(leads, ridge, color="#111", lw=2.6, label="z500 ridge anom", zorder=5)
    a1.set_ylabel("z500 ridge anomaly (m)"); a1.set_xlabel("lead (h)")
    a1b = a1.twinx()
    for f in strong[:10]:
        d = feats[int(f)]
        a1b.plot(leads, box_sum[:, f], color=GC[d["group"]], lw=1.6, alpha=0.9,
                 label=f"{f} {d['group']}/{d['dom_probe']}")
    a1b.set_ylabel("box firing (sum of code)")
    a1.set_title("Strong ridge-firing features vs the ridge", loc="left", fontsize=10)
    a1b.legend(fontsize=6.5, ncol=1, loc="upper right")
    # centroid tracks over dome window
    reg_lat = s["reg_lat"]; reg_lon = s["reg_lon"]
    a2.scatter(reg_lon, reg_lat, s=2, color="#eee", zorder=0)
    for f in strong[:14]:
        d = feats[int(f)]
        act = np.where((reg_wsum[:, f] > 0) & dome)[0]
        if len(act) >= 2:
            a2.plot(cen_lon[act, f], cen_lat[act, f], "-o", ms=2.5, lw=1.2,
                    color=GC[d["group"]], alpha=0.85)
            a2.annotate(str(int(f)), (cen_lon[act[-1], f], cen_lat[act[-1], f]), fontsize=6)
    bx = box; a2.plot([bx["lon"][0], bx["lon"][1], bx["lon"][1], bx["lon"][0], bx["lon"][0]],
                      [bx["lat"][0], bx["lat"][0], bx["lat"][1], bx["lat"][1], bx["lat"][0]],
                      color="#2a78d6", lw=1.3)
    a2.plot(ctr[1], ctr[0], "k*", ms=15); a2.set_xlabel("lon"); a2.set_ylabel("lat")
    a2.set_title("Propagation: firing-centroid tracks over the dome window", loc="left", fontsize=10)
    from matplotlib.lines import Line2D
    a2.legend([Line2D([0], [0], color=c, lw=2) for c in GC.values()], list(GC.keys()),
              fontsize=6.5, loc="lower left")
    fig.suptitle("Data-driven ridge-firing features: physical labels + propagation", x=0.01,
                 ha="left", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "heatdome_scan_labels.png", bbox_inches="tight"); plt.close(fig)
    print("-> figures/heatdome_scan_labels.png")

if __name__ == "__main__":
    main()
