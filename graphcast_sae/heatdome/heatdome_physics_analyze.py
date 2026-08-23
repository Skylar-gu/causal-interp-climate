"""Verdict + figures for the physics-guided collective ablation (CPU).

Consumes results/heatdome/{physics_ablate,era5_truth,scan_sets}.npy/json. Per set (core ->
core+flank -> core+jet -> full_physics -> union_all) and the random control, computes the ridge
z500-anomaly peak, box 2m-T peak, z500 & 2m-T skill RMSE vs ERA5 over the peak window, deltas vs
baseline, and union-feature internal suppression. Verdict: if even full_physics / union_all do NOT
collapse the ridge (beyond the random control) -> blocking is GENUINELY DISTRIBUTED (robust). If a
set does -> name it. Figures: heatdome_physics_traj.png, heatdome_physics_bars.png.

Paper: not in the paper (2021 heat-dome study; results shipped, demo notebook)
Inputs: none beyond the arguments above
Outputs: results/heatdome
Run:   # JAX env, CPU
    python -m graphcast_sae.heatdome.heatdome_physics_analyze
"""
import os, sys, json

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import graphcast_sae.common.fs_common as fc
import graphcast_sae.heatdome.heatdome_config as C
from graphcast_sae.heatdome.heatdome_analyze import box_rmse

RES = fc.ROOT / "results/heatdome"; FIG = fc.ROOT / "figures"
SETS = ["core", "core_flank", "core_jet", "full_physics", "union_all"]
COL = {"ERA5": "#111", "baseline": "#2a78d6", "core": "#f4a13a", "core_flank": "#1baf7a",
       "core_jet": "#1d6fb8", "full_physics": "#eb6834", "union_all": "#d1372b", "random": "#8a6d3b"}

def main():
    d = np.load(RES / "physics_ablate.npy", allow_pickle=True).item()
    truth = np.load(RES / "era5_truth.npy", allow_pickle=True).item()
    res = d["res"]; leads = d["leads_h"]; H = len(leads); union = d["sets"]["union_all"]
    lat = truth["box_lat"]
    win = (leads >= C.PEAK_WINDOW_H[0]) & (leads <= C.PEAK_WINDOW_H[1])
    era_z = truth["z500_box"][1:H+1]; era_t = truth["t2m_box"][1:H+1]
    base = res["baseline"]
    m = {}
    for a, r in res.items():
        ez = np.array([box_rmse(r["z500_box"][h], era_z[h], lat) for h in range(H)])
        et = np.array([box_rmse(r["t2m_box"][h], era_t[h], lat) for h in range(H)])
        ub = np.sum([r["box_feats"][f] for f in union], axis=0)
        m[a] = dict(ridge_peak=float(r["ridge"].max()), heat_peak=float(r["heat"].max()),
                    skill_z=float(ez[win].mean()), skill_t=float(et[win].mean()),
                    union_box_peak=float(ub.max()))
    b = m["baseline"]
    for a in m:
        if a == "baseline": continue
        m[a]["d_ridge"] = b["ridge_peak"] - m[a]["ridge_peak"]
        m[a]["d_heat"] = b["heat_peak"] - m[a]["heat_peak"]
        m[a]["d_skill_z"] = m[a]["skill_z"] - b["skill_z"]
        m[a]["d_skill_t"] = m[a]["skill_t"] - b["skill_t"]
        m[a]["supp"] = (b["union_box_peak"] - m[a]["union_box_peak"]) / max(b["union_box_peak"], 1e-6)

    # verdict: does union/full collapse ridge beyond random?
    rr = m["random"]
    def collapses(a):
        return (m[a]["d_ridge"] > 20 and m[a]["d_heat"] > 1.0 and m[a]["d_skill_z"] > 0
                and m[a]["d_ridge"] > 1.5 * abs(rr["d_ridge"]))
    hit = [a for a in SETS if collapses(a)]
    verdict = ("GENUINELY DISTRIBUTED (robust): even the full/union physics set does not collapse the ridge"
               if not hit else f"RIDGE HELD BY SET(S): {hit}")

    print(f"\nERA5 ridge peak {truth['ridge_zanom_max'].max():.0f} m; heat peak {truth['t2m_max_C'].max():.1f} C")
    print(f"baseline ridge {b['ridge_peak']:.0f} m; heat {b['heat_peak']:.1f} C; "
          f"skillZ {b['skill_z']:.1f} gpm; skillT {b['skill_t']:.2f} C; union_box_pk {b['union_box_peak']:.0f}")
    print(f"\n{'set':>13} {'nfeat':>5} {'ridge':>6} {'dR':>6} {'heat':>5} {'dH':>5} {'skZ':>5} {'dskZ':>6} "
          f"{'skT':>5} {'dskT':>6} {'supp%':>6}")
    for a in SETS + ["random"]:
        nf = len(d["sets"][a]) if a in d["sets"] else len(d["rand"])
        x = m[a]
        print(f"{a:>13} {nf:>5} {x['ridge_peak']:>6.0f} {x['d_ridge']:>+6.1f} {x['heat_peak']:>5.1f} "
              f"{x['d_heat']:>+5.1f} {x['skill_z']:>5.1f} {x['d_skill_z']:>+6.2f} {x['skill_t']:>5.2f} "
              f"{x['d_skill_t']:>+6.2f} {100*x['supp']:>6.0f}")
    print(f"\n=== VERDICT: {verdict}")
    json.dump({"metrics": m, "verdict": verdict, "sets": {k: v for k, v in d["sets"].items()},
               "random": d["rand"]}, open(RES / "physics_verdict.json", "w"), indent=2, default=float)
    print("-> results/heatdome/physics_verdict.json")

    # trajectories
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    tl = truth["leads_h"]
    a1.plot(tl, truth["ridge_zanom_max"], color=COL["ERA5"], lw=2.4, label="ERA5", zorder=6)
    a2.plot(tl, truth["t2m_max_C"], color=COL["ERA5"], lw=2.4, label="ERA5", zorder=6)
    for a in ["baseline"] + SETS + ["random"]:
        a1.plot(leads, res[a]["ridge"], color=COL[a], lw=1.7, marker="o", ms=2.5, label=a)
        a2.plot(leads, res[a]["heat"], color=COL[a], lw=1.7, marker="o", ms=2.5, label=a)
    a1.set_title("z500 ridge anomaly max over box", loc="left", fontsize=10)
    a2.set_title("2m-T max over box", loc="left", fontsize=10)
    a1.set_ylabel("m"); a2.set_ylabel("deg C"); a1.set_xlabel("lead (h)"); a2.set_xlabel("lead (h)")
    a1.legend(fontsize=7, ncol=2)
    fig.suptitle("Physics-guided collective ablation — ridge & heat vs lead vs ERA5", x=0.01,
                 ha="left", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / "heatdome_physics_traj.png", bbox_inches="tight")
    plt.close(fig); print("-> figures/heatdome_physics_traj.png")

    # bars
    keys = [("d_ridge", "ridge collapse (m)"), ("d_heat", "heat drop (C)"),
            ("d_skill_z", "z500 skill worsening (gpm)"), ("supp", "union feat suppression (frac)")]
    order = SETS + ["random"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for j, (k, ttl) in enumerate(keys):
        vals = [m[a][k] for a in order]
        axes[j].bar(range(len(order)), vals, color=[COL[a] for a in order])
        axes[j].set_xticks(range(len(order))); axes[j].set_xticklabels(order, rotation=30, ha="right", fontsize=7.5)
        axes[j].axhline(0, color="#888", lw=0.8); axes[j].set_title(ttl, fontsize=9, loc="left")
    fig.suptitle("Does any physics-motivated set collapse the ridge? (vs random control)", x=0.01,
                 ha="left", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / "heatdome_physics_bars.png", bbox_inches="tight")
    plt.close(fig); print("-> figures/heatdome_physics_bars.png")

if __name__ == "__main__":
    main()
