"""Analyze the heat-dome knockout -> verdict + figures.

Consumes results/heatdome/{phase1,phase2,era5_truth}.npy. Computes:
  BASELINE FIDELITY : does the model reproduce the ERA5 ridge & heat (before trusting ablation)?
  BLOCK (ridge)     : z500 ridge-anomaly-max peak, baseline vs block-normal/zero/rand -> collapse?
  HEAT              : box 2m-T max peak, same arms -> does the record heat dissipate?
  SKILL vs ERA5     : cos-lat box RMSE of z500(gpm) & 2m-T over the peak window; Dskill vs baseline.
  INTERNAL          : hd-feature box firing suppression baseline vs block-normal.
  CONTROLS          : rand-normal (specificity) and the non-block IC (should do little).
Frozen verdict (spec): blocking feature is a necessary handle iff block-normal collapses the ridge
AND reduces heat AND worsens skill, all beyond the random control, and the non-block IC does little.

Figures (figures/): heatdome_feature_firing.png, heatdome_trajectories.png, heatdome_maps.png,

Paper: not in the paper (2021 heat-dome study; results shipped, demo notebook)
Inputs: none beyond the arguments above
Outputs: results/heatdome
Run:   # JAX env, CPU
    python -m graphcast_sae.heatdome.heatdome_analyze  (CPU)
"""
import os, sys, json

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import graphcast_sae.common.fs_common as fc
import graphcast_sae.heatdome.heatdome_config as C

RES = fc.ROOT / "results/heatdome"; FIG = fc.ROOT / "figures"
ARMS = ["baseline", "block-normal", "block-zero", "rand-normal"]
COL = {"ERA5": "#111111", "baseline": "#2a78d6", "block-normal": "#eb6834",
       "block-zero": "#e34948", "rand-normal": "#1baf7a"}
LBL = {"baseline": "baseline", "block-normal": "block->normal", "block-zero": "block->0",
       "rand-normal": "random ctrl->normal", "ERA5": "ERA5 truth"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

def box_rmse(a, b, lat):
    w = np.cos(np.deg2rad(lat))[:, None]
    w = np.broadcast_to(w, a.shape)
    return float(np.sqrt(np.sum(w * (a - b) ** 2) / np.sum(w)))

def load():
    return (np.load(RES / "phase1.npy", allow_pickle=True).item(),
            np.load(RES / "phase2.npy", allow_pickle=True).item(),
            np.load(RES / "era5_truth.npy", allow_pickle=True).item())

def metrics(p2, truth):
    res = p2["res"]; hd = p2["hd_set"]; rand = p2["rand"]
    leads = p2["leads_h"]; H = len(leads)
    tl = truth["leads_h"]                      # 0..144
    era_ridge = truth["ridge_zanom_max"]; era_heat = truth["t2m_max_C"]
    lat = truth["box_lat"]
    # peak window mask on forecast leads
    win = (leads >= C.PEAK_WINDOW_H[0]) & (leads <= C.PEAK_WINDOW_H[1])
    # ERA5 aligned to forecast leads (+6..+144 -> truth idx 1..H)
    era_z = truth["z500_box"][1:H + 1]; era_t = truth["t2m_box"][1:H + 1]
    m = {"era_ridge_peak": float(era_ridge.max()), "era_heat_peak": float(era_heat.max()),
         "arms": {}}
    base = res["baseline"]
    for a in ARMS:
        if a not in res: continue
        r = res[a]
        ridge_peak = float(r["ridge"].max()); heat_peak = float(r["heat"].max())
        # skill RMSE over peak window
        ez = np.array([box_rmse(r["z500_box"][h], era_z[h], lat) for h in range(H)])
        et = np.array([box_rmse(r["t2m_box"][h], era_t[h], lat) for h in range(H)])
        d = dict(ridge_peak=ridge_peak, heat_peak=heat_peak,
                 skill_z=float(ez[win].mean()), skill_t=float(et[win].mean()),
                 hd_box_peak=float(np.max([r["box_feats"][f] for f in hd], axis=0).max()),
                 rand_box_peak=float(np.max([r["box_feats"][f] for f in rand], axis=0).max()))
        m["arms"][a] = d
    b = m["arms"]["baseline"]
    for a in ARMS:
        if a == "baseline" or a not in m["arms"]: continue
        d = m["arms"][a]
        d["d_ridge"] = b["ridge_peak"] - d["ridge_peak"]     # >0 => ridge collapses
        d["d_heat"] = b["heat_peak"] - d["heat_peak"]        # >0 => heat dissipates
        d["d_skill_z"] = d["skill_z"] - b["skill_z"]         # >0 => worse
        d["d_skill_t"] = d["skill_t"] - b["skill_t"]
        d["hd_supp"] = (b["hd_box_peak"] - d["hd_box_peak"]) / max(b["hd_box_peak"], 1e-6)
    # non-block control
    nb = p2["res_nonblock"]
    nbm = {}
    for a in ("baseline", "block-normal"):
        r = nb[a]
        nbm[a] = dict(ridge_peak=float(r["ridge"].max()), heat_peak=float(r["heat"].max()))
    nbm["d_ridge"] = nbm["baseline"]["ridge_peak"] - nbm["block-normal"]["ridge_peak"]
    nbm["d_heat"] = nbm["baseline"]["heat_peak"] - nbm["block-normal"]["heat_peak"]
    m["nonblock"] = nbm
    return m

def verdict(m):
    bn = m["arms"].get("block-normal", {}); rn = m["arms"].get("rand-normal", {})
    fidelity = m["arms"]["baseline"]["ridge_peak"] / max(m["era_ridge_peak"], 1e-6)
    necessary = (bn.get("d_ridge", 0) > 20 and bn.get("d_heat", 0) > 1.0 and
                 bn.get("d_skill_z", 0) > 0 and
                 bn.get("d_ridge", 0) > 1.5 * abs(rn.get("d_ridge", 0)) and
                 abs(m["nonblock"]["d_ridge"]) < 0.5 * bn.get("d_ridge", 1e9))
    return dict(baseline_ridge_fidelity=float(fidelity),
                block_normal_d_ridge=bn.get("d_ridge"), block_normal_d_heat=bn.get("d_heat"),
                block_normal_d_skill_z=bn.get("d_skill_z"), block_normal_d_skill_t=bn.get("d_skill_t"),
                block_normal_hd_supp=bn.get("hd_supp"),
                rand_d_ridge=rn.get("d_ridge"), rand_d_heat=rn.get("d_heat"),
                nonblock_d_ridge=m["nonblock"]["d_ridge"],
                verdict=("BLOCKING FEATURE NECESSARY" if necessary else
                         "DISTRIBUTED / NOT A SINGLE LEVER"))

def fig_feature_firing(p1, truth):
    hd = p1["hd_set"]; leads = p1["leads_h"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.plot(truth["leads_h"], truth["ridge_zanom_max"], color="#111", lw=2.4, label="ERA5 z500 ridge anom")
    ax1.plot(leads, p1["ridge"], color=COL["baseline"], lw=1.8, ls="--", label="GraphCast baseline ridge")
    ax1.set_ylabel("z500 ridge anomaly (m)"); ax1.set_xlabel("lead (h)")
    ax1b = ax1.twinx()
    for f in p1["cands"]:
        c = "#eb6834" if f in hd else "#bbb"; lw = 2.2 if f in hd else 1.0
        ax1b.plot(leads, p1["box_sum"][f], color=c, lw=lw, alpha=0.9,
                  label=f"feat {f}" + (" (hd)" if f in hd else ""))
    ax1b.set_ylabel("candidate box firing (sum of code)")
    ax1.set_title("Heat-dome feature firing tracks the ridge as it builds", loc="left", fontsize=10)
    ax1.legend(loc="upper left", fontsize=7); ax1b.legend(loc="lower right", fontsize=7, ncol=2)
    # spatial: hd feature firing map at ridge peak lead vs ridge box
    hpk = int(np.argmax(p1["ridge"]))
    f0 = hd[0]; nm = p1["node_maps"][f0][hpk]
    mlat = p1["mlat"]; mlon = p1["mlon"]
    reg = (mlat >= 30) & (mlat <= 72) & (mlon >= -160) & (mlon <= -85)
    act = nm[reg] > 0
    ax2.scatter(mlon[reg][~act], mlat[reg][~act], s=4, color="#ddd", zorder=1)
    sc = ax2.scatter(mlon[reg][act], mlat[reg][act], c=nm[reg][act], s=42, cmap="Oranges",
                     vmin=0, zorder=3, edgecolors="#a33", linewidths=0.3)
    bx = C.BOX
    ax2.plot([bx["lon"][0], bx["lon"][1], bx["lon"][1], bx["lon"][0], bx["lon"][0]],
             [bx["lat"][0], bx["lat"][0], bx["lat"][1], bx["lat"][1], bx["lat"][0]],
             color="#2a78d6", lw=1.4, label="W-NA box")
    cx, cy = truth["ridge_center"]; ax2.plot(cy, cx, "k*", ms=14, label="ridge centre")
    ax2.set_title(f"heat-dome feature {f0} firing @ ridge peak (+{leads[hpk]}h)", loc="left", fontsize=10)
    ax2.set_xlabel("lon"); ax2.set_ylabel("lat"); ax2.legend(fontsize=7)
    plt.colorbar(sc, ax=ax2, fraction=0.04, pad=0.02, label="feature activation")
    fig.tight_layout(); fig.savefig(FIG / "heatdome_feature_firing.png", bbox_inches="tight")
    plt.close(fig); print("-> figures/heatdome_feature_firing.png")

def fig_trajectories(p2, truth):
    res = p2["res"]; leads = p2["leads_h"]; tl = truth["leads_h"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.plot(tl, truth["ridge_zanom_max"], color=COL["ERA5"], lw=2.4, label=LBL["ERA5"], zorder=5)
    a2.plot(tl, truth["t2m_max_C"], color=COL["ERA5"], lw=2.4, label=LBL["ERA5"], zorder=5)
    for a in ARMS:
        if a not in res: continue
        a1.plot(leads, res[a]["ridge"], color=COL[a], lw=1.8, marker="o", ms=3, label=LBL[a])
        a2.plot(leads, res[a]["heat"], color=COL[a], lw=1.8, marker="o", ms=3, label=LBL[a])
    a1.set_title("The block: z500 ridge anomaly max over box", loc="left", fontsize=10)
    a2.set_title("The heat: 2m-T max over box", loc="left", fontsize=10)
    a1.set_ylabel("z500 anomaly (m)"); a2.set_ylabel("2m-T max (deg C)")
    a1.set_xlabel("lead (h)"); a2.set_xlabel("lead (h)"); a1.legend(fontsize=7.5)
    fig.suptitle("Heat-dome knockout — ridge & heat vs lead vs ERA5", x=0.01, ha="left",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIG / "heatdome_trajectories.png", bbox_inches="tight")
    plt.close(fig); print("-> figures/heatdome_trajectories.png")

def fig_maps(p2, truth):
    res = p2["res"]; leads = p2["leads_h"]; H = len(leads)
    lat = truth["box_lat"]; lon = truth["box_lon"]; lon180 = np.where(lon > 180, lon - 360, lon)
    zpk = int(np.argmax(truth["ridge_zanom_max"][1:H+1]))     # forecast-lead index for z500
    era_z = truth["z500_box"][1:H+1]; era_t = truth["t2m_box"][1:H+1] - 273.15
    cols = [("ERA5", None), ("baseline", res["baseline"]), ("block-normal", res["block-normal"])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    zlv = np.linspace(min(era_z[zpk].min(), 5300), max(era_z[zpk].max(), 6000), 13)
    for k, (nm, r) in enumerate(cols):
        zf = era_z[zpk] if r is None else r["z500_box"][zpk]
        tf = era_t[zpk] if r is None else (r["t2m_box"][zpk] - 273.15)
        c1 = axes[0][k].contourf(lon180, lat, zf, levels=zlv, cmap="RdBu_r", extend="both")
        axes[0][k].set_title(f"z500 (gpm) — {LBL[nm]}  @+{leads[zpk]}h", fontsize=9, loc="left")
        c2 = axes[1][k].contourf(lon180, lat, tf, levels=np.linspace(0, 45, 16), cmap="hot_r", extend="both")
        axes[1][k].set_title(f"2m-T (C) — {LBL[nm]}  @+{leads[zpk]}h", fontsize=9, loc="left")
        for ax in (axes[0][k], axes[1][k]):
            cx, cy = truth["ridge_center"]; ax.plot(cy, cx, "k*", ms=11)
            ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.colorbar(c1, ax=axes[0].tolist(), fraction=0.02, pad=0.01, label="gpm")
    plt.colorbar(c2, ax=axes[1].tolist(), fraction=0.02, pad=0.01, label="deg C")
    fig.suptitle("Ridge & heat fields: ERA5 vs baseline vs block->normal", x=0.01, ha="left",
                 fontsize=12, weight="bold")
    fig.savefig(FIG / "heatdome_maps.png", bbox_inches="tight")
    plt.close(fig); print("-> figures/heatdome_maps.png")

def fig_skill_bars(m):
    arms3 = ["block-normal", "block-zero", "rand-normal"]
    keys = [("d_ridge", "ridge collapse (m)  [base-arm]"),
            ("d_heat", "heat drop (C)  [base-arm]"),
            ("d_skill_z", "z500 skill worsening (gpm)"),
            ("d_skill_t", "2m-T skill worsening (C)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for j, (k, ttl) in enumerate(keys):
        vals = [m["arms"].get(a, {}).get(k, np.nan) for a in arms3]
        axes[j].bar(range(len(arms3)), vals, color=[COL[a] for a in arms3])
        axes[j].set_xticks(range(len(arms3))); axes[j].set_xticklabels([LBL[a] for a in arms3], rotation=25, ha="right", fontsize=7.5)
        axes[j].axhline(0, color="#888", lw=0.8); axes[j].set_title(ttl, fontsize=9, loc="left")
    fig.suptitle("Is the blocking feature a necessary causal handle? (block vs controls)",
                 x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / "heatdome_skill_bars.png", bbox_inches="tight")
    plt.close(fig); print("-> figures/heatdome_skill_bars.png")

def main():
    p1, p2, truth = load()
    m = metrics(p2, truth); V = verdict(m)
    print("\n=== HEAT-DOME KNOCKOUT METRICS ===")
    print(f"ERA5 ridge peak {m['era_ridge_peak']:.0f} m ; heat peak {m['era_heat_peak']:.1f} C")
    for a in ARMS:
        d = m["arms"].get(a)
        if not d: continue
        extra = ""
        if a != "baseline":
            extra = (f"  d_ridge {d['d_ridge']:+.0f}m  d_heat {d['d_heat']:+.1f}C  "
                     f"d_skillZ {d['d_skill_z']:+.1f}gpm  d_skillT {d['d_skill_t']:+.2f}C  hd_supp {100*d['hd_supp']:+.0f}%")
        print(f"  {a:13s} ridge_peak {d['ridge_peak']:.0f}m  heat_peak {d['heat_peak']:.1f}C  "
              f"skillZ {d['skill_z']:.1f}gpm  skillT {d['skill_t']:.2f}C  hd_box {d['hd_box_peak']:.0f}{extra}")
    print(f"  non-block IC: baseline ridge {m['nonblock']['baseline']['ridge_peak']:.0f}m -> "
          f"block-normal {m['nonblock']['block-normal']['ridge_peak']:.0f}m (d_ridge {m['nonblock']['d_ridge']:+.0f}m)")
    print("\n=== VERDICT ===")
    print(json.dumps(V, indent=2, default=float))
    json.dump({"phase1": {"hd_set": p1["hd_set"], "top": int(p1["top"]),
                          "peak": {int(k): float(v) for k, v in p1["peak"].items()},
                          "corr": {int(k): float(v) for k, v in p1["corr"].items()}},
               "metrics": m, "verdict": V}, open(RES / "verdict.json", "w"), indent=2, default=float)
    print("-> results/heatdome/verdict.json")
    fig_feature_firing(p1, truth); fig_trajectories(p2, truth); fig_maps(p2, truth); fig_skill_bars(m)

if __name__ == "__main__":
    main()
