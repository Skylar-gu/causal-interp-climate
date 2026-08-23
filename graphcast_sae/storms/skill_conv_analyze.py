"""Analyze the convection-skill runs and build figures + verdict.

Consumes results/skill/convection/run_<name>.npy and era5_truth.npy. Computes per storm:
  INTERNAL : TC feature 3243 peak in box, baseline vs conv-normal (suppression fraction)
  PHYSICAL : MSLP deepening (IC->min) and peak 10m wind per arm; Delta-deepening from ablation
  SKILL    : intensity error vs ERA5 (RMSE of MSLP-min trajectory over the intensification window),
             baseline vs conv-normal vs conv-zero vs rand-normal; Delta-skill = err(arm)-err(base)
Compares convection ablation vs random-feature control vs the non-developing storm.

Figures (figures/): skill_conv_trajectories.png (per-storm MSLP & wind vs lead, all arms + ERA5),
skill_conv_summary.png (Delta-deepening & Delta-skill bars, conv vs controls), and
skill_conv_colocation.png (mid-intensification node overlay: convection 2401 firing on the TC low).

Paper: Sec. 3 'The intervention contrast' (Table tab:mechanism-interventions)
Inputs: none beyond the arguments above
Outputs: results/skill/<MECH_RES|convection>/verdict.json; figures/skill_<tag>_{trajectories,summary,colocation}.png
Run:   # JAX env, CPU
    python -m graphcast_sae.storms.skill_conv_analyze   (CPU)
"""
import os, sys, json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import graphcast_sae.common.fs_common as fc
import importlib
S = importlib.import_module("graphcast_sae.common." + os.environ.get("SKILL_STORMS", "skill_conv_storms"))

RES = fc.ROOT / f"results/skill/{os.environ.get('MECH_RES','convection')}"
FIG = fc.ROOT / "figures"
# figure names carry the arm, or scoring a second arm silently overwrites the
# first arm's committed figures (it did, 2026-08-16, when moisture2 was scored)
TAG = "conv" if RES.name == "convection" else RES.name
TC = S.TC
ARMS = ["baseline", "conv-normal", "conv-zero", "rand-normal"]
COL = {"ERA5": "#111111", "baseline": "#2a78d6", "conv-normal": "#eb6834",
       "conv-zero": "#e34948", "rand-normal": "#1baf7a"}
LBL = {"baseline": "baseline", "conv-normal": "conv->normal", "conv-zero": "conv->0",
       "rand-normal": "random ctrl->normal", "ERA5": "ERA5 truth"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

def load():
    truth = np.load(RES / "era5_truth.npy", allow_pickle=True).item()
    runs = {}
    for name in S.STORMS:
        p = RES / f"run_{name}.npy"
        if p.exists():
            runs[name] = np.load(p, allow_pickle=True).item()
    return truth, runs

def metrics(runs, truth):
    """Per-storm metric table."""
    rows = {}
    for name, r in runs.items():
        t = truth[name]
        ic_mslp = float(t["mslp_min"][0])
        era_dp = ic_mslp - float(np.min(t["mslp_min"]))
        era_peak_lead = int(np.argmin(t["mslp_min"]))          # in units of 6h from IC
        # window: intensification up to ERA5 peak (at least 6 leads / +36h)
        wmax_h = max(era_peak_lead, 6)
        res = r["res"]
        # ERA5 aligned to forecast leads (+6h..+96h) => truth indices 1..H
        H = len(res["baseline"]["mslp_min"])
        truth_aligned = t["mslp_min"][1:H + 1]
        widx = np.arange(min(wmax_h, H))
        out = {"ic_mslp": ic_mslp, "era_deepen": era_dp, "era_peak_lead_h": era_peak_lead * 6,
               "nondev": bool(r.get("nondev", False)), "arms": {}}
        base_dp = None; base_err = None
        for a in ARMS:
            if a not in res: continue
            mm = res[a]["mslp_min"]; ww = res[a]["wind_max"]
            dp = ic_mslp - float(np.min(mm))
            pk_wind = float(np.max(ww))
            err = float(np.sqrt(np.mean((mm[widx] - truth_aligned[widx]) ** 2)))
            tc_peak = float(np.max(res[a]["box_feats"][TC]))
            # feature group comes from the RUN, not the module constant: the same analyzer
            # scores the moisture arm (results/skill/moisture), whose group is disjoint from CONV.
            _grp = [int(f) for f in r.get("conv", S.CONV)]
            conv_box = float(np.max(np.mean([res[a]["box_feats"][f] for f in _grp], axis=0)))
            # ...and so does the CONTROL group. This line used to hard-code S.RANDOM_CTRL
            # while the line above it correctly read the run, so every "the control fires at
            # X% of the treatment" number ever printed for a battery with a per-storm
            # MECH_INBOX_CTL described a control that was not the one ablated. In
            # conv_corectl2 that understated the matched controls by an order of magnitude:
            # reported 10-36% of the convection group, actual 150-380%.
            _ctl = [int(f) for f in r.get("rand", S.RANDOM_CTRL)]
            rand_box = (float(np.max(np.mean([res[a]["box_feats"][f] for f in _ctl], axis=0)))
                        if _ctl else float("nan"))
            d = dict(deepen=dp, peak_wind=pk_wind, err_mslp=err, tc_peak=tc_peak,
                     conv_box=conv_box, rand_box=rand_box)
            if a == "baseline":
                base_dp = dp; base_err = err; base_tc = tc_peak
            out["arms"][a] = d
        # deltas relative to baseline
        for a in ARMS:
            if a in out["arms"] and a != "baseline":
                out["arms"][a]["d_deepen"] = base_dp - out["arms"][a]["deepen"]   # >0 => less deepening
                out["arms"][a]["d_err"] = out["arms"][a]["err_mslp"] - base_err   # >0 => worse skill
                out["arms"][a]["tc_supp"] = (base_tc - out["arms"][a]["tc_peak"]) / max(base_tc, 1e-6)
        rows[name] = out
    return rows

def fig_trajectories(runs, truth):
    names = [n for n in S.STORMS if n in runs]
    ncol = 2  # MSLP, wind
    fig, axes = plt.subplots(len(names), ncol, figsize=(11, 2.4 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        r = runs[name]; t = truth[name]; res = r["res"]
        H = len(res["baseline"]["mslp_min"])
        fl = (np.arange(H) + 1) * 6         # forecast valid leads (h)
        tl = np.arange(len(t["mslp_min"])) * 6
        axm, axw = axes[i][0], axes[i][1]
        axm.plot(tl, t["mslp_min"], color=COL["ERA5"], lw=2.2, label=LBL["ERA5"], zorder=5)
        axw.plot(tl, t["wind_max"], color=COL["ERA5"], lw=2.2, label=LBL["ERA5"], zorder=5)
        for a in ARMS:
            if a not in res: continue
            axm.plot(fl, res[a]["mslp_min"], color=COL[a], lw=1.8, marker="o", ms=3, label=LBL[a])
            axw.plot(fl, res[a]["wind_max"], color=COL[a], lw=1.8, marker="o", ms=3, label=LBL[a])
        tag = " (non-developer control)" if r.get("nondev") else ""
        axm.set_title(f"{name}{tag}  —  min MSLP", loc="left", fontsize=9.5)
        axw.set_title(f"{name}  —  max 10m wind", loc="left", fontsize=9.5)
        axm.set_ylabel("hPa"); axw.set_ylabel("m/s")
        if i == len(names) - 1:
            axm.set_xlabel("lead (h)"); axw.set_xlabel("lead (h)")
        if i == 0:
            axm.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    fig.suptitle("Convection ablation vs baseline vs ERA5 truth — forecast intensity by lead",
                 x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(FIG / f"skill_{TAG}_trajectories.png", bbox_inches="tight")
    plt.close(fig); print(f"-> figures/skill_{TAG}_trajectories.png")

def fig_summary(M):
    names = [n for n in M if not M[n]["nondev"]]
    nd = [n for n in M if M[n]["nondev"]]
    order = names + nd
    x = np.arange(len(order)); w = 0.26
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    arms3 = ["conv-normal", "conv-zero", "rand-normal"]
    for j, a in enumerate(arms3):
        dd = [M[n]["arms"].get(a, {}).get("d_deepen", np.nan) for n in order]
        de = [M[n]["arms"].get(a, {}).get("d_err", np.nan) for n in order]
        a1.bar(x + (j - 1) * w, dd, w, color=COL[a], label=LBL[a])
        a2.bar(x + (j - 1) * w, de, w, color=COL[a], label=LBL[a])
    for ax, ttl, yl in ((a1, "Loss of deepening from ablation (hPa)  — higher = more suppression",
                         "baseline deepening - arm deepening (hPa)"),
                        (a2, "Skill degradation vs ERA5 (hPa RMSE)  — higher = worse forecast",
                         "arm error - baseline error (hPa)")):
        ax.set_title(ttl, fontsize=9.5, loc="left"); ax.set_ylabel(yl, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="#888", lw=0.8)
        ax.axvline(len(names) - 0.5, color="#bbb", ls="--", lw=1)
    a1.legend(fontsize=8)
    fig.suptitle("Is convection necessary for intensity skill?  Convection ablation vs random control",
                 x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / f"skill_{TAG}_summary.png", bbox_inches="tight")
    plt.close(fig); print(f"-> figures/skill_{TAG}_summary.png")

def fig_colocation(runs):
    # pick storms that have a mid snapshot; show up to 3 in a row
    have = [(n, r["snap"]["baseline_mid"]) for n, r in runs.items()
            if "baseline_mid" in r.get("snap", {})]
    if not have:
        print("no mid snapshots for co-location"); return
    have = have[:3]
    fig, axes = plt.subplots(1, len(have), figsize=(5.2 * len(have), 4.6), squeeze=False)
    for k, (name, snap) in enumerate(have):
        ax = axes[0][k]
        mlat = snap["mlat"]; mlon = snap["mlon"]
        c2401 = snap["node_2401"]; c3243 = snap["node_3243"]
        box = runs[name]["box"]
        m = (mlat >= box["lat"][0] - 5) & (mlat <= box["lat"][1] + 5) & \
            (mlon >= box["lon"][0] - 5) & (mlon <= box["lon"][1] + 5)
        # MSLP low contour (box grid)
        lo = S.norm_lon(box["lon"])
        glon = snap["mslp_lon"].copy()
        glon = np.where(glon > 180, glon - 360, glon)
        ax.contour(glon, snap["mslp_lat"], snap["mslp_grid"] / 100.0, levels=8,
                   colors="#888", linewidths=0.6, alpha=0.7, zorder=1)
        # convection 2401 firing (fill) where active
        act = c2401[m]
        sc = ax.scatter(mlon[m][act > 0], mlat[m][act > 0], c=act[act > 0], s=26,
                        cmap="Oranges", vmin=0, zorder=2, edgecolors="none", label="conv 2401")
        # TC feature 3243 firing (outline)
        tcm = c3243[m]
        ax.scatter(mlon[m][tcm > 0], mlat[m][tcm > 0], s=70, facecolors="none",
                   edgecolors="#2a78d6", linewidths=1.3, zorder=3, label="TC 3243")
        cx, cy = runs[name]["center"]
        ax.plot(cx, cy, "kx", ms=9, mew=2, zorder=4)
        ax.set_title(f"{name} @ +48h", fontsize=10, loc="left")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        if k == 0: ax.legend(fontsize=7, loc="upper right")
        plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label="conv 2401 act")
    fig.suptitle("Co-location: convection feature 2401 fires ON the TC low (feature 3243) at mid-intensification",
                 x=0.01, ha="left", fontsize=11.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / f"skill_{TAG}_colocation.png", bbox_inches="tight")
    plt.close(fig); print(f"-> figures/skill_{TAG}_colocation.png")

def verdict(M):
    dev = [n for n in M if not M[n]["nondev"]]
    def med(a, key):
        vals = [M[n]["arms"].get(a, {}).get(key, np.nan) for n in dev]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if vals else np.nan
    summ = {"storms_developing": dev,
            "conv_normal": {"med_d_deepen": med("conv-normal", "d_deepen"),
                            "med_d_err": med("conv-normal", "d_err"),
                            "med_tc_supp": med("conv-normal", "tc_supp")},
            "conv_zero": {"med_d_deepen": med("conv-zero", "d_deepen"),
                          "med_d_err": med("conv-zero", "d_err"),
                          "med_tc_supp": med("conv-zero", "tc_supp")},
            "rand_normal": {"med_d_deepen": med("rand-normal", "d_deepen"),
                            "med_d_err": med("rand-normal", "d_err"),
                            "med_tc_supp": med("rand-normal", "tc_supp")}}
    nd = [n for n in M if M[n]["nondev"]]
    if nd:
        n = nd[0]
        summ["nondev"] = {"name": n,
                          "conv_normal_d_deepen": M[n]["arms"].get("conv-normal", {}).get("d_deepen", np.nan),
                          "conv_normal_d_err": M[n]["arms"].get("conv-normal", {}).get("d_err", np.nan)}
    return summ

def main():
    truth, runs = load()
    if not runs:
        print("no run_<name>.npy yet — run skill_conv_run.py first"); return
    M = metrics(runs, truth)
    # print table
    print("\n=== PER-STORM METRICS ===")
    for name, m in M.items():
        tag = "CTRL" if m["nondev"] else "dev"
        print(f"\n{name} [{tag}] IC_MSLP {m['ic_mslp']:.1f} hPa; ERA5 deepen {m['era_deepen']:.1f} hPa "
              f"@+{m['era_peak_lead_h']}h")
        for a in ARMS:
            d = m["arms"].get(a)
            if not d: continue
            extra = ""
            if a != "baseline":
                extra = (f"  Ddeepen {d['d_deepen']:+.1f}hPa  Derr {d['d_err']:+.1f}hPa  "
                         f"TCsupp {100*d['tc_supp']:+.0f}%")
            print(f"   {a:12s} deepen {d['deepen']:5.1f}hPa  peakwind {d['peak_wind']:4.1f}  "
                  f"err {d['err_mslp']:4.1f}  TCpeak {d['tc_peak']:5.0f}  convbox {d['conv_box']:4.0f}  "
                  f"randbox {d['rand_box']:4.0f}{extra}")
    V = verdict(M)
    print("\n=== VERDICT (median over developing storms) ===")
    print(json.dumps(V, indent=2, default=float))
    json.dump({"metrics": M, "verdict": V}, open(RES / "verdict.json", "w"), indent=2, default=float)
    print(f"-> {RES.relative_to(fc.ROOT)}/verdict.json")
    fig_trajectories(runs, truth)
    fig_summary(M)
    fig_colocation(runs)

if __name__ == "__main__":
    main()
