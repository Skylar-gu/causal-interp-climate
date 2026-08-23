"""Score the convection disk-radius sweep against docs/prereg/prereg_conv_radius_sweep.md.

Reads results/skill/conv_r{R}/run_<storm>.npy for R in {500, 750, 1000, 2500} plus the
committed 1500 km battery in results/skill/convection/, reusing skill_conv_analyze.metrics()
so Delta-deepening and Delta-error are the committed definitions. Exposure per radius is the
fraction of the convection group's node-level firing (baseline +48 h snapshot, which is
radius-independent) that lies inside the disk, normalised to the largest (2500 km) disk.

Writes results/skill/conv_radius_sweep.json and figures/conv_radius_sweep.png.

Paper: Sec. 3 (1500 km disk justification)
Inputs: results/skill/convection (shipped)
Outputs: figures/conv_radius_sweep.png; results/skill/conv_radius_sweep.json
Run:   # JAX env, CPU
    python -m graphcast_sae.storms.conv_radius_sweep_analyze   (CPU)
"""
import os, sys, json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import graphcast_sae.common.fs_common as fc
import graphcast_sae.storms.skill_conv_analyze as A
import graphcast_sae.common.skill_conv_storms as S
from scipy.stats import spearmanr
from graphcast_sae.common.signature_physics import gc_km

RADII = [500, 750, 1000, 1500, 2500]
REF = 2.794          # committed median conv-normal Delta-deepening at 1500 km
BASE = fc.ROOT / "results/skill/convection"
RI = [n for n, c in S.STORMS.items() if not c.get("nondev", False)]
NONDEV = [n for n, c in S.STORMS.items() if c.get("nondev", False)]

def load_radius(R):
    d = BASE if R == 1500 else fc.ROOT / f"results/skill/conv_r{R}"
    runs = {}
    for n in S.STORMS:
        p = d / f"run_{n}.npy"
        if p.exists():
            runs[n] = np.load(p, allow_pickle=True).item()
    return runs

def exposure(run1500, R):
    """Convection-group firing (+48 h baseline) inside R km, as a fraction of that inside the largest disk."""
    s = run1500["snap"]["baseline_mid"]
    conv = s["node_conv"].sum(1); lat = s["mlat"]; lon = s["mlon"]
    lon = np.where(lon > 180, lon - 360, lon)
    c = run1500["center"]
    d = gc_km(lat, lon, c[0], c[1])
    tot = conv[d < max(RADII)].sum()          # normalise to the largest disk in the sweep
    return float(conv[d < R].sum() / tot) if tot > 0 else float("nan")

def main():
    truth = np.load(BASE / "era5_truth.npy", allow_pickle=True).item()
    base_runs = load_radius(1500)
    out = {"radii_km": [], "per_radius": {}, "per_storm": {}}
    for R in RADII:
        runs = load_radius(R)
        if not runs:
            print(f"R={R}: no runs yet"); continue
        M = A.metrics(runs, truth)
        # determinism gate: baseline MSLP trace must match the committed battery
        det = {}
        for n in runs:
            if n in base_runs:
                det[n] = bool(np.array_equal(runs[n]["res"]["baseline"]["mslp_min"],
                                             base_runs[n]["res"]["baseline"]["mslp_min"]))
        def med(arm, key, names):
            v = [M[n]["arms"][arm][key] for n in names if n in M and arm in M[n]["arms"]]
            return float(np.median(v)) if v else float("nan"), len(v)
        cn, k = med("conv-normal", "d_deepen", RI)
        row = {"n_ri": k, "missing_ri": [n for n in RI if n not in M],
               "conv_normal_d_deepen": cn, "r": cn / REF,
               "conv_normal_d_err": med("conv-normal", "d_err", RI)[0],
               "conv_zero_d_deepen": med("conv-zero", "d_deepen", RI)[0],
               "rand_normal_d_deepen": med("rand-normal", "d_deepen", RI)[0],
               "rand_normal_abs_med": float(np.median([abs(M[n]["arms"]["rand-normal"]["d_deepen"])
                                                       for n in RI if n in M])) if k else float("nan"),
               "nondev_conv_normal": {n: M[n]["arms"]["conv-normal"]["d_deepen"] for n in NONDEV if n in M},
               "exposure_med": float(np.median([exposure(base_runs[n], R) for n in RI if n in base_runs])),
               "baseline_matches_committed": det,
               "disk_nodes": {n: runs[n]["disk_nodes"] for n in runs}}
        out["radii_km"].append(R); out["per_radius"][str(R)] = row
        for n in M:
            out["per_storm"].setdefault(n, {})[str(R)] = {
                a: {k2: M[n]["arms"][a].get(k2) for k2 in ("d_deepen", "d_err")}
                for a in M[n]["arms"] if a != "baseline"}
            out["per_storm"][n][str(R)]["exposure"] = exposure(base_runs[n], R) if n in base_runs else None
        print(f"R={R:5d} km  n={k}  conv-normal med dDeepen {cn:+.3f} (r={cn/REF:.2f})  "
              f"dErr {row['conv_normal_d_err']:+.2f}  rand {row['rand_normal_d_deepen']:+.3f}  "
              f"nondev {row['nondev_conv_normal']}  exposure {row['exposure_med']:.2f}  "
              f"baseline-det {all(det.values()) if det else 'n/a'}")

    # ---- frozen reading ----
    P = out["per_radius"]; rd = {int(R): P[R]["r"] for R in P}
    verdict = "INCOMPLETE"
    if all(R in rd for R in (1000, 1500, 2500)):
        if rd[1000] >= 0.80 and rd[2500] <= 1.25:
            verdict = "SATURATES"; scale = min(R for R in sorted(rd) if rd[R] >= 0.80)
            out["scale_km"] = scale
        elif rd[2500] >= 1.25:
            verdict = "STILL GROWING"
        elif rd[1000] < 0.80 and rd[2500] <= 1.25:
            verdict = "COLLAPSES INWARD"
    ctrl_ok = all(P[R]["rand_normal_abs_med"] < max(0.1 * abs(P[R]["conv_normal_d_deepen"]), 0.0)
                  or P[R]["rand_normal_abs_med"] < 0.3 for R in P)
    ctrl_ok = ctrl_ok and all(abs(v) < 0.3 for R in P for v in P[R]["nondev_conv_normal"].values())
    rs = sorted(rd)
    if len(rs) >= 4:
        rho = spearmanr([rd[R] for R in rs], [P[str(R)]["exposure_med"] for R in rs]).correlation
        out["r_vs_exposure_spearman"] = float(rho)
    out["verdict"] = verdict; out["controls_hold"] = bool(ctrl_ok)
    print(f"\nVERDICT {verdict}; controls hold: {ctrl_ok}; "
          f"r-vs-exposure rho {out.get('r_vs_exposure_spearman', float('nan')):.2f}")
    json.dump(out, open(fc.ROOT / "results/skill/conv_radius_sweep.json", "w"), indent=1, default=float)

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for n in RI:
        ys = [out["per_storm"].get(n, {}).get(str(R), {}).get("conv-normal", {}).get("d_deepen", np.nan) for R in rs]
        ax[0].plot(rs, ys, "-o", ms=3, lw=0.8, alpha=0.5, label=n)
    ax[0].plot(rs, [P[str(R)]["conv_normal_d_deepen"] for R in rs], "k-o", lw=2, label="median")
    ax[0].plot(rs, [P[str(R)]["rand_normal_d_deepen"] for R in rs], "g--s", lw=1.2, label="random ctrl")
    ax[0].axhline(REF, color="grey", ls=":", lw=0.8)
    ax[0].set_xlabel("disk radius (km)"); ax[0].set_ylabel("Δ-deepening, conv→normal (hPa)")
    ax[0].set_xscale("log"); ax[0].set_xticks(rs); ax[0].set_xticklabels(rs); ax[0].legend(fontsize=6, ncol=2)
    ax[1].plot([P[str(R)]["exposure_med"] for R in rs], [P[str(R)]["r"] for R in rs], "k-o")
    for R in rs:
        ax[1].annotate(f"{R}", (P[str(R)]["exposure_med"], P[str(R)]["r"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("median convection exposure inside disk (frac of 2500 km)"); ax[1].set_ylabel("r = Δ(R) / Δ(1500)")
    fig.suptitle(f"Convection restore-to-normal: disk-radius sweep — {verdict}", fontsize=10)
    fig.tight_layout(); fig.savefig(fc.ROOT / "figures/conv_radius_sweep.png")

if __name__ == "__main__":
    main()
