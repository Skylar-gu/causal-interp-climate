"""Every TC amplification number the paper quotes, on min-MSLP trajectory RMSE against ERA5.

The revision prereg scored its bars on deepening error (rev_analyze `err_red`). The paper reports
the trajectory metric instead (rev_analyze `err_mslp` / `rmse_red`: RMSE of the box minimum MSLP
against ERA5 over leads up to max(ERA5 peak, +36 h), forecast step k aligned with analysis k+1).
This script re-reads the shipped verdicts and states each number on that metric, including the
pre-registered bar re-scored on it.

Floor. J4 measured the largest per-step difference between identical runs, 0.20 hPa
(`traj_max_hpa`); an arm's RMSE moves by at most that much, so a difference of two arms has a
floor of 2 x 0.20 = 0.40 hPa (the same doubling as the 0.25 hPa deepening floor).

Paper: Sec. 3.2 (group-level MSLP-error summary; input to offset_baseline)
Inputs: results/skill/{suff_ctl,sh_suff_ctl,dirs_j2,determinism_j4}/rev_verdict.json,
        results/skill/gain_3316/ (shipped)
Outputs: results/rmse_summary.json
Run:   # any env with numpy, CPU
    python -m graphcast_sae.storms.rmse_summary
"""
import json

import numpy as np

from graphcast_sae.paths import RESULTS

SK = RESULTS / "skill"
FLOOR = 2 * json.load(open(SK / "determinism_j4/rev_verdict.json"))["floor"]["traj_max_hpa"]
GAINS = ["0", "0.5", "1.25", "1.5", "2", "2.5", "3"]


def boot_ci(x, n=10000, seed=0):        # identical to rev_analyze.boot_ci
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return dict(median=float(np.median(x)), lo=float(np.percentile(meds, 2.5)),
                hi=float(np.percentile(meds, 97.5)), n=int(len(x)))


def sweep(run):
    M = json.load(open(SK / f"{run}/rev_verdict.json"))["metrics"]
    dev = [s for s in M if not M[s]["nondev"]]
    out = {"storms": dev, "by_gain": {}, "per_storm_g2": {}}
    for g in GAINS:
        red, spec = [], []
        for s in dev:
            a = M[s]["arms"]; b = a["baseline"]["err_mslp"]
            t, c = a[f"gain-{g}"]["err_mslp"], a[f"rand-gain-{g}"]["err_mslp"]
            red.append(b - t); spec.append((b - t) - (b - c))
        out["by_gain"][g] = dict(
            median_red=float(np.median(red)),
            improved=int(sum(r > FLOOR for r in red)), worsened=int(sum(r < -FLOOR for r in red)),
            beats_ctl=int(sum(x > FLOOR for x in spec)),
            ctl_max_abs=float(max(abs(M[s]["arms"]["baseline"]["err_mslp"] - M[s]["arms"][f"rand-gain-{g}"]["err_mslp"]) for s in dev)),
            per_storm={s: round(r, 3) for s, r in zip(dev, red)},
            spec={s: round(x, 3) for s, x in zip(dev, spec)})
    for s in dev:
        a = M[s]["arms"]; b = a["baseline"]["err_mslp"]
        out["per_storm_g2"][s] = dict(baseline=b, g2=a["gain-2"]["err_mslp"],
                                      pct=100 * (b - a["gain-2"]["err_mslp"]) / b)
    out["median_pct_g2"] = float(np.median([v["pct"] for v in out["per_storm_g2"].values()]))
    return out


def directions():
    V = json.load(open(SK / "dirs_j2/rev_verdict.json"))["metrics"]
    dev = [s for s in V if not V[s]["nondev"]]
    res = {}
    for g in ["1.5", "2", "3"]:
        tr, td = [], []
        for s in dev:
            a = V[s]["arms"]
            t = a[f"gain-{g}"]["rmse_red"]
            rr = [a[k]["rmse_red"] for k in a if k.startswith("dir-rand") and k.endswith(f"-{g}")]
            tr.append(t - float(np.median(rr)))
            tr_d = a.get(f"dir-dom-{g}", {}).get("rmse_red")
            td.append(None if tr_d is None else t - tr_d)
        res[g] = dict(t_minus_rand=boot_ci(tr), t_minus_dom=boot_ci(td))
    return res


def spin():
    T = np.load(SK / "gain_3316/era5_truth.npy", allow_pickle=True).item()
    out = {}
    for s in ["ida2021", "haishen2020", "patricia2015"]:
        res = np.load(SK / f"gain_3316/run_{s}.npy", allow_pickle=True).item()["res"]
        t = np.asarray(T[s]["mslp_min"])

        def rmse(arm):   # rev_analyze.arm_metrics err_mslp
            mm = np.asarray(res[arm]["mslp_min"]); H = len(mm)
            w = np.arange(min(max(int(np.argmin(t)), 6), H))
            return float(np.sqrt(np.mean((mm[w] - t[1:H + 1][w]) ** 2)))
        arms = {a: rmse(a) for a in res}
        out[s] = dict(rmse=arms, pct_g3=100 * (arms["baseline"] - arms["gain-3"]) / arms["baseline"],
                      ctl_g3=arms["baseline"] - arms["rand-gain-3"])
    return out


if __name__ == "__main__":
    out = dict(floor=FLOOR, nh=sweep("suff_ctl"), sh=sweep("sh_suff_ctl"), j2=directions(), spin=spin())
    json.dump(out, open(RESULTS / "rmse_summary.json", "w"), indent=1)
    print("floor", round(FLOOR, 3))
    for k in ("nh", "sh"):
        o = out[k]
        print(f"\n{k}: gain  median_red  improved/worsened  beats_ctl  ctl_max")
        for g, v in o["by_gain"].items():
            print(f"   {g:>4}  {v['median_red']:+6.2f}   {v['improved']}/{v['worsened']}   {v['beats_ctl']}   {v['ctl_max_abs']:.2f}")
        print("   g2 per storm %:", {s: round(v["pct"], 1) for s, v in o["per_storm_g2"].items()},
              "median", round(o["median_pct_g2"], 1))
        print("   g2 spec:", o["by_gain"]["2"]["spec"])
    print("\nJ2:", json.dumps({g: {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in d.items()}
                              for g, d in out["j2"].items()}))
    print("spin:", {s: (round(v["rmse"]["baseline"], 2), round(v["rmse"]["gain-3"], 2), round(v["pct_g3"], 1),
                        round(v["ctl_g3"], 2)) for s, v in out["spin"].items()})
