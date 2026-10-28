"""Post-hoc bias-correction baseline for the amplification skill claim (camera-ready).

For each storm, add to the unmodified forecast's min-MSLP trajectory the mean ERA5-minus-forecast
gap at each lead, fitted on the OTHER six storms (leave-one-out), and score it with the paper's
MSLP error (rev_analyze.arm_metrics: forecast step k vs analysis k+1, leads up to
max(ERA5 peak, 6 steps)). No model run; reads results/skill/convection.
Also scores the convection group at a gain chosen leave-one-out (the gain with the best median
MSLP-error reduction on the other six storms), from results/rmse_summary.json.

Run:   python -m graphcast_sae.storms.offset_baseline
Outputs: results/skill/offset_baseline.json
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/skill/convection"
STORMS = ["ida2021", "michael2018", "haishen2020", "goni2020", "haiyan2013", "patricia2015", "wilma2005"]


def main():
    T = np.load(RES / "era5_truth.npy", allow_pickle=True).item()
    D = {}
    for s in STORMS:
        mm = np.asarray(np.load(RES / f"run_{s}.npy", allow_pickle=True).item()["res"]["baseline"]["mslp_min"])
        t = np.asarray(T[s]["mslp_min"]); H = len(mm)
        D[s] = (mm, t[1:H + 1], np.arange(min(max(int(np.argmin(t)), 6), H)))
    err = lambda s, m: float(np.sqrt(np.mean((m[D[s][2]] - D[s][1][D[s][2]]) ** 2)))
    out = {}
    for s in STORMS:
        off = np.mean([D[x][1] - D[x][0] for x in STORMS if x != s], axis=0)
        out[s] = dict(unmodified=err(s, D[s][0]), offset=err(s, D[s][0] + off))
    nh = json.load(open(ROOT / "results/rmse_summary.json"))["nh"]
    red = {g: nh["by_gain"][g]["per_storm"] for g in nh["by_gain"] if float(g) >= 1}
    for s in STORMS:
        g = max(red, key=lambda g: np.median([red[g][x] for x in STORMS if x != s]))
        b = nh["per_storm_g2"][s]["baseline"]
        out[s].update(loo_gain=float(g), loo_unmodified=b, loo_sae=b - red[g][s])
    out["median"] = {k: float(np.median([out[s][k] for s in STORMS])) for k in ("unmodified", "offset", "loo_sae")}
    (ROOT / "results/skill/offset_baseline.json").write_text(json.dumps(out, indent=1))
    for s, v in out.items(): print(s, {k: round(x, 2) for k, x in v.items()})


if __name__ == "__main__":
    main()
