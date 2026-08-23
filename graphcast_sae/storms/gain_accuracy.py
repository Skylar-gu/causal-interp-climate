"""Does AMPLIFYING a feature group make the forecast more ACCURATE, not just deeper?

Motivation. GraphCast under-deepens these storms badly: it reaches a median 50% of
ERA5's box-minimum deepening and ~13% of the real (best-track) deepening, and it
flatlines exactly when a storm enters rapid intensification. That is a KNOWN,
SIGNED bias -- always too shallow, never too deep. A signed bias is the one case
where a causal handle can be pointed at accuracy rather than only at mechanism: if
convection features are what produce deepening, amplifying them should move the
forecast toward the truth, and there should be an optimum gain beyond which it
overshoots.

Two references, deliberately both:
  ERA5      the field's standard verification target, and what GraphCast was trained
            to reproduce. Improving against ERA5 is improvement by the metric the
            literature actually uses.
  BEST TRACK the real storm. ERA5 is 40-60 hPa too shallow for these cores, so it is
            not ground truth for intensity -- but it IS ground truth for the resolved
            state, so a model can be right about ERA5 and wrong about the storm.
            Reporting only one of these would hide exactly that distinction.

RMSE-vs-ERA5 is computed over the same intensification window skill_conv_analyze
uses, so the numbers are comparable to the committed d_err column.

Paper: Fig. fig:gain (dose-response of the convection intervention)
Inputs: none beyond the arguments above
Outputs: results/gain_accuracy.npy
Run:   # demo env (numpy/scipy/matplotlib)
    python -m graphcast_sae.storms.gain_accuracy
"""
import json
import os
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "skill"
# agency best-track minimum central pressure (NHC TCR / JMA). Reference values.
# Atlantic/E-Pac figures are aircraft-measured; W-Pac are satellite estimates and
# agencies differ, so haishen carries real uncertainty.
BT = {"ida2021": 929.0, "haishen2020": 910.0, "patricia2015": 872.0}
ARMS = ["gain_conv", "gain_asc17", "gain_asc09", "gain_moist2"]
SIG = {"gain_conv": 28.5, "gain_asc17": 16.8, "gain_asc09": 9.0, "gain_moist2": 2.1}

def main():
    verdict = json.load(open(RES / "convection" / "verdict.json"))
    truth = np.load(RES / "convection" / "era5_truth.npy", allow_pickle=True).item()

    out = {}
    for arm in ARMS:
        d = RES / arm
        if not d.exists():
            continue
        for storm in BT:
            p = d / f"run_{storm}.npy"
            if not p.exists():
                continue
            r = np.load(p, allow_pickle=True).item()
            m = verdict["metrics"][storm]
            ic = m["ic_mslp"]
            era = truth[storm]["mslp_min"]
            H = len(r["res"]["baseline"]["mslp_min"])
            era_al = era[1:H + 1]
            widx = np.arange(min(max(m["era_peak_lead_h"] // 6, 6), H))
            rows = []
            for aname, a in r["res"].items():
                mm = np.asarray(a["mslp_min"])
                g = (1.0 if aname == "baseline" else
                     float(re.fullmatch(r"gain-([0-9.]+)", aname).group(1))
                     if re.fullmatch(r"gain-([0-9.]+)", aname) else None)
                if g is None:
                    continue
                rows.append(dict(
                    g=g, mn=float(np.min(mm)),
                    err_era=float(np.sqrt(np.mean((mm[widx] - era_al[widx]) ** 2))),
                    err_bt=float(abs(np.min(mm) - BT[storm]))))
            rows.sort(key=lambda x: x["g"])
            out.setdefault(arm, {})[storm] = dict(rows=rows, ic=ic,
                                                  era_min=float(np.min(era)), bt=BT[storm])

    for arm, storms in out.items():
        print(f"\n{'='*76}\n{arm}   (calibrated ascent {SIG[arm]:+.1f} sigma)")
        for storm, s in storms.items():
            base = [x for x in s["rows"] if x["g"] == 1.0][0]
            print(f"\n  {storm}:  ERA5 min {s['era_min']:.1f}   best track {s['bt']:.0f}"
                  f"   GraphCast baseline {base['mn']:.1f}")
            print(f"    {'gain':>6}{'GC min':>9}{'vs ERA5 RMSE':>14}{'vs best track':>15}")
            for x in s["rows"]:
                tag = "  <- baseline" if x["g"] == 1.0 else (
                      "  <- ablate" if x["g"] == 0.0 else "")
                d_era = x["err_era"] - base["err_era"]
                d_bt = x["err_bt"] - base["err_bt"]
                print(f"    {x['g']:>6.2f}{x['mn']:>9.1f}"
                      f"{x['err_era']:>9.2f} ({d_era:+5.2f}){x['err_bt']:>9.1f} ({d_bt:+6.1f})"
                      f"{tag}")
    np.save(ROOT / "results" / "gain_accuracy.npy", out, allow_pickle=True)
    print(f"\n-> results/gain_accuracy.npy")

if __name__ == "__main__":
    main()
