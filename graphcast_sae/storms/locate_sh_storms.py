"""Locate the Southern-Hemisphere candidates in ERA5 and gate them on a closed deepening low.

Selection criteria and the bar are frozen in docs/prereg/prereg_ps5_southern.md, written and
committed BEFORE this script ran. This does not compute any outcome quantity -- no
displacement, no angle, no ablation effect. It reads MSLP only, to (a) fix the storm centre
and tracking box from data rather than from recall, and (b) reject any candidate ERA5 does
not show as a closed deepening minimum inside the window. That is a data gate on the input.

Paper: not in the paper (Southern-Hemisphere battery)
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: results/sh_storm_gate.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.storms.locate_sh_storms
"""
import os
import sys

import numpy as np

import graphcast_sae.common.fs_common as fc

STEP = np.timedelta64(6, "h")
NLEAD = 17                                  # IC .. +96 h

# wide BASIN boxes, deliberately loose: the point is to let ERA5 place the storm, not to
# place it myself. lat is negative (SH); lon in 0..360 to match the ERA5 grid.
CAND = {
    "winston2016":  ("2016-02-16", "spac", (-25, -8), (170, 200)),
    "harold2020":   ("2020-04-01", "spac", (-25, -8), (150, 180)),
    "fantala2016":  ("2016-04-14", "sind", (-25, -8), (45, 75)),
    "ambali2019":   ("2019-12-03", "sind", (-25, -8), (45, 75)),
    "marcus2018":   ("2018-03-17", "aus",  (-25, -8), (110, 140)),
    "veronica2019": ("2019-03-19", "aus",  (-25, -8), (105, 135)),
}
BT = dict(winston2016=884, harold2020=920, fantala2016=910,
          ambali2019=916, marcus2018=905, veronica2019=928)

def main():
    ds, _ = fc.open_wb2()
    lat = np.asarray(ds.lat.values, float)
    lon = np.asarray(ds.lon.values, float)
    print(f"{'storm':<14}{'basin':>6}{'IC lat':>8}{'IC lon':>8}{'IC mslp':>9}"
          f"{'min mslp':>10}{'at lead':>9}{'deepen':>8}{'drift km':>10}  gate")
    keep = {}
    for k, (ic, basin, blat, blon) in CAND.items():
        t = np.datetime64(ic) + np.arange(NLEAD) * STEP
        sub = ds["mean_sea_level_pressure"].sel(
            time=t, lat=slice(blat[0], blat[1]), lon=slice(blon[0], blon[1])).load()
        v = np.asarray(sub.values, float) / 100.0          # Pa -> hPa
        la = np.asarray(sub.lat.values, float)
        lo = np.asarray(sub.lon.values, float)
        tr = []
        for i in range(NLEAD):
            j = np.unravel_index(np.nanargmin(v[i]), v[i].shape)
            tr.append((la[j[0]], lo[j[1]], v[i][j]))
        tr = np.array(tr)
        imin = int(np.nanargmin(tr[:, 2]))
        deep = tr[0, 2] - tr[imin, 2]
        # drift of the located minimum: a genuine TC translates; a stationary artifact
        # (a monsoon trough minimum pinned to a coastline) does not.
        d = np.hypot((tr[imin, 0] - tr[0, 0]) * 111.0,
                     (tr[imin, 1] - tr[0, 1]) * 111.0 * np.cos(np.deg2rad(tr[0, 0])))
        # GATE: a closed deepening low that translates, with the centre inside 8-25 S at IC.
        ok = (deep >= 5.0) and (-25 <= tr[0, 0] <= -8) and (d > 100)
        # AND the minimum must not sit on the box edge, which would mean the real centre is
        # outside the box and the "minimum" is a boundary artifact.
        edge = (abs(tr[imin, 0] - blat[0]) < 0.3 or abs(tr[imin, 0] - blat[1]) < 0.3 or
                abs(tr[imin, 1] - blon[0]) < 0.3 or abs(tr[imin, 1] - blon[1]) < 0.3)
        ok = ok and not edge
        print(f"{k:<14}{basin:>6}{tr[0,0]:>8.1f}{tr[0,1]:>8.1f}{tr[0,2]:>9.1f}"
              f"{tr[imin,2]:>10.1f}{6*imin:>7}h{deep:>8.1f}{d:>10.0f}  "
              f"{'PASS' if ok else 'REJECT'}{' (edge)' if edge else ''}")
        if ok:
            keep[k] = dict(ic=ic, basin=basin, center=(float(tr[0, 0]), float(tr[0, 1])),
                           track=tr.tolist(), bt=BT[k], deepen=float(deep))
    print(f"\n{len(keep)} of {len(CAND)} pass the data gate: {sorted(keep)}")
    print("\nERA5 deepening vs agency best track (ERA5 is 40-60 hPa too shallow for TCs, so")
    print("this is a sanity check on the LOW, not an intensity verification):")
    for k, v in keep.items():
        print(f"  {k:<14} ERA5 min {min(r[2] for r in v['track']):>7.1f}   "
              f"best track {v['bt']:>4}   gap {min(r[2] for r in v['track'])-v['bt']:>+6.1f}")
    np.save(fc.ROOT / "results/sh_storm_gate.npy", keep, allow_pickle=True)
    print("\n-> results/sh_storm_gate.npy")

if __name__ == "__main__":
    main()
