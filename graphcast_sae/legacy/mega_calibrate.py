"""Calibrate the warm-core bar on BOTH sides before it is used as a gate (the control-must-be-able-to-fail rule).

(i)  It must PASS the storms already in the battery -- 8 NH + 5 SH tropical cyclones.
(ii) It must FAIL a negative control -- the explosive EXTRATROPICAL cyclones already
     screened in results/event_screen.json, which are cold-core by construction.
A bar that cannot fail is vacuous; a bar that kills the positive controls is broken.

Paper: not in the paper; kept for provenance only
Inputs: none beyond the arguments above
Outputs: printed report
Run:   # JAX env, CPU
    python -m graphcast_sae.legacy.mega_calibrate
"""
import importlib
import sys

import numpy as np

from graphcast_sae.legacy.mega_sweep import STEP, open_zarr, to180, warm_core

# negative controls: extratropical bombs (name, IC, approximate centre lat/lon in -180..180)
ETC = [
    ("dennis2020", "2020-02-15", 57.0, -20.0),
    ("ciara2020", "2020-02-09", 57.0, -15.0),
    ("eastcoast2018", "2018-01-04", 38.0, -71.0),
    ("greatlakes2010", "2010-10-26", 45.0, -92.0),
    ("alex2020", "2020-10-02", 48.0, -20.0),
]

def track_peak(ds, box, ic, norm_lon):
    lo = np.asarray(norm_lon(np.asarray(box["lon"], float)), float)
    t = np.datetime64(ic) + np.arange(17) * STEP
    sub = ds["mean_sea_level_pressure"].sel(time=t, lat=slice(*box["lat"]))
    sub = (sub.sel(lon=slice(lo[0], lo[1])) if lo[0] <= lo[1]
           else sub.sel(lon=(ds.lon >= lo[0]) | (ds.lon <= lo[1])))
    sub = sub.load()
    v = np.asarray(sub.values, float) / 100.0
    la = np.asarray(sub.lat.values, float)
    ln = np.asarray(sub.lon.values, float)
    i = int(np.nanargmin(v.reshape(v.shape[0], -1).min(axis=1)))
    j = np.unravel_index(np.nanargmin(v[i]), v[i].shape)
    return float(la[j[0]]), float(ln[j[1]]), str(t[i]), float(v[i][j])

def main():
    ds = open_zarr()
    print(f"{'case':<16}{'kind':<6}{'lat':>7}{'peak mslp':>11}{'warm core m':>13}")
    vals = {"tc": [], "etc": []}
    for mod in ("skill_conv_storms", "skill_sh_storms"):
        S = importlib.import_module("graphcast_sae.common." + mod)
        for name, cfg in S.STORMS.items():
            la, lo, tt, p = track_peak(ds, cfg["box"], cfg["ic"], S.norm_lon)
            w = warm_core(ds, la, lo, tt)
            kind = "nondev" if cfg.get("nondev") else "tc"
            vals.setdefault(kind, []).append(w)
            print(f"{name:<16}{kind:<6}{la:>7.1f}{p:>11.1f}{w:>13.1f}")
    for name, ic, la0, lo0 in ETC:
        t = np.datetime64(ic) + np.arange(9) * STEP
        sub = ds["mean_sea_level_pressure"].sel(
            time=t, lat=slice(la0 - 8, la0 + 8),
            lon=slice((lo0 - 10) % 360, (lo0 + 10) % 360)).load()
        v = np.asarray(sub.values, float) / 100.0
        i = int(np.nanargmin(v.reshape(v.shape[0], -1).min(axis=1)))
        j = np.unravel_index(np.nanargmin(v[i]), v[i].shape)
        la, lo = float(sub.lat.values[j[0]]), float(sub.lon.values[j[1]])
        w = warm_core(ds, la, lo, str(t[i]))
        vals["etc"].append(w)
        print(f"{name:<16}{'etc':<6}{la:>7.1f}{v[i][j]:>11.1f}{w:>13.1f}")
    print(f"\ntropical cyclones: min {min(vals['tc']):.1f}  median "
          f"{np.median(vals['tc']):.1f}  max {max(vals['tc']):.1f}   (n={len(vals['tc'])})")
    print(f"extratropical    : min {min(vals['etc']):.1f}  median "
          f"{np.median(vals['etc']):.1f}  max {max(vals['etc']):.1f}   (n={len(vals['etc'])})")
    if "nondev" in vals and vals["nondev"]:
        print(f"non-developer    : {vals['nondev']}")

if __name__ == "__main__":
    main()
