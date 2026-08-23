"""Side-by-side: GraphCast vs IFS HRES vs ERA5 on the storm-box MSLP minimum.

Reads the WeatherBench2 operational HRES archive (0.25 deg, inits 00/12 UTC,
2016-2022) and scores it with the SAME box-minimum statistic the convection
experiment already applies to GraphCast, so the two are directly comparable.

    hres/2016-2022-0012-1440x721.zarr   time = INIT, prediction_timedelta = LEAD

WHAT IS AND IS NOT FAIR HERE

- Both models are scored inside the same storm box, over the same +0..96 h window,
  from the same calendar IC, by the same statistic. That part is like-for-like.
- They are NOT initialised from the same analysis. GraphCast starts from ERA5
  HRES starts from ECMWF's own 9 km analysis, which resolves a tropical cyclone far
  better than 0.25 deg ERA5 does. Some of HRES's advantage at lead 0 is inherited,
  not forecast, which is why DEEPENING (IC -> minimum, each model from its own t=0)
  is reported next to the absolute minimum. Deepening is the fairer column.
- Verifying HRES against ERA5 is what WeatherBench2 itself does, but ERA5 is not
  ground truth for TC intensity -- it is 40-60 hPa too shallow for these storms.
  Best-track minima are printed alongside so the reanalysis gap stays visible.
- Four of the seven storms predate the HRES archive (2016) and are omitted.

Paper: Appendix app:taxonomy (GraphCast vs IFS-HRES on the storm box)
Inputs: results/skill/convection/era5_truth.npy (shipped); results/skill/convection/verdict.json (shipped); WeatherBench2 ERA5 (GCS, streamed)
Outputs: results/hres_compare.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.storms.hres_compare
"""
import json
import os
import sys

import numpy as np

import graphcast_sae.common.fs_common as fc
import graphcast_sae.common.skill_conv_storms as S

HRES = "weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
STEP = np.timedelta64(6, "h")
NLEAD = 16                      # +96 h
# agency best-track minimum central pressure (NHC TCR / JMA), reference values
BT = {"ida2021": 929, "michael2018": 919, "haishen2020": 910, "goni2020": 905}

def box_min(da, box):
    la0, la1 = box["lat"]
    lo = S.norm_lon(box["lon"])
    d = da.sel(lat=slice(la0, la1))
    d = d.sel(lon=slice(lo[0], lo[1])) if lo[0] <= lo[1] else d
    v = d.values / 100.0
    return np.nanmin(v.reshape(v.shape[0], -1), axis=1)

def main():
    import gcsfs
    import xarray as xr
    fs = gcsfs.GCSFileSystem(token="anon")
    hs = xr.open_zarr(fs.get_mapper(HRES), consolidated=True)
    hs = hs.rename({"latitude": "lat", "longitude": "lon"})
    if hs.lat[0] > hs.lat[-1]:
        hs = hs.reindex(lat=hs.lat[::-1])

    verdict = json.load(open(fc.ROOT / "results/skill/convection/verdict.json"))
    truth = np.load(fc.ROOT / "results/skill/convection/era5_truth.npy",
                    allow_pickle=True).item()

    rows = []
    for name in BT:
        cfg = S.STORMS[name]
        t0 = np.datetime64(cfg["ic"] + "T00")
        if t0 not in hs.time.values:
            print(f"{name}: IC not in HRES archive, skip"); continue
        sub = hs["mean_sea_level_pressure"].sel(
            time=t0, prediction_timedelta=slice(np.timedelta64(0, "h"),
                                                np.timedelta64(6 * NLEAD, "h"))).load()
        h = box_min(sub, cfg["box"])

        m = verdict["metrics"][name]
        era = truth[name]["mslp_min"][:NLEAD + 1]
        gc = np.asarray(m["arms"]["baseline"]["deepen"])           # scalar
        gc_min = m["ic_mslp"] - float(gc)
        # intensity RMSE against ERA5 over the intensification window, as analyze does
        wmax = max(m["era_peak_lead_h"] // 6, 6)
        idx = np.arange(min(wmax, NLEAD))
        h_err = float(np.sqrt(np.mean((h[1:NLEAD + 1][idx] - era[1:NLEAD + 1][idx]) ** 2)))
        rows.append(dict(
            name=name, bt=BT[name], era_ic=float(era[0]), era_min=float(np.min(era)),
            era_dp=float(era[0] - np.min(era)),
            gc_min=gc_min, gc_dp=float(gc), gc_err=m["arms"]["baseline"]["err_mslp"],
            h_ic=float(h[0]), h_min=float(np.min(h)), h_dp=float(h[0] - np.min(h)),
            h_err=h_err))

    print("\nGraphCast vs IFS HRES vs ERA5 — storm-box MSLP minimum, IC to +96 h\n")
    print(f"{'storm':<13}{'best':>6} | {'ERA5 min':>9}{'HRES min':>9}{'GC min':>8} | "
          f"{'ERA5 dp':>8}{'HRES dp':>8}{'GC dp':>7} | {'HRES err':>9}{'GC err':>8}")
    for r in rows:
        print(f"{r['name']:<13}{r['bt']:>6.0f} | {r['era_min']:>9.1f}{r['h_min']:>9.1f}"
              f"{r['gc_min']:>8.1f} | {r['era_dp']:>8.1f}{r['h_dp']:>8.1f}{r['gc_dp']:>7.1f}"
              f" | {r['h_err']:>9.1f}{r['gc_err']:>8.1f}")
    a = lambda k: np.median([r[k] for r in rows])
    print(f"{'median':<13}{a('bt'):>6.0f} | {a('era_min'):>9.1f}{a('h_min'):>9.1f}"
          f"{a('gc_min'):>8.1f} | {a('era_dp'):>8.1f}{a('h_dp'):>8.1f}{a('gc_dp'):>7.1f}"
          f" | {a('h_err'):>9.1f}{a('gc_err'):>8.1f}")

    print("\nshare of REAL (best-track) deepening captured, each from its own IC:")
    for r in rows:
        real_e = r["era_ic"] - r["bt"]; real_h = r["h_ic"] - r["bt"]
        print(f"  {r['name']:<13} ERA5 {100*r['era_dp']/real_e:>4.0f}%   "
              f"HRES {100*r['h_dp']/real_h:>4.0f}%   GC {100*r['gc_dp']/real_e:>4.0f}%")
    np.save(fc.ROOT / "results/hres_compare.npy", rows, allow_pickle=True)
    print("\n-> results/hres_compare.npy")

if __name__ == "__main__":
    main()
