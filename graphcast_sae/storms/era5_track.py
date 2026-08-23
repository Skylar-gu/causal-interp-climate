"""CPU-only: the ERA5 TRACK and MSLP field for each storm, which era5_truth.npy lacks.

`skill_conv_verify_era5.py` stores only the box minimum of MSLP per lead -- an intensity
series with no position. That is enough to score deepening and not enough to answer the
question the scalar readout hides: when an intervention changes the forecast, does it move
the storm's INTENSITY or its POSITION? Those are the two axes operational centres verify
separately, and data-driven models are known to behave very differently on them.

So this pulls, per storm and per lead over the same IC..+96 h window and the same box:
    mslp      the field itself (hPa), for the propagation maps
    clat/clon argmin-MSLP centre, the same estimator the model side uses in
              skill_conv_run.box_fields, so model and truth tracks are comparable
Writes results/skill/era5_track.npy. No GPU:

Paper: figures/paper_fig_track.py (ERA5 track per storm)
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: results/skill/era5_track.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.storms.era5_track
"""
import os
import sys

import numpy as np
import graphcast_sae.common.fs_common as fc
import importlib

S = importlib.import_module("graphcast_sae.common." + os.environ.get("SKILL_STORMS", "skill_conv_storms"))
norm_lon = S.norm_lon
OUT = fc.ROOT / "results/skill/era5_track.npy"
STEP = np.timedelta64(6, "h")

def box_fields(ds, box, t0, nlead):
    la0, la1 = box["lat"]
    lons = norm_lon(box["lon"])
    times = np.datetime64(t0) + np.arange(nlead + 1) * STEP
    sub = ds[["mean_sea_level_pressure"]].sel(time=times).sel(lat=slice(la0, la1))
    if lons[0] <= lons[1]:
        sub = sub.sel(lon=slice(lons[0], lons[1]))
    else:
        sub = sub.sel(lon=(ds.lon >= lons[0]) | (ds.lon <= lons[1]))
    sub = sub.load()
    mslp = np.asarray(sub["mean_sea_level_pressure"].values, np.float32) / 100.0
    lat = np.asarray(sub.lat.values, np.float32)
    lon = np.asarray(sub.lon.values, np.float32)
    clat = np.empty(mslp.shape[0], np.float32)
    clon = np.empty(mslp.shape[0], np.float32)
    for k in range(mslp.shape[0]):
        j, i = np.unravel_index(int(np.nanargmin(mslp[k])), mslp[k].shape)
        clat[k], clon[k] = lat[j], lon[i]
    return dict(mslp=mslp, grid_lat=lat, grid_lon=lon, clat=clat, clon=clon,
                times=times.astype("datetime64[h]").astype(str))

def main():
    ds, _ = fc.open_wb2()
    print("zarr open", flush=True)
    out = {}
    for name, cfg in S.STORMS.items():
        try:
            r = box_fields(ds, cfg["box"], cfg["ic"], S.H)
        except Exception as e:
            print(f"{name}: ERROR {e}", flush=True)
            continue
        out[name] = r
        d = np.sqrt((np.diff(r["clat"]) * 111.0) ** 2 +
                    (np.diff(r["clon"]) * 111.0 * np.cos(np.radians(r["clat"][:-1]))) ** 2)
        print(f"{name}: grid {r['mslp'].shape}  centre "
              f"({r['clat'][0]:.1f},{r['clon'][0]:.1f}) -> ({r['clat'][-1]:.1f},{r['clon'][-1]:.1f})  "
              f"median 6-h step {np.median(d):.0f} km", flush=True)
        np.save(OUT, out, allow_pickle=True)
    print("->", OUT, flush=True)

if __name__ == "__main__":
    main()
