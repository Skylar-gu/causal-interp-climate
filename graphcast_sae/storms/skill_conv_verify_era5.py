"""CPU-only prep: verify each storm has a REAL deepening in ERA5 over IC..+96h.

For each storm we pull MSLP and 10m winds from the same WB2 zarr the model trains on,
in the storm-tracking box, at leads 0..+96h (every 6h), and report:
  - MSLP minimum trajectory (does it drop? by how much?)
  - max 10m wind trajectory (does it rise?)
A storm with real deepening (MSLP drop >~10 hPa toward peak) is IN; a flat one is a
non-developer. This is the ERA5-truth reference the skill comparison scores against.
No GPU: run with plain `python` in the JAX env (JAX stays on CPU).
Writes results/skill/$MECH_RES/era5_truth.npy (crash-safe per storm; default convection).

Paper: Sec. 3 'The intervention contrast' (Table tab:mechanism-interventions)
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: results/skill/<MECH_RES|convection>/era5_truth.npy (crash-safe per storm)
Run:   # JAX env, CPU
    python -m graphcast_sae.storms.skill_conv_verify_era5
"""
import os, sys, json

import numpy as np
import graphcast_sae.common.fs_common as fc
import importlib
S = importlib.import_module("graphcast_sae.common." + os.environ.get("SKILL_STORMS", "skill_conv_storms"))
norm_lon = S.norm_lon

OUT = fc.ROOT / f"results/skill/{os.environ.get('MECH_RES', 'convection')}"
OUT.mkdir(parents=True, exist_ok=True)
STEP = np.timedelta64(6, "h")

def box_series(ds, box, t0, nlead):
    """MSLP min & max 10m wind in box at leads 0..nlead*6h. Returns dict of arrays."""
    la0, la1 = box["lat"]; lo0, lo1 = box["lon"]
    lons = norm_lon([lo0, lo1])
    times = np.datetime64(t0) + np.arange(nlead + 1) * STEP
    sub = ds[["mean_sea_level_pressure", "10m_u_component_of_wind",
              "10m_v_component_of_wind"]].sel(time=times)
    # latitude ascending after open_wb2 reindex
    sub = sub.sel(lat=slice(la0, la1))
    # longitude box: handle wrap (all our boxes are contiguous in 0..360)
    if lons[0] <= lons[1]:
        sub = sub.sel(lon=slice(lons[0], lons[1]))
    else:
        sub = sub.sel(lon=(ds.lon >= lons[0]) | (ds.lon <= lons[1]))
    sub = sub.load()
    mslp = sub["mean_sea_level_pressure"].values / 100.0  # hPa
    u = sub["10m_u_component_of_wind"].values
    v = sub["10m_v_component_of_wind"].values
    wind = np.sqrt(u * u + v * v)
    mslp_min = np.nanmin(mslp.reshape(mslp.shape[0], -1), axis=1)
    wind_max = np.nanmax(wind.reshape(wind.shape[0], -1), axis=1)
    # locate MSLP min lat/lon at each lead for tracking sanity
    return dict(mslp_min=mslp_min, wind_max=wind_max, times=times.astype("datetime64[h]").astype(str))

def main():
    ds, _ = fc.open_wb2()
    print("zarr open. time range:", str(ds.time.values[0])[:10], "..", str(ds.time.values[-1])[:10], flush=True)
    truth = {}
    for name, cfg in S.STORMS.items():
        try:
            r = box_series(ds, cfg["box"], cfg["ic"], int(os.environ.get("MECH_H", S.H)))
        except Exception as e:
            print(f"{name}: ERROR {e}", flush=True); continue
        m = r["mslp_min"]; w = r["wind_max"]
        drop = float(m[0] - m.min()); ipk = int(np.argmin(m))
        wgain = float(w.max() - w[0])
        r["drop_hpa"] = drop; r["peak_lead_h"] = ipk * 6; r["wind_gain"] = wgain
        r["nondev"] = bool(cfg.get("nondev", False))
        verdict = "DEEPENS" if drop >= 10 else ("flat/nondev" if cfg.get("nondev") else "WEAK-DROP?")
        truth[name] = r
        print(f"\n{name} IC {cfg['ic']} box {cfg['box']}", flush=True)
        print(f"  MSLP min (hPa): {np.array2string(m, precision=1, max_line_width=200)}", flush=True)
        print(f"  wind max (m/s): {np.array2string(w, precision=1, max_line_width=200)}", flush=True)
        print(f"  -> deepening {drop:.1f} hPa by +{ipk*6}h, wind +{wgain:.1f} m/s  [{verdict}]", flush=True)
        np.save(OUT / "era5_truth.npy", truth, allow_pickle=True)
    # summary
    print("\n=== SUMMARY (IN if deepens >=10 hPa; control expected flat) ===", flush=True)
    for name, r in truth.items():
        tag = "control" if r["nondev"] else ("IN" if r["drop_hpa"] >= 10 else "OUT")
        print(f"  {name:14s} drop {r['drop_hpa']:5.1f} hPa  peak +{r['peak_lead_h']:>3d}h  wind+{r['wind_gain']:4.1f}  [{tag}]", flush=True)
    print(f"\n-> {(OUT / 'era5_truth.npy').relative_to(fc.ROOT)}", flush=True)

if __name__ == "__main__":
    main()
