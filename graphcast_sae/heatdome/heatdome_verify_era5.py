"""CPU prep: verify the 2021 PNW heat-dome is a real ridge + heat event in ERA5, and
build the ERA5-truth reference the skill comparison scores against.

For IC..+Hh (every 6h) over the W-NA box we pull, from the same WB2 zarr the model
trains on:
  - geopotential @500 hPa -> geopotential height (m); zonal anomaly (vs full-circle
    zonal mean at each latitude); ridge metric = MAX zonal anomaly over the box.
  - 2m_temperature -> MAX over the box (the record heat).
  - full box fields of z500 (gpm) and 2m-T, saved per lead, for the skill RMSE and maps.
Also locates the ridge centre (z500-anomaly-max lat/lon) averaged over the peak window
-> the disk centre for the local counterfactual (Phase 2 reads it).

No GPU. Writes results/heatdome/era5_truth.npy.

Paper: not in the paper (2021 heat-dome study; results shipped, demo notebook)
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: results/heatdome
Run:   # JAX env, CPU
    python -m graphcast_sae.heatdome.heatdome_verify_era5
"""
import os, sys

import numpy as np
import graphcast_sae.common.fs_common as fc
import graphcast_sae.heatdome.heatdome_config as C

OUT = fc.ROOT / "results/heatdome"; OUT.mkdir(parents=True, exist_ok=True)
STEP = np.timedelta64(6, "h")

def main():
    ds, _ = fc.open_wb2()
    print("zarr open. time range:", str(ds.time.values[0])[:10], "..",
          str(ds.time.values[-1])[:10], flush=True)
    la0, la1 = C.BOX["lat"]; lo = C.norm_lon(C.BOX["lon"])
    times = np.datetime64(C.IC) + np.arange(C.H + 1) * STEP

    sub = ds[["geopotential", "2m_temperature"]].sel(time=times, level=500)
    # zonal mean uses the FULL longitude circle at each latitude, over the box latitudes
    zfull = sub["geopotential"].sel(lat=slice(la0, la1)) / C.G           # gpm, all lon
    zonal = zfull.mean("lon")                                            # (time,lat)
    zanom_full = zfull - zonal                                          # (time,lat,lon) anomaly

    # box selection (contiguous in 0..360)
    def boxsel(da):
        da = da.sel(lat=slice(la0, la1))
        return da.sel(lon=slice(lo[0], lo[1])) if lo[0] <= lo[1] else \
               da.sel(lon=(ds.lon >= lo[0]) | (ds.lon <= lo[1]))

    zanom_box = boxsel(zanom_full).load()                               # (time,lat,lon)
    z_box = boxsel(zfull).load()                                        # gpm absolute in box
    t2m_box = boxsel(sub["2m_temperature"]).load()                      # K

    ridge = zanom_box.max(dim=("lat", "lon")).values                    # ridge metric per lead
    t2max = t2m_box.max(dim=("lat", "lon")).values - 273.15             # deg C per lead

    # ridge centre: argmax of zonal anomaly averaged over the peak window
    h0, h1 = C.PEAK_WINDOW_H
    pk = (np.arange(C.H + 1) * 6 >= h0) & (np.arange(C.H + 1) * 6 <= h1)
    zpk = zanom_box.isel(time=pk).mean("time")
    j, i = np.unravel_index(int(np.argmax(zpk.values)), zpk.shape)
    clat = float(zpk.lat.values[j]); clon = float(zpk.lon.values[i])
    clon180 = clon - 360 if clon > 180 else clon

    truth = dict(
        ic=C.IC, box=C.BOX, times=times.astype("datetime64[h]").astype(str),
        leads_h=(np.arange(C.H + 1) * 6),
        ridge_zanom_max=ridge, t2m_max_C=t2max,
        z500_box=z_box.values.astype(np.float32), t2m_box=t2m_box.values.astype(np.float32),
        box_lat=z_box.lat.values, box_lon=z_box.lon.values,
        ridge_center=(clat, clon180))
    np.save(OUT / "era5_truth.npy", truth, allow_pickle=True)

    print(f"\n2021 PNW heat-dome, IC {C.IC}, box {C.BOX}", flush=True)
    print("  leads (h):        ", np.array2string(np.arange(C.H+1)*6, max_line_width=200), flush=True)
    print("  z500 ridge anom(m):", np.array2string(ridge, precision=0, max_line_width=200), flush=True)
    print("  2m-T max (C):      ", np.array2string(t2max, precision=1, max_line_width=200), flush=True)
    print(f"\n  peak ridge anomaly {ridge.max():.0f} m at +{int(np.argmax(ridge)*6)}h", flush=True)
    print(f"  peak box 2m-T max  {t2max.max():.1f} C at +{int(np.argmax(t2max)*6)}h", flush=True)
    print(f"  ridge centre (peak window): ({clat:.1f}N, {clon180:.1f}E)", flush=True)
    verdict = "REAL RIDGE+HEAT" if (ridge.max() >= 150 and t2max.max() >= 35) else "WEAK?"
    print(f"  -> [{verdict}]  (expect a big ridge ~200+ m and box heat >40 C)", flush=True)
    print("\n-> results/heatdome/era5_truth.npy", flush=True)

if __name__ == "__main__":
    main()
