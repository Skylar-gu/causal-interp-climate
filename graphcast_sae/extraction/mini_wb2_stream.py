"""Stream ERA5 windows from WeatherBench 2 in graphcast_small's exact input format.

Data source (public GCS, anonymous, no landing — design §2 "stream from WB2"):
    gs://weatherbench2/datasets/era5/
        1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr
This is native 1 deg (360x181), 6-hourly, the 13 GraphCast pressure levels, 1959-2023.

Every physical GraphCast input/target var is present under the same name. Two
adaptations, both validated against the local 2022-01-01 sample:
  * coords are latitude/longitude -> renamed lat/lon
  * `toa_incident_solar_radiation` is NOT stored (its ERA5 flux var is all-NaN in
    this product), so it is computed analytically with graphcast's own
    `solar_radiation` using integration_period="1h" -- reproduces the sample TISR
    exactly (ratio 1.000, corr 1.000000, relMAE 0.0000).

A "window" is 3 consecutive 6-h steps (time rebased to [0,6,12]h): inputs at
-6h,0h and target at +6h -- one teacher-forced step, matching extract_layer8.py.
The 4 progress forcings (year/day sin/cos) are derived downstream by
`data_utils` from the absolute `datetime` coord, so they are not built here.

Paper: graphcast_small lane; not in the paper
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: printed report
Run:   # JAX env, CPU
    python -m graphcast_sae.extraction.mini_wb2_stream
"""
import functools

import numpy as np
import xarray as xr

from graphcast import solar_radiation as sr

WB2_URL = ("gs://weatherbench2/datasets/era5/"
           "1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr")

# Physical vars graphcast_small consumes (statics handled separately; TISR computed).
SURFACE_VARS = ("2m_temperature", "mean_sea_level_pressure",
                "10m_v_component_of_wind", "10m_u_component_of_wind",
                "total_precipitation_6hr")
ATMOS_VARS = ("temperature", "geopotential", "u_component_of_wind",
              "v_component_of_wind", "vertical_velocity", "specific_humidity")
STATIC_VARS = ("geopotential_at_surface", "land_sea_mask")
STEP = np.timedelta64(6, "h")
INPUT_WINDOW = 3

@functools.lru_cache(maxsize=1)
def open_wb2():
    """Open the zarr once; return (ds renamed to lat/lon, statics Dataset)."""
    ds = xr.open_zarr(WB2_URL, storage_options={"token": "anon"}, chunks=None)
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    statics = ds[list(STATIC_VARS)].load()          # time-invariant, tiny
    return ds, statics

def valid_start_times(t0, t1, stride_steps=1):
    """6-h start datetimes in [t0, t1] leaving room for a 3-step window."""
    ds, _ = open_wb2()
    t = ds.time.sel(time=slice(np.datetime64(t0), np.datetime64(t1))).values
    # drop the last 2 so every start has a full 3-step window
    return t[: len(t) - (INPUT_WINDOW - 1) : stride_steps]

def build_window(start_dt):
    """Assemble one graphcast_small-format window starting at `start_dt`.

    Returns a Dataset with dims (batch, time, level, lat, lon), time rebased to
    [0,6,12]h, absolute timestamps on the `datetime` coord, TISR computed
    analytically, and the static fields broadcast in. Byte-compatible with the
    local sample's structure.
    """
    ds, statics = open_wb2()
    start = np.datetime64(start_dt)
    abs_times = start + np.arange(INPUT_WINDOW) * STEP
    win = ds[list(SURFACE_VARS) + list(ATMOS_VARS)].sel(time=abs_times)

    # analytic TISR (1h accumulation, matches ERA5 / the sample exactly)
    templ = win["2m_temperature"].assign_coords(datetime=("time", abs_times))
    tisr = sr.get_toa_incident_solar_radiation_for_xarray(
        templ, integration_period="1h", num_integration_bins=360)
    win["toa_incident_solar_radiation"] = tisr

    # statics (lat, lon) -- no time/batch dim, matching the sample
    for v in STATIC_VARS:
        win[v] = statics[v]

    # rebase time -> timedelta [0,6,12]h; add batch dim; set absolute datetime
    win = win.assign_coords(
        time=(abs_times - abs_times[0]).astype("timedelta64[ns]"))
    win = win.expand_dims(batch=1)
    win = win.assign_coords(
        datetime=(("batch", "time"), abs_times[None, :].astype("datetime64[ns]")))
    return win.load()
