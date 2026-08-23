"""ERA5-only screening gate for candidate intervention events. No model, no GPU.

Purpose. The convection spec applied this gate by hand ("verify each IC has a real
deepening in ERA5; drop any GraphCast doesn't forecast to intensify"). This
automates the ERA5 half so twenty candidates can be screened cheaply and only the
strong ones cost a flagship forward.

Design note. The candidate battery is built to break a degeneracy, not to add
storms. Two results exist — convection (moist/local/fast) is a sparse causal
handle, blocking (dry/hemispheric/slow) is a distributed set — and moist-vs-dry,
local-vs-large and fast-vs-slow all explain them equally well. The new cells are
explosive cyclogenesis (dry/local) and atmospheric rivers (moist/large); whichever
of those behaves like a handle identifies the real axis.

SCREENER CALIBRATION: events already verified downstream (Ida, Michael, Haiyan, the
2021 heat dome) are included as POSITIVE CONTROLS. If the screener does not rank
them strong on their own class metric, the screener is broken and its verdicts on
the new candidates are void.

DATA LIMIT, measured not assumed: the WB2 zarr runs 1959-01-01 .. 2021-12-31, which
is why Ian 2022 appears in the convection spec but not in its results. Every
candidate here is <= 2021. No SST / boundary-layer-height variable exists in this
source, so family-E diagnostics are not computable from it.

Paper: supporting: ERA5 screening gate for candidate storms (not a paper figure)
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: results/event_screen.json
Run:   # JAX env, CPU
    python -m graphcast_sae.storms.event_screen
"""
import json
import os
import sys

import numpy as np

import graphcast_sae.common.fs_common as fc

OUT = fc.ROOT / "results" / "event_screen.json"
G = 9.80665

# name, class, IC (analysis time), lat0, lat1, lon0, lon1 (deg E, 0-360), hours
EVENTS = [
    # ── positive controls: already verified downstream ───────────────────────
    ("ida2021",       "cyclone", "2021-08-26T00", 18, 32, 268, 285,  96),
    ("michael2018",   "cyclone", "2018-10-07T00", 18, 32, 263, 280,  96),
    ("haiyan2013",    "cyclone", "2013-11-05T00",  4, 16, 125, 147,  96),
    ("heatdome2021",  "heat",    "2021-06-24T00", 44, 56, 232, 250, 144),
    # ── new cell: DRY + LOCAL — explosive extratropical cyclogenesis ─────────
    ("dennis2020",    "cyclone", "2020-02-14T12", 45, 65, 320, 360,  72),
    ("ciara2020",     "cyclone", "2020-02-08T00", 46, 64, 330, 375,  72),
    ("eastcoast2018", "cyclone", "2018-01-03T00", 30, 46, 283, 302,  72),
    ("greatlakes2010","cyclone", "2010-10-25T12", 38, 52, 262, 282,  72),
    ("alex2020",      "cyclone", "2020-10-01T00", 42, 56, 340, 370,  72),
    # ── new cell: MOIST + LARGE — atmospheric river ──────────────────────────
    ("oroville2017",  "ar",      "2017-02-07T00", 34, 43, 230, 242,  96),
    ("pnw2021",       "ar",      "2021-11-13T00", 44, 52, 228, 240,  96),
    ("calif2019",     "ar",      "2019-02-12T00", 34, 43, 230, 242,  96),
    # ── new cell: SLOW + HEMISPHERIC + DRY, stratospheric driver ─────────────
    ("texas2021",     "cold",    "2021-02-10T00", 26, 38, 253, 270, 144),
]

def box(ds, e, var, levels=None):
    _, _, ic, la0, la1, lo0, lo1, hrs = e
    t0 = np.datetime64(ic)
    sel = dict(time=slice(t0, t0 + np.timedelta64(hrs, "h")), lat=slice(la0, la1))
    d = ds[var].sel(**sel)
    if levels is not None:
        d = d.sel(level=levels)
    if lo1 > 360:                                   # box crosses the prime meridian
        a = d.sel(lon=slice(lo0, 360))
        b = d.sel(lon=slice(0, lo1 - 360))
        import xarray as xr
        d = xr.concat([a, b], dim="lon")
    else:
        d = d.sel(lon=slice(lo0, lo1))
    return d.load()

def main():
    ds, _ = fc.open_wb2()
    tmin, tmax = ds.time.values[0], ds.time.values[-1]
    print(f"ERA5 zarr {str(tmin)[:10]} .. {str(tmax)[:10]}\n")

    rows = []
    for e in EVENTS:
        name, cls, ic, *_rest, hrs = e
        t0 = np.datetime64(ic)
        if t0 < tmin or t0 + np.timedelta64(hrs, "h") > tmax:
            print(f"  {name:<15} OUT OF RANGE — skipped")
            rows.append(dict(name=name, cls=cls, ic=ic, in_range=False))
            continue

        r = dict(name=name, cls=cls, ic=ic, hours=hrs, in_range=True)
        try:
            if cls == "cyclone":
                p = box(ds, e, "mean_sea_level_pressure") / 100.0
                v = p.values
                la, lo = p.lat.values, p.lon.values
                # CONTINUITY GATE. A box minimum is not a storm: if a second,
                # deeper low moves into the box the naive drop is ADVECTIVE, not
                # developmental. Caught by the positive controls — Ida and
                # Michael never jump, while three new candidates did, and their
                # headline deepening vanished once continuity was enforced.
                pos, series = [], []
                for t in range(v.shape[0]):
                    i, j = np.unravel_index(np.nanargmin(v[t]), v[t].shape)
                    pos.append((la[i], lo[j]))
                    series.append(v[t, i, j])
                pos, series = np.array(pos), np.array(series)
                step = []
                for t in range(1, len(pos)):
                    dla = np.radians(pos[t, 0] - pos[t - 1, 0])
                    dlo = np.radians((pos[t, 1] - pos[t - 1, 1] + 180) % 360 - 180)
                    ml = np.radians(0.5 * (pos[t, 0] + pos[t - 1, 0]))
                    step.append(6371 * np.hypot(dla, dlo * np.cos(ml)))
                step = np.array(step)
                k = max(1, int(24 / 6))
                r["mslp_min"] = float(series.min())
                r["naive_24h_drop"] = float(max(series[i] - series[i + k]
                                                for i in range(len(series) - k)))
                tracked = max([series[t] - series[t + k]
                               for t in range(len(series) - k)
                               if (step[t:t + k] < 500).all()] or [0.0])
                r["tracked_24h_drop"] = float(tracked)
                r["min_jumps_gt500km"] = int((step > 500).sum())
                r["min_step_med_km"] = float(np.median(step))
                r["continuous"] = bool((step > 500).sum() == 0)
                lat_mid = 0.5 * (e[3] + e[4])
                # 1 Bergeron == 24 hPa/24 h at 60 deg N, so the /24 is part of the
                # unit. Without it this field read 22.8 for Ida and 81.6 for
                # eastcoast2018 -- an order of magnitude above any observed
                # cyclone, which is how a normalization bug announces itself.
                r["bergeron"] = float(tracked / 24.0 * np.sin(np.radians(60))
                                      / np.sin(np.radians(lat_mid)))
                r["primary"] = r["tracked_24h_drop"]

            elif cls == "ar":
                lev = [l for l in ds.level.values if l >= 300]
                q = box(ds, e, "specific_humidity", lev)
                u = box(ds, e, "u_component_of_wind", lev)
                v = box(ds, e, "v_component_of_wind", lev)
                import xarray as xr
                dp = xr.DataArray(np.abs(np.gradient(np.asarray(lev, float) * 100.0)),
                                  coords={"level": lev}, dims="level")
                w = q * np.sqrt(u ** 2 + v ** 2)
                ivt = (w * dp).sum("level") / G
                bm = ivt.mean(("lat", "lon")).values
                r["ivt_box_mean_max"] = float(bm.max())
                r["ivt_box_max"] = float(ivt.max().values)
                pr = box(ds, e, "total_precipitation_6hr")
                r["precip_box_mean_total_mm"] = float(pr.mean(("lat", "lon")).sum().values * 1000)
                r["primary"] = r["ivt_box_mean_max"]

            elif cls == "heat":
                z = box(ds, e, "geopotential", [500]) / G
                t2 = box(ds, e, "2m_temperature") - 273.15
                r["z500_box_mean_max_m"] = float(z.mean(("lat", "lon", "level")).max().values)
                r["t2m_box_max_C"] = float(t2.max().values)
                r["primary"] = r["t2m_box_max_C"]

            elif cls == "cold":
                t2 = box(ds, e, "2m_temperature") - 273.15
                r["t2m_box_min_C"] = float(t2.min().values)
                r["t2m_box_mean_min_C"] = float(t2.mean(("lat", "lon")).min().values)
                r["primary"] = -r["t2m_box_mean_min_C"]
        except Exception as ex:                     # a bad box should not kill the sweep
            r["error"] = f"{type(ex).__name__}: {ex}"
            print(f"  {name:<15} ERROR {r['error'][:70]}")
            rows.append(r)
            continue

        extra = "  ".join(f"{k}={v:.1f}" for k, v in r.items()
                          if isinstance(v, float) and k != "primary")
        print(f"  {name:<15} {cls:<8} {extra}", flush=True)
        rows.append(r)

    json.dump(dict(zarr_start=str(tmin), zarr_end=str(tmax), events=rows),
              open(OUT, "w"), indent=1)

    print("\nRANKED WITHIN CLASS (primary metric; ● = already-verified positive control)")
    ctrl = {"ida2021", "michael2018", "haiyan2013", "heatdome2021"}
    for cls, unit in [("cyclone", "max 24-h MSLP drop, hPa"),
                      ("ar", "peak box-mean IVT, kg/m/s"),
                      ("heat", "box max 2 m T, °C"),
                      ("cold", "box-mean min 2 m T, °C (sign-flipped)")]:
        g = [r for r in rows if r.get("cls") == cls and "primary" in r]
        if not g:
            continue
        print(f"\n  {cls}  — {unit}")
        for r in sorted(g, key=lambda r: -r["primary"]):
            print(f"    {'●' if r['name'] in ctrl else ' '} {r['name']:<15} "
                  f"{r['primary']:>8.1f}")
    print(f"\nwrote {OUT.relative_to(fc.ROOT)}")

if __name__ == "__main__":
    main()
