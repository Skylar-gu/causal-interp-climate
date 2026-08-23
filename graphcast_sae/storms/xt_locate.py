"""CPU probe: locate the eastcoast2018 low at IC, and find a NON-DEVELOPING control
in the same box/season.

Why this runs before the battery. The extratropical battery cannot reuse
`nondev2013` (a tropical July wave) as its non-developing control: the point of the
battery is to give extratropical features exposure, and a control drawn from a
different season and basin cannot fail in the same way. Guardrail #9 wants a
control that CAN fail, so it has to be a January North Atlantic case in the same
box where nothing bombs.

Also fixes a guess: the ablation disk is centred on the storm at IC, and I do not
know the January-2018 low's position to 1 degree from memory. Read it.

Paper: supporting: extratropical battery control (Sec. 3 mentions the extratropical arms)
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: printed report
Run:   # JAX env, CPU
    python -m graphcast_sae.storms.xt_locate
"""
import os
import sys

import numpy as np

import graphcast_sae.common.fs_common as fc

STEP = np.timedelta64(6, "h")
BOX = dict(lat=(30, 46), lon=(283, 302))          # eastcoast2018, deg E 0..360
NLEAD = 12                                        # 72 h

# Candidate non-developers: early-January windows in other years, same box.
# Screened, not assumed -- a January North Atlantic box is a busy place.
CANDIDATES = ["2018-01-03", "2014-01-03", "2013-01-03", "2011-01-03", "2010-01-03",
              "2009-01-03", "2006-01-03", "2005-01-03", "2004-01-03", "2003-01-03",
              "2016-01-12", "2012-01-12", "2007-01-12", "2015-01-20", "2011-01-20",
              # second sweep: need 5 QUIET dates for the normal-level reference
              "2002-01-08", "2001-01-08", "2000-01-08", "1999-01-08", "1998-01-08",
              "1997-01-08", "1996-01-08", "2008-01-08", "2017-01-08", "2019-01-08",
              "2020-01-08", "2021-01-08", "1995-01-15", "1994-01-15", "1993-01-15"]

def series(ds, t0, nlead=NLEAD):
    times = np.datetime64(t0) + np.arange(nlead + 1) * STEP
    sub = ds[["mean_sea_level_pressure"]].sel(time=times)
    sub = sub.sel(lat=slice(*BOX["lat"]), lon=slice(*BOX["lon"])).load()
    m = sub["mean_sea_level_pressure"].values / 100.0
    la, lo = sub.lat.values, sub.lon.values
    pos, val = [], []
    for t in range(m.shape[0]):
        i, j = np.unravel_index(np.nanargmin(m[t]), m[t].shape)
        pos.append((float(la[i]), float(lo[j]) - 360.0)); val.append(float(m[t, i, j]))
    return np.array(val), pos

def main():
    ds, _ = fc.open_wb2()
    print(f"box lat {BOX['lat']} lon {BOX['lon']} (= {BOX['lon'][0]-360}..{BOX['lon'][1]-360} degE)\n")
    rows = []
    for c in CANDIDATES:
        try:
            v, pos = series(ds, c)
        except Exception as e:
            print(f"{c}: ERROR {e}"); continue
        drop = float(v[0] - v.min())
        # 24 h drop, and whether the box minimum stays put (advection vs development)
        k = 4
        d24 = float(max(v[t] - v[t + k] for t in range(len(v) - k)))
        rows.append((c, v[0], v.min(), drop, d24, pos[0]))
        print(f"{c}  IC_min={v[0]:7.1f}  min={v.min():7.1f}  drop={drop:5.1f}  "
              f"max24h={d24:5.1f}  low_at_IC=({pos[0][0]:.1f}, {pos[0][1]:.1f})")
    print("\nRANKED by deepening over 72 h:")
    for r in sorted(rows, key=lambda x: -x[3]):
        tag = "DEVELOPS" if r[3] >= 20 else ("quiet -> nondev candidate" if r[3] < 8 else "marginal")
        print(f"  {r[0]}  drop {r[3]:5.1f} hPa  (24h {r[4]:5.1f})  [{tag}]")
    print("\nA nondev control must be QUIET (drop < 8 hPa) in THIS box and season.")

if __name__ == "__main__":
    main()
