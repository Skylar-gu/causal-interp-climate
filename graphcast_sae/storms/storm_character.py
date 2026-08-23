"""What distinguishes the storms where amplification HELPS from those where it HURTS?

The benefit at g=2 tracks how much of ERA5's deepening GraphCast already captured
(Spearman +0.94 excluding the inert Patricia). But "already accurate" is not an
explanation -- it restates the question. WHY is GraphCast accurate on Haiyan and
Haishen and hopeless on Ida and Patricia?

Two physical hypotheses, both testable from ERA5 alone:

  H1 COMPACTNESS. The processor mesh is ~111 km. A storm whose core is small
     relative to that is unresolvable, so the model cannot build it. Patricia's
     radius of maximum wind was ~15 km at peak and Wilma's eye was 3-4 km across --
     the smallest ever measured -- while Haiyan and Haishen were physically large
     systems. If H1 holds, benefit should scale with COMPACTNESS.

  H2 SPIN-UP vs MAINTENANCE. GraphCast is handed a vortex by the encoder. It may be
     able to PROPAGATE an already-formed storm while being unable to BUILD one. If
     H2 holds, benefit should scale with how undeveloped the storm is at IC, and
     with how fast it then intensifies.

These are not independent -- extreme RI happens in compact cores -- so the point is
which one orders the storms better, not to declare a single cause.

Measured per storm, all from the ERA5 the model was trained on:
  ic_deficit   environment MSLP minus box minimum at t=0    (how developed at IC)
  max_dp24     largest 24 h box deepening in the window     (RI rate)
  r_def10      mean radius at which MSLP rises 10 hPa above the peak-time minimum
               -- a size proxy at 0.25 deg; SMALL = compact
  r_maxwind    distance from the peak-time pressure minimum to the peak-time
               maximum 10 m wind -- a coarse RMW proxy
  peak_lead    hours from IC to the ERA5 minimum

Paper: Fig. fig:gain discussion (which storms amplification helps)
Inputs: WeatherBench2 ERA5 (GCS, streamed)
Outputs: results/storm_character.npy
Run:   # JAX env, CPU
    python -m graphcast_sae.storms.storm_character
"""
import json
import os
import sys

import numpy as np

import graphcast_sae.common.fs_common as fc
import graphcast_sae.common.skill_conv_storms as S

from graphcast_sae.common.signature_physics import gc_km

STEP = np.timedelta64(6, "h")
STORMS = ["ida2021", "michael2018", "haishen2020", "goni2020",
          "haiyan2013", "patricia2015", "wilma2005"]

def main():
    ds, _ = fc.open_wb2()
    rows = {}
    for name in STORMS:
        cfg = S.STORMS[name]
        la0, la1 = cfg["box"]["lat"]
        lo = S.norm_lon(cfg["box"]["lon"])
        times = np.datetime64(cfg["ic"]) + np.arange(S.H + 1) * STEP
        sub = ds[["mean_sea_level_pressure", "10m_u_component_of_wind",
                  "10m_v_component_of_wind"]].sel(time=times)
        sub = sub.sel(lat=slice(la0, la1))
        sub = (sub.sel(lon=slice(lo[0], lo[1])) if lo[0] <= lo[1]
               else sub.sel(lon=(ds.lon >= lo[0]) | (ds.lon <= lo[1])))
        sub = sub.load()
        p = sub["mean_sea_level_pressure"].values / 100.0
        u = sub["10m_u_component_of_wind"].values
        v = sub["10m_v_component_of_wind"].values
        w = np.sqrt(u * u + v * v)
        la = sub.lat.values
        ln = np.where(sub.lon.values > 180, sub.lon.values - 360, sub.lon.values)

        pmin = np.nanmin(p.reshape(p.shape[0], -1), axis=1)
        ipk = int(np.argmin(pmin))
        k = 4
        max_dp24 = float(max(pmin[t] - pmin[t + k] for t in range(len(pmin) - k)))
        # environment = 90th percentile of the box at IC, a stand-in for ambient MSLP
        ic_env = float(np.nanpercentile(p[0], 90))
        ic_deficit = ic_env - float(pmin[0])

        # geometry at the ERA5 peak
        f = p[ipk]
        i, j = np.unravel_index(np.nanargmin(f), f.shape)
        clat, clon = float(la[i]), float(ln[j])
        LA = np.repeat(la[:, None], len(ln), 1).ravel()
        LN = np.repeat(ln[None, :], len(la), 0).ravel()
        d = gc_km(LA, LN, clat, clon).reshape(f.shape)
        # mean radius of the +10 hPa contour: area inside it, converted to a radius
        inside = f <= (f[i, j] + 10.0)
        # area weight per cell
        dlat = abs(float(la[1] - la[0])); dlon = abs(float(ln[1] - ln[0]))
        cell = (111.0 * dlat) * (111.0 * dlon * np.cos(np.deg2rad(la))[:, None])
        area = float((cell * inside).sum())
        r_def10 = float(np.sqrt(area / np.pi))
        g = w[ipk]
        wi, wj = np.unravel_index(np.nanargmax(g), g.shape)
        r_maxwind = float(d[wi, wj])
        rows[name] = dict(ic_deficit=ic_deficit, max_dp24=max_dp24,
                          r_def10=r_def10, r_maxwind=r_maxwind,
                          peak_lead=ipk * 6, era_min=float(pmin[ipk]),
                          peak_wind=float(np.nanmax(w)))
        print(f"{name:<14} ic_deficit {ic_deficit:5.1f}  max_dp24 {max_dp24:5.1f}  "
              f"r_def10 {r_def10:6.0f} km  r_maxwind {r_maxwind:5.0f} km  "
              f"peak +{ipk*6:3d}h", flush=True)

    np.save(fc.ROOT / "results/storm_character.npy", rows, allow_pickle=True)
    print("\n-> results/storm_character.npy")

if __name__ == "__main__":
    main()
