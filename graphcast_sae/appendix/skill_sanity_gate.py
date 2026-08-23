"""Phase 2 sanity gate: reproduce GC>IFS on Z500, medium range, NH-extratropics.

Uses WB2 PRECOMPUTED GraphCast + IFS-HRES forecasts vs ERA5 truth (all 0.25 deg).
This is the FROZEN sanity check before any decomposition. adv = rmse(IFS)-rmse(GC).

Paper: Appendix app:taxonomy (skill decomposition, GC vs IFS-HRES)
Inputs: results/skill/cases.npy (not shipped, see docs/REPRODUCE.md); WeatherBench2 ERA5 (GCS, streamed)
Outputs: results/skill/sanity_gate.npy (--out)
Run:   # JAX env, CPU
    FS_DEVICE=cpu python -m graphcast_sae.appendix.skill_sanity_gate [--n 120]
"""
import os, sys, time, argparse

import numpy as np, xarray as xr, gcsfs
import graphcast_sae.common.fs_common as fc

HRES = "weatherbench2/datasets/hres/2016-2022-12h-6h-0p25deg-chunk-1.zarr"
GCPC = "weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr"
G = 9.80665

def latasc(ds, latname):
    if ds[latname].values[0] > ds[latname].values[-1]:
        ds = ds.reindex({latname: ds[latname].values[::-1]})
    return ds

def wrmse(a, b, lat, mask):
    w = np.cos(np.deg2rad(lat))[:, None]
    d = (a - b) ** 2
    d = d[mask.values] if False else d
    num = (d * w)[mask].sum()
    den = (np.broadcast_to(w, d.shape))[mask].sum()
    return float(np.sqrt(num / den))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default="results/skill/sanity_gate.npy")
    args = ap.parse_args()
    C = np.load(fc.ROOT / "results/skill/cases.npy", allow_pickle=True).item()
    dates = C["dates"][: args.n]
    leads = [72, 120, 168]

    fs = gcsfs.GCSFileSystem(token="anon")
    hres = latasc(xr.open_zarr(fs.get_mapper(HRES), consolidated=True), "latitude")
    hres = hres.rename({"latitude": "lat", "longitude": "lon"})
    gc = latasc(xr.open_zarr(fs.get_mapper(GCPC), consolidated=True), "lat")
    era5, _ = fc.open_wb2()

    lat = np.asarray(hres.lat.values)
    lon = np.asarray(hres.lon.values)
    # NH-extratropics band 30..75N
    nh = (lat >= 30) & (lat <= 75)
    latmask = np.zeros((len(lat), len(lon)), bool); latmask[nh, :] = True

    def z500(ds, init, lead):
        v = ds["geopotential"].sel(time=np.datetime64(init),
                                   prediction_timedelta=int(lead),
                                   level=500)
        return np.asarray(v.values, np.float64) / G  # gpm

    def z500_truth(init, lead):
        vt = np.datetime64(init) + np.timedelta64(lead, "h")
        v = era5["geopotential"].sel(time=vt, level=500)
        return np.asarray(v.values, np.float64) / G

    rows = []  # (case_idx, lead, rmse_hres, rmse_gc)
    t0 = time.time()
    for ci, init in enumerate(dates):
        for lead in leads:
            try:
                tr = z500_truth(init, lead)
                rh = wrmse(z500(hres, init, lead), tr, lat, latmask)
                rg = wrmse(z500(gc, init, lead), tr, lat, latmask)
            except Exception as e:
                print(f"  skip {init} +{lead}h: {repr(e)[:120]}", flush=True)
                continue
            rows.append((ci, lead, rh, rg))
        if (ci + 1) % 10 == 0:
            r = np.array(rows)
            for lead in leads:
                m = r[:, 1] == lead
                adv = (r[m, 2] - r[m, 3]).mean()
                print(f"  [{ci+1}/{len(dates)}] lead {lead}h  meanADV={adv:+.2f} gpm "
                      f"(rmseHRES={r[m,2].mean():.1f} rmseGC={r[m,3].mean():.1f})", flush=True)
            print(f"    elapsed {(time.time()-t0)/60:.1f}m", flush=True)
    R = np.array(rows)
    np.save(fc.ROOT / args.out, dict(rows=R, dates=dates, leads=leads,
            cols=["case", "lead_h", "rmse_hres", "rmse_gc"]), allow_pickle=True)
    print("\n===== SANITY GATE (Z500, NH-extratropics 30-75N) =====")
    from scipy import stats
    perlead = {}
    for lead in leads:
        m = R[:, 1] == lead
        rh, rg = R[m, 2], R[m, 3]
        adv = rh - rg
        t, p = stats.ttest_rel(rh, rg)
        perlead[lead] = (adv.mean(), p)
        print(f"lead {lead}h: rmseHRES={rh.mean():.2f}  rmseGC={rg.mean():.2f}  "
              f"adv={adv.mean():+.2f}+-{adv.std()/np.sqrt(len(adv)):.2f} gpm  "
              f"GC_better_frac={(adv>0).mean():.2f}  p={p:.1e}")
    # Canonical medium range = 5-7 day (120,168h): GC must beat IFS significantly.
    mr = np.isin(R[:, 1], [120, 168])
    adv_mr = R[mr, 2] - R[mr, 3]
    t, p = stats.ttest_1samp(adv_mr, 0)
    passed = (adv_mr.mean() > 0) and (p < 0.05)
    print(f"\nmedium-range (120+168h) adv={adv_mr.mean():+.2f} gpm  p={p:.1e}")
    print("GATE:", "PASS (GC beats IFS-HRES, Z500, medium range, extratropics)" if passed
          else "FAIL -> readout/alignment defect, STOP")

if __name__ == "__main__":
    main()
