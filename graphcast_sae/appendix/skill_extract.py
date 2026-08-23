"""Phase 3 (GPU): per-case GraphCast rollout -> layer-8 SAE feature vectors + own-GC skill.

For each frozen case: 28-step (7-day) autoregressive rollout (manual _get_next_inputs
pattern). Capture layer-8 acts at the forwards PRODUCING leads {init,72,120,168}h, encode
to SAE codes, reduce to region-mean feature vectors (global / NH-ext 30-75N / tropics 20S-20N).
Also score OUR GraphCast Z500/T850/MSLP vs ERA5 and vs IFS-HRES -> per-case adv.
Crash-safe: results/skill/case_XXXX.npy per case; skips existing.

Paper: Appendix app:taxonomy (skill decomposition)
Inputs: results/skill/cases.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS); WeatherBench2 ERA5 (GCS, streamed)
Outputs: results/skill/case_<ci>.npy (--outdir; crash-safe per case)
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.appendix.skill_extract
"""
import os, sys, time, argparse, functools
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

HRES = "weatherbench2/datasets/hres/2016-2022-12h-6h-0p25deg-chunk-1.zarr"
G = 9.80665
H = 28
LEADS = [72, 120, 168]
PROD_H = {72: 11, 120: 19, 168: 27}          # forward index producing each lead (h predicts (h+1)*6h)
SNAP_H = {"init": 0, "72": 11, "120": 19, "168": 27}

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def build_io(t0, tc):
    blk, times, statics = fc.load_block(np.datetime64(t0), nframes=2 + H)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = statics[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*H}h"), **dataclasses.asdict(tc))

def build_io_fast(t0, tc):
    """Same (inputs, targets_template, forcings) as build_io but downloads only the 2
    INPUT frames of prognostic vars + the forcing var for all frames. Target-frame
    prognostic VALUES are unused by GraphCast (it predicts them), so they are tiled from
    the last input frame. ~10x less network than loading all 30 frames of ERA5."""
    ds, statics = fc.open_wb2()
    times = np.datetime64(t0) - fc.STEP + np.arange(2 + H) * fc.STEP
    prog = list(fc.SURFACE_VARS) + list(fc.ATMOS_VARS)
    frcv = list(fc.FORCING_VARS)
    inp2 = ds[prog].sel(time=times[:2]).load()                 # real input frames
    frc_all = ds[frcv].sel(time=times).load()                  # real forcing, all frames
    last = inp2.isel(time=1)
    tile = xr.concat([last.assign_coords(time=tt) for tt in times[2:]], dim="time")
    blk = xr.merge([xr.concat([inp2, tile], dim="time"), frc_all])
    blk = blk.sel(time=times)                                  # ensure order
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = statics[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*H}h"), **dataclasses.asdict(tc))

def region_masks(lat, lon):
    latm = {"global": np.ones_like(lat, bool),
            "nhext": (lat >= 30) & (lat <= 75),
            "tropics": (lat >= -20) & (lat <= 20)}
    w = np.cos(np.deg2rad(lat))
    return latm, w

def wrmse(a, b, latmask, w):
    d = (a - b) ** 2
    ww = np.broadcast_to(w[:, None], d.shape)
    m = latmask[:, None] & np.ones((1, d.shape[1]), bool)
    return float(np.sqrt((d * ww)[m].sum() / ww[m].sum()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=120)
    ap.add_argument("--slow", action="store_true", help="use full-block loader (validation)")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--outdir", default="results/skill", help="output subdir under repo root")
    args = ap.parse_args()
    loader = build_io if args.slow else build_io_fast
    C = np.load(fc.ROOT / "results/skill/cases.npy", allow_pickle=True).item()
    dates = C["dates"]

    # mesh node region masks
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = np.asarray(geom["lat"])
    node_reg = {"global": np.ones_like(mlat, bool),
                "nhext": (mlat >= 30) & (mlat <= 75),
                "tropics": (mlat >= -20) & (mlat <= 20)}
    node_reg_j = {k: jnp.asarray(v) for k, v in node_reg.items()}

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=None, bf16=True)   # capture-only, exact model
    fn = functools.partial(rf.apply, params, {}, jax.random.PRNGKey(0))
    apply = jax.jit(lambda inp, tgt, frc: fn(inp, tgt, frc)[0])  # -> (preds, acts)

    @jax.jit
    def reduce_codes(acts):
        X = jnp.asarray(acts, jnp.float32).reshape(-1, fc.D_IN)
        codes = sae.codes(X)                                     # (nodes,4096)
        out = {}
        for k, m in node_reg_j.items():
            out[k] = (codes * m[:, None]).sum(0) / m.sum()
        return out

    # truth + hres
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    hres = xr.open_zarr(fs.get_mapper(HRES), consolidated=True).rename(
        {"latitude": "lat", "longitude": "lon"})
    if hres.lat.values[0] > hres.lat.values[-1]:
        hres = hres.reindex(lat=hres.lat.values[::-1])
    era5, _ = fc.open_wb2()

    outdir = fc.ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    for ci in range(args.start, args.end):
        fpath = outdir / f"case_{ci:04d}{args.suffix}.npy"
        if fpath.exists():
            continue
        init = dates[ci]
        t0 = time.time()
        inp, tgt, frc = loader(init, tc)
        lat = np.asarray(inp.lat.values); lon = np.asarray(inp.lon.values)
        latm, w = region_masks(lat, lon)

        tct = tgt.time.isel(time=slice(0, 1))
        for c in ("datetime",):
            if c in tgt.coords: tgt = tgt.drop_vars(c)
            if c in frc.coords: frc = frc.drop_vars(c)

        featvec = {}   # snapshot -> region -> (4096,)
        gcfields = {}  # lead -> dict(z500,t850,msl)
        cur = inp
        for h in range(H):
            ctgt = tgt.isel(time=slice(h, h + 1)).assign_coords(time=tct)
            cfrc = frc.isel(time=slice(h, h + 1)).assign_coords(time=tct)
            preds, acts = apply(cur, ctgt, cfrc)
            # capture feature vectors
            for name, hh in SNAP_H.items():
                if h == hh:
                    red = reduce_codes(acts)
                    featvec[name] = {k: np.asarray(v, np.float32) for k, v in red.items()}
            preds = numpyify(preds)
            for lead, hh in PROD_H.items():
                if h == hh:
                    gcfields[lead] = dict(
                        z500=np.asarray(preds["geopotential"].isel(batch=0, time=0).sel(level=500).values) / G,
                        t850=np.asarray(preds["temperature"].isel(batch=0, time=0).sel(level=850).values),
                        msl=np.asarray(preds["mean_sea_level_pressure"].isel(batch=0, time=0).values))
            if h < H - 1:
                cur = rollout._get_next_inputs(cur, xr.merge([preds, cfrc])).assign_coords(
                    time=cur.coords["time"])

        # score
        varmap = dict(z500="geopotential", t850="temperature", msl="mean_sea_level_pressure")
        rmse_gc, rmse_hres = {}, {}
        for lead in LEADS:
            vt = np.datetime64(init) + np.timedelta64(lead, "h")
            truth = dict(
                z500=np.asarray(era5["geopotential"].sel(time=vt, level=500).values) / G,
                t850=np.asarray(era5["temperature"].sel(time=vt, level=850).values),
                msl=np.asarray(era5["mean_sea_level_pressure"].sel(time=vt).values))
            hf = dict(
                z500=np.asarray(hres["geopotential"].sel(time=np.datetime64(init),
                     prediction_timedelta=int(lead), level=500).values) / G,
                t850=np.asarray(hres["temperature"].sel(time=np.datetime64(init),
                     prediction_timedelta=int(lead), level=850).values),
                msl=np.asarray(hres["mean_sea_level_pressure"].sel(time=np.datetime64(init),
                     prediction_timedelta=int(lead)).values))
            for var in ("z500", "t850", "msl"):
                for reg in ("global", "nhext", "tropics"):
                    rmse_gc[(var, lead, reg)] = wrmse(gcfields[lead][var], truth[var], latm[reg], w)
                    rmse_hres[(var, lead, reg)] = wrmse(hf[var], truth[var], latm[reg], w)

        np.save(fpath, dict(ci=ci, init=init, featvec=featvec,
                            rmse_gc=rmse_gc, rmse_hres=rmse_hres,
                            leads=LEADS), allow_pickle=True)
        adv_z5_nh = rmse_hres[("z500", 168, "nhext")] - rmse_gc[("z500", 168, "nhext")]
        print(f"[{ci}] {str(init)[:13]}  {time.time()-t0:.0f}s  "
              f"Z500@168h NHext rmseHRES={rmse_hres[('z500',168,'nhext')]:.1f} "
              f"rmseGC={rmse_gc[('z500',168,'nhext')]:.1f} adv={adv_z5_nh:+.1f}", flush=True)

if __name__ == "__main__":
    main()
