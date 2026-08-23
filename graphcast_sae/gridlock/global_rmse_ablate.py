"""Do the suspected grid-artifact features matter for FORECAST SKILL AT ALL?

The hurricane readout was far too narrow. A feature that fires over the Sahara cannot show up
in a tropical-cyclone pressure trace no matter how much it matters. The right question is the
one a weather centre would ask: ablate the features, roll GraphCast out, and measure GLOBAL
RMSE against ERA5 on the standard headline fields.

ARMS (ablation = coef -1, the error-preserving ablation, applied GLOBALLY at EVERY step, so
the feature is removed from the model for the whole rollout rather than perturbed once):
  baseline      no patch
  mesh_locked   27 features firing the coarse icosahedral skeleton 2x-16x more than chance
                the top ones put 86-99% of activations on levels 0-4 vs a 6.25% chance share
  ctrl_mesh     27 features matched EXACTLY on firing rate (ratio 1.00), passing both filters
  scatter_blob  127 features with the widest footprint spread; visual inspection shows this
                class contains the literal triangular/bowtie artifacts (f2075) and the polar
                mesh-convergence bands (f2235)
  ctrl_blob     127 firing-rate-matched controls

READING, pre-registered here before any number exists:
  RMSE unchanged vs its control  -> the features are inert; removable at no cost
  RMSE worse                     -> they carry real forecast information despite looking like
                                    grid artifacts, and the "artifact" reading is wrong
  RMSE BETTER                    -> they inject grid-shaped error into the forecast, and
                                    deleting them IMPROVES GraphCast. The outcome worth having.

The control arms are what make this readable: GPU non-determinism and the SAE's own
reconstruction error both move RMSE, so the comparison is arm-vs-matched-control, never
arm-vs-baseline alone.

    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \

Paper: Sec. 3 grid-lock ablations (results/fs_global_rmse.npy; demo notebook part 4)
Inputs: GraphCast params (GRAPHCAST_PARAMS)
Outputs: out/global_rmse_status.txt; results/fs_global_rmse.npy
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.gridlock.global_rmse_ablate
"""
import dataclasses
import json
import os
import sys
import time

os.environ["FS_DEVICE"] = "gpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import xarray as xr
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

S = int(os.environ.get("GR_S", "20"))                 # 20 x 6 h = 120 h
ICS = os.environ.get("GR_ICS", "2020-01-10,2020-03-12,2020-05-11,2020-07-10,"
                                "2020-09-08,2020-11-07,2021-02-05,2021-06-15").split(",")
OUT = fc.ROOT / os.environ.get("GR_OUT", "results/fs_global_rmse.npy")
STATUS = fc.ROOT / os.environ.get("GR_STATUS", "out/global_rmse_status.txt")
G0 = 9.80665

# WeatherBench2 headline fields: (name, variable, level or None)
FIELDS = [("z500", "geopotential", 500), ("t850", "temperature", 850),
          ("t2m", "2m_temperature", None), ("u850", "u_component_of_wind", 850),
          ("v850", "v_component_of_wind", 850), ("q700", "specific_humidity", 700),
          ("msl", "mean_sea_level_pressure", None)]

def log(m):
    print(m, flush=True)
    with open(STATUS, "a") as f:
        f.write(m + "\n")

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def build_io(t0, tc, s):
    blk, times, statics = fc.load_block(np.datetime64(t0), nframes=s + 2)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims:
            w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS:
        w[v] = statics[v]
    w = w.assign_coords(datetime=(("batch", "time"),
                                  times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*s}h"), **dataclasses.asdict(tc))

def grab(ds, var, lev):
    d = ds[var].isel(batch=0, time=0)
    if lev is not None:
        d = d.sel(level=lev)
    a = np.asarray(d.transpose("lat", "lon").values, np.float64)
    return a / G0 if var == "geopotential" else a

def main():
    open(STATUS, "w").close()
    GF = os.environ.get("GR_GROUPS", "/tmp/artifact_groups.json")
    G = json.load(open(GF))
    order = os.environ.get("GR_ARMS", "")
    keys = order.split(",") if order else list(G)
    ARMS = [("baseline", None)] + [(k, G[k]) for k in keys]
    log(f"GLOBAL RMSE ABLATION  S={S} ({6*S} h)  ICs={len(ICS)}  arms={len(ARMS)}")
    for k, v in ARMS:
        log(f"   {k:<14} {'-' if v is None else str(len(v)) + ' features'}")

    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply_fn = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)
    patches = {k: (noop if v is None else fc.coef_patch(sae, v, -1.0))
               for k, v in ARMS}

    # PER-IC storage, not a running mean. Every arm sees the SAME initial conditions, so the
    # comparison is PAIRED: for a given IC the arm and its control forecast identical weather
    # and their difference isolates the ablation from "was this a hard week". Keeping only the
    # mean throws that structure away and leaves a number with no testable uncertainty.
    acc = {k: np.full((len(ICS), S, len(FIELDS)), np.nan) for k, _ in ARMS}
    n = 0
    for ii, ic in enumerate(ICS):
        t0 = time.time()
        inp, tgt, frc = build_io(ic, tc, S)
        lat = np.asarray(tgt.lat.values, np.float64)
        w = np.cos(np.deg2rad(lat))[:, None]
        for name, _ in ARMS:
            cur, p = inp, patches[name]
            for s in range(S):
                tg = tgt.isel(time=slice(s, s + 1)).assign_coords(time=tgt.time[:1])
                fr = frc.isel(time=slice(s, s + 1)).assign_coords(time=frc.time[:1])
                pr = numpyify(apply_fn(cur, tg, fr, p)[0])    # patch EVERY step; apply returns (preds, acts)
                truth = tgt.isel(time=slice(s, s + 1))
                for fi, (nm, var, lev) in enumerate(FIELDS):
                    d = grab(pr, var, lev) - grab(truth, var, lev)
                    acc[name][ii, s, fi] = np.sqrt((w * d * d).sum() / (w.sum() * d.shape[1]))
                cur = rollout._get_next_inputs(cur, xr.merge([pr, fr])).assign_coords(
                    time=inp.coords["time"])
        n += 1
        log(f"  IC {ic} done ({time.time()-t0:.0f}s)  {n}/{len(ICS)}")
        np.save(OUT, dict(acc=acc, n=n,
                          fields=[f[0] for f in FIELDS], S=S, ics=ICS[:n],
                          arms=[a[0] for a in ARMS], groups=G), allow_pickle=True)

    R = {k: np.nanmean(v, 0) for k, v in acc.items()}      # mean over ICs
    log(f"\n{'':16}" + "".join(f"{f[0]:>10}" for f in FIELDS) + "   (RMSE at +120 h)")
    for k, _ in ARMS:
        log(f"  {k:<14}" + "".join(f"{R[k][-1, i]:>10.3f}" for i in range(len(FIELDS))))
    log(f"\n  ARM MINUS ITS MATCHED CONTROL, PAIRED over the {n} ICs")
    log(f"  (negative = ablation IMPROVES the forecast; p from a paired t-test on the "
        f"{n} per-IC differences, which removes the week-to-week variance)")
    from scipy import stats
    pairs = [(a, "ctrl_" + a) for a, _ in ARMS
             if a != "baseline" and not a.startswith("ctrl_") and "ctrl_" + a in acc]
    if not pairs:
        pairs = [(a, c) for a, c in (("mesh_locked", "ctrl_mesh"),
                                     ("scatter_blob", "ctrl_blob")) if a in acc and c in acc]
    for arm, ctl in pairs:
        d = acc[arm][:, -1, :] - acc[ctl][:, -1, :]           # (nIC, nfield) at the last lead
        mu = np.nanmean(d, 0); se = np.nanstd(d, 0, ddof=1) / np.sqrt(n)
        t, pv = stats.ttest_rel(acc[arm][:, -1, :], acc[ctl][:, -1, :], axis=0,
                                nan_policy="omit")
        win = (d < 0).sum(0)
        log(f"  {arm}")
        log(f"    {'mean diff':<12}" + "".join(f"{v:>+10.4f}" for v in mu))
        log(f"    {'+-se':<12}" + "".join(f"{v:>10.4f}" for v in se))
        log(f"    {'paired p':<12}" + "".join(f"{v:>10.3f}" for v in np.atleast_1d(pv)))
        log(f"    {'ICs better':<12}" + "".join(f"{v:>8}/{n}" for v in win))
    log(f"\n  each control minus baseline (the scale of ablating N arbitrary features)")
    for ctl in [c for _, c in pairs]:
        d = acc[ctl][:, -1, :] - acc["baseline"][:, -1, :]
        log(f"  {ctl:<14}" + "".join(f"{v:>+10.4f}" for v in np.nanmean(d, 0)))
    log(f"\n-> {OUT}")

if __name__ == "__main__":
    main()
