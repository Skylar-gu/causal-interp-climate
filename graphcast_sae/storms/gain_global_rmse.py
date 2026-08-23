"""Does amplifying the convection features break the forecast EVERYWHERE ELSE?

THE QUESTION THIS SETTLES. Scaling the convection group's excess cuts Ida's
intensity RMSE by 65% at g=2.5 and keeps closing on best track out to g=3. That is
worthless as a forecast result if the same intervention degrades the global field:
a gain that fixes one storm's core and wrecks the hemisphere is a trade, not an
improvement. Nothing measured so far leaves the storm box, so nothing so far can
tell the difference.

WHAT IS MEASURED. Latitude-weighted RMSE against ERA5 on the WeatherBench2 headline
fields, at every 6 h lead out to +240 h (10 days), PARTITIONED by distance from the
storm. ERA5 is the right reference HERE even though it is not ground truth for a TC
core: these are hemispheric resolved fields, which is exactly what a reanalysis is
good at. The core caveat applies to the storm-minimum readout, not to this one.

    INSIDE   <= 1500 km of the storm centre -- exactly the disk the patch acts on
    OUTSIDE  > 1500 km                      -- everywhere the patch never touched
    GLOBAL   the standard number

The intervention is spatially confined to the disk, so the inside/outside split is
the whole point: OUTSIDE error can only grow by propagation, and propagation is the
failure mode that would make this useless.

THE AFTERMATH QUESTION. The patch is released at +96 h and the model then runs free
for six more days. A storm forecast that is genuinely better -- rather than merely
deeper -- should leave a better downstream flow behind it: recurvature, extratropical
transition and the downstream ridge/trough couplet are all real teleconnections, so
fixing the storm should show up in z500 days later, far from the disk. If instead the
benefit evaporates the moment the patch is released, the intervention was holding the
storm up rather than correcting it.

READING, pre-registered before any number exists:
  OUTSIDE flat, INSIDE improves      -> a genuine local correction. The result stands.
  OUTSIDE flat, INSIDE degrades      -> the box-MSLP gain was a lucky projection of a
                                        worse field. The accuracy claim dies.
  OUTSIDE degrades                   -> a trade, not an improvement. Report the
                                        exchange rate (hPa of storm per unit of
                                        global RMSE) and stop calling it a fix.
  OUTSIDE improves                   -> the model is globally under-convective and
                                        this is a bias correction, not a storm hack.
                                        The outcome worth having.

The random-feature group at the same gain is carried as a control, because SAE
reconstruction error and GPU non-determinism both move RMSE: the comparison that
counts is convection-at-g vs random-at-g, never arm vs baseline alone.

`ftarget` (the analog-derived normal level) is READ BACK from the completed gain run
rather than recomputed, so the intervention here is bit-identical to the one that
produced the intensity numbers.

    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \

Paper: Fig. fig:gain, right panel (global RMSE under amplification)
Inputs: GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/gain_global_rmse.npy
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.storms.gain_global_rmse
"""
import dataclasses
import os
import sys
import time

os.environ["FS_DEVICE"] = "gpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import xarray as xr
import jax.numpy as jnp

import graphcast_sae.common.fs_common as fc
import graphcast_sae.common.skill_conv_storms as S
from graphcast import data_utils, rollout

from graphcast_sae.common.signature_physics import gc_km

STORMS = os.environ.get("GG_STORMS", "ida2021,haishen2020").split(",")
GAINS = [float(x) for x in os.environ.get("GG_GAINS", "1.5,2,3").split(",")]
SRC = os.environ.get("GG_SRC", "gain_conv")
# 40 steps = +240 h = 10 days. The storm peaks by +42..96 h, so this leaves ~6 days
# of AFTERMATH: does a better-forecast storm leave a better forecast behind it, or
# does the intervention's error compound once the storm is gone?
H = int(os.environ.get("GG_H", "40"))
# The patch is RELEASED after PW steps (gain -> 1.0, an exact no-op, verified in
# fs_common.delta_gain). So the intervention acts during intensification only and the
# model then runs free. A `persistent` arm that never releases is carried alongside,
# because the two answer different questions: "does fixing the storm help downstream"
# vs "what does a continuous local forcing do".
PW = int(os.environ.get("GG_PW", "16"))
G0 = 9.80665
OUT = fc.ROOT / "results" / "gain_global_rmse.npy"
FIELDS = [("z500", "geopotential", 500), ("t850", "temperature", 850),
          ("t2m", "2m_temperature", None), ("u850", "u_component_of_wind", 850),
          ("v850", "v_component_of_wind", 850), ("q700", "specific_humidity", 700),
          ("msl", "mean_sea_level_pressure", None)]

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def build_io(t0, tc, nsteps):
    blk, times, st = fc.load_block(np.datetime64(t0), nframes=2 + nsteps)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims:
            w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS:
        w[v] = st[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*nsteps}h"), **dataclasses.asdict(tc))

def grab(ds, var, lev):
    d = ds[var].isel(batch=0, time=0)
    if lev is not None:
        d = d.sel(level=lev)
    a = np.asarray(d.transpose("lat", "lon").values, np.float64)
    return a / G0 if var == "geopotential" else a

def main():
    geom = np.load(fc.MESH_GEOM, allow_pickle=True).item()
    mlat = geom["lat"]
    mlon = np.where(geom["lon"] > 180, geom["lon"] - 360, geom["lon"])
    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply_cond(mc, tc, stats, sae, bf16=True)
    apply_fn = fc.make_apply(params, rf, patched=True)

    res = {}
    for storm in STORMS:
        run = np.load(fc.ROOT / f"results/skill/{SRC}/run_{storm}.npy",
                      allow_pickle=True).item()
        cfg = S.STORMS[storm]
        clat, clon = run["center"]
        CONV = [int(f) for f in run["conv"]]
        RAND = [int(f) for f in run["rand"]]
        ft = np.zeros(sae.n_features, np.float32)
        for k, v in run["ftarget"].items():
            ft[int(k)] = v
        nmask = (gc_km(mlat, mlon, clat, clon) < S.RADIUS_KM).astype(np.float32)
        fsel_c = np.zeros(sae.n_features, np.float32); fsel_c[CONV] = 1.0
        fsel_r = np.zeros(sae.n_features, np.float32); fsel_r[RAND] = 1.0
        zF = np.zeros(sae.n_features, np.float32); zN = np.zeros(len(mlat), np.float32)

        # every arm is a 4-tuple so the jit traces ONE signature; gain 1.0 is an
        # exact no-op, which is what both the baseline and the post-release phase use
        gmax = max(GAINS)
        arms = {"baseline": (zF, zF, zN, np.float32(1.0), PW)}
        for g in GAINS:
            arms[f"conv-g{g:g}"] = (fsel_c, ft, nmask, np.float32(g), PW)
        arms[f"conv-g{gmax:g}-persist"] = (fsel_c, ft, nmask, np.float32(gmax), H)
        arms[f"rand-g{gmax:g}"] = (fsel_r, ft, nmask, np.float32(gmax), PW)

        inp, tgt, frc = build_io(cfg["ic"], tc, H)
        lat = np.asarray(tgt.lat.values, np.float64)
        lon = np.asarray(tgt.lon.values, np.float64)
        LO = np.where(lon > 180, lon - 360, lon)
        # distance of every GRID point from the storm centre -> the same 1500 km disk
        d_km = gc_km(np.repeat(lat[:, None], len(lon), 1).ravel(),
                     np.repeat(LO[None, :], len(lat), 0).ravel(),
                     clat, clon).reshape(len(lat), len(lon))
        inside = d_km <= S.RADIUS_KM
        wlat = np.cos(np.deg2rad(lat))[:, None]
        w_in = wlat * inside
        w_out = wlat * (~inside)
        print(f"[{storm}] disk covers {100*inside.mean():.2f}% of grid points, "
              f"{100*w_in.sum()/(wlat*np.ones_like(inside)).sum():.2f}% of area", flush=True)

        acc = {a: np.zeros((H, len(FIELDS), 3)) for a in arms}
        for aname, spec in arms.items():
            t0 = time.time()
            cur = inp
            fsel, ftg, nmk, gval, pw = spec
            pj_on = tuple(jnp.asarray(x) for x in (fsel, ftg, nmk, gval))
            pj_off = tuple(jnp.asarray(x) for x in (fsel, ftg, nmk, np.float32(1.0)))
            for s in range(H):
                pj = pj_on if s < pw else pj_off
                tg = tgt.isel(time=slice(s, s + 1)).assign_coords(time=tgt.time[:1])
                fr = frc.isel(time=slice(s, s + 1)).assign_coords(time=frc.time[:1])
                pr = numpyify(apply_fn(cur, tg, fr, pj)[0])   # apply returns (preds, acts)
                truth = tgt.isel(time=slice(s, s + 1))
                for fi, (nm, var, lev) in enumerate(FIELDS):
                    d = grab(pr, var, lev) - grab(truth, var, lev)
                    dd = d * d
                    acc[aname][s, fi, 0] = np.sqrt((wlat * dd).sum() / (wlat.sum() * d.shape[1]))
                    acc[aname][s, fi, 1] = np.sqrt((w_in * dd).sum() / max(w_in.sum(), 1e-9))
                    acc[aname][s, fi, 2] = np.sqrt((w_out * dd).sum() / max(w_out.sum(), 1e-9))
                cur = rollout._get_next_inputs(cur, xr.merge([pr, fr])).assign_coords(
                    time=inp.coords["time"])
            print(f"  [{aname}] done ({time.time()-t0:.0f}s)  msl@+96h "
                  f"in {acc[aname][15, 6, 1]:.3f} out {acc[aname][15, 6, 2]:.3f} | "
                  f"msl@+{6*H}h in {acc[aname][-1, 6, 1]:.3f} "
                  f"out {acc[aname][-1, 6, 2]:.3f}", flush=True)
        res[storm] = acc
        np.save(OUT, dict(res=res, fields=[f[0] for f in FIELDS], gains=GAINS,
                          H=H, split=["global", "inside", "outside"], src=SRC),
                allow_pickle=True)

    LEADS = [(3, "+24h"), (7, "+48h"), (15, "+96h"), (23, "+144h"),
             (31, "+192h"), (H - 1, f"+{6*H}h")]
    for storm, acc in res.items():
        print(f"\n{'='*78}\n{storm}  --  RMSE CHANGE vs BASELINE (negative = better)")
        for split, si in (("INSIDE the 1500 km patch disk", 1),
                          ("OUTSIDE it", 2), ("GLOBAL", 0)):
            print(f"\n  {split}   [msl, hPa-equivalent Pa]")
            print(f"    {'arm':<20}" + "".join(f"{lb:>10}" for _, lb in LEADS))
            for a in acc:
                if a == "baseline":
                    continue
                row = "".join(f"{acc[a][i, 6, si] - acc['baseline'][i, 6, si]:>+10.3f}"
                              for i, _ in LEADS)
                print(f"    {a:<20}{row}")
        print(f"\n  z500 (m), OUTSIDE the disk -- the cleanest downstream field")
        print(f"    {'arm':<20}" + "".join(f"{lb:>10}" for _, lb in LEADS))
        for a in acc:
            if a == "baseline":
                continue
            row = "".join(f"{acc[a][i, 0, 2] - acc['baseline'][i, 0, 2]:>+10.4f}"
                          for i, _ in LEADS)
            print(f"    {a:<20}{row}")
    print(f"\n-> {OUT.relative_to(fc.ROOT)}")

if __name__ == "__main__":
    main()
