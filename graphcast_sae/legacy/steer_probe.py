"""Causal steering atlas — impulse an SAE feature, watch GraphCast's world respond.

At t=0 we kick one feature (dose its decoder contribution) for a SINGLE step, then let the
model run free for H steps. Delta(t) = steered_rollout - baseline_rollout tells us the causal
response. For every feature we also run a matched-magnitude RANDOM-direction impulse as the
control. We record, per lead time, the response field (z500) plus three physics diagnostics:
  amp   rms|Δz500|                      -> does the kick grow into a real anomaly?
  loc   frac of Δ² energy <2500 km      -> is the response organized near the feature?
  east  signed zonal drift of |Δ| centroid (deg/day, midlat) -> does it propagate downstream?
The claim under test: feature impulses are coherent, localized, downstream-propagating physical
responses; matched random impulses are not.  bf16, JAX env.

  FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python \
     graphcast_sae/legacy/steer_probe.py --feats 28,1185,1555 --horizon 20 --start 2020-01-05

Paper: not in the paper; kept for provenance only
Inputs: candidates/fs_feature_catalog.npy (not shipped, see docs/REPRODUCE.md); GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/fs_steer_probe.npy (--out)
Run:   # JAX env, GPU (~46 GB)
    python -m graphcast_sae.legacy.steer_probe
"""
import argparse, os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np, jax, jax.numpy as jnp, xarray as xr, dataclasses
from scipy.ndimage import gaussian_filter
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout
from graphcast_sae.common.signature_physics import gc_km

def build_rollout_io(t0, H, tc):
    """inputs (2 frames) + targets_template + forcings for H six-hourly steps from real data."""
    blk, times, statics = fc.load_block(np.datetime64(t0), nframes=2 + H)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = statics[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*H}h"), **dataclasses.asdict(tc))

def z500(preds):
    return np.asarray(preds["geopotential"].isel(batch=0, time=0).sel(level=500).values) / 9.81

def numpyify(ds):
    """jax-backed xarray -> numpy-backed (so autoregressive concat/merge is version-safe)."""
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def rollout_fields(apply, sae, inp, tgt, frc, patch0, H):
    """Manual autoregressive rollout: impulse patch0 at step 0, noop after; z500 (H,lat,lon)."""
    noop = fc.noop_patch(sae)
    tct = tgt.time.isel(time=slice(0, 1))                     # single-step time coord
    for c in ("datetime",):
        if c in tgt.coords: tgt = tgt.drop_vars(c)
        if c in frc.coords: frc = frc.drop_vars(c)
    cur = inp; out = []
    for h in range(H):
        ctgt = tgt.isel(time=slice(h, h + 1)).assign_coords(time=tct)
        cfrc = frc.isel(time=slice(h, h + 1)).assign_coords(time=tct)
        preds = apply(cur, ctgt, cfrc, patch0 if h == 0 else noop)[0]
        preds = numpyify(preds)
        out.append(np.asarray(preds["geopotential"].isel(batch=0, time=0).sel(level=500).values) / 9.81)
        if h < H - 1:
            nxt = rollout._get_next_inputs(cur, xr.merge([preds, cfrc]))
            cur = nxt.assign_coords(time=cur.coords["time"])
    return np.stack(out)                                      # (H, lat, lon)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="28,1185,1555")
    ap.add_argument("--start", default="2020-01-05"); ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--dose-scale", type=float, default=1.0)
    ap.add_argument("--rand-decoder", type=int, default=-1)   # >=0: randomize W_dec (fair control)
    ap.add_argument("--out", default="results/fs_steer_probe.npy")
    args = ap.parse_args()
    feats = [int(x) for x in args.feats.split(",")]
    cat = np.load(fc.ROOT / "candidates/fs_feature_catalog.npy", allow_pickle=True).item()

    sae = fc.SAEJax()
    if args.rand_decoder >= 0:
        rr = np.random.default_rng(args.rand_decoder)
        Wr = rr.standard_normal((fc.D_IN, sae.n_features)).astype(np.float32)
        Wr /= np.linalg.norm(Wr, axis=0, keepdims=True)       # same magnitude, random direction
        sae.W_dec = jnp.asarray(Wr)
        print(f"[FAIR CONTROL] decoder randomized (seed {args.rand_decoder}); "
              f"same firing locations+magnitude, random 512-direction", flush=True)
    params, mc, tc, stats = fc.load_model()
    rf, _ = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    inp, tgt, frc = build_rollout_io(args.start, args.horizon, tc)

    # grid geometry for diagnostics
    lat = np.asarray(fc.load_block(np.datetime64(args.start))[0]["lat"].values)
    lon = np.asarray(fc.load_block(np.datetime64(args.start))[0]["lon"].values)
    LO, LA = np.meshgrid(lon, lat)
    coslat = np.cos(np.radians(LA))

    print(f"baseline rollout H={args.horizon} from {args.start}", flush=True)
    t0 = time.time()
    base = rollout_fields(apply, sae, inp, tgt, frc, fc.noop_patch(sae), args.horizon)
    print(f"  baseline done {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(0)
    rec = {"leads_h": (np.arange(args.horizon) + 1) * 6, "feats": feats,
           "start": args.start, "fields": {}, "diag": {}}
    for fi in feats:
        dose = float(cat["dose90"][fi]) * args.dose_scale
        clat, clon = float(cat["clat"][fi]), float(cat["clon"][fi])
        D = gc_km(LA.ravel(), LO.ravel(), clat, clon).reshape(LA.shape); near = D < 2500
        # matched-magnitude random direction impulse
        u = rng.standard_normal(fc.D_IN).astype(np.float32); u /= np.linalg.norm(u)
        randp = (jnp.zeros(sae.n_features, jnp.float32), jnp.float32(0.0), jnp.asarray(u * dose))
        for tag, patch in [("dose", fc.coef_patch(sae, [fi], dose)), ("rand", randp)]:
            tt = time.time()
            roll = rollout_fields(apply, sae, inp, tgt, frc, patch, args.horizon)
            dz = roll - base                                   # (H,lat,lon)
            amp, loc, east, coh = [], [], [], []
            for h in range(args.horizon):
                f = dz[h]; e2 = (f**2 * coslat)
                amp.append(np.sqrt(e2.mean()))
                loc.append(e2[near].sum() / (e2.sum() + 1e-12))
                m = np.abs(f) * coslat; midl = np.abs(LA) > 25
                east.append((LO[midl] * m[midl]).sum() / (m[midl].sum() + 1e-12))  # energy-weighted lon
                sm = gaussian_filter(f, sigma=8, mode="wrap")  # ~2deg smooth: wave survives, grid-noise dies
                coh.append((sm**2).sum() / ((f**2).sum() + 1e-12))
            rec["diag"][(fi, tag)] = dict(amp=np.array(amp), loc=np.array(loc), east=np.array(east),
                                          coh=np.array(coh), clat=clat, clon=clon, dose=dose)
            rec["fields"][(fi, tag)] = dz[[args.horizon//4, args.horizon//2, args.horizon-1]].astype(np.float32)
            print(f"  feat {fi} {tag}: amp {amp[0]:.2f}->{amp[-1]:.2f} gpm  "
                  f"loc {loc[0]:.2f}->{loc[-1]:.2f}  {time.time()-tt:.0f}s", flush=True)
    rec["lat"] = lat; rec["lon"] = lon
    np.save(fc.ROOT / args.out, rec, allow_pickle=True)
    print(f"-> {args.out}", flush=True)

if __name__ == "__main__":
    main()
