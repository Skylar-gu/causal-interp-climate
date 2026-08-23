"""Replicate MacMillan & Ouellette's TC steering (feature 3243) + the missing random control.

Their protocol: scale a feature's activation by (1+gamma), gamma in [-0.5,0.5], error-preserving
(add gamma * alpha_f * W_dec[:,f] to the layer-8 activation), and read the tropical-cyclone
intensity. We do exactly that (coef=gamma, rho=0 reproduces it), for the REAL decoder direction and
for a RANDOM unit direction at the same firing sites+magnitude. The sharp question their paper flags
but doesn't test ('true abstraction vs the model's self-repair'):
  * if REAL scaling changes TC intensity monotonically with gamma and RANDOM does NOT -> the learned
    DIRECTION is a genuine causal handle (paper right, my earlier gross-energy test was too blunt).
  * if RANDOM reproduces it -> self-repair / location-driven; the direction is not special.

Paper: not in the paper; kept for provenance only
Inputs: GraphCast params (GRAPHCAST_PARAMS)
Outputs: results/fs_steer_tc.npy
Run:   # JAX env, GPU (~46 GB)
    FS_DEVICE=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -m graphcast_sae.legacy.steer_tc
"""
import os, sys, time
os.environ["FS_DEVICE"] = "gpu"; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np, jax, jax.numpy as jnp, xarray as xr, dataclasses
import graphcast_sae.common.fs_common as fc
from graphcast import data_utils, rollout

FI = 3243
GAMMAS = [-0.5, -0.25, 0.25, 0.5]
CANDIDATE_ICS = ["2020-09-01", "2020-09-05", "2020-10-31", "2020-11-11",
                 "2019-10-10", "2018-09-10", "2017-09-06", "2013-11-07"]  # strong W-Pac typhoons
HORIZON = 8                                                              # 48 h
BOX = dict(lat=(3, 32), lon=(115, 168))                                 # W Pacific TC basin

def numpyify(ds):
    return xr.Dataset({v: (ds[v].dims, np.asarray(ds[v].values)) for v in ds.data_vars},
                      coords={k: ds.coords[k] for k in ds.coords})

def surf(preds, name):
    return np.asarray(preds[name].isel(batch=0, time=0).values)

def build_io(t0, H, tc):
    blk, times, statics = fc.load_block(np.datetime64(t0), nframes=2 + H)
    w = blk.assign_coords(time=(times - times[0]).astype("timedelta64[ns]"))
    for v in list(w.data_vars):
        if "time" in w[v].dims: w[v] = w[v].expand_dims("batch")
    for v in fc.STATIC_VARS: w[v] = statics[v]
    w = w.assign_coords(datetime=(("batch", "time"), times[None, :].astype("datetime64[ns]")))
    return data_utils.extract_inputs_targets_forcings(
        w, target_lead_times=slice("6h", f"{6*H}h"), **dataclasses.asdict(tc))

def rollout_surface(apply, sae, inp, tgt, frc, patch0, H, lat, lon):
    """rollout with a CONSTANT patch each step (sustained scaling, as in the paper); return
       per-lead (mslp, wspd) cropped to the TC box."""
    tct = tgt.time.isel(time=slice(0, 1))
    for c in ("datetime",):
        if c in tgt.coords: tgt = tgt.drop_vars(c)
        if c in frc.coords: frc = frc.drop_vars(c)
    la = (lat >= BOX["lat"][0]) & (lat <= BOX["lat"][1])
    lo = (lon >= BOX["lon"][0]) & (lon <= BOX["lon"][1])
    cur = inp; ms, ws = [], []
    for h in range(H):
        ctgt = tgt.isel(time=slice(h, h + 1)).assign_coords(time=tct)
        cfrc = frc.isel(time=slice(h, h + 1)).assign_coords(time=tct)
        preds = numpyify(apply(cur, ctgt, cfrc, patch0)[0])
        mslp = surf(preds, "mean_sea_level_pressure")
        u = surf(preds, "10m_u_component_of_wind"); v = surf(preds, "10m_v_component_of_wind")
        ms.append(mslp[np.ix_(la, lo)]); ws.append(np.sqrt(u**2 + v**2)[np.ix_(la, lo)])
        if h < H - 1:
            cur = rollout._get_next_inputs(cur, xr.merge([preds, cfrc])).assign_coords(time=cur.coords["time"])
    return np.stack(ms), np.stack(ws)                                    # (H, nlat_box, nlon_box)

def main():
    sae = fc.SAEJax()
    params, mc, tc, stats = fc.load_model()
    rf, cap = fc.build_apply(mc, tc, stats, sae=sae, bf16=True)
    apply = fc.make_apply(params, rf, patched=True)
    noop = fc.noop_patch(sae)

    # --- phase 1: pick the IC where feature 3243 fires hardest (a real, strong TC) ---
    print("scanning ICs for feature 3243 activation ...", flush=True)
    best = None
    for t0 in CANDIDATE_ICS:
        blk = fc.load_block(np.datetime64(t0), nframes=fc.INPUT_WINDOW)
        inp, tgt, frc = fc.build_batch_inputs([blk], 0, tc)
        _, acts = apply(inp, tgt, frc, noop)
        X = np.asarray(acts, np.float32).reshape(-1, fc.D_IN)
        code = np.asarray(sae.codes(jnp.asarray(X)))[:, FI]
        act = float(code.sum()); nfire = int((code > 0).sum())
        print(f"  {t0}: feat{FI} total={act:8.1f}  nodes_firing={nfire}", flush=True)
        if best is None or act > best[1]: best = (t0, act, nfire)
    ic = best[0]; print(f"-> chosen IC {ic} (feat activation {best[1]:.1f})\n", flush=True)

    lat = np.asarray(fc.load_block(np.datetime64(ic))[0]["lat"].values)
    lon = np.asarray(fc.load_block(np.datetime64(ic))[0]["lon"].values)
    inp, tgt, frc = build_io(ic, HORIZON, tc)

    # baseline TC center (box MSLP min at +24h)
    base_ms, base_ws = rollout_surface(apply, sae, inp, tgt, frc, noop, HORIZON, lat, lon)
    lead = 4                                                            # +24 h
    print(f"baseline @+{6*(lead+1)}h: MSLP_min={base_ms[lead].min()/100:.1f} hPa  "
          f"wind_max={base_ws[lead].max():.1f} m/s\n", flush=True)

    # random-direction control SAE: randomize ONLY feature FI's decoder column
    rr = np.random.default_rng(0); r = rr.standard_normal(fc.D_IN).astype(np.float32); r /= np.linalg.norm(r)
    Wd = np.array(sae.W_dec); Wd[:, FI] = r; sae_rand = fc.SAEJax(); sae_rand.W_dec = jnp.asarray(Wd)
    rf2, _ = fc.build_apply(mc, tc, stats, sae=sae_rand, bf16=True)
    apply_rand = fc.make_apply(params, rf2, patched=True)

    rec = {"ic": ic, "gammas": GAMMAS, "lead_h": 6*(lead+1),
           "base": dict(mslp=base_ms[lead].min()/100, wind=base_ws[lead].max())}
    print(f"{'gamma':>6} {'arm':>6} {'MSLP_min(hPa)':>14} {'d_hPa':>7} {'wind_max':>9} {'d_wind':>7}")
    for arm, ap, sobj in [("real", apply, sae), ("rand", apply_rand, sae_rand)]:
        for g in GAMMAS:
            patch = fc.coef_patch(sobj, [FI], g)                        # scale feature by (1+gamma)
            ms, ws = rollout_surface(ap, sobj, inp, tgt, frc, patch, HORIZON, lat, lon)
            dms = ms[lead].min()/100 - rec["base"]["mslp"]; dws = ws[lead].max() - rec["base"]["wind"]
            rec[(arm, g)] = dict(mslp=ms[lead].min()/100, wind=ws[lead].max(), dmslp=dms, dwind=dws)
            print(f"{g:>6.2f} {arm:>6} {ms[lead].min()/100:>14.1f} {dms:>+7.2f} {ws[lead].max():>9.1f} {dws:>+7.2f}", flush=True)

    # verdict: is REAL monotonic in gamma (stronger TC = lower MSLP for +gamma) and RANDOM not?
    def mono(arm):
        d = [rec[(arm, g)]["dmslp"] for g in GAMMAS]                    # expect decreasing (intensify) with gamma
        return np.corrcoef(GAMMAS, d)[0, 1]
    cr, cc = mono("real"), mono("rand")
    swing = lambda arm: abs(rec[(arm, GAMMAS[-1])]["dmslp"] - rec[(arm, GAMMAS[0])]["dmslp"])
    sr, sn = swing("real"), swing("rand"); ratio = sr / max(sn, 1e-6)
    print(f"\nMSLP-vs-gamma correlation: real={cr:+.2f}  random={cc:+.2f}")
    print(f"MSLP swing over gamma: real={sr:.1f} hPa  random={sn:.1f} hPa  ->  efficacy ratio {ratio:.1f}x")
    print("(negative correlation = +gamma intensifies the TC, the paper's claim)")
    # what matters is EFFECT SIZE, not just sign: a random dir can nudge the same way but far weaker
    if cr < -0.7 and ratio > 2.5:
        print(f"VERDICT: real scaling steers TC intensity monotonically and {ratio:.1f}x more strongly "
              "than a random direction at the same sites -> the learned DIRECTION is a genuine causal "
              "handle (replicates the paper; the random residual is the 'self-repair' floor).")
    elif cr < -0.7 and ratio <= 2.5:
        print("VERDICT: real and random intensify comparably -> location/self-repair dominates; "
              "the specific direction is not special here.")
    else:
        print("VERDICT: no clean monotonic TC steering at this IC/horizon; inconclusive, investigate.")
    np.save(fc.ROOT / "results/fs_steer_tc.npy", rec, allow_pickle=True)
    print("-> results/fs_steer_tc.npy")

if __name__ == "__main__":
    main()
